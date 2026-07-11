# Wideband detection + channelization front-end — design

**Status:** approved (design), 2026-07-11. Branch `wideband-frontend`.
**Scope (sub-project 1 + 2):** turn signus from a single-signal demodulator into a wideband
*survey* tool: detect every emitter in a capture, extract each to its own baseband channel, decide
whether it is digitally demodulable, and demodulate the digital ones with the existing engine —
returning a **list** of per-emitter results. Includes the non-digital ("analog / unknown") verdict.
**Out of scope (later sub-projects):** GB-scale streaming I/O (3), short-burst/data-aided
estimation (4), framing/message decode e.g. AIS (5), web-UI overhaul (6), absolute RF centre
frequency reporting (7). Captures assumed to fit in RAM (tens–hundreds of MB).

## Why (evidence)

Empirically verified (`scratchpad/wideband_audit.py`): a capture of 3 emitters summed at one
fs=1e6 — A qpsk 25 kBd @−300 kHz (+6 dB), B 16qam 50 kBd @+180 kHz (0 dB), C msk 9.6 kBd @+350 kHz
(−6 dB) — run through the current `analyze()` returns ONE confidently-wrong answer (fc=−300 k,
mod=qpsk, **lock=100, MER=30 dB**), locking the loudest emitter and **silently dropping B and C**.
Worse, a **single** narrowband emitter alone in a wide capture already fails (emitter C alone →
bpsk, baud 303 kHz, lock 0): the trigger is signal-fraction-of-band, not emitter collision.

29 breakages were confirmed by adversarial verification (17 fatal). All share one root cause: every
estimator is **winner-take-all on the whole sampled band** — `est_carrier` (one M-th-power argmax),
`est_baud` (one |x|² argmax, plus carrier-difference beat notes as phantom lines), `fsk_gate`
(whole-band envelope CV inflated to Rayleigh 0.52 by *out-of-channel* noise), `occupied_bw` (returns
≈fs for any narrowband signal, disabling the baud sanity guard), `find_bursts` (whole-band total
power ≈ constant → one burst). There is no channel concept anywhere in the data model.

The fix direction is **proven necessary and sufficient**: mix each emitter to baseband →
LPF/decimate so it occupies ~1/4–1/3 of the reduced band → run the **unmodified** pipeline →
all three recover (qpsk/16qam/msk, correct baud, lock 100/83/100). So the engine is correct; only a
front-end is missing.

## Architecture

```
capture x[n], Meta(fs, …)
   │
   ▼  detect.detect(x, fs)                         [NEW  signus/detect.py]
   │     STFT (Blackman-Harris) → power spectrogram
   │     per-frequency noise floor (OS-CFAR along frequency, robust to occupied neighbours)
   │     threshold → binary TF mask
   │     ndimage.binary_closing (anisotropic) + binary_opening → label → find_objects
   │     merge freq-overlapping blobs, reject spurs (area/bandwidth/duration/fill, DC/image)
   │     per-blob robust fc (floor-subtracted centroid) + bandwidth (99%-power) + time extent
   ▼  → List[Detection]  {fc, bw, t0, t1, power_db, snr_db}
   │
   ▼  for each Detection:
   │     channelize.extract(x, fs, det)            [NEW  signus/channelize.py]
   │        mix to det.fc → FIR LPF → decimate (D sized so bw ≈ fs_ch/4..fs_ch/3)
   │        → (ch_iq, fs_ch)
   │     triage.family(ch_iq, fs_ch)               [NEW  signus/triage.py]
   │        digital?  → run existing pipeline.analyze(ch_iq, Meta(fs_ch,'iq')) UNCHANGED
   │        analog-fm / cw-tone / pulsed / unknown → record a non-digital entry, DO NOT demod
   ▼
   Survey  {captures Meta, detections, per-emitter results (linear/fsk) + non-digital entries}
      → to_json (list), CLI `signus survey`, batch table
```

**Invariant:** `pipeline.analyze()` and every DSP/classify/fsk/eq function are unchanged. The
existing 53 CORE / 233 pytest cases keep passing because "one emitter fills the band" is the special
case `detect` returns as a single full-band detection whose extracted channel ≈ the original signal.

## New modules

### `signus/detect.py`
- `@dataclass Detection`: `fc, bw, t0, t1` (Hz / sample indices), `power_db`, `snr_db`, plus a
  `baud_hint` derived from `bw` via signus's own B/Rs table (α 0.05→1.09 … 0.5→1.37; use ~1.2 mid).
- `detect(x, fs, *, nfft=4096, ...) -> list[Detection]`, time-ordered then strongest-first option.
- Stages, all numpy + scipy.signal/scipy.ndimage only:
  1. **Spectrogram** `scipy.signal.stft`, window `get_window('blackmanharris', nfft)` (−92 dB
     sidelobes → weak neighbour not buried by a strong emitter's leakage), 50–75 % overlap. Power
     `|STFT|²`.
  2. **Noise floor** per frequency bin: **OS-CFAR along the frequency axis** (sort the N=24–32
     training cells each side, G=1–2 guard, take the k=0.75 N order statistic; α = solved for
     Pfa≈1e-4). OS not CA because a second emitter inside the training window wrecks a mean but not
     the 75th percentile — essential in a congested band. (Cheaper fallback documented: per-bin
     median-over-time ×/ln2, valid when occupancy < 50 %.)
  3. **Threshold** `P > floor·α` → binary mask.
  4. **Morphology** `ndimage.binary_opening` (kill single-cell false alarms) then
     `binary_closing` with an **anisotropic** element (wider in time than frequency) so one emitter
     keying on/off is one blob, without bridging a frequency guard band.
  5. **Label + bbox** `ndimage.label` (8-connectivity) → `find_objects`; merge blobs whose
     frequency extents overlap and time gap is small.
  6. **Reject** spurs: min area / min bandwidth (≥ a few bins) / min duration / fill-factor; DC-bin
     (LO leakage) and IQ-image mirrors (weaker twin at −f about DC with identical time structure).
  7. **Estimate** per surviving blob: `fc` = floor-subtracted power-weighted centroid of the
     time-averaged PSD slice; `bw` = 99 %-power bandwidth (cross-checked against 26-dB-down);
     `t0,t1` from the blob time extent.

### `signus/channelize.py`
- `extract(x, fs, det, *, target_frac=0.28, min_taps=64) -> tuple[np.ndarray, float]`:
  - mix: reuse `dsp.mix(x, fs, det.fc)`.
  - decimation `D = clip(floor(fs / (det.bw / target_frac)), 1, ...)` so the signal lands at
    ~1/4–1/3 of the reduced rate `fs_ch = fs/D` — the empirically-validated sweet spot (too tight
    and `fsk_gate` misfires the other way; too wide and noise dominates the discriminator).
  - LPF: `scipy.signal.firwin` sized to passband ≈ 1.15·bw, stopband at `fs_ch/2`; apply then
    `scipy.signal.resample_poly` (handles non-integer D). Trim to `det.t0:det.t1` (± guard) so a
    bursty channel isn't diluted by empty time.
  - returns baseband complex IQ and its `fs_ch`.

### `signus/triage.py`
- `family(ch, fs_ch) -> str` in `{'linear','fsk','analog','tone','pulsed','unknown'}`.
  - Reuse the **existing** `fsk.fsk_gate` — but now on the extracted channel, which *fixes the
    whole-band CV inflation for free*.
  - `linear` when an M-th-power carrier tone exists (`est_carrier` peak-to-mean above a floor) AND a
    cyclostationary baud line exists (`est_baud` confidence above a floor).
  - `analog` (e.g. FM voice): constant-ish envelope but no bimodal IF and no baud line → **the
    escape hatch the current linear path lacks** (today it force-fits a PSK/QAM constellation).
  - `tone` (CW/spur): ~zero bandwidth, no modulation. `pulsed` (radar): low duty cycle / PRF comb.
    `unknown`: survives detection but matches nothing → reported, not demodulated.

### `signus/pipeline.py` — add `survey()`
- `survey(x, meta, *, diff=False) -> Survey`: run `detect` → per detection `extract` → `triage` →
  digital ⇒ existing `analyze(ch_iq, Meta(fs_ch,'iq',...))`; non-digital ⇒ a lightweight entry.
- `@dataclass Survey`: `meta`, `detections: list[Detection]`, `emitters: list[Result]` (each tagged
  with its `Detection` and channel fs), `nondigital: list[dict]`. `to_json` returns a **list-shaped**
  report (top-level detections + per-emitter blocks), reusing `Result.to_json` per emitter.
- `analyze()` stays as the single-signal entry point (backward compatible, keeps CORE green).

## CLI / IO
- `signus survey FILE [--strongest N] [--json out.json]`: prints a per-emitter table
  (idx │ fc │ bw │ class │ mod │ baud │ lock), one row per detection. `analyze` unchanged.
- No server/web changes in this spec (UI is sub-project 6). The Python `survey()` + JSON is the
  deliverable surface.
- **Bug fix (in scope):** `dsp.find_burst` docstring says "most energetic" but returns
  `min(..., key=energy)` = least energetic. Fix to `max`; adjust the test that currently passes
  only because it never checks strength.

## Testing (TDD)
1. **Fixture helper** in tests: `wideband(*specs)` sums `gen.generate` outputs at distinct fc into
   one array at a shared fs (+ one AWGN floor), returns the array and the ground-truth list.
2. **Detection** finds the right *count* and each `fc`/`bw` within tolerance; ordered; rejects a
   pure-noise capture (0 detections) and a DC spur.
3. **Extraction + recovery**: for the 3-emitter fixture, `survey` returns 3 emitters with correct
   `mod`/`baud` and lock above threshold (the empirical proof, now locked as a regression).
4. **Analog escape**: a wideband-FM / unmodulated-tone channel is classed `analog`/`tone`, NOT
   force-fit to a constellation.
5. **Single-signal regression**: every existing CORE case, when run through `survey`, yields exactly
   one detection whose emitter result matches today's `analyze` output (mod/baud/lock within tol).
   The 233 existing tests continue to pass unchanged (proves the engine was untouched).
6. **Robustness**: unequal powers (loud neighbour must not hide a weak one — Blackman-Harris +
   OS-CFAR), adjacent channels within ~2·bw (must separate), a bursty emitter under an always-on one
   (time-frequency detection finds both).

## Risks / decisions
- **Always-on vs bursty floor:** per-bin median-over-time sits *on* an always-on carrier; OS-CFAR
  *along frequency* uses noise neighbours instead, so it detects both. Chosen as primary.
- **Dense uniform bands** (many equal channels) would favour a polyphase channelizer; deferred —
  mix+decimate per detection is right for the sparse/heterogeneous maritime case and the memory-fits
  scope. Documented as a future path.
- **`nfft` blind sizing:** fixed default (4096) with morphology tolerance; a signal narrower than a
  few bins is logged as possibly under-resolved rather than silently split. No silent caps.
