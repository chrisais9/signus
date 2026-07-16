# Blind repeated-preamble synchronization — design

**Date:** 2026-07-16
**Status:** approved (design), pending implementation
**Goal:** raise the decode rate of *hard* bursts (short / low-SNR / dense-constellation) that
carry a **generic repeated preamble**, by using autocorrelation-based blind synchronization to
estimate carrier-frequency offset (CFO), timing and phase from the preamble — succeeding where the
blind M-th-power carrier estimator fails because it has too few symbols to average.

## Motivation

signus is a **blind** demodulator: with no prior it estimates carrier/baud/mod from the signal
alone. This works on long records but fails on short bursts — the M-th-power carrier tone needs
many symbols to rise out of noise. The deep-QA pass documented these as *fundamental limits* for
**random-data** bursts (short 8psk/32qam/64qam carrier, pi/4-DQPSK branch ambiguity).

Real **unknown** signals (the primary mission: catching unidentified emitters) are almost never
random — they are packetized and begin with a **preamble**: a short block repeated several times so
a receiver can lock on. That repetition is detectable and exploitable *without knowing its content*:
the delay-and-correlate (Schmidl–Cox) metric peaks where two period-separated windows match, and the
phase of that correlation is a low-variance CFO estimate that works from only a few symbols.

**Key framing:** this is not "repair the random-data short-burst limit" (information-theoretically
impossible) — it is "**add support for realistic packetized signals**", which is exactly what the
unknown-signal mission needs.

## Non-goals (YAGNI — separate roadmap items)

- Protocol-specific preamble libraries (LoRa/AIS/BLE known sequences). This design is **generic**
  (repetition only, no definitions). A known-sequence matched filter is a later, opt-in add-on.
- Streaming / online sync. Batch only, matching the current pipeline.
- Full message framing / de-whitening / FEC. We recover **symbols/bits**, not application payloads.

## Architecture — three pieces

### 1. Generator extension (`gen.py`) — realistic packet fixtures

Today the generator emits a random symbol stream through a realistic *channel* (multipath `taps`,
`drift_ppm`, `dc`/LO-leakage, `timing`, `snr`, `pad`). Real emissions also have realistic *content*:
a repeated preamble, a fixed sync word, and a structured (non-random) payload. We add both so a
fixture looks like an actual packet on an actual channel.

Add to `GenParams`:

```python
preamble: tuple[int, int] = (0, 0)   # (block_len_syms L, repeats R); (0,0) = none (current behaviour)
sync_word: tuple = ()                # optional fixed symbol-index pattern after the preamble (e.g. a marker)
payload: object = None               # None = random data (current); or explicit bits/pattern to transmit
```

Packet layout on the wire (before shaping & channel):

```
[ preamble: L-symbol block × R ]  [ sync_word ]  [ payload (explicit) or random data ]
```

- **preamble** — an `L`-symbol base block **tiled `R` times**. The base block uses a **dedicated,
  seed-derived RNG stream**, so it is a fixed pseudo-random training sequence identical across
  repeats (what `sync.find_preamble` locks onto). Total preamble symbols = `L * R`.
- **sync_word** — an explicit tuple of symbol indices placed right after the preamble (models the
  fixed marker real frames use to delimit "data starts here"). Empty = omitted.
- **payload** — lets a fixture carry *specific bits/patterns* instead of random data: accept either a
  bit array/sequence (packed to symbols for the mod) or a symbol-index sequence. `None` keeps today's
  random behaviour. This is the "특정 비트/패턴" the fixtures should be able to embed.
- `generate()` returns `(x, data_bits)` — **payload/data bits only** (preamble + sync_word are
  overhead, excluded from the ground-truth bits). The sample stream is `[preamble][sync][data]`.
- All three default to none/empty ⇒ **byte-identical** to today's output (guarded by the sweep).

Differential mods: the phase `cumsum` runs continuously through preamble → sync → data, so the data
transitions decode correctly from the boundary.

These structured-content knobs compose with the **existing channel realism** (`taps` multipath,
`drift_ppm`, `dc`, `timing`, `snr`, `pad`) — a fixture can be, e.g., "a 64qam packet with an 8×
repeated preamble and a fixed sync word, sent through a 2-ray channel at SNR 18 with LO leakage".

### 2. Blind sync module (`sync.py`, new)

```python
@dataclass
class Preamble:
    start: int          # sample index where the repeated region begins
    end: int            # sample index where it ends (data starts ~here)
    period: int         # repetition period in samples (= L * sps)
    cfo_hz: float       # carrier-frequency offset estimated from the preamble
    conf: float         # 0..1 plateau confidence

def find_preamble(x, fs, baud_hint=None) -> Preamble | None
```

Algorithm (delay-and-correlate / Schmidl–Cox):

- For a candidate period `P`: `Pmet(d) = Σ_{k<W} conj(x[d+k]) · x[d+k+P]`, energy
  `E(d) = Σ_{k<W} |x[d+k+P]|²`, metric `M(d) = |Pmet(d)|² / E(d)²  ∈ [0,1]`. `M≈1` where two
  `P`-separated windows are identical → inside the repeated preamble. Window `W ≈ P`.
- The preamble yields a **plateau** of `M(d)` of length `≈ (R-1)·P`; its extent gives `start/end`
  and `conf` (plateau length / P, and height).
- **CFO** `= angle(Pmet(d*)) · fs / (2π · P)` at the plateau centre `d*` — coherent, low-variance.
- Candidate `P`: with `baud_hint`, `sps = fs/baud`, try `P = round(k·sps)` for small `k`
  (1,2,3,4,6,8); pick the strongest plateau. Fallback: coarse lag scan of the normalized
  autocorrelation. Thresholds (`conf` floor, plateau height) are **calibrated empirically** the way
  the chirp `sweeps_band` discriminator was (a labelled preamble-vs-no-preamble grid).

Guardrails: `x.size < _MIN`, no plateau, or CFO beyond ±fs/(2P) → return `None`.

### 3. Pipeline integration (`pipeline.py`) — gated rescue, keep-best

Mirror the existing short-burst / equalizer rescue pattern:

```python
if q.lock < _SYNC_LOCK:                       # weak after the normal blind chain
    ps = find_preamble(xb, fs, baud_hint=baud)
    if ps is not None and ps.conf >= _SYNC_CONF:
        # correct carrier with the preamble CFO, sync timing, re-demod the DATA span
        t = _demod(dsp.mix(xb, fs, fc + ps.cfo)[ps.end:], fs, baud, symmetry)
        if t.lock > q.lock + _SYNC_GAIN:      # adopt ONLY if clearly better
            adopt(t); sync_applied = True
```

- **Keep-best-lock** ⇒ a signal with no preamble (every current test/sweep case) never improves here
  and is discarded → **CORE+STRETCH byte-identical**.
- Expose `Result.preamble` (start/period/repeats/cfo/conf) and surface it in `to_json()` — a bonus
  for the survey/characterization use case (an analyst sees "structured, R× repeated preamble").

## Data flow

```
detect/burst → est_carrier → est_baud → demod → [short-burst rescue] → [eq rescue]
                                                        │ lock still low?
                                                        ▼
                                          find_preamble(xb, fs, baud)  ── none ──▶ report as-is
                                                        │ preamble + conf ok
                                                        ▼
                                   mix by CFO, drop preamble, re-demod DATA → keep best lock
```

## Testing

- **New (positive):** *realistic* packetized short 8psk/32qam/64qam/pi4dqpsk — `preamble=(L,R)` +
  `sync_word` + (some with an explicit `payload`), sent through real-channel impairments (`taps`
  multipath, `drift_ppm`, `dc`, moderate SNR) → the sync rescue recovers them (BER < ~0.02) where
  the no-preamble twin fails. Include an explicit-payload case to verify a known pattern round-trips.
- **Byte-identical:** full CORE+STRETCH sweep unchanged (no-preamble signals skip the rescue).
- **No false-trigger:** pure noise, unmodulated tone, and no-preamble bursts → `find_preamble`
  returns `None` (or the rescue is discarded by keep-best); no confident-wrong is introduced.
- **CFO accuracy:** estimated CFO within a few % of truth across the preamble grid.

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| False preamble on self-correlated data / chirps | conf floor + keep-best (a non-improving rescue is dropped) |
| CFO aliasing (±fs/2P range) | combine with the coarse blind carrier; small `P` widens range |
| Threshold brittleness | empirical calibration on a labelled grid, margin both sides (as with `sweeps_band`) |
| Generator drift from current output | `preamble=(0,0)` default is byte-identical; guarded by the sweep |

## Success criteria

1. Packetized short bursts that today fail (lock<60 / wrong) decode correctly via the preamble rescue.
2. Zero regression: CORE+STRETCH sweep byte-identical; 290 existing tests still pass; ruff clean.
3. No new confident-wrong or crash on noise/no-preamble inputs (verified by an adversarial pass).
