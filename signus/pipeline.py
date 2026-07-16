"""End-to-end blind demodulation. Two families:
  linear (PSK/QAM): front-end -> carrier -> baud -> matched filter -> timing ->
                    classify -> fine sync -> [equalizer rescue] -> quality
  fsk (CPFSK/MSK):  frequency discriminator (gated by constant envelope)
Classification runs BEFORE fine sync so the decision-directed loop knows the
constellation. Differential demapping is an option, not a detection: dqpsk and
qpsk share a constellation."""

import json
from dataclasses import dataclass, field

import numpy as np

from . import classify as cl
from . import dsp, triage
from .channelize import extract
from .chirp import analyze_chirp
from .constellations import demap_bits, demap_diff_bits, mod_order
from .detect import Detection, detect
from .eq import equalize, equalize_fse
from .fsk import analyze_fsk, fsk_gate
from .sigio import Meta, parse_name, read
from .spectrum import spectrum, waterfall
from .sync import find_preamble

_BITS_CAP = 65536
_EQ_LOCK = 60.0    # below this a linear signal is worth an equalizer attempt
_EQ_GAIN = 3.0     # ...keep it only if lock improves by this much
_EQ_FLOOR = 50.0   # ...and clears this floor, so a wrong-mod fit is never locked in
_BAUD_GAIN = 5.0   # an in-band baud hypothesis must beat the global one by this lock
_SHORT_LOCK = 60.0  # below this, retry the baud line with fewer blocks (short-burst rescue)
_SYNC_LOCK = 60.0   # below this, try a repeated-preamble carrier/timing sync (packetized bursts)
_ALIAS_MARGIN = 8.0  # a preamble carrier alias must beat the next by this lock, else it's ambiguous
_DIFF_OF = {"bpsk": "dbpsk", "qpsk": "dqpsk"}  # pi4dqpsk arrives as qpsk -> dqpsk


@dataclass
class Result:
    meta: Meta
    family: str                    # 'linear' | 'fsk'
    burst: tuple[int, int]
    fc: float
    baud: float
    mod: str
    lock: float
    symbols: np.ndarray            # aligned symbols (linear) / normalized levels (fsk)
    bits: np.ndarray
    iq_corr: np.ndarray            # exportable corrected baseband
    burst_x: np.ndarray = field(repr=False, default=None)  # for spectrum views
    symmetry: int = 0
    carrier_ambiguous: bool = False
    baud_conf: float = 0.0
    rolloff: float | None = None
    h: float | None = None         # FSK modulation index
    evm: float = float("nan")
    mer_db: float = float("nan")
    occupied: int = 0
    snr_est: float = float("nan")
    eq_applied: bool = False
    eq_mode: str | None = None     # 'sym' | 'fse'
    alias_resolved: bool = False
    baud_fallback: bool = False
    bursts: list = field(default_factory=list)
    burst_idx: int = 0
    preamble: object = None        # sync.Preamble when a repeated preamble drove the decode

    def to_json(self, max_points: int = 6000, views: bool = True) -> dict:
        z = np.nan_to_num(self.symbols)   # a dead/DC capture -> NaN symbols -> invalid JSON
        if z.size > max_points:  # keep TIME ORDER: the UI animates this sequence
            z = z[np.linspace(0, z.size - 1, max_points).astype(int)]
        det = {"fc": round(self.fc, 3), "baud": round(self.baud, 3), "mod": self.mod,
               "symmetry": self.symmetry, "carrier_ambiguous": self.carrier_ambiguous,
               "baud_conf": round(self.baud_conf, 1),
               "alias_resolved": self.alias_resolved,
               "baud_fallback": self.baud_fallback}
        det["rolloff"] = None if self.rolloff is None else round(self.rolloff, 3)
        rf = self.meta.rf_center                      # real RF = capture centre + baseband fc
        det["rf_hz"] = None if rf is None else round(rf + self.fc, 3)
        if self.h is not None:
            det["h"] = round(self.h, 3)
        if self.preamble is not None:                 # a repeated preamble drove the sync
            p = self.preamble
            det["preamble"] = {"period": p.period, "cfo_hz": round(p.cfo_hz, 1),
                               "conf": round(p.conf, 2), "start": p.start, "end": p.end}
        doc = {
            "fs": self.meta.fs, "fmt": self.meta.fmt, "rf_center": self.meta.rf_center,
            "family": self.family, "n_samples": int(self.iq_corr.size),
            "burst": {"start": self.burst[0], "end": self.burst[1]},
            "bursts": [{"start": s, "end": e} for s, e in self.bursts],
            "burst_idx": self.burst_idx,
            "detected": det,
            "quality": {"lock": _r(self.lock, 1) or 0.0, "mer_db": _r(self.mer_db),
                        "evm": _r(self.evm, 4)},
            "snr_est_db": _r(self.snr_est),
            "eq": {"applied": self.eq_applied, "mode": self.eq_mode},
            "constellation": {"i": np.round(z.real, 4).tolist(),
                              "q": np.round(z.imag, 4).tolist()},
            "bits": "".join(map(str, self.bits[:_BITS_CAP])),
        }
        if views and self.burst_x is not None:
            doc["spectrum"] = spectrum(self.burst_x, self.meta.fs)
            doc["waterfall"] = waterfall(self.burst_x, self.meta.fs)
        return doc

    def save_iq(self, path: str) -> None:
        np.column_stack([self.iq_corr.real, self.iq_corr.imag]).astype("<f4").tofile(path)

    def save_symbols(self, path: str) -> None:
        np.save(path, self.symbols.astype(np.complex64))

    def save_bits(self, path: str, packed: bool = False) -> None:
        if packed:
            np.packbits(self.bits).tofile(path)
        else:
            with open(path, "w") as fh:
                fh.write("".join(map(str, self.bits)))

    def save_report(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(self.to_json(), fh, indent=1)


def _r(v: float, nd: int = 2) -> float | None:
    return None if v is None or not np.isfinite(v) else round(float(v), nd)


def _fine(z: np.ndarray, mod: str) -> np.ndarray:
    """Bulk-phase removal, decision-directed carrier loop, final rotation lock."""
    return cl.align(dsp.ddsync(cl.align(z, mod), mod), mod)


def _blocks(n: int) -> int:
    """M-th-power spectra are averaged INCOHERENTLY, so a weak tone (high symmetry,
    low SNR) wants fewer, longer segments; only long records need more for
    stationarity. Worth ~2 dB at the 8PSK edge over a fixed blocks=4."""
    return int(np.clip(n // 30000, 1, 8))


def _demod(xd: np.ndarray, fs: float, baud: float, symmetry: int) -> tuple:
    """Matched filter + timing + classify + fine sync at one baud hypothesis."""
    alpha = dsp.est_rolloff(xd, fs, baud)
    ym = dsp.matched(dsp.to_sps(xd, fs, baud), 4, alpha)
    raw = dsp.timing(ym, 4)
    mod = cl.classify(raw, symmetry)
    # align BEFORE ddsync: the coarse carrier leaves only micro-Hz residual, so removing
    # the bulk phase first kills the loop transient that would corrupt the first symbols
    syms = _fine(raw, mod)
    return alpha, ym, raw, mod, syms, cl.quality(syms, mod)


def _rescue(eq_fn, z: np.ndarray, seed_mod: str, symmetry: int) -> tuple:
    """Equalize with a seed constellation, then let the OPENED eye pick the mod:
    heavy ISI corrupts the ring test and even the 2/8 symmetry vote, so re-classify
    with the estimated symmetry AND the neutral order-4 gate, keep the best lock."""
    z1 = eq_fn(z, seed_mod)
    cands = []
    for m in {cl.classify(z1, symmetry), cl.classify(z1, 4)}:   # symmetry too, per the docstring
        s = _fine(z1 if m == seed_mod else eq_fn(z, m), m)
        cands.append((cl.quality(s, m), s, m))
    best = max(cands, key=lambda c: c[0].lock)
    # Occam de-fold for the PSK point-doubling: CMA is modulus-only, so a bpsk signal under a
    # multipath echo ALSO fits a qpsk ring at a marginally higher lock (phantom 2->4 points).
    # A bpsk candidate that clears the accept floor cannot be an over-fit of qpsk, so prefer it.
    # Only fires when symmetry==2 put bpsk in the set; a genuine qpsk reads symmetry 4 -> the
    # candidate set is the qpsk singleton -> this branch is unreachable and it stays untouched.
    lo = [c for c in cands if c[2] == "bpsk" and c[0].lock >= _EQ_FLOOR]
    if lo and best[2] == "qpsk":
        return lo[0]
    return best


def analyze(x: np.ndarray, meta: Meta, diff: bool = False,
            burst: int | None = None) -> Result:
    """Run the blind chain. `diff` demaps phase transitions (D-BPSK/D-QPSK);
    `burst` selects one detected burst (default: the most energetic)."""
    x = dsp.analytic(x)
    x = x - x.mean()  # DC block: LO leakage otherwise corrupts every estimator
    if x.size < 64:   # empty / trivially short: nothing to estimate, and it crashes downstream
        raise ValueError(f"신호가 너무 짧습니다 ({x.size} 샘플)")
    bursts = dsp.find_bursts(x, meta.fs)
    if burst is None or not 0 <= burst < len(bursts):
        burst = int(np.argmax([np.sum(np.abs(x[s:e]) ** 2) for s, e in bursts]))
    s, e = bursts[burst]
    xb = x[s:e] if e - s >= 64 else x

    if fsk_gate(xb, meta.fs):
        r = analyze_fsk(xb, meta.fs)
        sym = np.asarray(r["symbols"], dtype=np.complex128)
        return Result(meta, "fsk", (s, e), r["fc"], r["baud"], r["mod"], r["lock"],
                      sym, r["bits"], xb, burst_x=xb, h=r["h"],
                      bursts=bursts, burst_idx=burst)

    fc0, symmetry, _ = dsp.est_carrier(xb, meta.fs, blocks=_blocks(xb.size))
    fc = dsp.resolve_alias(xb, meta.fs, fc0, symmetry)
    amb = abs(symmetry * fc) > 0.4 * meta.fs
    xd = dsp.mix(xb, meta.fs, fc)

    baud, conf = dsp.est_baud(xd, meta.fs)
    fell = False
    alpha, ym, raw, mod, syms, q = _demod(xd, meta.fs, baud, symmetry)
    bw = dsp.occupied_bw(xd, meta.fs)
    if bw and not (0.72 * bw <= baud <= 1.05 * bw):
        # the global |x|^2 peak disagrees with the occupied band: below alpha~0.1
        # (and under harsh channels) data junk outruns the true line. Demod the
        # in-band hypotheses too and let lock decide, with a clear margin.
        for lo in (0.80, 0.67):
            b2, c2 = dsp.est_baud(xd, meta.fs, lo=lo * bw, hi=1.02 * bw)
            if abs(b2 - baud) <= 0.01 * b2:
                continue
            t = _demod(xd, meta.fs, b2, symmetry)
            if t[5].lock > q.lock + _BAUD_GAIN:
                alpha, ym, raw, mod, syms, q = t
                baud, conf, fell = b2, c2, True

    if q.lock < _SHORT_LOCK:
        # short / low-SNR burst rescue: the default 4-block |x|^2 line splits the record
        # too finely and locks a junk peak. Fewer, longer blocks give a stronger line (the
        # carrier _blocks logic, applied to baud). Run both and keep only a clearly-better
        # lock -- so a long, already-locked signal (CORE) never enters here and is untouched.
        for nb in (2, 1):
            b2, c2 = dsp.est_baud(xd, meta.fs, blocks=nb)
            if abs(b2 - baud) <= 0.01 * b2:
                continue
            t = _demod(xd, meta.fs, b2, symmetry)
            if t[5].lock > q.lock + _BAUD_GAIN:
                alpha, ym, raw, mod, syms, q = t
                baud, conf, fell = b2, c2, True

    eq_applied, eq_mode, pre = False, None, None
    if q.lock < _EQ_LOCK:  # ISI (multipath) is what a phase-only loop cannot fix
        # CMA is modulus-only, so it opens the eye without knowing the constellation.
        qe, se, me = _rescue(lambda z, m: equalize(z, m), raw, mod, symmetry)
        mode = "sym"
        if qe.lock < max(_EQ_FLOOR, q.lock + _EQ_GAIN):
            # symbol-spaced FIR could not invert the channel; retry T/2-spaced on
            # the clock-tracked 2 sps stream (unaliased spectrum, longer inverse)
            q2, s2, m2 = _rescue(lambda z, m: equalize_fse(z, m),
                                 dsp.timing(ym, 4, out=2), mod, symmetry)
            if q2.lock > qe.lock:
                qe, se, me, mode = q2, s2, m2, "fse"
        if qe.lock > q.lock + _EQ_GAIN and qe.lock >= _EQ_FLOOR:
            syms, mod, q, eq_applied, eq_mode = se, me, qe, True, mode
    elif mod in ("16qam", "32qam", "64qam"):
        # confident square-QAM at high lock: a benign post-echo can fold a LOWER-order signal
        # onto a QAM lattice with a clean-looking eye, so the low-lock rescue above never fires.
        # Equalize once and accept ONLY if a strictly lower order emerges -- verified never to
        # demote a genuine QAM (0/36 across 16/32/64qam x snr x seeds), so real QAM is untouched.
        qe, se, me = _rescue(lambda z, m: equalize(z, m), raw, mod, symmetry)
        mode = "sym"
        if mod_order(me) < mod_order(mod) and qe.lock < _EQ_FLOOR:
            # a lower order emerged but the symbol-spaced eye stayed shut (echo > ~2 symbols);
            # retry T/2 fractionally-spaced. Only runs once a de-fold is INDICATED, so a genuine
            # QAM (no lower order) never pays for the FSE.
            qe, se, me = _rescue(lambda z, m: equalize_fse(z, m),
                                 dsp.timing(ym, 4, out=2), mod, symmetry)
            mode = "fse"
        if mod_order(me) < mod_order(mod) and qe.lock >= _EQ_FLOOR:
            syms, mod, q, eq_applied, eq_mode = se, me, qe, True, mode

    if q.lock < _SYNC_LOCK:
        # packetized-burst rescue -- LAST, after the equalizer: a repeated preamble pins the
        # carrier/baud/timing from only a few symbols, where the blind M-th-power estimator (needing
        # many symbols to average) fails. Runs only if the eq rescue above did NOT lift the lock, so
        # a multipath signal (which the eq handles, and whose ISI can fake a short period) never
        # reaches here. Detect the preamble, mix by its CFO, demod the DATA past it, keep-best -- a
        # random / no-preamble signal yields no plateau or a CFO that does not improve lock and is
        # dropped (the sweep stays byte-identical). The preamble period P = L*sps supplies the baud
        # (fs*L/P per block length L), the phase slope gives the carrier (+-fs/P resolves the L-fold
        # alias); every PSK/QAM symmetry is tried -- the right (carrier, baud, sym) locks clean.
        ps = find_preamble(xb, meta.fs, baud_hint=baud)
        if ps is not None:
            # The Schmidl-Cox CFO is ambiguous modulo fs/P = baud/L: an M-PSK carrier off by baud/L
            # rotates a whole constellation position per symbol, so the eye still looks locked but
            # the bits shift -> a confident WRONG decode if the wrong alias is trusted. The coarse
            # M-th-power fc lands within ~1 alias step of the truth even on these short bursts, so
            # CENTRE the alias scan on it (k0) and sweep +-3 steps -- this reaches the correct alias
            # for any offset (not just |fc|<baud/2); keep-best over them picks the cleanest eye.
            step = meta.fs / ps.period
            k0 = round((fc - ps.cfo_hz) / step)
            cands = []
            for b2 in sorted({round(meta.fs * L / ps.period) for L in (3, 4, 6, 8)}):
                for k in range(k0 - 3, k0 + 4):
                    cfo = ps.cfo_hz + k * step
                    data = dsp.mix(xb, meta.fs, cfo)[ps.end:]
                    if data.size < 256:
                        continue
                    for sm in (2, 4, 8):
                        t = _demod(data, meta.fs, float(b2), sm)
                        cands.append((t[5].lock, cfo, float(b2), sm, t))
            cands.sort(key=lambda c: -c[0])
            # Adjacent aliases both look like clean M-PSK, so lock alone can pick the WRONG one by a
            # hair. Require the winner to beat the best DIFFERENT-carrier candidate by a clear gap;
            # otherwise the alias is genuinely ambiguous -> leave the honest low-lock result rather
            # than emit a confident wrong decode. (Same-carrier baud/sym variants do not compete.)
            if cands and cands[0][0] > q.lock + _EQ_GAIN:
                top = cands[0]
                other = next((c for c in cands if abs(c[1] - top[1]) > step / 2), None)
                if other is None or top[0] - other[0] >= _ALIAS_MARGIN:
                    alpha, ym, raw, mod, syms, q = top[4]
                    fc, baud, symmetry = top[1], top[2], top[3]
                    eq_applied, eq_mode, pre = False, None, ps

    bits = demap_diff_bits(syms, _DIFF_OF[mod]) if diff and mod in _DIFF_OF \
        else demap_bits(syms, mod)
    return Result(meta, "linear", (s, e), fc, baud, mod, q.lock, syms, bits, ym,
                  burst_x=xb, symmetry=symmetry, carrier_ambiguous=amb, baud_conf=conf,
                  rolloff=alpha, evm=q.evm, mer_db=q.mer_db, occupied=q.occupied,
                  snr_est=cl.snr_m2m4(syms, mod), eq_applied=eq_applied, eq_mode=eq_mode,
                  alias_resolved=abs(fc - fc0) > 1.0, baud_fallback=fell,
                  bursts=bursts, burst_idx=burst, preamble=pre)


def analyze_file(path: str, fs: float | None = None, fmt: str | None = None,
                 dtype: str | None = None, endian: str | None = None,
                 bitrev: bool | None = None, diff: bool = False,
                 burst: int | None = None, rf: float | None = None) -> Result:
    """Analyze a file; explicit args override SigMF/filename tokens."""
    from .sigio import parse_sigmf
    m = parse_sigmf(path) or parse_name(path)
    meta = Meta(fs or m.fs, fmt or m.fmt, dtype or m.dtype,
                endian or m.endian, m.bitrev if bitrev is None else bitrev,
                rf_center=rf if rf is not None else m.rf_center)
    x, meta = read(path, meta)
    return analyze(x, meta, diff, burst)


# --- wideband survey: many emitters in one capture --------------------------

@dataclass
class Emitter:
    detection: Detection
    kind: str                      # linear|fsk|chirp|analog|tone|tooshort|error
    abs_fc: float                  # emitter carrier vs capture centre (Hz)
    result: Result | None = None   # demod result for digital kinds
    info: dict | None = None       # characterization for non-digital kinds (e.g. chirp params)

    def to_json(self) -> dict:
        d = self.detection
        doc = {"kind": self.kind, "abs_fc": round(self.abs_fc, 3),
               "det": {"fc": round(d.fc, 3), "bw": round(d.bw, 3),
                       "t0": d.t0, "t1": d.t1, "snr_db": _r(d.snr_db),
                       "baud_hint": round(d.baud_hint, 1)}}
        if self.info is not None:
            doc["info"] = self.info
        if self.result is not None:
            r = self.result
            doc.update(mod=r.mod, baud=round(r.baud, 3), lock=round(r.lock, 1),
                       mer_db=_r(r.mer_db), evm=_r(r.evm, 4), family=r.family)
        return doc

    def to_detail(self, rf_center: float | None = None) -> dict:
        """Web drill-down payload: the box summary plus, for a digital emitter, the
        full analyze result so the UI reuses its single-signal detail view verbatim.
        The channel is demodulated at baseband (no rf_center), and r.fc is only the
        residual within-channel offset -- so the emitter's absolute RF is the capture
        centre plus abs_fc (the full offset from capture centre), injected here."""
        doc = self.to_json()
        if self.result is not None:
            doc["result"] = self.result.to_json()
            if rf_center is not None:
                doc["result"]["detected"]["rf_hz"] = round(rf_center + self.abs_fc, 3)
        return doc


@dataclass
class Survey:
    meta: Meta
    emitters: list[Emitter] = field(default_factory=list)

    def to_json(self) -> dict:
        return {"fs": self.meta.fs, "fmt": self.meta.fmt,
                "n_emitters": len(self.emitters),
                "emitters": [e.to_json() for e in self.emitters]}


def survey(x: np.ndarray, meta: Meta, *, diff: bool = False, **detect_kw) -> Survey:
    """Detect every emitter in a wideband capture, extract each to its own baseband
    channel, and demodulate the digital ones with the unchanged single-signal chain.
    A capture holding one signal that fills the band collapses to a single emitter."""
    xa = dsp.analytic(x)
    xa = xa - xa.mean()
    emitters = []
    for d in detect(xa, meta.fs, **detect_kw):
        ch, fs_ch = extract(xa, meta.fs, d)
        if ch.size < 256:
            emitters.append(Emitter(d, "tooshort", d.fc))
            continue
        kind = triage.family(ch, fs_ch)
        if kind in ("linear", "fsk"):
            # triage only gates digital vs not; analyze's own family is authoritative.
            # Isolate each channel: a degenerate one must not abort the whole survey.
            try:
                r = analyze(ch, Meta(fs_ch, "iq", "f32", "le", False), diff=diff)
                emitters.append(Emitter(d, r.family, d.fc + r.fc, r))
            except Exception:
                emitters.append(Emitter(d, "error", d.fc))
        else:
            info = analyze_chirp(ch, fs_ch) if kind == "chirp" else None
            emitters.append(Emitter(d, kind, d.fc, info=info))
    return Survey(meta, emitters)


def survey_web(x: np.ndarray, meta: Meta, *, diff: bool = False) -> dict:
    """Web survey payload for /api/survey. When detect sees <=1 emitter the capture
    is effectively one signal -> 'single' mode returns the UNCHANGED direct analyze()
    result (correct fc, matches /api/analyze). Otherwise 'survey' mode adds a whole-
    capture overview waterfall and every emitter's drill-down detail. Additive: does
    not alter survey()/Survey/Emitter, so CLI + existing tests are unaffected."""
    xa = dsp.analytic(x)
    xa = xa - xa.mean()
    if len(detect(xa, meta.fs)) <= 1:
        r = analyze(x, meta, diff=diff)
        out = {"mode": "single", "result": r.to_json()}
        if len(r.bursts) > 1:   # multi-burst signal -> whole-record map for burst selection
            out["overview"] = {"n": int(xa.size), "fs": meta.fs,
                               "waterfall": waterfall(xa, meta.fs)}
        return out
    sv = survey(x, meta, diff=diff)
    return {"mode": "survey", "fs": meta.fs, "fmt": meta.fmt, "rf_center": meta.rf_center,
            "overview": {"fs": meta.fs, "n": int(xa.size), "spectrum": spectrum(xa, meta.fs),
                         "waterfall": waterfall(xa, meta.fs)},
            "emitters": [e.to_detail(meta.rf_center) for e in sv.emitters]}


def survey_file(path: str, fs: float | None = None, fmt: str | None = None,
                dtype: str | None = None, endian: str | None = None,
                bitrev: bool | None = None, diff: bool = False,
                rf: float | None = None) -> Survey:
    """Survey a capture file; explicit args override SigMF/filename tokens."""
    from .sigio import parse_sigmf
    m = parse_sigmf(path) or parse_name(path)
    meta = Meta(fs or m.fs, fmt or m.fmt, dtype or m.dtype,
                endian or m.endian, m.bitrev if bitrev is None else bitrev,
                rf_center=rf if rf is not None else m.rf_center)
    x, meta = read(path, meta)
    return survey(x, meta, diff=diff)
