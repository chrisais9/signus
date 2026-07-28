"""Blind repeated-preamble synchronization (definition-free).

A real packetized emission begins with a preamble: a short block repeated a few times so a
receiver can lock. The repetition is detectable and exploitable WITHOUT knowing its content --
the delay-and-correlate (Schmidl-Cox) metric M(d) = |Sum conj(x[d+k]) x[d+k+P]|^2 / (Sum |x|^2)^2
rises to ~1 over the span where two period-P windows coincide (inside the preamble), and the PHASE
of that correlation is a low-variance carrier-frequency-offset (CFO) estimate that works from only a
few symbols -- where the blind M-th-power carrier estimator, needing many symbols to average, fails.

Used as a gated, keep-best rescue in pipeline.analyze(): a signal with no preamble yields no
plateau -> no rescue -> the existing blind chain is untouched (sweep stays byte-identical)."""

from dataclasses import dataclass

import numpy as np

_CONF_MIN = 0.80    # plateau strength floor (median M over the run); calibrated vs no-preamble
_RUN_MIN = 4.0      # plateau length >= this * period. RRC pulse memory spans a FIXED ~4.5 symbols,
#                     so in periods it exceeds this only at a 1-2 symbol lag and dies by P~=3 syms;
#                     a real preamble's plateau is (R-1) periods long -> this rejects the short
#                     memory (needs R>=~5 repeats; residual random false-alarms are caught by the
#                     pipeline keep-best-lock gate -- a spurious CFO lowers lock and is discarded).
_HARM_MIN = 0.50    # the SAME plateau must also correlate at lag 2P (a real preamble, >=3 repeats)
_VALLEY_MAX = 0.80  # ...AND DROP at lag 1.5P (between harmonics): a real preamble is peaked at
#                     multiples of P (a valley in between), so median(M@1.5P) < 0.85*conf. RRC pulse
#                     memory is a smooth decay (~equal at P and 1.5P) -> rejected. Peak/valley +
#                     run length is what lets the period be scanned directly, WITHOUT a reliable
#                     baud (est_baud is exactly what fails on the short bursts we target).
_PMIN = 12          # smallest period to scan (samples); below this is intra-symbol pulse memory
_PMAX = 512         # largest period to scan; a preamble block is short (L<=~12 syms, sps<=~40)
_WMAX = 8192        # only the leading window is scanned -- a preamble sits at the burst START


@dataclass
class Preamble:
    start: int      # sample index where the repeated region begins
    end: int        # sample index where the preamble ends (data starts ~here)
    period: int     # repetition period in samples (= L * sps)
    cfo_hz: float   # carrier-frequency offset estimated from the preamble phase slope
    conf: float     # 0..1 plateau confidence (median metric over the detected run)


def _metric(x: np.ndarray, P: int) -> tuple[np.ndarray, np.ndarray]:
    """Normalized delay-correlation M(d) in [0,1] and the raw correlation Pmet(d), lag P, window
    W=P. Normalized by BOTH windows' energy (Cauchy-Schwarz) so M<=1 -- a true correlation
    coefficient, ~1 only where the two period-P windows are genuinely identical (a preamble)."""
    prod = np.conj(x[:-P]) * x[P:]           # conj(x[n]) x[n+P]
    ea = np.abs(x[:-P]) ** 2                 # first-window energy
    eb = np.abs(x[P:]) ** 2                  # second-window energy
    cp = np.concatenate([[0.0 + 0j], np.cumsum(prod)])
    ca = np.concatenate([[0.0], np.cumsum(ea)])
    cb = np.concatenate([[0.0], np.cumsum(eb)])
    W = P
    pm = cp[W:] - cp[:-W]                     # Sum prod[d:d+W]
    da = ca[W:] - ca[:-W]
    db = cb[W:] - cb[:-W]
    return np.abs(pm) ** 2 / (da * db + 1e-30), pm


def _seg_median(x: np.ndarray, lag: int, s: int, ln: int) -> float:
    """Median of the delay-correlation metric at `lag` over the plateau [s, s+ln)."""
    m = _metric(x, lag)[0]
    e = min(s + ln, m.size)
    return float(np.median(m[s:e])) if e > s else 0.0


def _longest_run(mask: np.ndarray) -> tuple[int, int]:
    """(start, length) of the longest True run in a boolean array; (0, 0) if none."""
    if not mask.any():
        return 0, 0
    d = np.diff(np.concatenate([[0], mask.view(np.int8), [0]]))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    i = int(np.argmax(ends - starts))
    return int(starts[i]), int(ends[i] - starts[i])


def find_preamble(x: np.ndarray, fs: float, baud_hint: float | None = None) -> Preamble | None:
    """Detect a repeated preamble and estimate its (start, end, period, CFO, confidence).
    Returns None when no repeated block stands out. The period is scanned DIRECTLY (not derived
    from baud) so it works even when est_baud fails on the short burst -- exactly the target case.
    baud_hint, if given, only tightens the scan's upper bound for speed."""
    x = np.asarray(x, dtype=np.complex128)
    x = x - x.mean()
    if x.size < 256 or not np.all(np.isfinite(x)):
        return None
    xw = x[:_WMAX]                            # a preamble sits at the burst start
    n = xw.size
    pmax = _PMAX if not (baud_hint and baud_hint > 0) else min(_PMAX, int(12 * fs / baud_hint))
    pmax = min(pmax, (n - 8) // 3)            # need >=3 windows for the 2P harmonic check
    best = None
    for P in range(_PMIN, max(_PMIN + 1, pmax + 1)):
        M, pm = _metric(xw, P)
        run_start, run_len = _longest_run(M >= _CONF_MIN)
        if run_len < _RUN_MIN * P:
            continue
        conf = float(np.median(M[run_start:run_start + run_len]))
        # peak/valley gate: a genuinely periodic preamble correlates at 2P (harmonic) but DROPS at
        # 1.5P (between harmonics); RRC pulse memory (which fakes a plateau at a 1-2 symbol lag) is
        # a smooth decay -- high at both -> rejected, letting the period be scanned directly.
        harm = _seg_median(xw, 2 * P, run_start, run_len)
        valley = _seg_median(xw, int(round(1.5 * P)), run_start, run_len)
        if harm < _HARM_MIN or valley > _VALLEY_MAX * conf:
            continue
        score = run_len * conf               # long AND strong wins
        if best is None or score > best[0]:
            ang = float(np.angle(pm[run_start:run_start + run_len].sum()))  # coherent, wrap-safe
            cfo = ang * fs / (2 * np.pi * P)
            end = run_start + run_len + 2 * P            # past the last correlated repeat -> data
            best = (score, Preamble(run_start, min(end, x.size), P, cfo, conf))
    return best[1] if best else None
