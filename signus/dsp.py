"""Receiver DSP stages for blind PSK/QAM demodulation. Vectorized; per the house
anti-shared-bug rule this imports only constellation reference data, never gen.py."""

from fractions import Fraction

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import hilbert, resample_poly, welch

from ._accel import dd_carrier
from .constellations import ideal_points


def analytic(x: np.ndarray) -> np.ndarray:
    """Complex passthrough as complex128; real input -> Hilbert analytic signal. Non-finite
    samples are zeroed first: a single NaN/Inf otherwise spreads through every FFT (and Inf
    overflows on the |x|^2 the estimators take)."""
    x = np.nan_to_num(x, posinf=0.0, neginf=0.0)
    x = np.where(np.abs(x) > 1e150, 0.0, x)   # a finite spike that overflows |x|^2 is corrupt too
    if np.iscomplexobj(x):
        return x.astype(np.complex128)
    return hilbert(np.asarray(x, dtype=np.float64)).astype(np.complex128)


def _parab(ydb: np.ndarray, k: int) -> float:
    """Sub-bin peak offset from a 3-point parabola fit (inputs already in dB)."""
    if k <= 0 or k >= ydb.size - 1:
        return 0.0
    a, b, c = ydb[k - 1], ydb[k], ydb[k + 1]
    d = a - 2 * b + c
    return 0.5 * (a - c) / d if d != 0 else 0.0


def _kmeans1d(a: np.ndarray, k: int, iters: int = 50) -> tuple[np.ndarray, np.ndarray, float]:
    """1-D k-means on quantile-seeded centers; return (centers, labels, rms spread).
    Assignment is a running min over the k centers (no (N, k) broadcast temporary);
    strict `<` keeps the lowest-index center on ties, so labels are identical to a
    broadcast argmin. Shared by the classify amplitude-ring test and the FSK level demod."""
    c = np.quantile(a, (np.arange(k) + 0.5) / k)
    lab = np.zeros(a.size, dtype=int)
    for _ in range(iters):
        d = np.abs(a - c[0])
        lab = np.zeros(a.size, dtype=int)
        for j in range(1, k):
            dj = np.abs(a - c[j])
            closer = dj < d
            lab[closer] = j
            d[closer] = dj[closer]
        nc = np.array([a[lab == j].mean() if np.any(lab == j) else c[j] for j in range(k)])
        if np.allclose(nc, c):
            break
        c = nc
    return c, lab, float(np.sqrt(np.mean((a - c[lab]) ** 2)))


# --- burst detection -------------------------------------------------------

def find_bursts(x: np.ndarray, fs: float) -> list[tuple[int, int]]:
    """Dual-threshold energy detector with an Otsu log-power floor; return every
    merged burst in time order, or [(0, size)] when nothing stands out."""
    n = x.size
    win = max(64, n // 1000)
    ps = uniform_filter1d(np.abs(x) ** 2, win)
    lp = np.log10(ps + 1e-20)

    # Otsu split of the log-power histogram: a floor that survives an 80%-full
    # record because it is drawn from the (small) below-threshold noise class.
    hist, edges = np.histogram(lp, bins=128)
    centers = (edges[:-1] + edges[1:]) / 2
    w = np.cumsum(hist).astype(float)
    m = np.cumsum(hist * centers)
    mb = m / np.where(w > 0, w, 1)
    mf = (m[-1] - m) / np.where(w[-1] - w > 0, w[-1] - w, 1)
    btw = w * (w[-1] - w) * (mb - mf) ** 2
    thr = centers[int(np.argmax(btw))]
    noise = lp[lp < thr]
    floor = noise.mean() if noise.size else lp.min()

    above_hi = lp >= floor + 1.0  # 10 dB (power decade) over floor
    above_lo = lp >= floor + 0.6  # 6 dB
    dif = np.diff(above_lo.astype(np.int8))
    starts = list(np.where(dif == 1)[0] + 1)
    ends = list(np.where(dif == -1)[0] + 1)
    if above_lo[0]:
        starts.insert(0, 0)
    if above_lo[-1]:
        ends.append(n)
    runs = [(s, e) for s, e in zip(starts, ends, strict=True) if above_hi[s:e].any()]
    if not runs:
        return [(0, n)]

    gap = mlen = max(256, n // 200)
    merged = [list(runs[0])]
    for s, e in runs[1:]:
        if s - merged[-1][1] < gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    merged = [(int(s), int(e)) for s, e in merged if e - s >= mlen]
    return merged or [(0, n)]


def find_burst(x: np.ndarray, fs: float) -> tuple[int, int]:
    """The most energetic burst of find_bursts."""
    return max(find_bursts(x, fs), key=lambda b: float(np.sum(np.abs(x[b[0]:b[1]]) ** 2)))


# --- carrier estimation ----------------------------------------------------

def est_carrier(
    x: np.ndarray, fs: float, powers: tuple[int, ...] = (2, 4, 8),
    nfft: int = 65536, blocks: int = 4,
) -> tuple[float, int, bool]:
    """Blind M-th-power carrier estimate; returns (fc, symmetry, ambiguous). No DC
    nulling: the front-end is DC-blocked, so a peak at DC is a genuine fc=0."""
    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, 1 / fs))
    best_par, best_s, sym = -1.0, None, powers[0]
    for p in powers:
        xp = x ** p
        blk = xp.size // blocks
        segs = [xp[i * blk:(i + 1) * blk] for i in range(blocks)] if blk >= 8 else [xp]
        acc = np.zeros(nfft)
        for s in segs:
            acc += np.abs(np.fft.fftshift(np.fft.fft(s * np.hanning(s.size), nfft)))
        spec = acc / len(segs)
        par = spec.max() / spec.mean()
        if par > best_par:
            best_par, best_s, sym = par, spec, p

    if best_s is None:  # degenerate input (all-zero / no energy): no M-th-power tone exists
        return 0.0, int(powers[0]), True
    ydb = 20 * np.log10(best_s + 1e-20)
    k = int(np.argmax(best_s))
    f_peak = freqs[k] + _parab(ydb, k) * (freqs[1] - freqs[0])
    fc = f_peak / sym
    ambiguous = abs(sym * fc) > 0.4 * fs
    return float(fc), int(sym), bool(ambiguous)


def resolve_alias(x: np.ndarray, fs: float, fc: float, sym: int) -> float:
    """The M-th-power tone wraps mod fs, so fc is only known mod fs/sym. Candidates tile the
    whole band at spacing fs/sym; pick the one whose alias CELL holds the most signal power
    (wrap-aware). Cell-energy beats a spectral centroid at the band edge, where a signal
    straddling +-fs/2 corrupts the two-sided centroid (bpsk fc=0.495*fs picked fc=-5000)."""
    f, pxx = welch(x, fs=fs, nperseg=min(4096, x.size), return_onesided=False)
    p = np.maximum(pxx - np.median(pxx), 0)
    # range must reach +-fs/2 for EVERY sym: |k| up to sym//2 (an old fixed range(-2,3) left
    # 8psk beyond 0.3125*fs with no candidate).
    cands = [fc + k * fs / sym for k in range(-(sym // 2) - 1, sym // 2 + 2)
             if abs(fc + k * fs / sym) < 0.5 * fs]
    half = fs / (2 * sym)                        # each candidate owns a fs/sym-wide cell
    return max(cands, key=lambda c: float(p[np.abs(((f - c + fs / 2) % fs) - fs / 2) < half].sum()))


def mix(x: np.ndarray, fs: float, fc: float) -> np.ndarray:
    """Downconvert by fc (complex heterodyne)."""
    return x * np.exp(-2j * np.pi * fc * np.arange(x.size) / fs)


# --- symbol rate + rolloff -------------------------------------------------

def est_baud(
    x: np.ndarray, fs: float, lo: float | None = None, hi: float | None = None,
    blocks: int = 4,
) -> tuple[float, float]:
    """Cyclostationary line at the symbol rate in |x|^2; returns (baud, confidence).
    Low confidence flags a weak line (near-zero rolloff); caller decides. Fewer blocks =
    a longer, stronger line (worth trying on a short burst where the default 4 is too weak)."""
    u = np.abs(x) ** 2
    u = u - u.mean()
    blk = u.size // blocks
    segs = [u[i * blk:(i + 1) * blk] for i in range(blocks)] if blk >= 8 else [u]
    nfft = 1 << max(16, int(np.ceil(np.log2(max(s.size for s in segs)))))
    acc = np.zeros(nfft // 2 + 1)
    for s in segs:
        acc += np.abs(np.fft.rfft(s * np.hanning(s.size), nfft))
    spec = acc / len(segs)
    freqs = np.fft.rfftfreq(nfft, 1 / fs)

    lo = lo if lo is not None else 0.005 * fs
    hi = hi if hi is not None else 0.45 * fs
    band = (freqs >= lo) & (freqs <= hi)
    if not band.any():  # degenerate window (e.g. noise-derived prior): default band
        band = (freqs >= 0.005 * fs) & (freqs <= 0.45 * fs)
    idx = np.where(band)[0]
    k = idx[int(np.argmax(spec[idx]))]
    ydb = 20 * np.log10(spec + 1e-20)
    baud = freqs[k] + _parab(ydb, k) * (freqs[1] - freqs[0])
    conf = spec[k] / (np.median(spec[band]) + 1e-20)
    return float(baud), float(conf)


def occupied_bw(x: np.ndarray, fs: float) -> float:
    """Two-sided occupied bandwidth without a baud prior: threshold 10% of the way
    from the 25th-percentile floor to the 90th-percentile in-band level, walk
    outward from DC, stop at the first >20-bin below-threshold gap (notch tolerant).
    Returns 0.0 when no band hugs DC (caller must not trust the prior then)."""
    f, pxx = welch(x, fs=fs, nperseg=min(8192, x.size), return_onesided=False)
    floor = np.percentile(pxx, 25)
    order = np.argsort(np.abs(f))
    idx = np.flatnonzero(pxx[order] > floor + 0.10 * (np.percentile(pxx, 90) - floor))
    if idx.size == 0:
        return 0.0
    last = idx[-1]
    return float(2 * np.abs(f[order][last]))


def est_rolloff(x: np.ndarray, fs: float, baud: float) -> float:
    """Occupied-bandwidth band-edge estimate mapped through the v1 rolloff calibration."""
    n = x.size
    nper = min(8192, max(256, n // 8))
    f, pxx = welch(x, fs=fs, nperseg=nper, return_onesided=False)
    f, pxx = np.fft.fftshift(f), np.fft.fftshift(pxx)
    af = np.abs(f)
    inband = np.median(pxx[af < 0.25 * baud])
    noise = np.median(pxx[af > 0.9 * baud])
    thr = noise + 0.10 * (inband - noise)
    band = (af > 0.3 * baud) & (af < 1.0 * baud) & (pxx > thr)
    if not band.any():
        return 0.35
    raw = 2 * af[band].max() / baud - 1
    return float(np.clip(1.5 * raw + 0.026, 0.05, 1.0))


# --- resampling + matched filter -------------------------------------------

def to_sps(x: np.ndarray, fs: float, baud: float, sps: int = 4) -> np.ndarray:
    """Rational resample to exactly sps samples/symbol."""
    r = Fraction(sps * baud / fs).limit_denominator(2000)
    return resample_poly(x, r.numerator, r.denominator)


def _rx_rrc(alpha: float, span: int, sps: int) -> np.ndarray:
    """RX matched RRC taps, unit energy, odd length; singularities at t=0 and
    |t|=1/4a patched. Written independently of gen.py's TX shaper (anti-shared-bug)."""
    m = (span * sps) // 2
    t = np.arange(-m, m + 1) / sps
    a = alpha
    num = np.sin(np.pi * t * (1 - a)) + 4 * a * t * np.cos(np.pi * t * (1 + a))
    den = np.pi * t * (1 - (4 * a * t) ** 2)
    h = np.divide(num, den, out=np.zeros_like(t), where=den != 0)
    h[t == 0] = 1 - a + 4 * a / np.pi
    if a > 0:
        s = np.isclose(np.abs(t), 1 / (4 * a))
        h[s] = a / np.sqrt(2) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * a))
                                 + (1 - 2 / np.pi) * np.cos(np.pi / (4 * a)))
    return h / np.sqrt(np.sum(h ** 2))


def matched(x: np.ndarray, sps: int, alpha: float, span: int = 10) -> np.ndarray:
    """Apply the RX matched RRC filter."""
    return np.convolve(x, _rx_rrc(alpha, span, sps), "same")


# --- timing recovery -------------------------------------------------------

def timing(x: np.ndarray, sps: int, block: int = 256, out: int = 1) -> np.ndarray:
    """Block-wise Oerder-Meyr timing recovery; returns `out` samples/symbol
    (1 = decisions, 2 = clock-tracked T/2 stream for the fractionally-spaced eq).
    Per block the spectral-line phase gives the offset; unwrapped across blocks
    to track clock drift, then interpolated per symbol."""
    n = sps
    nsym = x.size // n
    if nsym < 1:
        return x[:0].astype(np.complex128)
    xt = x[:nsym * n]
    p = np.abs(xt) ** 2
    ph = np.exp(-2j * np.pi * np.arange(xt.size) / n)
    bs = block * n
    if xt.size <= bs:  # single block: one global fractional offset
        eps = np.full(nsym, -np.angle(np.sum(p * ph)) / (2 * np.pi))
    else:
        nblk = xt.size // bs
        c = (p[:nblk * bs].reshape(nblk, bs) * ph[:nblk * bs].reshape(nblk, bs)).sum(1)
        eps_b = np.unwrap(-np.angle(c)) / (2 * np.pi)
        eps = np.interp(np.arange(nsym), (np.arange(nblk) + 0.5) * block, eps_b)
    pos = (np.arange(nsym * out) // out + np.repeat(eps, out)) * n
    idx = np.arange(xt.size)
    return np.interp(pos, idx, xt.real) + 1j * np.interp(pos, idx, xt.imag)


# --- decision-directed carrier sync ----------------------------------------

def ddsync(symbols: np.ndarray, mod: str, alpha: float = 0.05, beta: float = 0.002) -> np.ndarray:
    """Decision-directed PI carrier loop, general across PSK/QAM (never a PSK-only
    Costas loop, which jitters QAM at the correct points). Sequential feedback (each
    phase update depends on the prior decision) -> numba kernel when available."""
    pts = ideal_points(mod)
    z = symbols / np.sqrt(np.mean(np.abs(symbols) ** 2))  # AGC to unit power
    return dd_carrier(np.ascontiguousarray(z, dtype=np.complex128),
                      np.ascontiguousarray(pts, dtype=np.complex128), alpha, beta)
