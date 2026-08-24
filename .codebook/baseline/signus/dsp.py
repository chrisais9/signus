"""Receiver DSP stages for blind PSK/QAM demodulation. Vectorized; per the house
anti-shared-bug rule this imports only constellation reference data, never gen.py."""

from fractions import Fraction

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import get_window, hilbert, resample_poly, stft, welch

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
    # clamp to +-half a bin: a true peak's sub-bin offset cannot exceed that, but a near-flat
    # (d~=0) or non-max triple can explode the ratio and drive baud negative -> a resample crash.
    return float(np.clip(0.5 * (a - c) / d, -0.5, 0.5)) if d != 0 else 0.0


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

_SPIKE_PAR = 60.0  # raw peak-to-mean power above which a candidate run is an impulse smear, not a
#                    burst. Modulated bursts sit low: constant-envelope FSK ~1, RRC-shaped PSK/QAM
#                    peaks ~6-12, high-order QAM tails to ~20; a single-sample impulse smeared over
#                    a ~win-wide run scores ~win (hundreds+). 60 clears every real mod with margin.


def _otsu(v: np.ndarray, bins: int = 128) -> float:
    """Otsu split threshold of a 1-D value distribution (log powers, column scores)."""
    hist, edges = np.histogram(v, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    w = np.cumsum(hist).astype(float)
    m = np.cumsum(hist * centers)
    mb = m / np.where(w > 0, w, 1)
    mf = (m[-1] - m) / np.where(w[-1] - w > 0, w[-1] - w, 1)
    return float(centers[int(np.argmax(w * (w[-1] - w) * (mb - mf) ** 2))])


def _cell_power(x: np.ndarray, fs: float, nperseg: int = 256) -> tuple[np.ndarray, int, int]:
    """Waterfall cell power for detection: (P[fbin, col], hop, nperseg). nperseg
    auto-shrinks on short records; shared core for find_bursts (and, later, survey)."""
    nperseg = int(min(nperseg, max(64, 1 << int(np.log2(max(x.size // 2, 64))))))
    hop = nperseg // 2
    win = get_window("blackmanharris", nperseg)
    _, _, z = stft(x, fs=fs, window=win, nperseg=nperseg, noverlap=nperseg - hop,
                   return_onesided=False, boundary=None, padded=False)
    return np.abs(z) ** 2, hop, nperseg


def find_bursts(x: np.ndarray, fs: float) -> list[tuple[int, int]]:
    """Waterfall (cell-level) burst detector. Per-bin noise floors give the same
    narrowband processing gain the operator's eye gets on a spectrogram (a burst at
    wideband 0 dB is +12 dB per cell when it occupies 1/16 of the band); column
    scores then drive the run/merge/guard machinery on the time axis.
    Returns bursts in time order, or [(0, size)] when nothing stands out."""
    n = x.size
    pw = np.abs(x) ** 2
    # A burst is a TIME-ENERGY event. A continuous signal that merely shuffles its spectrum
    # (multi-tone FSK) lights different cells per column and once shattered into 78 phantom
    # bursts -- but its wideband envelope is FLAT. Envelope Otsu separation measured 0.014-
    # 0.052 decades on continuous FSK vs >=0.30 on every real burst regime: nothing turns
    # on or off -> the whole record is the burst.
    lp = np.log10(uniform_filter1d(pw, max(64, n // 1000)) + 1e-20)
    e_thr = _otsu(lp)
    e_lo, e_hi = lp[lp < e_thr], lp[lp >= e_thr]
    if not e_lo.size or not e_hi.size or float(e_hi.mean() - e_lo.mean()) < 0.12:
        return [(0, n)]
    # nperseg 128: bin 7.8 kHz keeps the narrowband gain for real signal widths while the
    # 64-sample hop resolves inter-burst gaps down to ~200 samples (256 could not split a
    # 300-sample gap -- no column fits inside it).
    P, hop, nperseg = _cell_power(x, fs, nperseg=128)
    if P.shape[1] < 4:                   # too few columns to tell bursts from record
        return [(0, n)]
    # Per-bin floor at a LOW percentile: tracks coloured noise bin by bin and stays on the
    # noise even when a bin is signal-occupied up to ~90% of the time (high-duty trains).
    # A record-filling signal owns its bins' floors entirely -> flat score -> [(0, n)].
    floor_f = np.percentile(P, 10, axis=1)[:, None]
    r = P / (floor_f + 1e-30)
    k = max(3, P.shape[0] // 16)
    sc = np.log10(np.sort(r, axis=0)[-k:].mean(axis=0) + 1e-30)  # column score (decades)
    # A single noise column can spike +0.53 decades over base -- OVERLAPPING the weakest
    # real bursts (+0.57). Like the eye, the discriminator is PERSISTENCE, not height: a
    # 3-column smooth drops the noise worst-case to 0.31-0.33 while a real burst (>=3
    # columns) keeps its level (0.56+) -- the gap the fixed bars below live in.
    sc = uniform_filter1d(sc, 3)
    # The noise base is the LOWEST MODE of the score histogram (Otsu, as the wideband
    # version proved out): fixed quantile anchors cannot serve every duty regime, and
    # spread ESTIMATES broke three ways (median blind >50% duty, two-sided MAD inflated by
    # skirts, one-sided by the score's left tail). The margins are FIXED instead: a noise
    # column's score is distribution-pinned at C = log10((ln(nbins/k)+1)/0.105) whatever
    # the noise level, so its fluctuation is a constant of (nbins, k) -- measured worst
    # 0.33 over 24 noise records vs 0.56 for the weakest keeper burst.
    thr = _otsu(sc, bins=64)
    quiet = sc[sc < thr]
    if not quiet.size:
        return [(0, n)]
    base = float(np.median(quiet))
    # No noise-quiet column at all (base far above the pinned C): a continuous signal
    # shuffling its spectrum (a 4-tone FSK read as 78 phantom bursts before this guard).
    c_noise = float(np.log10((np.log(P.shape[0] / k) + 1) / 0.105))
    if base > c_noise + 0.25:
        return [(0, n)]
    hi = base + 0.45
    lo = base + 0.25
    above_lo = sc >= lo
    dif = np.diff(above_lo.astype(np.int8))
    starts = list(np.where(dif == 1)[0] + 1)
    ends = list(np.where(dif == -1)[0] + 1)
    if above_lo[0]:
        starts.insert(0, 0)
    if above_lo[-1]:
        ends.append(sc.size)
    runs = [(s, e) for s, e in zip(starts, ends, strict=True) if (sc[s:e] >= hi).any()]
    if not runs:
        return [(0, n)]
    merged = [list(runs[0])]
    for s, e in runs[1:]:
        # A real inter-burst gap returns to the score base; a marginal-level burst dips just
        # below lo without reaching it and would otherwise shatter into fragments. Merge
        # across any valley whose median stays off the base.
        valley = sc[merged[-1][1]:s]
        if s - merged[-1][1] < 2 or float(np.median(valley)) > base + 0.12:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    # Shape gate: a modulated burst has a near-flat raw envelope (peak/mean ~ a few); an
    # impulse smear is one spike over noise (ratio ~span). Reject spike-dominated candidates.
    out, spiked = [], 0
    for c0, c1 in merged:
        if c1 - c0 < 2:                  # single-column flicker (noise / impulse smear)
            continue
        s, e = max(0, c0 * hop), min(n, (c1 - 1) * hop + nperseg)
        if c1 - c0 >= 6:
            # long bursts: trim one hop of smear/quantization slack per edge. The noise
            # tail otherwise fed to the demod sits on a knife edge for dense QAM (58 extra
            # leading samples flipped a 16qam@24dB slice to 64qam lock 18). Short runs keep
            # their full span -- trimming there would eat into the burst itself.
            s, e = s + hop, e - hop
        seg = pw[s:e]
        if float(seg.max() / (seg.mean() + 1e-30)) > _SPIKE_PAR:
            spiked += e - s
            continue
        out.append((int(s), int(e)))
    # A detection must EXPLAIN the candidate mass the SPIKE gate threw away. If an impulse
    # inside the strongest burst killed its candidate, returning the survivors would hand
    # analyze() the wrong emitter -- fall back instead. Only spike kills count: the
    # single-column drops above are routine flicker filtering.
    covered = sum(e - s for s, e in out)
    if out and covered < 0.6 * (covered + spiked):
        return [(0, n)]
    # A continuous signal OWNS its bins' floors and is invisible to the cell path -- but its
    # edge transients are not, and once returned [(0, 192)] for a 64k channel whose 16qam
    # ran the whole record. If the detections cover almost none of the envelope's high-power
    # mass, the cell path missed the main event: return the ENVELOPE's high span instead
    # (not the raw record -- a dead tail fed to the demod flipped that 16qam to 64qam).
    # 0.1 keeps partial detections (a strong burst next to a lo-only sibling covers ~30%).
    if out and covered < 0.1 * int((lp >= e_thr).sum()):
        idx = np.where(lp >= e_thr)[0]
        return [(int(idx[0]), int(idx[-1]) + 1)]
    return out or [(0, n)]


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
    gaps = np.flatnonzero(np.diff(idx) > 20)   # 틈 정지가 없으면 대역 밖 스퍼(믹싱돼
    last = idx[gaps[0]] if gaps.size else idx[-1]   # -fc 에 떨어진 LO 누설)가 bw 를 부풀린다
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
