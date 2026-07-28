"""FSK/CPM receive path: constant-envelope gate + frequency-discriminator demod.
Runs on the DC-blocked analytic burst, before any carrier work (linear front-end
does not apply). Imports only constellation reference data (anti-shared-bug rule)."""

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import firwin, resample_poly, welch

from .constellations import fsk_levels, to_bits
from .dsp import _kmeans1d, _parab, analytic, mix

_CV_MAX = 0.24    # envelope coeff-of-variation ceiling. Recalibrated on a broad grid: real
# FSK (all h x SNR>=10 x high baud) tops out at cv 0.224, but constant-modulus PSK at very
# high baud (fs/baud~3) or near +-fs/2 dips to cv>=0.250 and used to slip the old 0.30 gate
# (false FSK). 0.24 sits in the 0.026 gap -> keeps every real FSK, rejects the PSK misfires.
_SEP_MIN = 3.1    # instantaneous-freq 2-means separation floor (FSK>=3.39, rest<=2.85)
_K4_RATIO = 2.6   # sp(k=2)/sp(k=4) collapse above which 4 levels beat 2
_MIN_SYMS = 8     # fewer symbols than this cannot cluster 2/4 levels (empty-quantile crash)
_PROM_MIN = 10.0  # a symbol-rate line peak/median below this is unreliable -- BUT only confident-
#                   wrong when paired with too few symbols (a long low-index burst has a weak line
#                   yet a correct baud). The sweep sits at prom 23-73.
_NSYM_MIN = 200   # ...so reject a weak line only below this symbol count. A 100 floor was tried
#                   (to recover more short bursts) but an independent grid found weak-prominence
#                   half-rate folds in the 100-199 band that still cluster tight enough to lock >=60
#                   (msk h=0.5 baud 24-32% off, lock 61-65) -- confident-wrong. 200 keeps them out;
#                   precision over recall (the short-burst confident-wrong is the cardinal sin).
_DECIM_FRAC = 0.30  # decimate a heavily-oversampled burst so the tones fill ~this of Nyquist
_DECIM_MIN = 3      # ...but only when that needs a factor >= this, so the sps~10 demod grid
#                     (fsk2/fsk4/msk baud=fs/10) is untouched -> the sweep stays byte-identical


def _bw99(x: np.ndarray, fs: float) -> float:
    """Two-sided 99%-power occupied bandwidth (Hz), noise-pedestal subtracted."""
    f, p = welch(x, fs=fs, nperseg=min(4096, x.size), return_onesided=False, detrend=False)
    f, p = np.fft.fftshift(f), np.fft.fftshift(p)
    p = np.maximum(p - np.median(p), 0.0)
    cs = np.cumsum(p) / (p.sum() + 1e-30)
    lo = f[min(np.searchsorted(cs, 0.005), f.size - 1)]  # clamp BOTH edges: a degenerate PSD
    hi = f[min(np.searchsorted(cs, 0.995), f.size - 1)]  # (all-zero after floor subtraction)
    return float(abs(hi - lo))                           # otherwise indexes one past the end


def _dcblock(x: np.ndarray) -> np.ndarray:
    """Complex128 analytic burst, DC-blocked (front-end convention)."""
    x = analytic(x)
    return x - x.mean()


def _ifreq(x: np.ndarray, fs: float) -> np.ndarray:
    """Instantaneous frequency (Hz) from sample-to-sample phase differences."""
    return np.angle(x[1:] * np.conj(x[:-1])) * fs / (2 * np.pi)


def _sep(a: np.ndarray) -> float:
    """2-means between/within separation ratio of a 1-D sample set."""
    c, _, sp = _kmeans1d(a, 2)
    return abs(c[1] - c[0]) / (sp + 1e-9)


def _est_baud(f: np.ndarray, fs: float) -> tuple[float, float]:
    """Symbol rate from the spectral line of |diff(smoothed f)| (transition impulses),
    harmonic-folded to the fundamental (2-level lines alias to 2*baud). Returns
    (baud, prominence) where prominence = peak/median of the in-band line: a weak line
    (few symbols, no clean symbol clock) means the baud is UNRELIABLE -> caller rejects."""
    d = np.abs(np.diff(uniform_filter1d(f, 2)))
    d = d - d.mean()
    n = 1 << int(np.ceil(np.log2(d.size)))
    spec = np.abs(np.fft.rfft(d * np.hanning(d.size), n))
    freqs = np.fft.rfftfreq(n, 1 / fs)
    df = freqs[1] - freqs[0]
    lo, hi = 0.002 * fs, 0.45 * fs
    idx = np.where((freqs >= lo) & (freqs <= hi))[0]
    prom = float(spec[idx].max() / (np.median(spec[idx]) + 1e-20)) if idx.size else 0.0
    k = idx[int(np.argmax(spec[idx]))]

    def _hp(kk: int) -> float:                       # harmonic-comb energy: the TRUE fundamental
        return float(sum(spec[m * kk] for m in range(1, 6) if m * kk < spec.size))  # captures most
    #   in-range harmonics ONLY: clamping out-of-range ones to the last bin multi-counted the
    #   Nyquist bin, inflating _hp at a 2*baud peak enough to veto the legitimate fold -> 2x baud.

    for div in (3, 2):
        ks = int(round(freqs[k] / div / df))
        if freqs[ks] >= lo:
            w = spec[max(0, ks - 2):ks + 3]
            kk = max(0, ks - 2) + int(np.argmax(w)) if w.size else k
            # fold to the sub-harmonic ONLY if it is genuinely the fundamental -- its harmonic comb
            # must beat the current peak's. A real baud/2 fundamental captures MORE harmonic lines
            # than the 2*baud peak; spurious low-h/low-SNR energy at baud/2 does not (it once
            # slipped the 0.34 gate and folded baud -> baud/2: a confident WRONG decode). The comb
            # guard vetoes that. On the sps~10 grid the fold never fired (0.34 unmet) -> sweep same.
            if w.size and w.max() > 0.34 * spec[k] and _hp(kk) >= _hp(k):
                k = kk
    return float(freqs[k] + _parab(20 * np.log10(spec + 1e-20), k) * df), prom


def fsk_gate(x: np.ndarray, fs: float) -> bool:
    """True when the burst looks like FSK/CPM: near-constant envelope AND a bimodal
    instantaneous frequency. Calibrated on generate() over all GEN_MODS x seeds 0..3
    x snr {25,15}: every linear mod fails one test with margin -- bpsk/dbpsk have
    cv~0.40 (>_CV_MAX), the rest have freq-sep<=2.85 (<_SEP_MIN); fsk2/fsk4/msk keep
    cv<=0.22 and freq-sep>=3.39 down to snr 10 (margins ~0.08 and ~0.29).
    Recenter by the mean instantaneous frequency first: at a large carrier offset a
    linear mod's phase-transition spikes wrap past +-fs/2 and fake a second mode."""
    x = _dcblock(x)
    a = np.abs(x)
    if a.std() / (a.mean() + 1e-12) >= _CV_MAX:
        return False
    f = _ifreq(x, fs)
    x = x * np.exp(-2j * np.pi * np.mean(f) / fs * np.arange(x.size))
    return bool(_sep(uniform_filter1d(_ifreq(x, fs), 4)) > _SEP_MIN)


def analyze_fsk(x: np.ndarray, fs: float) -> dict:
    """Full FSK/MSK demod by frequency discrimination. Returns mod, fc, baud, h,
    lock (0-100), level_freqs (Hz, sorted, centered), symbols (normalized f per
    symbol) and Gray-demapped bits."""
    x = _dcblock(x)
    f = _ifreq(x, fs)
    # A carrier OFFSET (tones sit at fc +- dev, not around DC) corrupts the symbol-rate estimate two
    # ways -- off-centre it trips _est_baud's harmonic fold, and the tones fall outside the decim
    # LPF passband -- either way a confident WRONG baud. Recentre the discriminator on the carrier
    # (mean instantaneous freq) first; only for a significant offset, so a centred burst
    # (fc~0, the whole demod/sweep grid) is byte-identical. Levels are already centred downstream.
    fc0 = float(np.mean(f))
    if abs(fc0) > 0.005 * fs:
        x = mix(x, fs, fc0)
        f = _ifreq(x, fs)
    else:
        fc0 = 0.0
    baud, prom = _est_baud(f, fs)
    # heavy oversampling (fs/baud >> 10) buries the symbol-rate line under broadband transition
    # artifacts -> _est_baud locks a spurious peak (baud ~25% off) while the clean tones still
    # cluster (lock ~94): a confident WRONG decode. Re-estimate baud on a decimated copy (tones ->
    # ~_DECIM_FRAC of Nyquist, as the survey channelizer does). Gated at _DECIM_MIN so the sps~10
    # grid is untouched (fsk2/fsk4/msk baud=fs/10 -> d<3 -> no re-estimate -> sweep byte-identical).
    bw = _bw99(x, fs)
    d = int(_DECIM_FRAC * fs / bw) if bw > 0 else 1
    if d >= _DECIM_MIN:
        xd = np.convolve(x, firwin(129, min(0.9 * (fs / d) / 2, bw) / (fs / 2)), "same")
        baud, prom = _est_baud(_ifreq(resample_poly(xd, 1, d), fs / d), fs / d)
    if baud <= 0:
        raise ValueError("FSK 버스트에서 심볼율을 추정할 수 없습니다 (너무 짧음)")
    sps = fs / baud
    f = uniform_filter1d(f, max(1, int(round(sps / 2))))
    resid = float(f.mean())
    fc = fc0 + resid                    # absolute carrier = recentre offset + in-band residual
    fcen = f - resid
    nsym = int(fcen.size / sps) - 1
    # a WEAK symbol-rate line (prom < _PROM_MIN) is unreliable ONLY when symbols are also too few to
    # average (nsym < _NSYM_MIN): a short burst's spurious peak still clusters tight (few points ->
    # high lock) = a confident WRONG baud. A long low-index (h~0.35) burst also has a weak line but
    # MANY symbols -> its baud is correct -> NOT rejected. Sweep: nsym~3000/prom 23-73 -> untouched.
    if nsym < _MIN_SYMS or (prom < _PROM_MIN and nsym < _NSYM_MIN):
        raise ValueError(f"FSK 버스트가 너무 짧거나 심볼율이 불확실합니다 ({nsym} 심볼)")
    ctr = (np.arange(nsym) + 0.5) * sps

    # symbol sampling phase: the offset whose sampled levels separate best
    v, best = fcen[:nsym], -1.0
    for o in np.linspace(0, sps, 8, endpoint=False):
        s = fcen[np.clip((ctr + o).astype(int), 0, fcen.size - 1)]
        sep = _sep(s)
        if sep > best:
            best, v = sep, s

    # level count via spread-collapse (k=2 vs k=4), then cluster
    k = 4 if _kmeans1d(v, 2)[2] / (_kmeans1d(v, 4)[2] + 1e-9) > _K4_RATIO else 2
    c, lab, spread = _kmeans1d(v, k)
    order = np.argsort(c)
    ranks = np.argsort(order)[lab]           # per-symbol level index, 0 = lowest freq
    csort = c[order]
    mid = float(csort.mean())                # debias center: symmetric levels sum ~0
    fc, csort, v = fc + mid, csort - mid, v - mid
    spacing = float(np.mean(np.diff(csort)))
    h = spacing / baud

    mod = "msk" if (k == 2 and abs(h - 0.5) < 0.15) else f"fsk{k}"
    _, glabels = fsk_levels(mod)
    bits = to_bits(glabels[ranks], mod).astype(np.uint8)
    lock = float(np.clip((spacing / (spread + 1e-9) - 1.5) / 6.5 * 100, 0, 100))
    return dict(mod=mod, fc=fc, baud=baud, h=h, lock=lock, level_freqs=csort,
                symbols=v / (spacing / 2), bits=bits)
