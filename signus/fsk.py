"""FSK/CPM receive path: constant-envelope gate + frequency-discriminator demod.
Runs on the DC-blocked analytic burst, before any carrier work (linear front-end
does not apply). Imports only constellation reference data (anti-shared-bug rule)."""

import numpy as np
from scipy.ndimage import uniform_filter1d

from .constellations import fsk_levels, to_bits
from .dsp import _kmeans1d, _parab, analytic

_CV_MAX = 0.24    # envelope coeff-of-variation ceiling. Recalibrated on a broad grid: real
# FSK (all h x SNR>=10 x high baud) tops out at cv 0.224, but constant-modulus PSK at very
# high baud (fs/baud~3) or near +-fs/2 dips to cv>=0.250 and used to slip the old 0.30 gate
# (false FSK). 0.24 sits in the 0.026 gap -> keeps every real FSK, rejects the PSK misfires.
_SEP_MIN = 3.1    # instantaneous-freq 2-means separation floor (FSK>=3.39, rest<=2.85)
_K4_RATIO = 2.6   # sp(k=2)/sp(k=4) collapse above which 4 levels beat 2
_MIN_SYMS = 8     # fewer symbols than this cannot cluster 2/4 levels (empty-quantile crash)


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


def _est_baud(f: np.ndarray, fs: float) -> float:
    """Symbol rate from the spectral line of |diff(smoothed f)| (transition
    impulses), harmonic-folded to the fundamental (2-level lines alias to 2*baud)."""
    d = np.abs(np.diff(uniform_filter1d(f, 2)))
    d = d - d.mean()
    n = 1 << int(np.ceil(np.log2(d.size)))
    spec = np.abs(np.fft.rfft(d * np.hanning(d.size), n))
    freqs = np.fft.rfftfreq(n, 1 / fs)
    df = freqs[1] - freqs[0]
    lo, hi = 0.002 * fs, 0.45 * fs
    idx = np.where((freqs >= lo) & (freqs <= hi))[0]
    k = idx[int(np.argmax(spec[idx]))]
    for div in (3, 2):
        ks = int(round(freqs[k] / div / df))
        if freqs[ks] >= lo:
            w = spec[max(0, ks - 2):ks + 3]
            if w.size and w.max() > 0.34 * spec[k]:
                k = max(0, ks - 2) + int(np.argmax(w))
    return float(freqs[k] + _parab(20 * np.log10(spec + 1e-20), k) * df)


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
    baud = _est_baud(f, fs)
    if baud <= 0:                       # a too-short / degenerate burst -> no baud line
        raise ValueError("FSK 버스트에서 심볼율을 추정할 수 없습니다 (너무 짧음)")
    sps = fs / baud
    f = uniform_filter1d(f, max(1, int(round(sps / 2))))
    fc = float(f.mean())
    fcen = f - fc
    nsym = int(fcen.size / sps) - 1
    if nsym < _MIN_SYMS:                # too few symbols to cluster levels / empty quantile
        raise ValueError(f"FSK 버스트가 너무 짧습니다 ({nsym} 심볼)")
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
