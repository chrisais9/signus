"""Wideband signal detection: find every emitter in a capture as a frequency/time
box. Turns signus from a single-signal demodulator into a survey tool. Frequency
detection runs on the time-averaged spectrogram (robust to signal width); the time
extent per band comes from the spectrogram. numpy + scipy only."""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.signal import get_window, stft


@dataclass
class Detection:
    fc: float          # centre frequency (Hz, baseband offset)
    bw: float          # 99%-power bandwidth (Hz)
    t0: int            # burst start / end (sample indices into the capture)
    t1: int
    snr_db: float      # peak-to-floor of the band (dB)

    @property
    def baud_hint(self) -> float:
        """Rough symbol-rate prior: 99%-power BW is ~0.86 of occupied BW Rs(1+a)."""
        return self.bw / (0.86 * 1.25)


def _floor(s: np.ndarray, frac: float = 0.5, pct: int = 20) -> np.ndarray:
    """Per-bin noise floor via a wide low-percentile filter. A 20th percentile over
    a half-band window tracks the front-end tilt yet stays on the noise even when a
    signal fills up to ~half the window -- so it survives both a narrowband emitter
    and a single signal occupying 10-50% of the band."""
    w = max(11, int(frac * s.size) | 1)
    return ndimage.percentile_filter(s, pct, size=w, mode="nearest")


def detect(x: np.ndarray, fs: float, *, nfft: int = 4096, overlap: float = 0.5,
           thr_db: float = 6.0, min_bw_bins: int = 3) -> list[Detection]:
    """Detect emitters as (fc, bw, t0, t1) boxes, time/frequency ordered by fc.
    Empty capture -> []; a single signal filling the band -> one whole-band box."""
    nfft = int(min(nfft, 1 << max(8, int(np.log2(max(x.size // 16, 256))))))
    win = get_window("blackmanharris", nfft)
    f, t, z = stft(x, fs=fs, window=win, nperseg=nfft, noverlap=int(nfft * overlap),
                   return_onesided=False, boundary=None, padded=False)
    f = np.fft.fftshift(f)
    p = np.fft.fftshift(np.abs(z) ** 2, axes=0)          # (freq, time) linear power
    s = p.mean(axis=1)                                    # time-averaged PSD
    floor = _floor(s)
    mask = s > floor * 10 ** (thr_db / 10)
    close = max(3, int(0.012 * s.size) | 1)              # bridge intra-signal nulls
    mask = ndimage.binary_opening(mask, structure=np.ones(2))
    mask = ndimage.binary_closing(mask, structure=np.ones(close))

    lab, n = ndimage.label(mask)
    regions = [np.where(lab == i)[0] for i in range(1, n + 1)]
    regions = [r for r in regions if r.size >= min_bw_bins]
    merged: list[np.ndarray] = []
    for r in sorted(regions, key=lambda r: r[0]):
        # heal an intra-signal split (gap small vs the signal's OWN width) without
        # swallowing a genuinely separate neighbour -- cap the merge gap at half the
        # narrower blob, so adjacent channels (e.g. the 25 kHz marine raster) survive
        if merged and r[0] - merged[-1][-1] <= max(close, min(merged[-1].size, r.size) // 2):
            merged[-1] = np.arange(merged[-1][0], r[-1] + 1)
        else:
            merged.append(r)

    dets = [_measure(r, s, floor, p, f, t, fs, x.size) for r in merged]
    if not dets and s.max() > 10 ** (thr_db / 10) * np.percentile(s, 20):
        # floor defeated by a signal filling most of the band: one whole-band box,
        # so survey() falls back to analysing the entire capture (legacy behaviour)
        dets = [Detection(0.0, fs * 0.5, 0, x.size, 0.0)]
    return dets


def _measure(r: np.ndarray, s: np.ndarray, floor: np.ndarray, p: np.ndarray,
             f: np.ndarray, t: np.ndarray, fs: float, n: int) -> Detection:
    """Estimate fc (floor-subtracted centroid), 99%-power bw, and time extent."""
    lo, hi = r[0], r[-1]
    w0, w1 = max(0, lo - 3), min(s.size, hi + 4)         # widen for skirts
    idx = np.arange(w0, w1)
    wgt = np.maximum(s[idx] - floor[idx], 0)
    fc = float(np.sum(f[idx] * wgt) / (np.sum(wgt) + 1e-30))
    cs = np.cumsum(wgt) / (np.sum(wgt) + 1e-30)
    f_lo = f[idx][np.searchsorted(cs, 0.005)]
    f_hi = f[idx][min(np.searchsorted(cs, 0.995), idx.size - 1)]
    bw = float(abs(f_hi - f_lo))
    band = p[lo:hi + 1, :].sum(axis=0)                   # time profile of the band
    on = np.where(band > max(np.median(band) * 2.0, band.max() * 0.1))[0]
    t0 = int(t[on[0]] * fs) if on.size else 0
    t1 = int(t[min(on[-1], t.size - 1)] * fs) if on.size else n
    snr = float(10 * np.log10(s[lo:hi + 1].max() / (np.median(floor) + 1e-30)))
    return Detection(fc, bw, t0, t1, snr)
