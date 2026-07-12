"""Spectrum + waterfall views of the burst (display only, never used for detection)."""

import numpy as np
from scipy.signal import welch


def _db(p: np.ndarray) -> np.ndarray:
    return 10 * np.log10(p + 1e-20)


def spectrum(x: np.ndarray, fs: float, bins: int = 256) -> dict:
    """Averaged PSD of the burst, decimated to `bins` points. Frequencies in kHz."""
    nper = int(min(4096, max(64, x.size // 8)))
    f, p = welch(x, fs=fs, nperseg=nper, return_onesided=False, detrend=False)
    f, p = np.fft.fftshift(f), _db(np.fft.fftshift(p))
    if f.size > bins:  # max-pool so narrow tones survive decimation
        k = f.size // bins
        f, p = f[: k * bins].reshape(bins, k).mean(1), p[: k * bins].reshape(bins, k).max(1)
    return {"f": np.round(f / 1e3, 3).tolist(), "db": np.round(p, 2).tolist()}


def waterfall(x: np.ndarray, fs: float, rows: int = 72, bins: int = 144) -> dict:
    """Time-frequency magnitude (dB), row-major, robustly scaled by the caller."""
    nfft = 1 << int(np.ceil(np.log2(max(bins * 2, 64))))
    step = max(1, (x.size - nfft) // max(rows - 1, 1))
    idx = np.arange(rows) * step
    idx = idx[idx + nfft <= x.size]
    if idx.size == 0:
        return {"rows": 0, "bins": bins, "db": [], "dt": 0.0}
    win = np.hanning(nfft)
    seg = np.stack([x[i:i + nfft] * win for i in idx])
    s = np.abs(np.fft.fftshift(np.fft.fft(seg, axis=1), axes=1)) ** 2
    k = nfft // bins
    s = s[:, : k * bins].reshape(idx.size, bins, k).mean(2)
    return {"rows": int(idx.size), "bins": bins, "dt": float(step / fs),
            "db": np.round(_db(s).ravel(), 2).tolist()}
