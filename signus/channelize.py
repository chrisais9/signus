"""Channel extraction: pull one detected emitter out of a wideband capture as a
baseband stream, sized so the signal occupies ~1/4-1/3 of the reduced rate -- the
empirically-validated sweet spot for the downstream blind estimators (fill the band
and the matched filter starves; leave it too wide and noise dominates)."""

import numpy as np
from scipy.signal import firwin, resample_poly

from . import dsp
from .detect import Detection


def extract(x: np.ndarray, fs: float, det: Detection, *, target_frac: float = 0.28,
            pad: float = 1.5, ntaps: int = 129) -> tuple[np.ndarray, float]:
    """Mix det.fc to DC, low-pass to the channel, decimate; trim to the burst.
    Returns (baseband IQ, channel sample rate)."""
    bb = dsp.mix(x, fs, det.fc)
    want = max(det.bw * pad, fs * 1e-4)                  # passband to keep
    d = max(1, int(fs / (want / target_frac)))          # want ~= target_frac * fs/d
    fs_ch = fs / d
    if d > 1:
        cutoff = min(0.9 * fs_ch / 2, want / 2) / (fs / 2)
        bb = np.convolve(bb, firwin(ntaps, cutoff), "same")
        bb = resample_poly(bb, 1, d)
    a, b = int(det.t0 // d), int(det.t1 // d)
    if b - a >= 256:                                     # trim empty time, keep guard
        g = (b - a) // 20
        bb = bb[max(0, a - g):b + g]
    return bb.astype(np.complex128), fs_ch
