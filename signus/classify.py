"""Decision layer: modulation classification, rotation alignment, lock quality, SNR."""

from dataclasses import dataclass

import numpy as np

from .constellations import ideal_points, mod_symmetry
from .dsp import _kmeans1d

# Amplitude-ring separators, calibrated through the real front-end (seeds 0-3, snr 14-25):
# 16qam <=0.80 and 64qam <=0.99 vs 32qam >=1.16 on s3*s9/s5**2.
_D32 = 1.07
_C42_PSK = 0.80   # constant-modulus 4th cumulant: qpsk ~1.0, QAM <0.7
_CM_TOL = 0.2     # 1-ring spread guard for constant-modulus signals
_R39 = 2.7        # s3/s9: 16qam ~2.1, 64qam >=3.0


def _agc(z: np.ndarray) -> np.ndarray:
    """Scale to unit average power."""
    return z / np.sqrt(np.mean(np.abs(z) ** 2))


def _c42(z: np.ndarray) -> float:
    """Magnitude of the normalized 4th-order cumulant C42 (rotation invariant)."""
    z = _agc(z)
    p = np.mean(np.abs(z) ** 2)
    return float(abs(np.mean(np.abs(z) ** 4) - abs(np.mean(z**2)) ** 2 - 2 * p**2))


def _spread(a: np.ndarray, k: int) -> float:
    """RMS within-cluster spread from the shared 1-D k-means (sorted-value ring test)."""
    return _kmeans1d(a, k)[2]


def classify(z: np.ndarray, symmetry: int) -> str:
    """Symmetry 2->bpsk, 8->8psk; 4-> qpsk/16qam/32qam/64qam via C42 then amplitude rings.

    Not blind classes (documented): oqpsk collides with square-QAM on every ring/cumulant
    feature; pi4dqpsk's 4th power carries a pi-per-symbol rotation, so the carrier estimate
    absorbs baud/8 and it arrives here looking exactly like qpsk (its bits still decode
    correctly via differential demapping).
    """
    if symmetry == 2:
        return "bpsk"
    if symmetry == 8:
        return "8psk"
    z = _agc(z)
    if _c42(z) >= _C42_PSK:
        return "qpsk"
    a = np.sort(np.abs(z))
    s1 = _spread(a, 1)
    if s1 <= _CM_TOL:  # constant-modulus guard (a PSK cloud that slipped the C42 gate)
        return "qpsk"
    s3, s5, s9 = _spread(a, 3), _spread(a, 5), _spread(a, 9)
    if s3 * s9 / (s5 * s5 + 1e-18) > _D32:  # only a 5-ring cross collapses at k=5
        return "32qam"
    # amplitude-ring ratio, NEVER best-fit EVM (a denser grid always fits closer)
    return "16qam" if s3 / (s9 + 1e-12) < _R39 else "64qam"


_SNR_CEIL = 40.0  # above this the moments cannot resolve the noise term


def snr_m2m4(z: np.ndarray, mod: str) -> float:
    """Blind symbol-SNR (dB) from the 2nd/4th moments, corrected for the
    constellation kurtosis. Saturates at _SNR_CEIL (noise below the moments'
    resolution); NaN when the moments leave the valid region entirely."""
    ka = np.mean(np.abs(ideal_points(mod)) ** 4)  # unit-power points => already /M2**2
    m2, m4 = np.mean(np.abs(z) ** 2), np.mean(np.abs(z) ** 4)
    disc = 2 * m2**2 - m4
    if disc <= 0:
        return float("nan")
    s = np.sqrt(disc / (2 - ka))
    n = m2 - s
    return min(float(10 * np.log10(s / n)), _SNR_CEIL) if n > 0 else _SNR_CEIL


def _nearest_dist(w: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Distance to nearest ideal point. Running min over the M points instead of a
    (..., M) broadcast temporary: identical values, far less memory, ~1.5-5x faster."""
    d = np.abs(w - pts[0])
    for p in pts[1:]:
        d = np.minimum(d, np.abs(w - p))
    return d


def align(z: np.ndarray, mod: str) -> np.ndarray:
    """AGC + resolve the M-fold rotation ambiguity by minimizing nearest-point distance."""
    z = _agc(z)
    pts = ideal_points(mod)
    sub = z[:1000]
    ang = np.linspace(0, 2 * np.pi / mod_symmetry(mod), 64, endpoint=False)
    rot = sub[None, :] * np.exp(1j * ang)[:, None]
    cost = _nearest_dist(rot, pts).mean(axis=1)
    i = int(np.argmin(cost))
    c0, c1, c2 = cost[(i - 1) % 64], cost[i], cost[(i + 1) % 64]  # parabolic sub-grid refine
    denom = c0 - 2 * c1 + c2
    # clip: on a flat cost surface (e.g. pure noise) the fit can explode
    delta = float(np.clip(0.5 * (c0 - c2) / denom, -1, 1)) if denom else 0.0
    return z * np.exp(1j * (ang[i] + delta * (ang[1] - ang[0])))


@dataclass
class Quality:
    evm: float
    mer_db: float
    lock: float
    occupied: int


def quality(z: np.ndarray, mod: str) -> Quality:
    """Alignment + EVM/MER, plus an occupancy-gated lock score in [0, 100]."""
    z = align(z, mod)
    pts = ideal_points(mod)
    m = pts.size
    idx = np.argmin(np.abs(z[:, None] - pts[None, :]), axis=1)
    d = pts[idx]
    evm = float(np.sqrt(np.mean(np.abs(z - d) ** 2) / np.mean(np.abs(d) ** 2)))
    mer_db = float(-20 * np.log10(evm + 1e-12))
    thr = max(3, 0.6 * len(z) / m)
    occupied = int(np.sum(np.bincount(idx, minlength=m) >= thr))
    lock = float(np.clip((mer_db - 6) / 19 * 100, 0, 100) * (occupied / m))
    return Quality(evm, mer_db, lock, occupied)
