"""Spectrum + spectrogram-strip views (display only, never used for detection)."""

import numpy as np
from scipy.signal import stft, welch


def _db(p: np.ndarray) -> np.ndarray:
    return 10 * np.log10(p + 1e-20)


def spectrum(x: np.ndarray, fs: float, bins: int = 512) -> dict:
    """sa 프로브의 PSD 판과 같은 조판: 실수부 welch(nperseg<=2048), 단측 0..fs/2 (kHz).
    표시 칸으로 줄일 때는 최대 풀링 -- 좁은 바늘이 살아남게."""
    xd = x.real if np.iscomplexobj(x) else x
    f, p = welch(xd, fs=fs, nperseg=int(min(2048, max(64, xd.size))))
    p = _db(p)
    if f.size > bins:  # max-pool so narrow tones survive decimation
        k = f.size // bins
        f, p = f[: k * bins].reshape(bins, k).mean(1), p[: k * bins].reshape(bins, k).max(1)
    return {"f": np.round(f / 1e3, 3).tolist(), "db": np.round(p, 2).tolist()}


def strip(x: np.ndarray, fs: float, max_cols: int = 1600) -> dict:
    """sa 프로브의 PNG 스펙트로그램 띠와 같은 조판의 회색조 이미지 (0..235, row-major,
    위 행 = +fs/2 쪽). hamming 256/128 STFT, 바닥 p25 + 대비폭 10..35 dB 클램프, 열은
    최대 풀링으로만 줄인다 -- 보간이 짧은 버스트를 뭉개지 않게."""
    xd = x.real if np.iscomplexobj(x) else x
    nper = int(min(256, max(64, 1 << int(np.log2(max(xd.size // 2, 64))))))
    _, _, z = stft(xd, fs=fs, window="hamming", nperseg=nper, noverlap=nper // 2,
                   return_onesided=True, boundary=None, padded=False)
    db = 10 * np.log10(np.abs(z) ** 2 + 1e-12)
    lo = float(np.percentile(db, 25))
    rg = float(max(10.0, min(35.0, np.percentile(db, 99.5) - lo)))
    g = np.clip((db - lo) / rg, 0, 1)[::-1]
    kp = max(1, g.shape[1] // max_cols)
    g = g[:, : g.shape[1] // kp * kp].reshape(g.shape[0], -1, kp).max(2)
    return {"rows": int(g.shape[0]), "cols": int(g.shape[1]),
            "g": np.round(g * 235).astype(int).ravel().tolist()}
