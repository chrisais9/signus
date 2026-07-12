"""Constellations, Gray labels, differential tables, FSK levels (shared reference)."""

import numpy as np

MODS = ("bpsk", "qpsk", "8psk", "16qam", "64qam")          # blind-classify targets
FSK_MODS = ("fsk2", "fsk4", "msk")
DIFF_MODS = ("dbpsk", "dqpsk", "pi4dqpsk")
GEN_MODS = MODS + ("32qam", "oqpsk") + DIFF_MODS + FSK_MODS  # generator vocabulary

# (data-alphabet order, constellation rotational symmetry). pi4dqpsk: 4 transition
# values per symbol but an 8-point composite constellation.
_INFO = {"bpsk": (2, 2), "qpsk": (4, 4), "8psk": (8, 8), "16qam": (16, 4),
         "64qam": (64, 4), "32qam": (32, 4), "oqpsk": (4, 4), "pi4dqpsk": (4, 8),
         "dbpsk": (2, 2), "dqpsk": (4, 4), "fsk2": (2, 0), "fsk4": (4, 0), "msk": (2, 0)}

# differential phase increment per Gray symbol value
DIFF_PHASES = {
    "dbpsk": np.array([0.0, np.pi]),
    "dqpsk": np.array([0.0, np.pi / 2, np.pi, -np.pi / 2]),
    "pi4dqpsk": np.array([np.pi / 4, 3 * np.pi / 4, -3 * np.pi / 4, -np.pi / 4]),
}


def family(mod: str) -> str:
    return "fsk" if mod in FSK_MODS else "linear"


def mod_order(mod: str) -> int:
    return _INFO[mod][0]


def mod_symmetry(mod: str) -> int:
    """Lowest power p for which x**p strips the modulation (square QAM: 4)."""
    return _INFO[mod][1]


def ideal_points(mod: str) -> np.ndarray:
    if mod in ("bpsk", "dbpsk"):
        return np.array([1.0 + 0j, -1.0 + 0j])
    if mod in ("qpsk", "oqpsk", "dqpsk"):
        return np.exp(1j * (np.pi / 4 + np.arange(4) * np.pi / 2))
    if mod in ("8psk", "pi4dqpsk"):
        return np.exp(2j * np.pi * np.arange(8) / 8)
    if mod == "32qam":  # 6x6 cross: grid minus the four corners
        lv = np.arange(6) * 2.0 - 5
        i, q = np.meshgrid(lv, lv)
        pts = (i + 1j * q).ravel()
        pts = pts[np.abs(pts.real * pts.imag) != 25]
        return pts / np.sqrt(np.mean(np.abs(pts) ** 2))
    n = 4 if mod == "16qam" else 8
    lv = np.arange(n) * 2.0 - (n - 1)
    i, q = np.meshgrid(lv, lv)
    pts = (i + 1j * q).ravel()
    return pts / np.sqrt(np.mean(np.abs(pts) ** 2))


def _gray(k: np.ndarray) -> np.ndarray:
    return k ^ (k >> 1)


def bit_labels(mod: str) -> np.ndarray:
    """Gray label per ideal_points index (PSK: phase Gray; square QAM: per-axis).
    32qam uses plain index labels (cross grid has no clean per-axis Gray)."""
    m = mod_order(mod)
    k = np.arange(m)
    if mod in ("16qam", "64qam"):
        n = 4 if mod == "16qam" else 8
        return _gray(k // n) * n + _gray(k % n)
    if mod == "32qam":
        return k
    return _gray(k)


def fsk_levels(mod: str) -> tuple[np.ndarray, np.ndarray]:
    """(frequency levels, Gray label per level) — e.g. fsk4: [-3,-1,1,3], [0,1,3,2]."""
    m = mod_order(mod)
    return np.arange(m) * 2.0 - (m - 1), _gray(np.arange(m))


def to_bits(labels: np.ndarray, mod: str) -> np.ndarray:
    """Integer labels -> flat 0/1 array, MSB first, log2(M) bits per label."""
    k = mod_order(mod).bit_length() - 1
    return (labels[:, None] >> np.arange(k - 1, -1, -1) & 1).ravel().astype(np.uint8)


def demap_bits(symbols: np.ndarray, mod: str) -> np.ndarray:
    """Hard-decide to nearest ideal point, return Gray bits."""
    pts = ideal_points(mod)
    idx = np.argmin(np.abs(symbols[:, None] - pts[None, :]), axis=1)
    return to_bits(bit_labels(mod)[idx], mod)


def demap_diff_bits(symbols: np.ndarray, mod: str) -> np.ndarray:
    """Differential demap: quantize successive phase differences to the mod's
    transition set. Rotation-ambiguity-free (no absolute phase reference)."""
    ph = DIFF_PHASES[mod]
    d = np.angle(symbols[1:] * np.conj(symbols[:-1]))
    err = np.angle(np.exp(1j * (d[:, None] - ph[None, :])))  # wrapped distance
    return to_bits(np.argmin(np.abs(err), axis=1), mod)
