"""Tests for the decision layer: classify / align / quality."""

import numpy as np
import pytest
from scipy.signal import resample_poly

from signus.classify import _c42, _spread, classify, quality
from signus.constellations import ideal_points, mod_symmetry
from signus.gen import GenParams, generate

MODS = ("bpsk", "qpsk", "8psk", "16qam", "64qam")


def _cloud(mod: str, snr: float, seed: int, n: int = 4000) -> np.ndarray:
    """Ideal symbols + AWGN at the given sample-power SNR."""
    rng = np.random.default_rng(seed)
    pts = ideal_points(mod)
    s = pts[rng.integers(0, pts.size, n)]
    nvar = np.mean(np.abs(s) ** 2) / 10 ** (snr / 10)
    return s + np.sqrt(nvar / 2) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))


def _ring_ratio(z: np.ndarray) -> float:
    a = np.sort(np.abs(z / np.sqrt(np.mean(np.abs(z) ** 2))))
    return _spread(a, 3) / (_spread(a, 9) + 1e-12)


def _frontend(mod: str, snr: float = 25, cfo: float = 8000) -> np.ndarray:
    """Symbols recovered through a real front-end: mix off exact cfo, 4 sps, best phase."""
    p = GenParams(mod=mod, snr=snr, fc=cfo, seed=0)
    x, _ = generate(p)
    x = x - x.mean()  # front-end DC block
    x = x * np.exp(-1j * 2 * np.pi * cfo / p.fs * np.arange(x.size))
    x = resample_poly(x, 4, int(p.fs / p.baud))  # -> 4 samples/symbol
    k = int(np.argmax([np.mean(np.abs(x[j::4])) for j in range(4)]))
    return x[k::4]


@pytest.mark.parametrize("mod", MODS)
@pytest.mark.parametrize("snr", (25, 15))
@pytest.mark.parametrize("seed", range(4))
def test_classify_clouds(mod: str, snr: float, seed: int) -> None:
    assert classify(_cloud(mod, snr, seed), mod_symmetry(mod)) == mod


@pytest.mark.parametrize("snr", (25, 15))
@pytest.mark.parametrize("seed", range(4))
def test_c42_margins(snr: float, seed: int) -> None:
    assert _c42(_cloud("qpsk", snr, seed)) > 0.9
    assert _c42(_cloud("16qam", snr, seed)) < 0.7
    assert _c42(_cloud("64qam", snr, seed)) < 0.7


@pytest.mark.parametrize("seed", range(4))
def test_ring_ratio_margins(seed: int) -> None:
    assert _ring_ratio(_cloud("16qam", 20, seed)) < 2.4
    assert _ring_ratio(_cloud("64qam", 20, seed)) > 2.9


@pytest.mark.parametrize("mod", ("qpsk", "16qam", "64qam"))
def test_classify_frontend(mod: str) -> None:
    assert classify(_frontend(mod), mod_symmetry(mod)) == mod


@pytest.mark.parametrize("mod", ("qpsk", "16qam"))
def test_align_rotation_invariant(mod: str) -> None:
    z = _cloud(mod, 25, 0)
    q0 = quality(z, mod)
    q1 = quality(z * np.exp(1j * 1.2345), mod)
    assert abs(q0.mer_db - q1.mer_db) < 0.5
    assert q0.occupied == q1.occupied
    assert abs(q0.lock - q1.lock) < 1.0


@pytest.mark.parametrize("mod", MODS)
def test_quality_locks_on_clean(mod: str) -> None:
    assert quality(_cloud(mod, 40, 0), mod).lock > 90


def test_quality_noise_and_collapse() -> None:
    rng = np.random.default_rng(0)
    noise = (rng.standard_normal(4000) + 1j * rng.standard_normal(4000)) / np.sqrt(2)
    assert quality(noise, "16qam").lock < 30
    collapsed = np.full(4000, 0.7 + 0.7j)
    assert quality(collapsed, "64qam").lock < 5
