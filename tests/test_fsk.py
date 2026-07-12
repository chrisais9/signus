"""FSK/MSK gate + demod against generator ground truth."""

import numpy as np
import pytest

from signus.constellations import FSK_MODS, GEN_MODS, mod_order
from signus.fsk import analyze_fsk, fsk_gate
from signus.gen import GenParams, generate

_LINEAR = [m for m in GEN_MODS if m not in FSK_MODS]


def _ber(rx: np.ndarray, tx: np.ndarray, bps: int, crop: int = 20) -> float:
    """Min BER over integer symbol shifts in [-8, 8], edge symbols cropped."""
    best = 1.0
    for sh in range(-8, 9):
        s = sh * bps
        a = rx[s:] if s > 0 else rx
        b = tx[-s:] if s < 0 else tx
        n = min(a.size, b.size)
        c = crop * bps
        if n <= 2 * c:
            continue
        best = min(best, float(np.mean(a[c:n - c] != b[c:n - c])))
    return best


# --- gate: both directions across every generator vocabulary mod ------------

@pytest.mark.parametrize("mod", _LINEAR)
@pytest.mark.parametrize("snr", [25, 15])
def test_gate_false_for_linear(mod, snr):
    for seed in range(4):
        p = GenParams(mod=mod, n_symbols=4000, snr=snr, seed=seed)
        x, _ = generate(p)
        assert not fsk_gate(x, p.fs), (mod, snr, seed)


@pytest.mark.parametrize("mod", FSK_MODS)
@pytest.mark.parametrize("snr", [25, 15, 10])
def test_gate_true_for_fsk(mod, snr):
    for seed in range(4):
        p = GenParams(mod=mod, n_symbols=4000, snr=snr, seed=seed)
        x, _ = generate(p)
        assert fsk_gate(x, p.fs), (mod, snr, seed)


def test_gate_noise_is_false():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(40000) + 1j * rng.standard_normal(40000)
    assert not fsk_gate(x, 1e6)


def test_gate_immune_to_large_cfo():
    # regression: at fc=0.1625*fs a linear mod's phase-transition spikes wrapped
    # past +-fs/2 and faked a bimodal instantaneous frequency
    p = GenParams(mod="qpsk", n_symbols=6000, snr=14, fc=162500.0, seed=2, phase=0.8)
    x, _ = generate(p)
    assert not fsk_gate(x, p.fs)
    pf = GenParams(mod="fsk2", n_symbols=4000, snr=18, h=0.7, fc=50000.0, seed=0)
    xf, _ = generate(pf)
    assert fsk_gate(xf, pf.fs)  # recentering must not cost genuine CFO'd FSK


# --- full demod -------------------------------------------------------------

_CASES = [("fsk2", 0.7), ("fsk2", 1.0), ("fsk4", 0.5), ("fsk4", 0.7),
          ("fsk4", 1.0), ("msk", 0.5)]


@pytest.mark.parametrize("mod,h", _CASES)
@pytest.mark.parametrize("snr", [20, 14])
def test_demod(mod, h, snr):
    bps = mod_order(mod).bit_length() - 1
    for seed in range(4):
        p = GenParams(mod=mod, n_symbols=4000, snr=snr, h=h, seed=seed)
        x, tx = generate(p)
        r = analyze_fsk(x, p.fs)
        tag = (mod, h, snr, seed)
        assert r["mod"] == mod, tag
        assert abs(r["fc"]) < 300, tag
        assert abs(r["baud"] - p.baud) / p.baud < 0.02, tag
        assert abs(r["h"] - h) < 0.15, tag
        assert r["lock"] >= 60, tag
        ber = _ber(r["bits"], tx, bps)
        # fsk4 Gray levels: an adjacent-level slip flips a single bit
        limit = 0.002 if mod == "fsk4" else (0.0 if snr >= 20 else 0.003)
        assert ber <= limit, (tag, ber)


@pytest.mark.parametrize("mod,h", [("fsk2", 1.0), ("fsk4", 0.7), ("msk", 0.5)])
def test_demod_cfo(mod, h):
    """A large carrier offset is absorbed by recentering on mean(f)."""
    bps = mod_order(mod).bit_length() - 1
    p = GenParams(mod=mod, n_symbols=4000, snr=20, h=h, seed=0, fc=8000)
    x, tx = generate(p)
    r = analyze_fsk(x, p.fs)
    assert r["mod"] == mod
    assert abs(r["baud"] - p.baud) / p.baud < 0.02
    assert abs(r["h"] - h) < 0.15
    assert r["lock"] >= 60
    assert _ber(r["bits"], tx, bps) <= (0.002 if mod == "fsk4" else 0.0)
