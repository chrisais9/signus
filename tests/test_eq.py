"""Blind CMA/decision-directed equalizer against symbol-spaced multipath."""

import numpy as np
import pytest

from signus import classify as cl
from signus import dsp
from signus.eq import equalize
from signus.gen import GenParams, generate
from signus.lab import ber
from signus.pipeline import analyze
from signus.sigio import Meta

# Symbol-spaced echoes (tap_sym=1.0): the ISI a symbol-rate equalizer can undo.
CHANS = {
    "A": (1.0, 0.0, 0.35 * np.exp(0.8j)),   # sparse 2-symbol echo
    "B": (1.0, 0.45 * np.exp(1j * 0.8), 0.2j),
    "C": (0.4, 1.0, 0.3),                   # non-minimum phase (hardest)
}
SNR = {"qpsk": 18, "16qam": 24}


def _raw_symbols(mod, taps):
    """Front-end up to timing recovery (pre fine-sync), like pipeline.analyze."""
    p = GenParams(mod=mod, n_symbols=6000, fs=1e6, baud=1e5, snr=SNR[mod], fc=8000.0,
                  phase=0.5, timing=0.3, taps=taps, tap_sym=1.0, seed=0)
    x, tx = generate(p)
    x = dsp.analytic(x)
    x = x - x.mean()
    s, e = dsp.find_burst(x, p.fs)
    xb = x[s:e]
    fc, _, _ = dsp.est_carrier(xb, p.fs, blocks=2)
    xd = dsp.mix(xb, p.fs, fc)
    baud, _ = dsp.est_baud(xd, p.fs)
    a = dsp.est_rolloff(xd, p.fs, baud)
    return dsp.timing(dsp.matched(dsp.to_sps(xd, p.fs, baud), 4, a), 4), tx


def _fine(z, mod):
    return cl.align(dsp.ddsync(cl.align(z, mod), mod), mod)


@pytest.mark.parametrize("mod", ["qpsk", "16qam"])
@pytest.mark.parametrize("ch", ["A", "B"])
def test_equalize_recovers_minimum_phase(mod, ch):
    """The equalizer must lock a channel the phase-only loop cannot."""
    raw, _ = _raw_symbols(mod, CHANS[ch])
    before = cl.quality(_fine(raw, mod), mod).lock
    after = cl.quality(_fine(equalize(raw, mod), mod), mod).lock
    assert after >= (50 if mod.endswith("qam") else 60), (before, after)
    assert after > before + 3


def test_equalize_is_harmless_on_a_clean_channel():
    raw, _ = _raw_symbols("qpsk", ())
    before = cl.quality(_fine(raw, "qpsk"), "qpsk").lock
    after = cl.quality(_fine(equalize(raw, "qpsk"), "qpsk"), "qpsk").lock
    assert before - after < 3


def test_equalize_zero_ber_on_channel_a():
    raw, tx = _raw_symbols("qpsk", CHANS["A"])
    z = _fine(equalize(raw, "qpsk"), "qpsk")
    assert ber(z, "qpsk", tx) == 0.0


@pytest.mark.parametrize("mod", ["qpsk", "16qam"])
def test_nonminimum_phase_is_honestly_rejected(mod):
    """A symbol-spaced equalizer cannot invert a non-minimum-phase channel; the
    pipeline must not lock in a low-quality, possibly wrong-mod fit."""
    p = GenParams(mod=mod, n_symbols=6000, fs=1e6, baud=1e5, snr=SNR[mod], fc=8000.0,
                  taps=CHANS["C"], tap_sym=1.0, seed=0)
    x, _ = generate(p)
    r = analyze(x, Meta(p.fs, p.fmt, p.dtype))
    assert not r.eq_applied or r.lock >= 50


@pytest.mark.parametrize("mod", ["qpsk", "16qam"])
def test_pipeline_rescues_multipath_end_to_end(mod):
    """Full blind chain: heavy ISI misclassifies, CMA opens the eye, re-classify wins."""
    p = GenParams(mod=mod, n_symbols=6000, fs=1e6, baud=1e5, snr=SNR[mod], fc=8000.0,
                  phase=0.5, timing=0.3, taps=CHANS["B"], tap_sym=1.0, seed=0)
    x, _ = generate(p)
    r = analyze(x, Meta(p.fs, p.fmt, p.dtype))
    assert r.eq_applied and r.mod == mod
    assert r.lock >= (50 if mod.endswith("qam") else 60)


def test_fse_rescues_long_echo_end_to_end():
    """0.8-gain echo at 1.5 symbols: symbol-rate folding defeats `equalize` (and the
    channel fakes symmetry 8), so the T/2 FSE + neutral re-classify must win."""
    p = GenParams(mod="qpsk", n_symbols=6000, fs=1e6, baud=1e5, snr=18, fc=8000.0,
                  taps=(1.0, 0.8), tap_sym=1.5, seed=0)
    x, tx = generate(p)
    r = analyze(x, Meta(p.fs, p.fmt, p.dtype))
    assert r.eq_applied and r.eq_mode == "fse" and r.mod == "qpsk"
    assert r.lock >= 60
    assert ber(r.symbols, "qpsk", tx) <= 0.002
