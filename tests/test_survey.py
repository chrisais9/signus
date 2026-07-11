"""Wideband survey: detection, channel extraction, triage, and per-emitter demod.
Regression-locks the empirical multi-emitter break/fix and the single-signal path."""

import numpy as np
import pytest

from signus.detect import detect
from signus.gen import GenParams, generate
from signus.pipeline import survey
from signus.sigio import Meta
from signus.triage import family

FS = 1e6
N = 240000


def _emitter(mod, baud, fc, power_db, seed, n_symbols, **kw):
    x, _ = generate(GenParams(mod=mod, n_symbols=n_symbols, fs=FS, baud=baud, fc=fc,
                              snr=60.0, seed=seed, **kw))
    x = x[:N] if x.size >= N else np.concatenate([x, np.zeros(N - x.size, complex)])
    return x / np.sqrt(np.mean(np.abs(x) ** 2)) * 10 ** (power_db / 20)


def _mixture(noise_var=0.125, seed=0):
    """Three emitters at distinct carriers + a wideband noise floor (the audit fixture:
    qpsk -300k strong, 16qam +180k medium, msk +350k weak/AIS-like)."""
    rng = np.random.default_rng(seed)
    specs = [("qpsk", 25e3, -300e3, +6.0, 1, 6000, {"rolloff": 0.35}),
             ("16qam", 50e3, +180e3, 0.0, 2, 12000, {"rolloff": 0.35}),
             ("msk", 9.6e3, +350e3, -6.0, 3, 2304, {})]
    mix = sum(_emitter(m, b, fc, p, s, ns, **kw) for m, b, fc, p, s, ns, kw in specs)
    mix = mix + np.sqrt(noise_var / 2) * (rng.standard_normal(N) + 1j * rng.standard_normal(N))
    truth = [("qpsk", -300e3, 25e3), ("16qam", 180e3, 50e3), ("msk", 350e3, 9.6e3)]
    return mix, truth


def _match(emitters, fc, tol=8e3):
    return next((e for e in emitters if abs(e.abs_fc - fc) < tol), None)


def test_detect_finds_all_three_emitters():
    mix, truth = _mixture()
    dets = detect(mix, FS)
    assert len(dets) == 3
    for det, (_, fc, _) in zip(sorted(dets, key=lambda d: d.fc), truth, strict=True):
        assert abs(det.fc - fc) < 5e3            # centre within 5 kHz
        assert det.bw > 0


def test_detect_rejects_pure_noise():
    rng = np.random.default_rng(7)
    noise = np.sqrt(0.5) * (rng.standard_normal(N) + 1j * rng.standard_normal(N))
    assert detect(noise, FS) == []


def test_survey_recovers_every_emitter():
    # the audit's confidently-wrong single answer becomes three correct ones
    mix, _ = _mixture()
    s = survey(mix, Meta(FS, "iq", "f32", "le", False))
    assert len(s.emitters) == 3
    a = _match(s.emitters, -300e3)
    b = _match(s.emitters, 180e3)
    c = _match(s.emitters, 350e3)
    assert a and a.result.mod == "qpsk" and a.result.baud == pytest.approx(25e3, rel=0.02)
    assert a.result.lock > 60
    assert b and b.result.mod == "16qam" and b.result.baud == pytest.approx(50e3, rel=0.02)
    assert b.result.lock > 60
    assert c and c.kind == "fsk" and c.result.mod == "msk"
    assert c.result.baud == pytest.approx(9.6e3, rel=0.03)


@pytest.mark.parametrize("mod,baud,fc", [("qpsk", 1e5, 8e3), ("16qam", 1e5, 8e3)])
def test_survey_single_signal_fills_band(mod, baud, fc):
    # the legacy single-signal regime: exactly one emitter, correctly demodulated
    x, _ = generate(GenParams(mod=mod, n_symbols=8000, fs=FS, baud=baud, fc=fc,
                              snr=22, seed=0))
    s = survey(x, Meta(FS, "iq", "f32", "le", False))
    assert len(s.emitters) == 1
    e = s.emitters[0]
    assert e.result.mod == mod and e.result.baud == pytest.approx(baud, rel=0.02)
    assert e.result.lock > 60


def _fm_voice(n, fs, dev=8e3, seed=0):
    from scipy.signal import firwin
    rng = np.random.default_rng(seed)
    msg = np.convolve(rng.standard_normal(n), firwin(201, 3e3 / (fs / 2)), "same")
    return np.exp(1j * 2 * np.pi * dev * np.cumsum(msg / msg.std()) / fs)


def test_triage_flags_analog_and_tone_not_forced_to_constellation():
    fs = 2e5
    assert family(_fm_voice(20000, fs), fs) == "analog"
    tone = np.exp(2j * np.pi * 1e4 * np.arange(20000) / fs)
    assert family(tone, fs) == "tone"
    x, _ = generate(GenParams(mod="qpsk", n_symbols=6000, fs=fs, baud=4e4, snr=25, seed=0))
    assert family(x[:20000], fs) == "linear"


def test_survey_reports_analog_emitter():
    # an FM-voice channel dropped into a wide capture must be reported, not demodulated
    rng = np.random.default_rng(0)
    fm = _fm_voice(N, FS)
    x = fm * np.exp(2j * np.pi * 2e5 * np.arange(N) / FS)
    x = x + np.sqrt(0.02 / 2) * (rng.standard_normal(N) + 1j * rng.standard_normal(N))
    s = survey(x, Meta(FS, "iq", "f32", "le", False))
    analog = [e for e in s.emitters if e.kind == "analog"]
    assert analog and all(e.result is None for e in analog)
