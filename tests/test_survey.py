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


@pytest.mark.parametrize("sep", [150e3, 100e3, 50e3])
def test_detect_separates_adjacent_channels(sep):
    # two narrowband emitters a few bandwidths apart must NOT merge into one blob
    # (the merge gap heals intra-signal splits, never swallows a neighbour)
    rng = np.random.default_rng(0)
    a = _emitter("qpsk", 20e3, -sep / 2, 0.0, 1, 8000)
    b = _emitter("qpsk", 20e3, +sep / 2, 0.0, 2, 8000)
    mix = a + b + np.sqrt(0.03 / 2) * (rng.standard_normal(N) + 1j * rng.standard_normal(N))
    dets = detect(mix, FS)
    assert len(dets) == 2
    assert abs(dets[0].fc + sep / 2) < 5e3 and abs(dets[1].fc - sep / 2) < 5e3


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


def test_detect_degenerate_inputs_never_crash():
    assert detect(np.zeros(0, complex), FS) == []          # empty -> [], not AxisError
    for k in (1, 100, 255, 256):                           # short record -> no crash
        detect(np.random.default_rng(k).standard_normal(k).astype(complex), FS)


def test_detect_signal_at_band_edge_is_one_emitter():
    # a signal straddling +-fs/2 must be one detection, not two edge halves
    rng = np.random.default_rng(0)
    x = _emitter("qpsk", 40e3, 0.0, 0.0, 1, 8000)          # padded to length N
    edge = x * np.exp(2j * np.pi * (FS / 2) * np.arange(N) / FS)
    edge = edge + np.sqrt(0.02 / 2) * (rng.standard_normal(N) + 1j * rng.standard_normal(N))
    dets = detect(edge, FS)
    assert len(dets) == 1
    assert abs(abs(dets[0].fc) - FS / 2) < 5e3             # near the Nyquist edge


def test_survey_one_bad_channel_does_not_abort(monkeypatch):
    # a channel whose analyze() raises must be marked 'error', not sink the whole survey
    import signus.pipeline as pl
    mix, _ = _mixture()
    orig, seen = pl.analyze, {"n": 0}

    def flaky(ch, meta, diff=False):
        seen["n"] += 1
        if seen["n"] == 1:
            raise ValueError("simulated degenerate channel")
        return orig(ch, meta, diff=diff)

    monkeypatch.setattr(pl, "analyze", flaky)
    s = pl.survey(mix, Meta(FS, "iq", "f32", "le", False))
    assert len(s.emitters) == 3
    assert sum(e.kind == "error" for e in s.emitters) == 1
    assert any(e.result is not None for e in s.emitters)   # the others still demodulated


def test_survey_reports_analog_emitter():
    # an FM-voice channel dropped into a wide capture must be reported, not demodulated
    rng = np.random.default_rng(0)
    fm = _fm_voice(N, FS)
    x = fm * np.exp(2j * np.pi * 2e5 * np.arange(N) / FS)
    x = x + np.sqrt(0.02 / 2) * (rng.standard_normal(N) + 1j * rng.standard_normal(N))
    s = survey(x, Meta(FS, "iq", "f32", "le", False))
    analog = [e for e in s.emitters if e.kind == "analog"]
    assert analog and all(e.result is None for e in analog)


# --- survey_web (roadmap #6): web payload assembly ---------------------------

def test_survey_web_single_signal_matches_analyze():
    # <=1 detected emitter -> 'single' mode returns the UNCHANGED direct analyze result
    from signus.pipeline import analyze, survey_web
    x, _ = generate(GenParams(mod="qpsk", n_symbols=8000, fs=FS, baud=1e5, fc=8e3,
                              snr=22, seed=0))
    meta = Meta(FS, "iq", "f32", "le", False)
    web = survey_web(x, meta)
    assert web["mode"] == "single"
    assert web["result"] == analyze(x, meta).to_json()   # byte-identical: no regression


def test_survey_web_multi_has_overview_and_emitter_details():
    from signus.pipeline import survey_web
    mix, _ = _mixture()
    web = survey_web(mix, Meta(FS, "iq", "f32", "le", False))
    assert web["mode"] == "survey"
    assert {"spectrum", "waterfall", "fs"} <= set(web["overview"])
    assert web["overview"]["waterfall"]["rows"] > 0
    assert len(web["emitters"]) == 3
    for e in web["emitters"]:                             # box geometry present for every box
        assert {"kind", "abs_fc", "det"} <= set(e)
        assert {"fc", "bw", "t0", "t1", "snr_db"} <= set(e["det"])
    dig = [e for e in web["emitters"] if e["kind"] in ("linear", "fsk")]
    assert dig and all("constellation" in e["result"] for e in dig)


# --- RF absolute centre frequency (roadmap #7) -------------------------------

def test_rf_center_parsed_from_filename_and_sigmf(tmp_path):
    import json

    from signus.sigio import parse_name, parse_sigmf
    assert parse_name("cap_fs20e6_rf162e6_iq_i16.dat").rf_center == 162e6
    assert parse_name("cap_fs1000000_iq_f32.dat").rf_center is None      # absent -> None
    (tmp_path / "s.sigmf-meta").write_text(json.dumps({
        "global": {"core:datatype": "cf32_le", "core:sample_rate": 1e6},
        "captures": [{"core:sample_start": 0, "core:frequency": 161.975e6}]}))
    assert parse_sigmf(str(tmp_path / "s.sigmf-data")).rf_center == 161.975e6


def test_rf_reported_as_real_frequency_else_unchanged():
    from signus.pipeline import analyze
    x, _ = generate(GenParams(mod="qpsk", n_symbols=8000, fs=FS, baud=1e5, fc=8e3, snr=22, seed=0))
    d0 = analyze(x, Meta(FS, "iq", "f32", "le", False)).to_json()["detected"]
    d1 = analyze(x, Meta(FS, "iq", "f32", "le", False, rf_center=162e6)).to_json()["detected"]
    assert d0["rf_hz"] is None                                  # no rf -> unchanged
    assert d1["rf_hz"] == pytest.approx(162e6 + d1["fc"], abs=1)  # rf -> real frequency
    assert d0["fc"] == d1["fc"]                                 # baseband fc identical (no regress)


def test_survey_web_carries_rf_center():
    from signus.pipeline import survey_web
    mix, _ = _mixture()
    web = survey_web(mix, Meta(FS, "iq", "f32", "le", False, rf_center=162e6))
    assert web["mode"] == "survey" and web["rf_center"] == 162e6


# --- chirp / CSS (LoRa) detection (roadmap: chirp support) -------------------

def _lora(sf, bw, nsym, fc, snr, seed, fs=FS, n=N):
    """LoRa-like CSS test fixture: cyclically-shifted up-chirps + 8 base-chirp preamble.
    Independent of the receiver detector (anti-shared-bug)."""
    rng = np.random.default_rng(seed)
    sps = int(round(fs * 2 ** sf / bw))
    mu = bw * bw / 2 ** sf
    syms = np.concatenate([np.zeros(8, int), rng.integers(0, 2 ** sf, nsym)])
    parts = []
    for s in syms:
        f = -bw / 2 + s * bw / 2 ** sf + mu * np.arange(sps) / fs
        f = ((f + bw / 2) % bw) - bw / 2
        parts.append(np.exp(2j * np.pi * np.cumsum(f) / fs))
    x = np.concatenate(parts)
    x = x[:n] if x.size >= n else np.concatenate([x, np.zeros(n - x.size, complex)])
    x = x * np.exp(2j * np.pi * fc * np.arange(x.size) / fs)
    nv = 1 / 10 ** (snr / 10)
    return x + np.sqrt(nv / 2) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))


def test_chirp_detector_flags_lora_not_others():
    from signus.chirp import analyze_chirp, is_chirp
    lo = _lora(9, 125e3, 40, 0.0, 25, 1, n=100000)
    assert is_chirp(lo, FS)
    info = analyze_chirp(lo, FS)
    assert info["sf"] == 9 and info["up"]
    assert info["rs"] == pytest.approx(125e3 / 2 ** 9, rel=0.15)
    for mod in ("qpsk", "16qam", "64qam"):            # digital must NOT be flagged chirp
        x, _ = generate(GenParams(mod=mod, n_symbols=6000, fs=FS, baud=1e5, snr=20, seed=0))
        assert not is_chirp(x, FS)
    rng = np.random.default_rng(5)
    assert not is_chirp(np.sqrt(0.5) * (rng.standard_normal(60000)
                                        + 1j * rng.standard_normal(60000)), FS)


def test_survey_reports_lora_as_chirp():
    # a LoRa emitter in a wide capture -> kind 'chirp', characterized, NEVER demodulated
    lo = _lora(9, 125e3, 40, 2.2e5, 22, 2)
    rng = np.random.default_rng(0)
    x = lo + np.sqrt(0.02 / 2) * (rng.standard_normal(N) + 1j * rng.standard_normal(N))
    s = survey(x, Meta(FS, "iq", "f32", "le", False))
    chirps = [e for e in s.emitters if e.kind == "chirp"]
    assert chirps and all(e.result is None and e.info is not None for e in chirps)
    assert chirps[0].info["sf"] == 9


def test_server_survey_and_analyze_endpoints_smoke():
    import http.client
    import json
    import threading
    from http.server import ThreadingHTTPServer

    from signus.server import Handler
    mix, _ = _mixture()
    body = np.column_stack([mix.real, mix.imag]).ravel().astype("<f4").tobytes()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        qs = "?name=cap_fs1000000_iq_f32.dat"
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
        c.request("POST", "/api/survey" + qs + "&rf=161975000", body=body)
        doc = json.loads(c.getresponse().read())
        assert doc["mode"] == "survey" and len(doc["emitters"]) == 3
        assert doc["rf_center"] == 161975000.0        # ?rf= query -> real RF plumbed through
        c.request("POST", "/api/analyze" + qs, body=body)     # refactor must not break this
        r = c.getresponse()
        doc = json.loads(r.read())
        assert r.status == 200 and "detected" in doc and "constellation" in doc
        c.request("POST", "/api/survey?name=bad.dat", body=b"\x00\x00")  # unknown meta -> 400
        bad = c.getresponse()
        bad.read()
        assert bad.status == 400
    finally:
        srv.shutdown()


def test_survey_wide_box_is_isolated_not_duplicate():
    # a wide detection box (bw > ~0.28*fs) must still be low-passed to its own band, else extract
    # returns the whole capture and analyze re-demods a different, stronger emitter (confident dup).
    def _emit(mod, baud, fc, pdb, sd):
        x, _ = generate(GenParams(mod=mod, n_symbols=8000, fs=FS, baud=baud, fc=fc,
                                  snr=60, seed=sd))
        x = x[:N] if x.size >= N else np.concatenate([x, np.zeros(N - x.size, complex)])
        return x / np.sqrt(np.mean(np.abs(x) ** 2)) * 10 ** (pdb / 20)
    rng = np.random.default_rng(3)
    mix = _emit("16qam", 130e3, 250e3, -6, 1) + _emit("qpsk", 20e3, 120e3, 6, 2) \
        + 0.1 * (rng.standard_normal(N) + 1j * rng.standard_normal(N))
    s = survey(mix, Meta(FS, "iq", "f32", "le", False))
    wide = _match(s.emitters, 250e3, tol=12e3)
    assert wide and wide.result and wide.result.mod == "16qam"
