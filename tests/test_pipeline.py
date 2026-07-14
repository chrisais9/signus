"""End-to-end blind demodulation: param recovery, v1-bug regressions, negatives."""

import numpy as np
import pytest

from signus.constellations import MODS
from signus.gen import GenParams, generate
from signus.pipeline import analyze, analyze_file
from signus.sigio import Meta, decode

_SNR = {"bpsk": 14, "qpsk": 14, "8psk": 18, "16qam": 22, "64qam": 28}


def _run(p: GenParams):
    x, _ = generate(p)
    return analyze(x, Meta(p.fs, p.fmt, p.dtype))


def _check(r, p):
    assert r.mod == p.mod
    assert abs(r.fc - p.fc) < max(50, 1e-4 * p.fs)
    assert abs(r.baud - p.baud) / p.baud < 0.01
    assert r.lock >= (50 if p.mod.endswith("qam") else 60)


@pytest.mark.parametrize("mod", MODS)
def test_e2e_iq(mod):
    p = GenParams(mod=mod, fs=1e6, baud=1e5, snr=_SNR[mod], fc=8000.0,
                  phase=0.7, timing=0.35, seed=0)
    _check(_run(p), p)


@pytest.mark.parametrize("mod", ["bpsk", "qpsk"])
def test_e2e_real_passband(mod):
    p = GenParams(mod=mod, fs=1e6, baud=1e5, snr=_SNR[mod], fc=1e5, fmt="real", seed=1)
    _check(_run(p), p)


@pytest.mark.parametrize("kw", [
    {},                        # cfo = 0 (v1: DC-null destroyed on-tune captures)
    {"dc": 0.3 + 0.3j, "fc": 8000.0},  # LO leakage (v1: no DC block)
    {"pad": 0.5, "fc": 8000.0},        # bursty record (v1: burst detect was a no-op)
    {"baud": 1.3e5, "fc": 8000.0},     # non-integer samples/symbol
])
def test_v1_regressions(kw):
    p = GenParams(mod="qpsk", fs=1e6, baud=kw.pop("baud", 1e5), snr=16, seed=2, **kw)
    _check(_run(p), p)


def test_multi_burst_detect_and_select():
    p1 = GenParams(mod="qpsk", fs=1e6, baud=1e5, snr=18, fc=8000.0, pad=0.4,
                   seed=1, n_symbols=4000)
    p2 = GenParams(mod="16qam", fs=1e6, baud=1e5, snr=24, fc=8000.0, pad=0.4,
                   seed=2, n_symbols=4000)
    x = np.concatenate([generate(p1)[0], generate(p2)[0]])
    r0 = analyze(x, Meta(1e6, "iq", "f32"), burst=0)
    r1 = analyze(x, Meta(1e6, "iq", "f32"), burst=1)
    assert len(r0.bursts) == 2 and r0.burst_idx == 0 and r1.burst_idx == 1
    assert r0.mod == "qpsk" and r1.mod == "16qam"
    doc = r0.to_json(views=False)
    assert len(doc["bursts"]) == 2 and doc["burst_idx"] == 0


def test_carrier_alias_resolved_end_to_end():
    # 4*fc wraps past fs/2 AND the wrapped tone lands inside the unflagged zone
    p = GenParams(mod="qpsk", fs=1e6, baud=1e5, snr=14, fc=162500.0, seed=0)
    r = _run(p)
    assert r.alias_resolved and r.mod == "qpsk"
    assert abs(r.fc - p.fc) < 300 and r.lock >= 60


def test_low_rolloff_baud_fallback():
    # alpha=0.08 16qam: the |x|^2 line loses to low-frequency data junk without
    # the occupied-bandwidth prior (global peak lands ~75% low)
    p = GenParams(mod="16qam", fs=1e6, baud=1e5, snr=24, rolloff=0.08, fc=8000.0,
                  seed=1, phase=0.4, timing=0.3)
    r = _run(p)
    assert r.baud_fallback and r.mod == "16qam"
    assert abs(r.baud - p.baud) / p.baud < 0.01 and r.lock >= 50


def test_pure_noise_reports_no_lock():
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(60000) + 1j * rng.standard_normal(60000)) / np.sqrt(2)
    r = analyze(x, Meta(1e6, "iq", "f32"))
    assert r.lock < 40  # honest failure, no exception


def test_reader_negatives(tmp_path):
    f = tmp_path / "x_fs1000000_iq_i16.iq"
    f.write_bytes(b"")
    with pytest.raises(ValueError):
        analyze_file(str(f))
    assert decode(b"\x01\x02\x03", Meta(1e6, "iq", "i16")).size == 0  # odd bytes drop tail
    with pytest.raises(ValueError):
        analyze_file(str(tmp_path / "noname.bin"))


def test_blindness_invariance(tmp_path):
    """Detection must not change when the filename carries legacy truth tokens."""
    p = GenParams(mod="qpsk", fs=1e6, baud=1e5, snr=18, fc=8000.0, seed=3)
    x, _ = generate(p)
    blind, legacy = tmp_path / "a_fs1000000_iq_i16.iq", (
        tmp_path / "a_fs1000000_fc8000_baud100000_qpsk_snr18_iq_i16.iq")
    from signus.sigio import write
    write(str(blind), x, Meta(1e6, "iq", "i16"))
    write(str(legacy), x, Meta(1e6, "iq", "i16"))
    ra, rb = analyze_file(str(blind)), analyze_file(str(legacy))
    assert ra.to_json()["detected"] == rb.to_json()["detected"]


def test_exports_roundtrip(tmp_path):
    p = GenParams(mod="16qam", fs=1e6, baud=1e5, snr=24, fc=8000.0, seed=0)
    r = _run(p)
    r.save_symbols(str(tmp_path / "s.npy"))
    r.save_bits(str(tmp_path / "b.txt"))
    r.save_bits(str(tmp_path / "b.bin"), packed=True)
    r.save_iq(str(tmp_path / "c.f32"))
    r.save_report(str(tmp_path / "r.json"))
    syms = np.load(tmp_path / "s.npy")
    assert syms.dtype == np.complex64 and syms.size == r.symbols.size
    txt = (tmp_path / "b.txt").read_text()
    assert set(txt) <= {"0", "1"} and len(txt) == r.bits.size
    assert (tmp_path / "b.bin").stat().st_size == (r.bits.size + 7) // 8
    iq = np.fromfile(tmp_path / "c.f32", dtype="<f4")
    assert iq.size == 2 * r.iq_corr.size
    import json
    doc = json.loads((tmp_path / "r.json").read_text())
    assert doc["detected"]["mod"] == "16qam"


@pytest.mark.parametrize("mod,nsym", [("qpsk", 200), ("qpsk", 150), ("16qam", 200),
                                      ("bpsk", 200)])
def test_short_burst_baud_rescue(mod, nsym):
    # short + low-SNR burst: the default 4-block |x|^2 baud line picks junk and misclassifies;
    # the fewer-block rescue (lock-gated, keep-best) recovers it. Long signals never enter the
    # rescue -- the full sweep is byte-identical before/after (verified out-of-band).
    from signus.cli import ber
    x, tx = generate(GenParams(mod=mod, n_symbols=nsym, fs=1e6, baud=1e5, fc=8e4,
                               snr=15, seed=1))
    r = analyze(x, Meta(1e6, "iq", "f32", "le"))
    assert r.mod == mod
    assert ber(r.symbols, r.mod, tx) < 0.01
