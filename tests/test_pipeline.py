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


@pytest.mark.parametrize("mod", ["bpsk", "qpsk", "8psk", "16qam"])
@pytest.mark.parametrize("scale", [1e-20, 1e-13, 1e20, 1e30])
def test_scale_invariant(mod, scale):
    # regression: at a pathological amplitude the fsk_gate CV epsilon and est_carrier's x**p
    # over/underflowed -> wrong mod/family (bpsk*1e-13 -> confident fsk2). analyze() normalizes an
    # extreme-scale burst so every estimator is scale-invariant; a normal capture (rms~1) is kept.
    x, _ = generate(GenParams(mod=mod, fs=1e6, baud=1e5, fc=8e3, snr=22, seed=0))
    r = analyze(x * scale, Meta(1e6, "iq", "f32"))
    assert r.family == "linear" and r.mod == mod, (mod, scale, r.family, r.mod)


@pytest.mark.parametrize("fc", [0.008, 0.02, 0.03, 0.05])
@pytest.mark.parametrize("seed", [1, 3, 17, 19])
def test_32qam_not_confident_64qam(fc, seed):
    # regression: 32qam's weak p=4 tone let est_carrier read symmetry 2 -> bpsk -> the eq rescue
    # dragged the cloud onto a 64qam lattice at a marginal lock (~51) = confident WRONG order. A
    # dense-QAM eq fit must clear a higher floor now, so a genuine 32qam is never mislabelled 64qam
    # at lock >= 50 (it reports honest-low instead).
    x, _ = generate(GenParams(mod="32qam", fs=1e6, baud=1e5, fc=fc * 1e6, snr=15,
                              rolloff=0.35, n_symbols=5000, seed=seed))
    r = analyze(x, Meta(1e6, "iq", "f32"))
    assert not (r.lock >= 50 and r.mod == "64qam"), (fc, seed, r.mod, r.lock)


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
    from signus.lab import ber
    x, tx = generate(GenParams(mod=mod, n_symbols=nsym, fs=1e6, baud=1e5, fc=8e4,
                               snr=15, seed=1))
    r = analyze(x, Meta(1e6, "iq", "f32", "le"))
    assert r.mod == mod
    assert ber(r.symbols, r.mod, tx) < 0.01


def test_8psk_high_carrier_offset():
    # resolve_alias must tile the WHOLE band for sym=8 (was capped at 0.3125*fs -> wrong fc)
    from signus.lab import ber
    for fc in (0.35e6, 0.40e6, -0.45e6):
        x, tx = generate(GenParams(mod="8psk", baud=2e4, n_symbols=8000, fs=1e6,
                                   fc=fc, snr=25, seed=0))
        r = analyze(x, Meta(1e6, "iq", "f32", "le"))
        assert r.mod == "8psk" and abs(r.fc - fc) < 300 and r.lock > 60
        assert ber(r.symbols, "8psk", tx) < 0.01


def test_robust_to_nonfinite_and_degenerate():
    # a corrupt sample must not derail a good signal; a dead capture must not crash
    m = Meta(1e6, "iq", "f32", "le")
    x, _ = generate(GenParams(mod="qpsk", n_symbols=4000, fs=1e6, baud=1e5, snr=25, seed=1))
    xn = x.copy()
    xn[10], xn[20] = np.nan, np.inf
    assert analyze(xn, m).mod == "qpsk"
    from signus.pipeline import survey
    for bad in (np.zeros(4000, complex), np.full(4000, np.nan, complex),
                np.full(4000, np.inf, complex)):
        analyze(bad, m)                     # degenerate -> nonsense result, but NO crash
    survey(np.zeros(20000, complex), m)     # and the survey path too


@pytest.mark.parametrize("taps,ts", [((1.0, 0.5), 2.0), ((1.0, 0.5, 0.3), 1.0),
                                     ((1.0, 0.5), 3.0), ((1.0, 0.5), 6.0)])
def test_post_echo_defold_rescue(taps, ts):
    # a benign post-echo folds qpsk onto a QAM lattice at high lock (confident wrong); the
    # de-fold rescue equalizes and recovers the true lower order. Genuine QAM is never demoted
    # (the full sweep is byte-identical), so this only ever helps.
    from signus.lab import ber
    x, tx = generate(GenParams(mod="qpsk", n_symbols=6000, fs=1e6, baud=1e5, snr=25,
                               fc=8e3, taps=taps, tap_sym=ts, seed=0))
    r = analyze(x, Meta(1e6, "iq", "f32", "le"))
    assert r.mod == "qpsk"
    assert ber(r.symbols, "qpsk", tx) < 0.01


@pytest.mark.parametrize("mod,baud,fc", [("bpsk", 2e4, 0.495e6), ("bpsk", 1e5, 0.5e6),
                                         ("qpsk", 1e5, 0.49e6), ("qpsk", 1e5, 0.5e6),
                                         ("8psk", 2e4, 0.49e6)])
def test_band_edge_carrier_resolves(mod, baud, fc):
    # carriers near/at +-fs/2 straddle the band; the cell-energy alias pick resolves them
    # (a two-sided spectral centroid was corrupted by the wrap -> confident wrong decode).
    from signus.lab import ber
    x, tx = generate(GenParams(mod=mod, baud=baud, fs=1e6, n_symbols=8000, fc=fc, snr=25, seed=0))
    r = analyze(x, Meta(1e6, "iq", "f32", "le"))
    assert r.mod == mod
    assert ber(r.symbols, mod, tx) < 0.01


def test_marginal_bpsk_not_promoted_to_qpsk():
    # a marginal BPSK (lock just under _EQ_LOCK) triggers the eq rescue; _rescue must re-classify
    # with the estimated symmetry (2), not only order-4, else it is forced to a confident QPSK.
    from signus.lab import ber
    x, tx = generate(GenParams(mod="bpsk", snr=16, n_symbols=8000, fs=1e6, baud=4e5,
                               rolloff=0.05, seed=0))
    r = analyze(x, Meta(1e6, "iq", "i16", "le"))
    assert r.mod == "bpsk"
    assert ber(r.symbols, "bpsk", tx) < 0.01


@pytest.mark.parametrize("phase,echo", [(0.9, 0.2), (0.9, 0.5), (0.6, 0.3), (1.2, 0.2)])
def test_bpsk_multipath_not_promoted_to_qpsk(phase, echo):
    # BPSK + one symbol-spaced echo + carrier offset enters the eq rescue; CMA is modulus-only
    # so it ALSO fits a phantom qpsk ring at a marginally higher lock. symmetry==2 already put
    # bpsk in the candidate set -- the Occam de-fold must keep bpsk (a lower order can't be an
    # over-fit of qpsk), not report a confident wrong qpsk. Genuine qpsk (symmetry 4) is untouched.
    from signus.lab import ber
    x, tx = generate(GenParams(mod="bpsk", fs=1e6, baud=1e5, snr=16, fc=0.02e6, phase=phase,
                               taps=(1.0, echo * np.exp(1j * 0.5)), tap_sym=1.0, seed=0))
    r = analyze(x, Meta(1e6, "iq", "f32", "le"))
    assert r.mod == "bpsk"
    assert ber(r.symbols, "bpsk", tx) < 0.01


def test_extreme_magnitude_and_empty_do_not_crash():
    m = Meta(1e6, "iq", "f32", "le")
    x, _ = generate(GenParams(mod="qpsk", n_symbols=4000, fs=1e6, baud=1e5, snr=25, seed=1))
    xn = x.copy()
    xn[100] = 1e300 + 1e300j                          # finite but overflows |x|^2 -> was a crash
    assert analyze(xn, m).mod == "qpsk"
    with pytest.raises(ValueError):                   # empty -> clean rejection, not a crash
        analyze(np.zeros(0, complex), m)


@pytest.mark.parametrize("scale", [1e150, 1e152, 1e154, 1e-140])
def test_pathological_uniform_scale_still_decodes(scale):
    # regression in the scale-normalize guard itself: it ran AFTER analytic()'s magnitude
    # sanitizer, so a uniformly extreme capture (every sample near/over the 1e150 ceiling) had
    # its samples zeroed WHOLESALE before any normalize could help -- analyze returned garbage
    # mods and, at 1e154, a lock=nan / NaN-symbols Result. A robust (99th-percentile) rescale
    # BEFORE the sanitizer keeps the whole range decodable; spike outliers (see the test above)
    # leave the percentile untouched, so the sanitizer still handles them sample-wise.
    m = Meta(1e6, "iq", "f32", "le")
    x, _ = generate(GenParams(mod="qpsk", n_symbols=4000, fs=1e6, baud=1e5, snr=25, seed=1))
    r = analyze(x * scale, m)
    assert r.mod == "qpsk" and r.lock >= 90, (scale, r.mod, r.lock)
    assert not np.isnan(r.symbols).any()
