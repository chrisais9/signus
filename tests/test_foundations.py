"""Foundations: constellations / sigio (sample-type matrix) / gen (all mod families)."""

import os

import numpy as np
import pytest

from signus.constellations import (
    DIFF_MODS,
    DIFF_PHASES,
    FSK_MODS,
    GEN_MODS,
    MODS,
    bit_labels,
    demap_bits,
    demap_diff_bits,
    family,
    ideal_points,
    mod_order,
    to_bits,
)
from signus.gen import GenParams, generate, save
from signus.sigio import Meta, decode, make_name, parse_name, parse_sigmf, read, sidecar_read, write


@pytest.mark.parametrize("mod", MODS)
def test_labels_are_permutation_and_demap_roundtrips(mod):
    m = mod_order(mod)
    assert sorted(bit_labels(mod)) == list(range(m))
    idx = np.random.default_rng(0).integers(0, m, 500)
    rx = demap_bits(ideal_points(mod)[idx], mod)
    assert np.array_equal(rx, to_bits(bit_labels(mod)[idx], mod))


@pytest.mark.parametrize("mod", MODS)
def test_unit_power(mod):
    assert abs(np.mean(np.abs(ideal_points(mod)) ** 2) - 1.0) < 1e-9


def test_qam_gray_adjacency():
    lb = bit_labels("16qam").reshape(4, 4)
    for a, b in zip(lb[:-1].ravel(), lb[1:].ravel(), strict=True):
        assert bin(int(a) ^ int(b)).count("1") == 1


def test_parse_name_minimal_and_blind():
    m = parse_name(make_name("cap", Meta(2_400_000, "iq", "f32")))
    assert (m.fs, m.fmt, m.dtype) == (2_400_000, "iq", "f32") and m.ok()
    assert not parse_name("mystery.bin").ok()
    # legacy truth tokens are ignored, not parsed
    legacy = parse_name("sig_fs1000000_fc10000_baud9600_qpsk_snr20_real_i16.pcm")
    assert (legacy.fs, legacy.fmt) == (1_000_000, "real")


def test_make_name_is_dot_separated_and_ends_in_pcm():
    assert make_name("cap", Meta(2e6, "iq", "i16")) == "cap.cplx.2000000.16t.pcm"
    assert make_name("voice", Meta(48000, "real", "f32")) == "voice.real.48000.32f.pcm"
    assert make_name("x", Meta(1e6, "iq", "u8", "be", True)) == "x.cplx.1000000.8o.be.bitrev.pcm"


@pytest.mark.parametrize("name,want", [
    ("capture.cplx.2000000.16t.pcm", (2e6, "iq", "i16")),
    ("voice.real.48000.32f.pcm", (48000.0, "real", "f32")),
    ("rtl.cplx.2400000.8o.pcm", (2.4e6, "iq", "u8")),
    # 이름 자체가 숫자여도 샘플레이트를 헷갈리면 안 된다: fs 는 cplx|real 바로 다음 자리다
    ("20260801.cplx.1000000.16t.pcm", (1e6, "iq", "i16")),
    # 샘플타입이 없으면 i16 (기본값). 그래도 fs/fmt 는 읽힌다
    ("bare.cplx.1000000.pcm", (1e6, "iq", "i16")),
])
def test_parse_dot_format(name, want):
    m = parse_name(name)
    assert (m.fs, m.fmt, m.dtype) == want and m.ok()


@pytest.mark.parametrize("name", [
    "my_real.cplx.1000000.16t.pcm",     # 이름 끝이 real
    "x_real.cplx.1000000.32f.pcm",
    "cplx.1000000.16t.pcm",             # 라벨 없이 포맷이 0번 토막
])
def test_label_containing_a_format_word_never_flips_the_format(name):
    # 이름에 real/cplx 이 섞이면 어느 쪽이 진짜인지 헷갈린다. 복소 캡처를 real 로 읽으면
    # 조용히 쓰레기가 나온다(길이 2배, 성상도 붕괴) -- 이 저장소의 cardinal sin 이다.
    # 진짜 포맷은 샘플레이트를 데리고 다니는 쪽이라는 규칙으로 가른다.
    assert parse_name(name).fmt == "iq"


@pytest.mark.parametrize("name,fs,fmt", [
    # 라벨의 포맷 단어가 숫자까지 데리고 다니는 경우: 마지막 후보 규칙 + 제원 관문이 막는다.
    # 적대 감사에서 나온 실제 강탈 사례들 -- fs=20260801(날짜!)로 조용히 복조됐었다.
    ("rec_iq_20260801.cplx.2400000.16t.pcm", 2400000.0, "iq"),
    ("sdr_real_20260801.cplx.2400000.16t.pcm", 2400000.0, "iq"),
    ("real_2.cplx.2400000.16t.pcm", 2400000.0, "iq"),
    ("x_iq_48000_16t.cplx.1000000.16t.pcm", 1000000.0, "iq"),  # 관문까지 뚫는 라벨 -> 마지막 후보
])
def test_label_with_digits_never_hijacks_fs_or_fmt(name, fs, fmt):
    m = parse_name(name)
    assert (m.fs, m.fmt) == (fs, fmt)


@pytest.mark.parametrize("name,dtype,endian,bitrev", [
    # 제원 토막은 포맷 토막 **뒤**에만 산다: 라벨의 u8/f32/be/bitrev 는 제원이 아니다.
    ("u8_check.cplx.2400000.16t.pcm", "i16", "le", False),
    ("f32_test.cplx.1000000.16t.pcm", "i16", "le", False),
    ("be_run.cplx.1000000.16t.pcm", "i16", "le", False),
    ("bitrev_scan.cplx.1000000.16t.pcm", "i16", "le", False),
    ("rtl_u8.cplx.2400000.8o.pcm", "u8", "le", False),   # 진짜 제원(8o)은 그대로 읽힌다
])
def test_label_spec_words_never_pollute_dtype_endian_bitrev(name, dtype, endian, bitrev):
    m = parse_name(name)
    assert (m.dtype, m.endian, m.bitrev) == (dtype, endian, bitrev)


@pytest.mark.parametrize("name", [
    "cap.cplx.2.4e6.16t.pcm",       # 점 낀 샘플레이트: fs=2 Hz 로 조용히 오독됐었다
    "cap.cplx.48_000.16t.pcm",      # 밑줄 낀 샘플레이트: fs=48 Hz
    "capture_iq_20260728.raw",      # 라벨 숫자를 fs 로 발명했었다 (HEAD 는 크게 죽던 자리)
])
def test_broken_sample_rate_dies_loud_instead_of_shrinking(name):
    # 조용한 오답 대신 시끄러운 실패: fs 를 못 정하면 ok() 가 거짓이어야 read() 가 던진다
    assert not parse_name(name).ok()


def test_parse_dot_format_extras():
    m = parse_name("cap.cplx.1000000.16t.be.bitrev.pcm")
    assert m.endian == "be" and m.bitrev
    assert parse_name("cap.cplx.20000000.16t.rf162000000.pcm").rf_center == 162e6
    # 점 형식과 밑줄 형식이 같은 파일을 같게 읽는다 (마이그레이션 중 두 이름이 섞여도 안전)
    a, b = parse_name("x.cplx.1000000.16t.pcm"), parse_name("x_fs1000000_iq_i16.iq")
    assert (a.fs, a.fmt, a.dtype) == (b.fs, b.fmt, b.dtype)


def test_parse_baudline_aliases_endian_bitrev():
    m = parse_name("cap_fs1000000_iq_8o.iq")           # 8o = offset binary = u8
    assert m.dtype == "u8"
    assert parse_name("cap_fs1_iq_16t.iq").dtype == "i16"
    assert parse_name("cap_fs1_iq_8t.iq").dtype == "i8"
    m = parse_name("cap_fs1000000_iq_i16_be_bitrev.iq")
    assert m.endian == "be" and m.bitrev
    rt = parse_name(make_name("x", Meta(1e6, "iq", "u16", "be", True)))
    assert (rt.dtype, rt.endian, rt.bitrev) == ("u16", "be", True)


@pytest.mark.parametrize("junk", [
    "[1,2,3]",                                                    # JSON, but not an object
    '{"global": "hello"}',                                        # 'global' not a dict
    '{"global": {"core:datatype": "cf32_le", "core:sample_rate": null}}',  # null rate
    '{"global": {"core:datatype": 42, "core:sample_rate": 1e6}}',  # datatype not a str
    "not json at all",
])
def test_malformed_sigmf_falls_back_to_filename(tmp_path, junk):
    # a junk .sigmf-meta beside a valid token-named file must NOT crash read(): parse_sigmf
    # returns None (any malformed-but-parseable shape) and read() falls back to filename tokens.
    x, _ = generate(GenParams(mod="qpsk", n_symbols=300, snr=30, seed=1))
    f = str(tmp_path / make_name("cap", Meta(1e6, "iq", "f32")))
    write(f, x, Meta(1e6, "iq", "f32"))
    with open(os.path.splitext(f)[0] + ".sigmf-meta", "w") as fh:
        fh.write(junk)
    assert parse_sigmf(f) is None                 # malformed -> no Meta, no exception
    y, m = read(f)                                # falls back to filename tokens
    assert y.size > 0 and m.fs == 1e6 and m.dtype == "f32"


@pytest.mark.parametrize("caps", [
    '"captures": "x"',                                           # captures not a list
    '"captures": [{"core:frequency": {"value": 433920000.0}}]',  # rf a dict, not a number
    '"captures": [{"core:frequency": "junk"}]',                  # rf not parseable
])
def test_sigmf_malformed_optional_rf_keeps_mandatory_fields(tmp_path, caps):
    # regression: the OPTIONAL core:frequency parse sat inside the same try as the mandatory
    # fields, so ONE malformed rf value discarded the entire valid sidecar (fs/fmt/dtype) and the
    # reader silently fell back to filename tokens -- which can LIE about fs (here they do) ->
    # a quiet garbage decode. A malformed optional field must cost only that field.
    x, _ = generate(GenParams(mod="qpsk", n_symbols=300, snr=30, seed=1))
    f = str(tmp_path / "cap_fs2000000_iq_f32.iq")   # tokens deliberately CONTRADICT the sidecar
    write(f, x, Meta(1e6, "iq", "i16"))
    with open(os.path.splitext(f)[0] + ".sigmf-meta", "w") as fh:
        fh.write('{"global": {"core:datatype": "ci16_le", "core:sample_rate": 1e6}, ' + caps + "}")
    m = parse_sigmf(f)
    assert m is not None and m.fs == 1e6 and m.dtype == "i16" and m.fmt == "iq", caps
    assert m.rf_center is None


@pytest.mark.parametrize("dtype", ["i8", "u8", "i16", "u16", "f32", "f64"])
@pytest.mark.parametrize("endian", ["le", "be"])
@pytest.mark.parametrize("bitrev", [False, True])
def test_sample_type_matrix_roundtrip(tmp_path, dtype, endian, bitrev):
    """write -> read must preserve the waveform for every sample-type variant."""
    x, _ = generate(GenParams(mod="qpsk", n_symbols=300, snr=30, seed=5))
    meta = Meta(1e6, "iq", dtype, endian, bitrev)
    f = str(tmp_path / make_name("t", meta))
    write(f, x, meta)
    y, _ = read(f)
    m = min(x.size, y.size)
    r = np.abs(np.vdot(y[:m], x[:m])) / (np.linalg.norm(y[:m]) * np.linalg.norm(x[:m]))
    assert r > (0.98 if "8" in dtype else 0.999), (dtype, endian, bitrev, r)


def test_decode_u8_offset_binary():
    raw = np.array([128, 128, 255, 0], dtype=np.uint8).tobytes()  # (0,0), (~1,-1)
    z = decode(raw, Meta(1e6, "iq", "u8"))
    assert abs(z[0]) < 1e-9 and z[1].real > 0.9 and z[1].imag < -0.9


def test_generate_reproducible_and_snr():
    p = GenParams(mod="qpsk", n_symbols=3000, seed=7)
    (a, ba), (b, bb) = generate(p), generate(p)
    assert np.allclose(a, b) and np.array_equal(ba, bb)
    clean, _ = generate(GenParams(mod="qpsk", n_symbols=3000, seed=7, snr=100))
    noisy, _ = generate(GenParams(mod="qpsk", n_symbols=3000, seed=7, snr=10))
    snr = 10 * np.log10(np.mean(np.abs(clean) ** 2) / np.mean(np.abs(noisy - clean) ** 2))
    assert abs(snr - 10) < 1.5


@pytest.mark.parametrize("kw", [{}, {"pad": 0.4}, {"drift_ppm": 80}, {"dc": 0.4 + 0.2j},
                                {"fmt": "real", "fc": 90e3}])
def test_generate_impairments_finite(kw):
    x, bits = generate(GenParams(mod="8psk", n_symbols=400, seed=1, **kw))
    assert np.isfinite(x).all() and len(bits) == 400 * 3
    assert np.iscomplexobj(x) == (kw.get("fmt", "iq") == "iq")


@pytest.mark.parametrize("mod", GEN_MODS)
def test_all_gen_mods_produce_finite_signal(mod):
    kw = {"fc": 0.0} if family(mod) == "fsk" else {"fc": 8000.0}
    x, bits = generate(GenParams(mod=mod, n_symbols=400, seed=1, **kw))
    k = mod_order(mod).bit_length() - 1
    assert np.isfinite(x).all() and len(bits) == 400 * k


@pytest.mark.parametrize("mod", DIFF_MODS)
def test_diff_demap_loopback(mod):
    """Transition bits survive a noiseless diff encode -> demap round trip
    regardless of an arbitrary rotation (the whole point of differential)."""
    rng = np.random.default_rng(3)
    k = mod_order(mod).bit_length() - 1
    idx = rng.integers(0, mod_order(mod), 500)
    sym = np.exp(1j * (np.cumsum(DIFF_PHASES[mod][idx]) + 1.234))  # rotated!
    rx = demap_diff_bits(sym, mod)
    tx = to_bits(idx, mod)
    assert np.array_equal(rx, tx[k:])  # first symbol's transition needs the seed phase


@pytest.mark.parametrize("mod", FSK_MODS)
def test_fsk_constant_envelope_and_index(mod):
    p = GenParams(mod=mod, n_symbols=600, fs=8e5, baud=1e5, snr=90, h=0.7, seed=2)
    x, _ = generate(p)
    env = np.abs(x)
    assert np.std(env) / np.mean(env) < 0.05  # CPFSK: constant envelope
    f_inst = np.angle(x[1:] * np.conj(x[:1] and x[:-1])) * p.fs / (2 * np.pi)
    h = 0.5 if mod == "msk" else 0.7
    lv = np.sort(np.unique(np.round(f_inst / (h * p.baud / 2))))
    span = mod_order(mod) - 1
    assert lv.min() >= -span - 1 and lv.max() <= span + 1  # levels near +-(M-1)


def test_multipath_taps_and_sidecar(tmp_path):
    taps = (1, 0, 0.35 * np.exp(1j * 0.8))
    p = GenParams(mod="qpsk", n_symbols=500, fc=8000.0, taps=taps, seed=1)
    x, _ = generate(p)
    assert np.isfinite(x).all()
    path = save(p, str(tmp_path))
    sc = sidecar_read(path)
    assert len(sc["gen"]["taps"]) == 3 and abs(sc["gen"]["taps"][2][1] - 0.251) < 0.01


def test_32qam_cross_geometry():
    pts = ideal_points("32qam")
    assert pts.size == 32
    assert abs(np.mean(np.abs(pts) ** 2) - 1.0) < 1e-9
    assert sorted(bit_labels("32qam")) == list(range(32))


def test_save_writes_blind_name_and_sidecar(tmp_path):
    path = save(GenParams(mod="16qam", fc=8000, n_symbols=500, seed=2), str(tmp_path))
    name = os.path.basename(path)
    assert "16qam" not in name and "8000" not in name.replace("fs", "")
    x, meta = read(path)
    assert np.iscomplexobj(x) and meta.fs == 1e6
    sc = sidecar_read(path)
    assert sc["truth"] == {"mod": "16qam", "fc": 8000, "baud": 1e5, "rolloff": 0.35, "snr": 20}


def test_default_endian_comes_from_env_and_tokens_win(monkeypatch):
    """실장비 녹음기는 BE 16비트다(2026-08-21 확정 — 154초 캡처가 LE 로 읽혀 균등분포
    잡음으로 보였다). 파일마다 .be 를 붙이는 대신 SIGNUS_ENDIAN 으로 기본을 정하되,
    파일명 토큰(.be/.le)은 언제나 환경변수보다 우선한다."""
    from signus.sigio import parse_name
    monkeypatch.delenv("SIGNUS_ENDIAN", raising=False)
    assert parse_name("cap.real.10000.16t.pcm").endian == "le"
    monkeypatch.setenv("SIGNUS_ENDIAN", "be")
    assert parse_name("cap.real.10000.16t.pcm").endian == "be"
    assert parse_name("cap.real.10000.16t.le.pcm").endian == "le"     # 토큰이 이긴다
    monkeypatch.setenv("SIGNUS_ENDIAN", "le")
    assert parse_name("cap.real.10000.16t.be.pcm").endian == "be"
    monkeypatch.setenv("SIGNUS_ENDIAN", "big")                         # 오타는 시끄럽게
    import pytest
    with pytest.raises(ValueError):
        parse_name("cap.real.10000.16t.pcm")
