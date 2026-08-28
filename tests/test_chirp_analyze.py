"""처프/CSS(LoRa) 판독 — survey 제거 후 analyze 가 단일 파이프라인으로 흡수한 경로.
회귀 무장: FMCW·LoRa 를 자신만만한 엉터리 FSK 로 강복조하지 않고, 진짜 FSK(정수 h·
버스트열 포함)를 처프로 빼앗기지도 않는다. 웹 페이로드(analyze_web)의 버스트 지도도 잠근다."""

import json

import numpy as np
import pytest

from signus.gen import GenParams, generate
from signus.pipeline import analyze, analyze_web
from signus.sigio import Meta

FS = 1e6
N = 240000
M = Meta(FS, "iq", "f32", "le", False)


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


def _fmcw_saw(bw, T, fc, snr, seed, n=N):
    """Linear FMCW (sawtooth) chirp, independent of the receiver (anti-shared-bug)."""
    rng = np.random.default_rng(seed)
    P = int(T * FS)
    t = np.arange(n)
    ramp = np.exp(2j * np.pi * np.cumsum(-bw / 2 + bw * (t % P) / P) / FS)
    x = ramp * np.exp(2j * np.pi * fc * t / FS)
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


def test_analyze_reports_lora_with_characteristics():
    # LoRa 캡처 -> family 'chirp', SF·심볼레이트 특성 보고, 성상도 심볼 0개 -- 강복조 금지
    lo = _lora(9, 125e3, 40, 2.2e5, 22, 2)
    rng = np.random.default_rng(0)
    x = lo + np.sqrt(0.02 / 2) * (rng.standard_normal(N) + 1j * rng.standard_normal(N))
    r = analyze(x, M)
    assert r.family == "chirp" and r.mod == "lora" and r.symbols.size == 0
    assert r.chirp["sf"] == 9 and r.chirp["up"]
    assert r.baud == pytest.approx(125e3 / 2 ** 9, rel=0.15)     # baud = LoRa 심볼레이트
    assert abs(r.fc - 2.2e5) < 0.2 * 125e3                       # 스펙트럼 무게중심 ~ 반송파
    doc = r.to_json()
    assert doc["detected"]["chirp"]["sf"] == 9
    body = json.dumps(doc)                       # NaN 없는 유효 JSON (브라우저 JSON.parse)
    assert "NaN" not in body
    json.loads(body)


@pytest.mark.parametrize("bw,T", [(80e3, 2e-3), (40e3, 1e-3), (160e3, 5e-3)])
def test_analyze_fmcw_chirp_not_demodulated_as_fsk(bw, T):
    # FMCW 는 fsk_gate(양봉 IF)와 is_chirp 를 동시에 밟는다 -- IF-스윕 타이브레이크가 'chirp'
    # 로 보내야 하며, 자신만만한 엉터리 FSK 심볼을 절대 만들지 않는다 (구 ValueError 거부의 후신).
    # mod 는 단정하지 않는다: 칩 수(bw·T)가 2^k 근처인 FMCW 는 bw·μ 만으로 LoRa 와 같은
    # 신호라 측정 오차에 따라 정직하게 'LoRa 추정' 이 붙을 수 있다 (그래서 CLI 도 '추정').
    rng = np.random.default_rng(0)
    x = _fmcw_saw(bw, T, 2e5, 22, 0) + np.sqrt(0.02 / 2) * (
        rng.standard_normal(N) + 1j * rng.standard_normal(N))
    r = analyze(x, M)
    assert r.family == "chirp" and r.symbols.size == 0
    assert r.mod in ("chirp", "lora")


@pytest.mark.parametrize("sf", [7, 12])
def test_lora_sf7_sf12_not_stolen_by_fsk(sf):
    # SF7/SF12 는 fsk_gate 를 밟는다(SF9 와 달리) -- IF-스윕 타이브레이크 전에는 'fsk' 오판이었다.
    rng = np.random.default_rng(1)
    x = _lora(sf, 125e3, 40, 2.2e5, 22, 3) + np.sqrt(0.02 / 2) * (
        rng.standard_normal(N) + 1j * rng.standard_normal(N))
    r = analyze(x, M)
    assert r.family == "chirp" and r.symbols.size == 0


@pytest.mark.parametrize("pad,seed", [(0.25, 0), (0.4, 0), (0.6, 1)])
def test_bursty_integer_h_fsk_not_stolen_by_chirp(pad, seed):
    # 정수-h CPFSK 버스트열: 비트 톤이 is_chirp 를 밟지만 sweeps_band 타이브레이크가 막아
    # 복조로 흘려보내야 한다 -- 버스트 분리 후 깨끗한 FSK 로 복원된다.
    x, _ = generate(GenParams(mod="fsk2", fs=FS, baud=4e4, h=1.0, fc=2e5, snr=20,
                              n_symbols=5000, pad=pad, seed=seed))
    r = analyze(x, M)
    tag = (pad, seed, r.family, r.mod)
    assert r.family == "fsk", tag
    assert abs(r.baud - 4e4) / 4e4 < 0.02, (tag, r.baud)


@pytest.mark.parametrize("mod,fc,seed", [("fsk2", 3e4, 0), ("fsk4", 0.0, 0), ("fsk4", 3e4, 1)])
def test_integer_h_fsk_at_moderate_snr_not_refused_as_chirp(mod, fc, seed):
    # h=2.0 / snr~14: 잡음이 RAW IF 히스토그램을 스윕처럼 눌러도(sweeps_band 칼끝) 평활 IF 가
    # FSK 톤 스파이크를 되살린다 -- 두 시점이 모두 스윕이어야만 처프로 보낸다.
    x, _ = generate(GenParams(mod=mod, h=2.0, snr=14, fc=fc, baud=4e4,
                              n_symbols=5000, fs=FS, seed=seed))
    r = analyze(x, M)
    tag = (mod, fc, seed, r.mod, round(r.lock), round(r.baud))
    assert r.family == "fsk" and r.lock >= 60 and abs(r.baud - 4e4) / 4e4 < 0.02, tag


def test_analyze_chirp_degenerate_input_no_crash():
    # 잠복 크래시: _bw99 의 LO 쪽 searchsorted 미클램프 -- 퇴화 PSD(바닥 차감 후 전부 0)가
    # 끝+1 을 인덱싱했다. analyze 의 처프 갈래가 직접 부르므로 bw=0 으로 무해해야 한다.
    from signus.chirp import analyze_chirp
    info = analyze_chirp(np.zeros(4096, complex), FS)
    assert info["bw"] == 0.0 and info["sf"] is None


def test_analyze_web_overview_only_on_first_multi_burst_load():
    # 버스트 지도 데이터: 첫 로드(burst 미지정)의 다중버스트에만 overview 가 실린다.
    sg, _ = generate(GenParams(mod="qpsk", n_symbols=4000, fs=FS, baud=1e5, snr=90,
                               fc=8e4, seed=1))
    one = sg[:15000] / np.sqrt(np.mean(np.abs(sg[:15000]) ** 2))
    rng = np.random.default_rng(7)
    x = np.tile(np.concatenate([one * 10 ** (20 / 20), np.zeros(15000, complex)]), 3)
    x = x + (rng.standard_normal(x.size) + 1j * rng.standard_normal(x.size)) / np.sqrt(2)
    doc = analyze_web(x, M)
    assert len(doc["bursts"]) == 3
    assert "overview" in doc and doc["overview"]["strip"]["g"]
    assert doc["overview"]["n"] == x.size and doc["overview"]["fs"] == FS
    assert "overview" not in analyze_web(x, M, burst=0)       # 재선택은 지도 재전송 없음
    xs, _ = generate(GenParams(mod="qpsk", n_symbols=4000, fs=FS, baud=1e5, snr=20, seed=2))
    assert "overview" not in analyze_web(xs, M)               # 단일 버스트도 없음


def test_cli_analyze_prints_lora_and_brief_line(tmp_path, capsys):
    from signus.cli import check_code, main
    from signus.sigio import write
    lo = _lora(9, 125e3, 40, 2.2e5, 22, 2)
    rng = np.random.default_rng(0)
    x = lo + np.sqrt(0.02 / 2) * (rng.standard_normal(N) + 1j * rng.standard_normal(N))
    f = tmp_path / "cs.cplx.1000000.32f.pcm"
    write(str(f), x, M)
    assert main(["analyze", str(f), "--brief"]) == 0
    out = capsys.readouterr().out
    assert "LoRa 추정 SF9" in out and "처프율" in out
    line = next(ln for ln in out.splitlines() if ln.startswith("sig2 "))
    assert " lora " in line
    body, code = line.rsplit(" #", 1)
    assert check_code(body) == code
