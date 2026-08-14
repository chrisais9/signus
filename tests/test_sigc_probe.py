"""장비용 관찰 프로브(tools/sigc_probe.py)와 개발기 수신 도구(tools/sigc.py).

프로브는 장비에서 손으로 필사되는 파일이라 여기서는 서브프로세스(CLI 계약)로만 검증한다.
시나리오 합성은 프로브/도구와 코드를 공유하지 않고 이 파일 안에서 직접 조립한다
(발생기-수신기 쌍둥이 버그 방지 — 저장소 규칙)."""

import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from signus import dsp
from signus.cli import check_code
from signus.gen import GenParams, generate
from signus.sigio import Meta, make_name, write

ROOT = Path(__file__).resolve().parent.parent
PROBE = ROOT / "tools" / "sigc_probe.py"
TOOL = ROOT / "tools" / "sigc.py"
FS = 1e6


def _run(script: Path, args: list[str], stdin: str | None = None) -> tuple[int, str]:
    r = subprocess.run([sys.executable, str(script), *args], input=stdin,
                       capture_output=True, text=True, cwd=ROOT,
                       env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"})
    return r.returncode, r.stdout + r.stderr


def _mask(n: int, blen: int, gap: int, start: int = 0) -> np.ndarray:
    m = np.zeros(n, bool)
    s = start
    while s < n:
        m[s:s + blen] = True
        s += blen + gap
    return m


def _qpsk(n: int, fc: float, baud: float, seed: int) -> np.ndarray:
    p = GenParams(mod="qpsk", n_symbols=int(n * baud / FS) + 64, fs=FS, baud=baud,
                  snr=90.0, fc=fc, seed=seed, dtype="f32")
    x, _ = generate(p)
    x = x[:n]
    return x / np.sqrt(np.mean(np.abs(x) ** 2))


def _save(tmp: Path, x: np.ndarray) -> Path:
    meta = Meta(FS, "iq", "f32")
    path = tmp / make_name("cap", meta)
    write(str(path), x, meta)
    return path


@pytest.fixture(scope="module")
def burst_train(tmp_path_factory) -> tuple[Path, np.ndarray]:
    """건강한 버스트 열차: 협대역 qpsk 5회 (6000 켜짐 / 6000 꺼짐), 광대역 20dB."""
    rng = np.random.default_rng(11)
    n = 65536
    x = _qpsk(n, 2e5, 5e4, seed=3) * _mask(n, 6000, 6000)
    x += np.sqrt(0.01 / 2) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    return _save(tmp_path_factory.mktemp("bt"), x), x


@pytest.fixture(scope="module")
def cont_plus_burst(tmp_path_factory) -> tuple[Path, np.ndarray]:
    """실장비 가설 재현: 연속 방사체가 광대역 포락선을 평평하게 만들어 베토를 쏘는 캡처.
    연속 qpsk(전력 1.0, -150kHz) + 버스트 qpsk(전력 0.25, +200kHz, 6000/6000)."""
    rng = np.random.default_rng(12)
    n = 65536
    x = _qpsk(n, -1.5e5, 8e4, seed=5)
    x += np.sqrt(0.25) * _qpsk(n, 2e5, 5e4, seed=6) * _mask(n, 6000, 6000)
    x += np.sqrt(0.01 / 2) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    return _save(tmp_path_factory.mktemp("cb"), x), x


def _parse(out: str) -> dict[str, dict[str, int]]:
    """sigc 4줄 -> {줄이름: {키: 정수}}. 줄마다 검출 코드도 즉석 대조한다."""
    doc: dict[str, dict[str, int]] = {}
    for ln in out.strip().splitlines():
        m = re.fullmatch(r"(sigc ([a-d]) .*?) #([0-9a-z]{4})", ln.strip())
        if not m:
            continue
        assert check_code(m.group(1)) == m.group(3), f"검출 코드 불일치: {ln}"
        doc[m.group(2)] = {k: int(v) for k, v in
                           re.findall(r"([a-z]+)(-?\d+)", m.group(1)[7:])}
    return doc


# --- 요약 4줄 ------------------------------------------------------------------

def test_kat_is_deterministic_and_self_checking():
    rc1, out1 = _run(PROBE, ["kat"])
    rc2, out2 = _run(PROBE, ["kat"])
    assert rc1 == 0 and out1 == out2
    doc = _parse(out1)
    assert set(doc) == {"a", "b", "c", "d"}


def test_healthy_burst_train_features(burst_train):
    path, x = burst_train
    rc, out = _run(PROBE, [str(path)])
    assert rc == 0, out
    doc = _parse(out)
    a, c, d = doc["a"], doc["c"], doc["d"]
    assert a["n"] == 65536 and a["f"] == 1000000
    assert a["ev"] >= 12          # 포락선 분리 뚜렷 -> 베토 통과
    assert 4 <= c["r"] <= 6       # 버스트 5회
    assert 70 <= c["dn"] <= 120   # 6000샘플 / hop 64 = ~94열
    assert 70 <= c["gp"] <= 120
    assert d["kc"] == 0           # 연속 방사체 없음
    assert 22 <= d["p"] <= 30     # +200kHz -> 200e3/(fs/128) = 25.6빈
    # 특징이 건강하다고 말하면 find_bursts 도 실제로 잡아야 한다 (일관성)
    fb = dsp.find_bursts(x, FS)
    assert fb != [(0, x.size)] and 4 <= len(fb) <= 6


def test_continuous_coemitter_explains_full_record_fallback(cont_plus_burst):
    path, x = cont_plus_burst
    # 실장비 증상 재현: find_bursts 는 통짜 [(0,n)] 을 낸다
    assert dsp.find_bursts(x, FS) == [(0, x.size)]
    rc, out = _run(PROBE, [str(path)])
    assert rc == 0, out
    doc = _parse(out)
    assert doc["a"]["ev"] < 12    # 프로브가 이유를 말한다: 포락선 베토(E1)
    assert doc["d"]["kc"] >= 1    # 원인도 보인다: 연속 점유 빈 존재
    assert doc["c"]["r"] >= 1     # 셀 점수에는 버스트 구조가 살아 있다


# --- 관찰 단계 (matplotlib 없음 -> ASCII 경로) -----------------------------------

def test_step2_envelope_view_ascii(burst_train):
    rc, out = _run(PROBE, [str(burst_train[0]), "2"])
    assert rc == 0 and "분리도" in out and "|" in out


def test_step4_waterfall_and_step5_score_run_without_matplotlib(burst_train):
    for step in ("3", "4", "5", "6"):
        rc, out = _run(PROBE, [str(burst_train[0]), step])
        assert rc == 0 and out.strip(), f"단계 {step} 실패:\n{out}"


def test_kat_steps_run(tmp_path):
    for step in ("1", "2", "5"):
        rc, out = _run(PROBE, ["kat", step])
        assert rc == 0 and out.strip()


def test_transcription_budget():
    lines = [ln for ln in PROBE.read_text().splitlines() if ln.strip()]
    assert len(lines) <= 160, f"필사 분량 초과: {len(lines)}줄"


# --- 개발기 수신 도구 -----------------------------------------------------------

def test_check_accepts_valid_block_and_names_the_exit(cont_plus_burst):
    _, out = _run(PROBE, [str(cont_plus_burst[0])])
    rc, verdict = _run(TOOL, ["check"], stdin=out)
    assert rc == 0
    assert verdict.count("일치") >= 4
    assert "베토" in verdict            # E1 이 (0,n) 의 범인이라고 짚는다

def test_check_localizes_a_typo_to_its_line(burst_train):
    _, out = _run(PROBE, [str(burst_train[0])])
    lines = out.strip().splitlines()
    lines[2] = re.sub(r"(dn)(\d)", lambda m: f"{m.group(1)}{(int(m.group(2)) + 1) % 10}",
                      lines[2], count=1)
    rc, verdict = _run(TOOL, ["check"], stdin="\n".join(lines))
    assert rc != 0 and "sigc c" in verdict


def test_gen_reproduces_the_failure_from_features(cont_plus_burst, tmp_path):
    _, out = _run(PROBE, [str(cont_plus_burst[0])])
    rc, log = _run(TOOL, ["gen", "--out", str(tmp_path)], stdin=out)
    assert rc == 0, log
    made = sorted(tmp_path.glob("*.pcm"))
    assert len(made) >= 4                       # 특징 주변 변형 그리드
    assert "(0," in log and "재현" in log        # 최소 한 변형이 [(0,n)] 재현
