"""격리망 장비 시뮬레이션: 인쇄물(SECTIONS)에 실리는 파일만으로 장비가 도는가.

합성 생성기·채점(gen.py, lab.py)은 인쇄물에서 영구 제외됐다 (2026-08-03 사용자 결정) --
그 장비에는 이 두 파일이 아예 존재하지 않는다. 여기서는 그 상태를 그대로 재현한다:
SECTIONS 파일만 복사한 트리에서 analyze 가 끝까지 돌고, gen/dataset/sweep 명령은
"없는 명령"이어야 한다. 이 테스트가 깨지면 "장비가 못 도는 코드를 인쇄물에 실었다"는
뜻이다 -- 예컨대 필사 대상 모듈이 gen/lab 을 모듈 수준에서 임포트하기 시작한 경우."""

import re
import subprocess
import sys
from pathlib import Path

from signus.gen import GenParams, save

ROOT = Path(__file__).resolve().parent.parent
_SRC = (ROOT / "tools" / "codebook.py").read_text(encoding="utf-8")


def _sections() -> list[str]:
    m = re.search(r"^SECTIONS.*?^\]", _SRC, re.S | re.M)
    return re.findall(r'\("([^"]+)", "', m.group(0))


def _excluded() -> set[str]:
    m = re.search(r"^EXCLUDED.*?^\}", _SRC, re.S | re.M)
    return set(re.findall(r'"([^"]+)"', m.group(0)))


def test_codebook_segments_keep_leading_blank_lines():
    # pygments swallows a file's leading newlines: a file starting with a blank line shifted
    # every printed line number by one and crashed the diff issuer (IndexError in diff_rows).
    import pytest
    pytest.importorskip("pygments")
    sys.path.insert(0, str(ROOT / "tools"))
    import codebook
    segs = codebook.segments_of("\n\nx = 1\n", ".py")
    assert len(segs) == 3, segs
    assert segs[0] == [] and segs[1] == []


def test_codebook_excludes_generator_and_grader():
    rels = _sections()
    assert len(rels) >= 20, rels                     # 목록 파싱 자체가 깨졌는지 먼저 본다
    assert "signus/gen.py" not in rels
    assert "signus/lab.py" not in rels
    assert {"signus/gen.py", "signus/lab.py"} <= _excluded()


# cwd=device 라 sys.path[0] 이 장비 트리를 가리키지만, 개발기의 editable 설치(PEP 660)는
# sys.meta_path 에 _EditableFinder 를 심고 이 파인더는 부모 __path__ 를 무시한 채
# signus.* 전부를 원본 저장소에서 찾는다 -- 장비 트리에 없는 lab 이 여기서만 임포트되는
# 가짜 성공이 나온다. 실제 장비에는 그 파인더가 없으므로 걷어내는 쪽이 올바른 재현이다.
_BOOT = ("import sys;"
         "sys.meta_path[:] = [f for f in sys.meta_path"
         " if 'Editable' not in getattr(f, '__name__', type(f).__name__)];"
         f"sys.path[:] = [p for p in sys.path if p != {str(ROOT)!r}];"
         "from signus.cli import main; sys.exit(main(sys.argv[1:]))")


def _run_device_cli(device: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", _BOOT, *args],
                          cwd=device, capture_output=True, text=True, timeout=300)


def test_device_tree_analyzes_without_gen_and_lab(tmp_path):
    device = tmp_path / "device"
    for rel in _sections():
        dst = device / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes((ROOT / rel).read_bytes())
    assert not (device / "signus" / "gen.py").exists()
    assert not (device / "signus" / "lab.py").exists()

    # 입력 신호는 개발기의 생성기로 만들어 준다 -- 장비 트리에는 생성기가 없으니까
    f = save(GenParams(mod="qpsk", fs=1e6, baud=1e5, snr=18, fc=8000.0, seed=0),
             str(tmp_path), "airgap")

    r = _run_device_cli(device, "analyze", f, "--brief")
    assert r.returncode == 0, r.stderr
    assert "qpsk" in r.stdout and "sig2 an" in r.stdout

    for cmd in ("gen", "dataset", "sweep"):
        r = _run_device_cli(device, cmd)
        assert r.returncode == 2, (cmd, r.returncode, r.stderr)  # argparse: 없는 명령
        assert "invalid choice" in r.stderr, cmd
