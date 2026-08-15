"""장비용 스펙트로그램 띠 뷰어(tools/strip_probe.py) — GUI 대조 결정 실험용."""

import re
import struct
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from signus.cli import check_code  # noqa: E402
from signus.sigio import Meta, make_name, write  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PROBE = ROOT / "tools" / "strip_probe.py"


def _run(args, cwd):
    r = subprocess.run([sys.executable, str(PROBE), *args], capture_output=True, text=True,
                       cwd=cwd, env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"})
    return r.returncode, r.stdout + r.stderr


def _png_size(path: Path) -> tuple[int, int]:
    head = path.read_bytes()
    assert head[:8] == b"\x89PNG\r\n\x1a\n"
    w, h = struct.unpack(">II", head[16:24])
    zlib.decompress(head[head.index(b"IDAT") + 4:head.rindex(b"IEND") - 4])  # 스트림 무결성
    return w, h


def test_kat_is_deterministic_and_self_checking(tmp_path):
    rc1, out1 = _run(["kat"], tmp_path)
    rc2, out2 = _run(["kat"], tmp_path)
    assert rc1 == 0 and out1 == out2, out1
    line = out1.splitlines()[0]
    m = re.fullmatch(r"(strip kat .*) #([0-9a-z]{4})", line)
    assert m and check_code(m.group(1)) == m.group(2)
    assert " s6.0 " in line                      # 길이(초) — GUI 파일 길이와 대조하는 값
    assert _png_size(tmp_path / "strip-kat.png") == (467, 129)


def test_capture_render_matches_duration_and_size(tmp_path):
    rng = np.random.default_rng(1)
    n, fs = 200000, 10000.0
    t = np.arange(n) / fs
    x = np.sin(2 * np.pi * 2200 * t) * ((np.arange(n) // 2500) % 2 == 0)
    x = x + 0.2 * rng.standard_normal(n)
    meta = Meta(fs, "real", "i16")
    cap = tmp_path / make_name("cap", meta)
    write(str(cap), x, meta)
    rc, out = _run([cap.name], tmp_path)
    assert rc == 0, out
    assert " s20.0 " in out and "strip cap " in out
    w, h = _png_size(tmp_path / (cap.name + ".png"))
    assert h == 129 and 1500 <= w <= 1600        # (200000-256)//128+1 = 1561열


def test_transcription_budget():
    lines = [ln for ln in PROBE.read_text().splitlines() if ln.strip()]
    assert len(lines) <= 70, f"필사 분량 초과: {len(lines)}줄"
