"""종합 관찰 프로브(probes/sa.py) — 스펙트로그램+검출+버스트별 변조 판독 한 장."""

import re
import struct
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from signus.cli import check_code  # noqa: E402
from signus.gen import GenParams, generate  # noqa: E402
from signus.sigio import Meta, make_name, write  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PROBE = ROOT / "probes" / "sa.py"


def _run(args, cwd):
    r = subprocess.run([sys.executable, str(PROBE), *args], capture_output=True, text=True,
                       cwd=cwd, env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"})
    return r.returncode, r.stdout + r.stderr


def _png_ok(path):
    b = path.read_bytes()
    assert b[:8] == b"\x89PNG\r\n\x1a\n"
    zlib.decompress(b[b.index(b"IDAT") + 4:b.rindex(b"IEND") - 4])
    return struct.unpack(">II", b[16:24])


def test_kat_is_deterministic_and_self_checking(tmp_path):
    rc1, out1 = _run(["kat"], tmp_path)
    rc2, out2 = _run(["kat"], tmp_path)
    assert rc1 == 0 and out1 == out2, out1
    line = next(ln for ln in out1.splitlines() if ln.startswith("sa kat"))
    m = re.fullmatch(r"(sa kat .*) #([0-9a-z]{4})", line)
    assert m and check_code(m.group(1)) == m.group(2)
    assert " fb2 " in line                       # BPSK·QPSK 버스트 둘 다 검출
    blines = [ln for ln in out1.splitlines() if ln.startswith("sa b")]
    assert len(blines) == 2                     # 버스트별 판독 숫자줄 (검출 코드 포함)
    for ln in blines:
        mm = re.fullmatch(r"(sa b\d+ [a-z0-9]+ p2 f\d+ d-?\d+ p4 f\d+ d-?\d+ p8 f\d+ d-?\d+"
                          r" am f\d+ d-?\d+) #([0-9a-z]{4})", ln)
        assert mm and check_code(mm.group(1)) == mm.group(2), ln
    assert " bpsk " in blines[0] and " qpsk " in blines[1]   # 자동 판정 라벨
    assert " p2 f1100 " in blines[0]             # BPSK: x² 바늘이 정확히 2·fc=1100Hz
    assert " p4 f2200 " in blines[1]             # QPSK: x⁴ 바늘이 4·fc=2200Hz
    assert " am f294 " in blines[0]              # 심볼레이트 10000/34
    w, h = _png_ok(tmp_path / "sa-kat.png")
    assert w >= 2500 and h > 300                 # 스트립 + 판독 2행


def test_capture_run_reports_burst_table(tmp_path):
    rng = np.random.default_rng(3)
    n, fs = 120000, 10000.0
    p = GenParams(mod="qpsk", n_symbols=900, fs=fs, baud=300.0, fc=800.0, snr=90.0,
                  fmt="real", dtype="f32", seed=4)
    s, _ = generate(p)
    x = 0.15 * rng.standard_normal(n)
    x[20000:50000] += s[:30000] / np.sqrt(np.mean(s[:30000] ** 2))
    meta = Meta(fs, "real", "i16")
    cap = tmp_path / make_name("cap", meta)
    write(str(cap), x, meta)
    rc, out = _run([cap.name, "2"], tmp_path)
    assert rc == 0, out
    assert "find_bursts → 1개" in out and "★판독" in out
    _png_ok(tmp_path / (cap.name + ".sa.png"))


def test_transcription_budget():
    lines = [ln for ln in PROBE.read_text().splitlines() if ln.strip()]
    assert len(lines) <= 175, f"필사 분량 초과: {len(lines)}줄"
