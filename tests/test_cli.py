"""CLI, BER scorer, and the stdlib server."""

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

from signus.cli import _grid, ber, main
from signus.constellations import ideal_points
from signus.gen import GenParams, generate
from signus.server import Handler


def test_ber_scorer_sanity():
    p = GenParams(mod="qpsk", n_symbols=2000, seed=0)
    _, bits = generate(p)
    rng = np.random.default_rng(99)  # NOT seed 0: that replays the generator's draw
    pts = ideal_points("qpsk")
    # perfect symbols, rotated by one symmetry step and phase-offset half-way: ber 0
    labels = (bits.reshape(-1, 2) << np.arange(1, -1, -1)).sum(1)
    from signus.constellations import bit_labels
    inv = np.empty(4, dtype=int)
    inv[bit_labels("qpsk")] = np.arange(4)
    z = pts[inv[labels]] * np.exp(1j * np.pi / 2)
    assert ber(z, "qpsk", bits) == 0.0
    # random symbols: ~0.5
    zr = pts[rng.integers(0, 4, 2000)]
    assert 0.3 < ber(zr, "qpsk", bits) < 0.7


def test_web_assets_ship_inside_package():
    # regression: web/ sat at the repo top level, outside [tool.setuptools.packages.find] --
    # wheel and sdist omitted it entirely, so a pip-installed `signus serve` 404'd the whole UI
    # while /api/analyze kept working (a silent partial break the source-checkout server test
    # can never catch). The assets must live INSIDE the signus package, where setuptools
    # package-data actually ships them and server._WEB resolves in an install.
    from pathlib import Path

    import signus
    from signus.server import _WEB
    pkg = Path(signus.__file__).resolve().parent
    assert _WEB.resolve().is_relative_to(pkg), _WEB
    for asset in ("index.html", "app.js", "style.css"):
        assert (_WEB / asset).is_file(), asset


def test_grid_has_core_regressions():
    labels = [label for label, core, _ in _grid("core", 1) if core]
    for want in ("cfo0", "dc", "pad", "real"):
        assert any(want in label for label in labels)


def test_cli_gen_and_analyze(tmp_path, capsys):
    out = str(tmp_path)
    assert main(["gen", "--mod", "8psk", "--cfo", "8000", "--snr", "20",
                 "--out", out, "--label", "t"]) == 0
    path = capsys.readouterr().out.strip()
    assert path.endswith(".iq") and "8psk" not in path
    rep = str(tmp_path / "r.json")
    assert main(["analyze", path, "--report", rep,
                 "--save-bits", str(tmp_path / "b.txt")]) == 0
    printed = capsys.readouterr().out
    assert "8psk" in printed and "정답" in printed  # sidecar comparison shown
    assert json.loads((tmp_path / "r.json").read_text())["detected"]["mod"] == "8psk"
    assert set((tmp_path / "b.txt").read_text()) <= {"0", "1"}


def test_cli_real_pcm_roundtrip(tmp_path, capsys):
    """Real passband PCM: gen writes a .pcm + sidecar, analyze recovers it from disk."""
    out = str(tmp_path)
    assert main(["gen", "--mod", "qpsk", "--fmt", "real", "--cfo", "100000",
                 "--snr", "16", "--out", out, "--label", "pcm"]) == 0
    path = capsys.readouterr().out.strip()
    assert path.endswith(".pcm") and "qpsk" not in path
    assert main(["analyze", path]) == 0
    printed = capsys.readouterr().out
    assert "qpsk" in printed and "정답" in printed


def test_cli_gen_real_requires_carrier(capsys):
    assert main(["gen", "--fmt", "real", "--cfo", "0", "--out", "/tmp/x"]) == 1
    assert "반송파" in capsys.readouterr().out


def test_cli_analyze_with_overrides(tmp_path, capsys):
    from signus.sigio import Meta, write
    x, _ = generate(GenParams(mod="qpsk", snr=18, fc=8000.0, seed=0))
    f = tmp_path / "capture.bin"  # no metadata tokens at all
    write(str(f), x, Meta(1e6, "iq", "f32"))
    assert main(["analyze", str(f), "--fs", "1e6", "--fmt", "iq", "--dtype", "f32"]) == 0
    assert "qpsk" in capsys.readouterr().out


def test_cli_survey_wideband_file(tmp_path, capsys):
    """A capture file with two emitters at different carriers: survey finds both."""
    from signus.sigio import Meta, write
    fs, n = 1e6, 240000
    rng = np.random.default_rng(0)

    def emit(mod, baud, fc, seed):
        x, _ = generate(GenParams(mod=mod, n_symbols=8000, fs=fs, baud=baud, fc=fc,
                                  snr=60.0, seed=seed))
        x = x[:n] if x.size >= n else np.concatenate([x, np.zeros(n - x.size, complex)])
        return x / np.sqrt(np.mean(np.abs(x) ** 2))

    mix = emit("qpsk", 25e3, -250e3, 1) + emit("16qam", 50e3, 150e3, 2)
    mix = mix + np.sqrt(0.05 / 2) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    f = tmp_path / "wb_fs1000000_iq_i16.iq"
    write(str(f), mix, Meta(fs, "iq", "i16"))
    rep = str(tmp_path / "s.json")
    assert main(["survey", str(f), "--report", rep]) == 0
    printed = capsys.readouterr().out
    assert "2개 신호 감지" in printed and "qpsk" in printed and "16qam" in printed
    doc = json.loads((tmp_path / "s.json").read_text())
    assert doc["n_emitters"] == 2
    mods = {e.get("mod") for e in doc["emitters"]}
    assert {"qpsk", "16qam"} <= mods


@pytest.fixture
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_server_analyze_and_static(server):
    x, _ = generate(GenParams(mod="qpsk", snr=18, fc=8000.0, seed=0))
    raw = np.column_stack([x.real, x.imag]).astype("<f4").tobytes()
    req = urllib.request.Request(
        server + "/api/analyze?name=cap_fs1000000_iq_f32.iq", data=raw, method="POST")
    with urllib.request.urlopen(req, timeout=60) as res:
        doc = json.load(res)
    assert doc["detected"]["mod"] == "qpsk"
    assert len(doc["constellation"]["i"]) == len(doc["constellation"]["q"]) > 0
    with urllib.request.urlopen(server + "/", timeout=10) as res:
        assert res.status == 200 and b"signus" in res.read().lower()
    # real passband .pcm upload (int16 mono) detects through the same endpoint
    xr, _ = generate(GenParams(mod="bpsk", snr=16, fc=1e5, fmt="real", seed=1))
    raw16 = np.clip(np.round(xr * 32767 / np.percentile(np.abs(xr), 99.9)),
                    -32768, 32767).astype("<i2").tobytes()
    req = urllib.request.Request(
        server + "/api/analyze?name=cap_fs1000000_real_i16.pcm", data=raw16, method="POST")
    with urllib.request.urlopen(req, timeout=60) as res:
        assert json.load(res)["detected"]["mod"] == "bpsk"
    bad = urllib.request.Request(server + "/api/analyze?name=nometa.bin",
                                 data=b"\x00" * 4096, method="POST")
    with pytest.raises(urllib.error.HTTPError) as ei:
        urllib.request.urlopen(bad, timeout=10)
    assert ei.value.code == 400 and "error" in json.load(ei.value)


# --- 손으로 옮기는 한 줄 ------------------------------------------------------
# 결과는 사람이 화면을 보고 받아쳐서 나온다. 여기 잠그는 건 "조용히 틀린 채 통과"한
# 사고다 -- 검출기가 오타를 놓치면 잘못된 값 위에서 실신호 수사가 시작된다.

def test_check_code_catches_shifted_space_that_moves_a_digit():
    # 회귀: 공백을 전부 지우고 crc 를 걸던 판에서 'fs1000000 16qam' 과 'fs10000001 6qam'
    # (샘플레이트가 10배!) 이 같은 코드를 받아 "일치 ✓" 로 통과했다. 값이 바뀌는 오타
    # 32종이 이 경로로 샜다. 공백은 한 칸으로 '줄이되' 없애지 않는다.
    from signus.cli import check_code
    good = "sig2 an fs1000000 16qam fc8000 bd100000 lk100"
    bad = "sig2 an fs10000001 6qam fc8000 bd100000 lk100"
    assert check_code(good) != check_code(bad)


@pytest.mark.parametrize("typo", [
    "sig2 an fs1000000 16qam fc8000 bd10000 lk100",    # 자릿수 누락
    "sig2 an fs1000000 16qam fc8000 bd100001 lk100",   # 한 자리 치환
    "sig2 an fs1000000 16qam fc-8000 bd100000 lk100",  # 부호
    "sig2 an fs1000000 16qam fc8000 bd100000 lk10",    # 끝자리 누락
])
def test_check_code_catches_common_hand_copy_typos(typo):
    from signus.cli import check_code
    assert check_code("sig2 an fs1000000 16qam fc8000 bd100000 lk100") != check_code(typo)


def test_check_code_ignores_layout_only_differences():
    # 사람이 줄을 어떻게 띄우고 대소문자를 어떻게 쓰든 통과해야 한다 -- 헛경보가 잦으면
    # 진짜 불일치를 무시하게 된다. 잡아야 하는 건 오직 값이다.
    from signus.cli import check_code
    base = "sig2 an fs1000000 16qam fc8000"
    assert check_code(base) == check_code("  SIG2\tan  fs1000000   16QAM fc8000  ")


def test_brief_line_round_trips_through_its_own_check_code():
    from signus.cli import brief, check_code
    doc = {"fs": 1e6, "fmt": "iq", "dtype": "i16", "bursts": [{}], "burst_idx": 0,
           "detected": {"mod": "16qam", "fc": 8000.1, "baud": 100000.1, "rolloff": 0.33,
                        "alias_resolved": False, "baud_fallback": False,
                        "carrier_ambiguous": False},
           "quality": {"lock": 100.0, "mer_db": 29.8}, "eq": {"applied": False}}
    line = brief(doc, "an")
    body, _, code = line.rpartition(" #")
    assert check_code(body) == code
    assert "iq-i16" in body        # dtype 은 lock 0 진단의 1순위 용의자라 반드시 실린다
    assert len(line) < 100         # 한 줄로 받아칠 수 있어야 한다
