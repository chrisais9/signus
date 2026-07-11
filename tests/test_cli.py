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
