"""Blind repeated-preamble sync: detector, generator packet fixtures, and the
end-to-end packetized-burst decode rescue (raises the decode rate of short bursts)."""

import numpy as np
import pytest

from signus.cli import ber
from signus.gen import GenParams, generate
from signus.pipeline import analyze
from signus.sigio import Meta
from signus.sync import find_preamble

FS = 1e6
M = Meta(FS, "iq", "f32", "le")


# --- generator: realistic packet content -------------------------------------

def test_preamble_creates_detectable_repetition():
    x, bits = generate(GenParams(mod="qpsk", fs=FS, baud=1e5, fc=0, snr=30,
                                 n_symbols=200, seed=1, preamble=(4, 8)))
    assert bits.size == 400                        # data-only bits (preamble excluded)
    ps = find_preamble(x, FS)
    assert ps is not None and abs(ps.period - 40) <= 6   # L=4 * sps=10


def test_explicit_payload_round_trips():
    pattern = "1011001110100101" * 6
    x, bits = generate(GenParams(mod="qpsk", fs=FS, baud=1e5, fc=0, snr=40,
                                 seed=2, payload=pattern, preamble=(4, 4)))
    assert "".join(map(str, bits)) == pattern      # the exact bits we asked to transmit


def test_defaults_unchanged_when_no_preamble():
    # (0,0)/None must reproduce the legacy stream so the sweep stays byte-identical
    a, ba = generate(GenParams(mod="16qam", fs=FS, baud=1e5, fc=8e3, snr=24, seed=0))
    b, bb = generate(GenParams(mod="16qam", fs=FS, baud=1e5, fc=8e3, snr=24, seed=0,
                               preamble=(0, 0), sync_word=(), payload=None))
    assert np.array_equal(a, b) and np.array_equal(ba, bb)


# --- detector: no false alarm on non-packetized signals ----------------------

@pytest.mark.parametrize("mod", ["qpsk", "8psk", "16qam", "64qam"])
def test_no_preamble_random_data_not_detected(mod):
    x, _ = generate(GenParams(mod=mod, fs=FS, baud=1e5, fc=8e3, snr=20,
                              n_symbols=400, seed=0))
    assert find_preamble(x, FS) is None


def test_pure_noise_not_detected():
    rng = np.random.default_rng(0)
    z = (rng.standard_normal(50000) + 1j * rng.standard_normal(50000)) / np.sqrt(2)
    assert find_preamble(z, FS) is None


# --- end-to-end: short packetized bursts decode via the sync rescue ----------

@pytest.mark.parametrize("mod,nsym,snr,fc,seed", [
    ("8psk", 300, 20, 8e3, 1),      # blind M-th-power carrier fails on a short 8psk
    ("8psk", 400, 22, -3e4, 1),
    ("32qam", 400, 26, 8e3, 1),     # 32qam short + carrier offset
    ("64qam", 400, 28, 8e3, 3),
    ("8psk", 250, 19, 5e4, 2),      # large carrier offset -> +-fs/P alias search
    ("32qam", 500, 27, -2e4, 0),
])
def test_packetized_short_burst_recovered(mod, nsym, snr, fc, seed):
    # the SAME short data with no preamble fails the blind chain; a repeated preamble pins the
    # carrier/baud/symmetry so the data decodes. Verifies both (blind fails) and (packet recovers).
    blind_x, blind_tx = generate(GenParams(mod=mod, fs=FS, baud=1e5, fc=fc, snr=snr,
                                           n_symbols=nsym, seed=seed))
    rb = analyze(blind_x, M)
    blind_ber = ber(rb.symbols, rb.mod, blind_tx) if rb.mod == mod else 1.0
    assert blind_ber > 0.05, "fixture must be genuinely hard for the blind chain"

    x, tx = generate(GenParams(mod=mod, fs=FS, baud=1e5, fc=fc, snr=snr,
                               n_symbols=nsym, seed=seed, preamble=(4, 12)))
    r = analyze(x, M)
    assert r.mod == mod
    assert ber(r.symbols, r.mod, tx) < 0.02
    assert r.preamble is not None                  # the rescue drove this decode
    assert r.to_json(views=False)["detected"]["preamble"]["period"] > 0


def test_multipath_not_hijacked_by_preamble_rescue():
    # a 1-symbol echo fakes a short periodicity; the eq rescue runs FIRST and lifts the lock, so
    # the preamble rescue is skipped and the bpsk multipath signal is not mis-driven.
    x, tx = generate(GenParams(mod="bpsk", fs=FS, baud=1e5, snr=16, fc=0.02e6, phase=0.9,
                               taps=(1.0, 0.5 * np.exp(1j * 0.5)), tap_sym=1.0, seed=0))
    r = analyze(x, M)
    assert r.mod == "bpsk" and r.preamble is None
    assert ber(r.symbols, "bpsk", tx) < 0.01
