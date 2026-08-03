"""Blind repeated-preamble sync: detector, generator packet fixtures, and the
end-to-end packetized-burst decode rescue (raises the decode rate of short bursts)."""

import numpy as np
import pytest

from signus.gen import GenParams, generate
from signus.lab import ber
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


@pytest.mark.parametrize("mod,payload", [("qpsk", ""), ("qpsk", []), ("64qam", "101"),
                                         ("fsk2", ""), ("8psk", "01")])
def test_too_short_payload_rejected_cleanly(mod, payload):
    # a payload shorter than one symbol's bits used to reach np.convolve as an empty array
    # (raw ValueError) or emit an empty/NaN FSK stream -- it must be a clean, explained rejection.
    with pytest.raises(ValueError):
        generate(GenParams(mod=mod, payload=payload))


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


@pytest.mark.parametrize("baud,fc,preamble", [
    (5e4, 1.6e4, (8, 8)),           # fc inside +-baud/2
    (8e4, 2.6e4, (8, 8)),
    (1.25e5, 7.5e4, (8, 8)),        # fc = 0.6*baud, OUTSIDE +-baud/2 (the widened-scan case)
])
def test_large_carrier_offset_alias_resolved(baud, fc, preamble):
    # the preamble CFO is ambiguous modulo baud/L: an 8psk carrier off by baud/L rotates a whole
    # constellation position per symbol, so a WRONG alias still locks (~70) with garbage bits (ber
    # ~0.46). The rescue centres the alias scan on the coarse M-th-power fc and requires a clear
    # lock margin, so the CORRECT alias is reached and no rotated alias is confidently accepted.
    for sd in range(8):
        x, tx = generate(GenParams(mod="8psk", fs=FS, baud=baud, fc=fc, snr=16,
                                   n_symbols=300, seed=sd, preamble=preamble))
        r = analyze(x, M)
        assert not (r.lock >= 60 and ber(r.symbols, "8psk", tx) > 0.2), "confident rotated alias"


@pytest.mark.parametrize("fc,seed", [(1.15e5, 14), (1.4e5, 14), (2.0e5, 14)])
def test_lowsnr_8psk_rotated_alias_refused(fc, seed):
    # a 65 lock floor was tried (to recover more moderate bursts) but an independent snr-12 grid
    # found 8psk (8,10) rotated aliases that lock 71-72 -- exactly the 65-78 band the lowering
    # opened. A rotated alias keeps the right constellation but the carrier is off by one baud/L
    # step (adist ~1.0), so every symbol rotates one position: confident garbage bits. The floor
    # stays at 78 (precision over recall); these must NOT produce a confident preamble decode.
    x, tx = generate(GenParams(mod="8psk", fs=FS, baud=1e5, fc=fc, snr=12,
                               n_symbols=100, seed=seed, preamble=(8, 10)))
    r = analyze(x, M)
    if r.preamble is None:
        return                                         # refused outright: correct
    step = FS / r.preamble.period
    adist = abs(r.fc - fc) / step                      # rotated alias sits ~1 full step off truth
    assert not (r.lock >= 60 and (r.mod != "8psk" or adist > 0.4)), \
        (fc, seed, r.mod, round(r.lock, 1), round(adist, 2))


def test_rescue_never_reports_rotation_garbage():
    # a confidently-accepted alias that is off by baud/L gives ber ~0.44 (a whole-constellation
    # rotation). Across a broad packetized grid incl. large carrier offsets, no confident (lock>=60)
    # rescue decode may be rotation garbage. (Dense 32qam near its SNR edge can sit a hair over 0.05
    # with the CORRECT carrier -- that is a noisy recovery, not the alias bug, so we gate at 0.2.)
    bad = 0
    for mod in ("8psk", "16qam", "32qam", "64qam", "qpsk"):
        for fc in (8e3, 1.6e4, 4.5e4, 6e4, -7e4):
            for sd in range(3):
                baud = 1.25e5 if abs(fc) > 5e4 else 1e5
                x, tx = generate(GenParams(mod=mod, fs=FS, baud=baud, fc=fc, snr=22,
                                           n_symbols=350, seed=sd, preamble=(4, 10)))
                r = analyze(x, M)
                if r.preamble is not None and r.lock >= 60 and r.mod == mod:
                    bad += ber(r.symbols, r.mod, tx) > 0.2
    assert bad == 0


def test_rescued_decode_diagnostics_consistent():
    # regression: the rescue replaced fc/baud/symmetry but left the PRE-rescue diagnostics in
    # the Result -- alias_resolved compared the preamble-derived carrier against the pre-rescue
    # fc0 (spuriously True on every rescued decode), baud_conf reported the spectral-line
    # strength of a baud that was DISCARDED, and baud_fallback/carrier_ambiguous described the
    # abandoned blind attempt. Diagnostics must describe the decode actually returned.
    x, _ = generate(GenParams(mod="8psk", fs=FS, baud=1e5, fc=8e3, snr=20,
                              n_symbols=300, seed=1, preamble=(4, 12)))
    r = analyze(x, M)
    assert r.preamble is not None, "fixture must engage the rescue"
    assert not r.alias_resolved            # resolve_alias did not drive this carrier
    assert not r.baud_fallback             # nor did the fewer-blocks baud retry
    assert r.baud_conf == 0.0              # no spectral line backs the preamble-derived baud
    assert r.carrier_ambiguous == (abs(r.symmetry * r.fc) > 0.4 * FS)


@pytest.mark.parametrize("nsym,baud,preamble", [(160, 1.25e5, (6, 16)), (8, 1e5, (4, 4))])
def test_short_packet_negative_baud_does_not_crash(nsym, baud, preamble):
    # est_baud's sub-bin parabola could explode on a degenerate short-burst spectrum and return a
    # NEGATIVE baud, which reached scipy resample_poly ('up and down must be >= 1'). The offset is
    # now clamped to +-half a bin, so a bad estimate degrades (low lock), never crashes.
    x, _ = generate(GenParams(mod="qpsk", fs=FS, baud=baud, snr=20, n_symbols=nsym,
                              seed=0, preamble=preamble))
    r = analyze(x, M)                              # must not raise a scipy ValueError
    assert r.mod in ("bpsk", "qpsk", "8psk", "16qam", "32qam", "64qam")


@pytest.mark.parametrize("mod,baud,nsym,snr,fc,seed,pre", [
    ("16qam", 1e5, 120, 20, 1.4e5, 2, (4, 12)),  # true carrier flukes to 64qam -> rotation wins
    ("8psk", 1e5, 100, 12, 2.9e5, 1, (8, 10)),   # adjacent alias won by a hair over the old margin
    ("16qam", 1e5, 80, 16, 1.15e5, 5, (4, 12)),
    ("8psk", 5e4, 100, 16, 3e4, 1, (8, 8)),      # coarse ANCHOR itself aliases -> validates rot.
])
def test_preamble_hard_corner_refuses_rotated_alias(mod, baud, nsym, snr, fc, seed, pre):
    # short + low-SNR + large-offset packetized bursts: the carrier alias (off by baud/L) is
    # blind-ambiguous -- a rotated alias still locks (~55-80) with garbage bits (ber ~0.44), and
    # lock/margin/anchor-distance all overlap correct vs wrong (the coarse anchor can itself alias).
    # The absolute lock floor (_SYNC_ACCEPT) + margin (_ALIAS_MARGIN) + anchor distance refuse
    # them (precision over recall), so the honest low-lock blind result stands -- no confident
    # wrong preamble decode. Before these gates, all four decoded as confident rotation garbage.
    x, tx = generate(GenParams(mod=mod, fs=FS, baud=baud, fc=fc, snr=snr,
                               n_symbols=nsym, seed=seed, preamble=pre))
    r = analyze(x, M)
    assert r.preamble is None, (mod, fc, r.lock, r.mod)  # refused, not a confident rotated alias


def test_multipath_not_hijacked_by_preamble_rescue():
    # a 1-symbol echo fakes a short periodicity; the eq rescue runs FIRST and lifts the lock, so
    # the preamble rescue is skipped and the bpsk multipath signal is not mis-driven.
    x, tx = generate(GenParams(mod="bpsk", fs=FS, baud=1e5, snr=16, fc=0.02e6, phase=0.9,
                               taps=(1.0, 0.5 * np.exp(1j * 0.5)), tap_sym=1.0, seed=0))
    r = analyze(x, M)
    assert r.mod == "bpsk" and r.preamble is None
    assert ber(r.symbols, "bpsk", tx) < 0.01
