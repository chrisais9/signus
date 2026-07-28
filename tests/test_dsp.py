"""Receiver DSP stages against generator ground truth."""

import numpy as np
import pytest

from signus import dsp
from signus.constellations import MODS, ideal_points, mod_symmetry
from signus.gen import GenParams, generate


def _nearest(z: np.ndarray, mod: str) -> float:
    z = z / np.sqrt(np.mean(np.abs(z) ** 2))
    return float(np.abs(z[:, None] - ideal_points(mod)[None, :]).min(1).mean())


def _front(p: GenParams) -> np.ndarray:
    x, _ = generate(p)
    x = dsp.analytic(x)
    return x - x.mean()


def test_analytic_suppresses_negative_image():
    n = np.arange(4096)
    xa = dsp.analytic(np.cos(2 * np.pi * 200 * n / 4096))
    spec = np.abs(np.fft.fft(xa))
    assert spec[-200] < 1e-6 * spec[200]
    z = np.exp(1j * n[:64])
    assert np.array_equal(dsp.analytic(z), z.astype(np.complex128))


def test_find_burst_padded_and_fallbacks():
    p = GenParams(mod="qpsk", n_symbols=4000, snr=15, pad=0.5, seed=0)
    x = _front(p)
    n_burst = x.size // 2  # pad=0.5 -> noise|burst|noise thirds of burst length
    s, e = dsp.find_burst(x, p.fs)
    t0, t1 = x.size // 4, x.size // 4 + n_burst
    overlap = max(0, min(e, t1) - max(s, t0))
    assert overlap / n_burst >= 0.9
    full = _front(GenParams(mod="qpsk", n_symbols=2000, seed=1))
    assert dsp.find_burst(full, 1e6) == (0, full.size)
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(20000) + 1j * rng.standard_normal(20000)
    dsp.find_burst(noise, 1e6)  # must not raise


def test_find_bursts_multiple_and_strongest():
    p1 = GenParams(mod="qpsk", n_symbols=3000, snr=18, fc=8000.0, pad=0.4, seed=1)
    p2 = GenParams(mod="qpsk", n_symbols=3000, snr=18, fc=8000.0, pad=0.4, seed=2)
    x = np.concatenate([_front(p1), _front(p2)])
    bursts = dsp.find_bursts(x, 1e6)
    assert len(bursts) == 2 and bursts[0][1] <= bursts[1][0]  # time-ordered
    assert dsp.find_burst(x, 1e6) in bursts


def test_find_bursts_keeps_short_packet_in_long_record():
    # regression: the min-burst length scaled with the record (n//200), so a short packetized
    # burst -- the preamble-sync feature's whole target -- was dropped from a long record and
    # find_bursts fell back to [(0, n)]: analyze() then ran on a million noise samples and the
    # packet was undecodable. The min length is now capped absolutely; the merge gap (which
    # heals intra-signal splits and may stay proportional) is unchanged.
    rng = np.random.default_rng(0)
    n = 1_000_000
    x = 0.05 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    pkt, _ = generate(GenParams(mod="qpsk", n_symbols=100, preamble=(8, 8), snr=25, seed=1))
    pkt = pkt / np.sqrt(np.mean(np.abs(pkt) ** 2))
    mid = n // 2
    x[mid:mid + pkt.size] += pkt
    bursts = dsp.find_bursts(x, 1e6)
    assert bursts != [(0, n)], "packet burst was dropped (whole-record fallback)"
    s, e = max(bursts, key=lambda b: float(np.sum(np.abs(x[b[0]:b[1]]) ** 2)))
    assert mid - 2000 <= s <= mid + 200 and mid + pkt.size - 200 <= e <= mid + pkt.size + 2000, \
        (s, e, mid, pkt.size)


def test_find_bursts_rejects_impulse_transient_in_long_record():
    # regression from the mlen cap: capping the min burst length at 1024 let a SHORT high-energy
    # transient (an impulse smeared by the n//1000 smoother into a ~win-wide run) qualify as a
    # burst in a long record. Its total energy can exceed the real signal, so analyze()'s
    # argmax-energy auto-selection picks the blip and returns garbage. The impulse is rejected by
    # shape (its raw power is a single spike, unlike a modulated packet's flat envelope), so the
    # genuine burst remains the strongest -- while a real short packet (test above) still survives.
    rng = np.random.default_rng(0)
    n = 4_000_000
    x = 0.05 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    sig, _ = generate(GenParams(mod="qpsk", n_symbols=30000, snr=25, seed=1))
    sig = sig / np.sqrt(np.mean(np.abs(sig) ** 2))
    x[1_500_000:1_500_000 + sig.size] += sig
    x[3_000_000] += 6000 + 6000j                       # one huge impulse sample
    s, e = max(dsp.find_bursts(x, 2e6), key=lambda b: float(np.sum(np.abs(x[b[0]:b[1]]) ** 2)))
    assert s >= 1_450_000 and e <= 1_500_000 + sig.size + 50_000, (s, e)   # the genuine burst
    assert not (s <= 3_000_000 < e), (s, e)            # NOT the impulse


def test_resolve_alias_deep_wrap_and_identity():
    # 4*fc = 1.3*(fs/2): the 4th-power tone wraps, and the wrapped estimate lands
    # back inside the "unambiguous" zone, so no flag can catch it -- resolve always
    p = GenParams(mod="qpsk", n_symbols=8000, fs=1e6, baud=1e5, snr=25,
                  fc=162500.0, seed=0)
    x = _front(p)
    fc, sym, _ = dsp.est_carrier(x, p.fs)
    assert abs(fc - p.fc) > 1e4                 # raw estimate is wrapped-wrong
    assert abs(dsp.resolve_alias(x, p.fs, fc, sym) - p.fc) < 300
    # ...and it must NOT move a correct estimate (fc=0 offers +-fs/4 candidates)
    p0 = GenParams(mod="qpsk", n_symbols=8000, snr=25, seed=0)
    x0 = _front(p0)
    fc0, sym0, _ = dsp.est_carrier(x0, p0.fs)
    assert abs(dsp.resolve_alias(x0, p0.fs, fc0, sym0)) < 300


def test_occupied_bw_tracks_rolloff():
    for roll, lo, hi in ((0.05, 1.02, 1.18), (0.35, 1.18, 1.40)):
        p = GenParams(mod="qpsk", n_symbols=8000, snr=25, rolloff=roll,
                      fc=8000.0, seed=0)
        bw = dsp.occupied_bw(dsp.mix(_front(p), p.fs, p.fc), p.fs)
        assert lo < bw / p.baud < hi, roll


@pytest.mark.parametrize("mod", MODS)
def test_est_carrier_symmetry_and_fc(mod):
    p = GenParams(mod=mod, n_symbols=8000, fs=1e6, baud=5e4, snr=25, fc=12345.0, seed=0)
    fc, sym, amb = dsp.est_carrier(_front(p), p.fs)
    assert sym == mod_symmetry(mod)
    assert abs(fc - 12345.0) < 200 and not amb


def test_est_carrier_zero_cfo_and_dc():
    # v1 regression: DC-nulling destroyed genuine fc=0; DC offset broke detection
    for kw in ({}, {"dc": 0.5}):
        p = GenParams(mod="qpsk", n_symbols=8000, snr=25, seed=0, **kw)
        fc, sym, _ = dsp.est_carrier(_front(p), p.fs)
        assert sym == 4 and abs(fc) < 200, kw


def test_est_carrier_aliasing_flag():
    p = GenParams(mod="qpsk", n_symbols=8000, fs=1e6, baud=5e4, snr=25, fc=130e3, seed=0)
    _, _, amb = dsp.est_carrier(_front(p), p.fs)
    assert amb


@pytest.mark.parametrize("fs,baud", [(1e6, 1e5), (1e6, 1.37e5), (4.8e5, 4.8e4)])
def test_est_baud(fs, baud):
    p = GenParams(mod="qpsk", n_symbols=8000, fs=fs, baud=baud, snr=25, seed=0)
    est, _ = dsp.est_baud(_front(p), fs)
    assert abs(est - baud) / baud < 0.02


def test_baud_confidence_tracks_rolloff():
    def conf(roll):
        p = GenParams(mod="qpsk", n_symbols=8000, snr=25, rolloff=roll, seed=0)
        return dsp.est_baud(_front(p), p.fs)[1]
    assert conf(0.35) > conf(0.05)


@pytest.mark.parametrize("alpha", [0.2, 0.35, 0.5])
def test_est_rolloff(alpha):
    p = GenParams(mod="qpsk", n_symbols=8000, snr=24, rolloff=alpha, seed=0)
    assert abs(dsp.est_rolloff(_front(p), p.fs, p.baud) - alpha) < 0.1


def test_timing_recovers_offset_and_tracks_drift():
    p = GenParams(mod="qpsk", n_symbols=8000, fs=4e5, baud=1e5, snr=30, timing=0.4, seed=0)
    x, _ = generate(p)
    syms = dsp.timing(dsp.matched(x, 4, 0.35), 4)
    assert _nearest(syms, "qpsk") < 0.2
    pd = GenParams(mod="qpsk", n_symbols=8000, fs=4e5, baud=1e5, snr=30,
                   drift_ppm=100, seed=0)
    xd, _ = generate(pd)
    ym = dsp.matched(xd, 4, 0.35)
    blockwise = _nearest(dsp.timing(ym, 4), "qpsk")
    global_ = _nearest(dsp.timing(ym, 4, block=10**9), "qpsk")
    assert blockwise < global_  # block-wise O&M must beat a single global estimate


def test_timing_out2_interleaves_half_symbols():
    p = GenParams(mod="qpsk", n_symbols=4000, fs=4e5, baud=1e5, snr=30,
                  timing=0.4, seed=0)
    x, _ = generate(p)
    ym = dsp.matched(x, 4, 0.35)
    one, two = dsp.timing(ym, 4), dsp.timing(ym, 4, out=2)
    assert two.size == 2 * one.size
    assert np.allclose(two[::2], one)  # even samples are the decision instants


def test_ddsync_converges_and_is_qam_safe():
    rng = np.random.default_rng(0)
    pts = ideal_points("qpsk")
    z = pts[rng.integers(0, 4, 4000)]
    z = z * np.exp(1j * (2 * np.pi * 0.001 * np.arange(4000) + 0.7))
    out = dsp.ddsync(z, "qpsk")
    assert _nearest(out[2000:], "qpsk") < 0.1
    # v1 regression: fine sync must not damage an already-clean QAM cloud
    q = ideal_points("16qam")[rng.integers(0, 16, 4000)]
    q = q + 0.02 * (rng.standard_normal(4000) + 1j * rng.standard_normal(4000))
    before = _nearest(q, "16qam")
    after = _nearest(dsp.ddsync(q, "16qam"), "16qam")
    assert after < before * 1.2
