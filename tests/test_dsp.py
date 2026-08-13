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


def test_cell_power_shapes_and_energy():
    # 셀 전력 헬퍼: 모양이 맞고, 톤 하나가 정확히 한 빈 열(row)만 뜨겁게 한다
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(8192) + 1j * rng.standard_normal(8192)) * 0.01
    x += np.exp(2j * np.pi * 0.25 * np.arange(8192))     # fs/4 톤
    P, hop, nper = dsp._cell_power(x, 1e6)
    assert P.shape[0] == nper and hop == nper // 2
    assert P.shape[1] == 1 + (x.size - nper) // hop
    hot_bin = int(np.argmax(P.mean(axis=1)))
    assert P.mean(axis=1)[hot_bin] > 100 * np.median(P.mean(axis=1))
    # 짧은 레코드: nperseg가 자동으로 줄어 최소 1열은 나온다
    P2, hop2, nper2 = dsp._cell_power(x[:300], 1e6)
    assert nper2 <= 300 and P2.shape[1] >= 1


def test_find_bursts_sees_narrowband_bursts_like_the_eye():
    # the operator SEES these on a spectrogram (53% of cells lit, 2026-08-11 measurement):
    # narrowband qpsk bursts at wideband 0 dB are +12 dB per CELL (fs/bw processing gain).
    # wideband energy detection dilutes them to +3 dB -> invisible. cell detection must not.
    rng = np.random.default_rng(2)
    n = 65006
    x = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
    sig, _ = generate(GenParams(mod="qpsk", n_symbols=600, fs=1e6, baud=5e4,
                                snr=60, fc=2e5, seed=1))
    truth = []
    for k in range(4):
        s = 8000 + k * 14000
        b = sig[k * 1500:(k + 1) * 1500]
        x[s:s + 1500] += b / np.sqrt(np.mean(np.abs(b) ** 2))
        truth.append((s, s + 1500))
    bursts = dsp.find_bursts(x, 1e6)
    assert bursts != [(0, n)], "narrowband bursts invisible (wideband dilution)"
    hits = sum(any(bs <= s + 300 and e - 300 <= be for bs, be in bursts) for s, e in truth)
    assert hits >= 3, (hits, bursts[:6])


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


def test_find_bursts_detects_short_strong_bursts():
    # field case (65k-sample capture): a packet train of STRONG bursts shorter than the
    # record-scaled min length was dropped entirely -> [(0, n)] fallback -> analyze ran on
    # the whole record. The operator can SEE the bursts on a spectrogram; the detector must
    # not be blinder than the operator. Min length must not scale past the smoothing width.
    rng = np.random.default_rng(3)
    n = 65006
    x = 0.1 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    truth = []
    for k in range(10):
        s = 3000 + k * 6000
        x[s:s + 200] += np.exp(2j * np.pi * 0.11 * np.arange(200))
        truth.append((s, s + 200))
    bursts = dsp.find_bursts(x, 1e6)
    assert bursts != [(0, n)], "strong 200-sample bursts fell back to whole-record"
    hits = sum(any(bs <= s and e <= be for bs, be in bursts) for s, e in truth)
    assert hits >= 8, (hits, bursts[:4])


def test_find_bursts_detects_moderate_snr_bursts():
    # field case: bursts clearly visible on a spectrogram but below the +10 dB above_hi bar
    # (a 6 dB wideband burst sits ~7 dB over the smoothed floor) were invisible -> fallback.
    rng = np.random.default_rng(5)
    n = 65006
    amp = 10 ** (-6 / 20)  # wideband snr 6 dB
    x = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * amp / np.sqrt(2)
    truth = []
    for k in range(10):
        s = 3000 + k * 6000
        x[s:s + 1000] += np.exp(2j * np.pi * 0.11 * np.arange(1000))
        truth.append((s, s + 1000))
    bursts = dsp.find_bursts(x, 1e6)
    assert bursts != [(0, n)], "6 dB bursts fell back to whole-record"
    hits = sum(any(bs <= s + 100 and e - 100 <= be for bs, be in bursts) for s, e in truth)
    assert hits >= 8, (hits, bursts[:4])


def test_find_bursts_separates_close_bursts():
    # field case: two distinct bursts with a clear noise gap (300 samples, ~5x the smoothing
    # width) were merged by the record-scaled gap (n//200=325) into one span, so analyze
    # demodulated burst+noise+burst. A visible noise gap means separate bursts.
    rng = np.random.default_rng(7)
    n = 65006
    x = 0.1 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    a, gap, blen = 20000, 300, 2000
    x[a:a + blen] += np.exp(2j * np.pi * 0.11 * np.arange(blen))
    b = a + blen + gap
    x[b:b + blen] += np.exp(2j * np.pi * 0.13 * np.arange(blen))
    bursts = dsp.find_bursts(x, 1e6)
    inside = [bu for bu in bursts if bu[0] < b + blen and bu[1] > a]
    assert len(inside) == 2, (inside, bursts)
    (s1, e1), (s2, e2) = inside
    assert e1 <= a + blen + gap and s2 >= a + blen, (inside)


def test_find_bursts_marginal_contrast_does_not_fragment():
    # adversarial find: a burst whose level sits BETWEEN the lo and hi bars (Otsu class
    # separation ~0.40-0.43 decades) passed the separation guard but shattered into up to
    # 7-8 fragments -- find_burst then picked a ~3k piece of a 20k burst, worse for demod
    # than the old whole-record fallback. A detection that explains only a minority of the
    # above-lo energy is not a detection; it must fall back (or return one covering burst).
    rng = np.random.default_rng(6)
    n = 40000
    x = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
    sig, _ = generate(GenParams(mod="qpsk", n_symbols=2000, snr=60, seed=1))
    sig = sig / np.sqrt(np.mean(np.abs(sig) ** 2)) * 10 ** (1.9 / 20)
    x[10000:10000 + sig.size] += sig
    bursts = dsp.find_bursts(x, 1e6)
    if bursts != [(0, n)]:
        s, e = max(bursts, key=lambda b: b[1] - b[0])
        cover = (min(e, 10000 + sig.size) - max(s, 10000)) / sig.size
        assert cover >= 0.8, (bursts, cover)


def test_find_bursts_partial_detection_survives_weak_siblings():
    # regression (from the coverage guard's first form): a clean strong detection was thrown
    # away because OTHER energy -- weak bursts between the lo and hi bars, never candidates --
    # inflated the above-lo mass past the guard's denominator. Only energy that actually
    # qualified as a candidate run may count against the detection.
    def _qb(n_sym, seed, amp_db):
        s, _ = generate(GenParams(mod="qpsk", n_symbols=n_sym, snr=60.0, seed=seed))
        s = s / np.sqrt(np.mean(np.abs(s) ** 2))
        return s * 10 ** (amp_db / 20)

    rng = np.random.default_rng(31)
    n = 60000
    x = np.sqrt(0.5) * (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(complex)
    b1, b2 = _qb(800, 5, 12.0), _qb(800, 6, 2.0)     # strong / weak-between-the-bars
    x[15000:15000 + b1.size] += b1
    s2 = 15000 + b1.size + 6000
    x[s2:s2 + b2.size] += b2
    bursts = dsp.find_bursts(x, 1e6)
    assert bursts != [(0, n)], "guard threw away a clean strong detection"
    assert any(s < 15200 and e > 15000 + b1.size - 200 for s, e in bursts), bursts


def test_find_bursts_borderline_mlen_drops_keep_survivors():
    # long record (win=300, mlen=428): 200-sample bursts smear into ~500-sample runs that sit
    # right at the length gate, so noise jitter drops SOME of them. The coverage guard must
    # not count those routine mlen drops as "silently discarded candidates" and throw away
    # the survivors -- only spike-gate kills (suspicious content) justify the fallback.
    rng = np.random.default_rng(0)
    n = 300000
    x = 10 ** (-9 / 20) * (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
    truth = []
    for k in range(10):
        s = 5000 + k * 23800
        x[s:s + 200] += np.exp(2j * np.pi * 0.11 * np.arange(200))
        truth.append((s, s + 200))
    bursts = dsp.find_bursts(x, 1e6)
    assert bursts != [(0, n)], "survivors were thrown away with the mlen-dropped runs"
    hits = sum(any(bs <= s + 80 and e - 80 <= be for bs, be in bursts) for s, e in truth)
    assert hits >= 4, (hits, bursts[:4])


def test_find_bursts_impulse_killed_burst_falls_back_not_misselects():
    # two emitters; an impulse inside the STRONG burst trips the spike gate and kills it.
    # Returning only the weak burst would silently hand analyze() the wrong emitter -- the
    # guard must notice that the gates discarded most of the candidate mass and fall back.
    rng = np.random.default_rng(13)
    n = 60000
    x = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
    x[10000:20000] += 10 ** (15 / 20) * np.exp(2j * np.pi * 0.11 * np.arange(10000))
    x[40000:48000] += 10 ** (8 / 20) * np.exp(2j * np.pi * 0.13 * np.arange(8000))
    x[15000] += 400 + 400j
    bursts = dsp.find_bursts(x, 1e6)
    weak_only = all(s >= 30000 for s, e in bursts) and bursts != [(0, n)]
    assert not weak_only, bursts


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
