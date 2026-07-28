"""FSK/MSK gate + demod against generator ground truth."""

import numpy as np
import pytest

from signus.constellations import FSK_MODS, GEN_MODS, mod_order
from signus.fsk import analyze_fsk, fsk_gate
from signus.gen import GenParams, generate

_LINEAR = [m for m in GEN_MODS if m not in FSK_MODS]


def _ber(rx: np.ndarray, tx: np.ndarray, bps: int, crop: int = 20) -> float:
    """Min BER over integer symbol shifts in [-8, 8], edge symbols cropped."""
    best = 1.0
    for sh in range(-8, 9):
        s = sh * bps
        a = rx[s:] if s > 0 else rx
        b = tx[-s:] if s < 0 else tx
        n = min(a.size, b.size)
        c = crop * bps
        if n <= 2 * c:
            continue
        best = min(best, float(np.mean(a[c:n - c] != b[c:n - c])))
    return best


# --- gate: both directions across every generator vocabulary mod ------------

@pytest.mark.parametrize("mod", _LINEAR)
@pytest.mark.parametrize("snr", [25, 15])
def test_gate_false_for_linear(mod, snr):
    for seed in range(4):
        p = GenParams(mod=mod, n_symbols=4000, snr=snr, seed=seed)
        x, _ = generate(p)
        assert not fsk_gate(x, p.fs), (mod, snr, seed)


@pytest.mark.parametrize("mod", FSK_MODS)
@pytest.mark.parametrize("snr", [25, 15, 10])
def test_gate_true_for_fsk(mod, snr):
    for seed in range(4):
        p = GenParams(mod=mod, n_symbols=4000, snr=snr, seed=seed)
        x, _ = generate(p)
        assert fsk_gate(x, p.fs), (mod, snr, seed)


def test_gate_noise_is_false():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(40000) + 1j * rng.standard_normal(40000)
    assert not fsk_gate(x, 1e6)


def test_gate_immune_to_large_cfo():
    # regression: at fc=0.1625*fs a linear mod's phase-transition spikes wrapped
    # past +-fs/2 and faked a bimodal instantaneous frequency
    p = GenParams(mod="qpsk", n_symbols=6000, snr=14, fc=162500.0, seed=2, phase=0.8)
    x, _ = generate(p)
    assert not fsk_gate(x, p.fs)
    pf = GenParams(mod="fsk2", n_symbols=4000, snr=18, h=0.7, fc=50000.0, seed=0)
    xf, _ = generate(pf)
    assert fsk_gate(xf, pf.fs)  # recentering must not cost genuine CFO'd FSK


# --- full demod -------------------------------------------------------------

_CASES = [("fsk2", 0.7), ("fsk2", 1.0), ("fsk4", 0.5), ("fsk4", 0.7),
          ("fsk4", 1.0), ("msk", 0.5)]


@pytest.mark.parametrize("mod,h", _CASES)
@pytest.mark.parametrize("snr", [20, 14])
def test_demod(mod, h, snr):
    bps = mod_order(mod).bit_length() - 1
    for seed in range(4):
        p = GenParams(mod=mod, n_symbols=4000, snr=snr, h=h, seed=seed)
        x, tx = generate(p)
        r = analyze_fsk(x, p.fs)
        tag = (mod, h, snr, seed)
        assert r["mod"] == mod, tag
        assert abs(r["fc"]) < 300, tag
        assert abs(r["baud"] - p.baud) / p.baud < 0.02, tag
        assert abs(r["h"] - h) < 0.15, tag
        assert r["lock"] >= 60, tag
        ber = _ber(r["bits"], tx, bps)
        # fsk4 Gray levels: an adjacent-level slip flips a single bit
        limit = 0.002 if mod == "fsk4" else (0.0 if snr >= 20 else 0.003)
        assert ber <= limit, (tag, ber)


@pytest.mark.parametrize("mod,baud,h", [("fsk2", 2e4, 0.35), ("fsk2", 2e4, 0.7),
                                        ("msk", 2e4, 0.5), ("fsk4", 2e4, 0.7),
                                        ("fsk2", 5e4, 0.7)])
@pytest.mark.parametrize("fc", [0.0, 5e4, 1e5])       # carrier offset: the missed corner
def test_oversampled_fsk_not_confident_wrong(mod, baud, h, fc):
    # regression: at fs/baud >= 20 the discriminator's symbol-rate line is buried under broadband
    # transition artifacts, so _est_baud locked a spurious peak (baud ~25% off) while the clean
    # tones still clustered -> a confident (lock ~94) WRONG decode. A decimated baud re-estimate
    # pins the true rate; at a carrier OFFSET the discriminator is recentred first (else the
    # baseband LPF and harmonic fold corrupt it). Decode right, or reject to linear -- never wrong.
    from signus.pipeline import analyze
    from signus.sigio import Meta
    for seed in range(3):
        x, tx = generate(GenParams(mod=mod, fs=1e6, baud=baud, h=h, snr=25, fc=fc,
                                   n_symbols=3000, seed=seed))
        r = analyze(x, Meta(1e6, "iq", "f32", "le", False))
        tag = (mod, baud, h, fc, seed, r.mod, round(r.lock), round(r.baud))
        if r.family != "fsk":
            continue                       # an honest reject to linear is acceptable
        assert abs(r.baud - baud) / baud < 0.02, tag
        ber = _ber(r.bits, tx, mod_order(r.mod).bit_length() - 1)
        assert not (r.lock >= 60 and ber > 0.1), tag  # correct rate never gives a confident-wrong


@pytest.mark.parametrize("h,nsym,seed", [(0.5, 320, 0), (0.35, 210, 3), (0.35, 400, 2)])
def test_midlength_fsk_true_baud_not_doubled(h, nsym, seed):
    # regression: the harmonic-comb fold guard summed out-of-range harmonics onto the LAST spectrum
    # bin (index clamp), multi-counting the Nyquist bin -- that inflated _hp(k) at the 2*baud peak
    # and vetoed the legitimate 2*baud -> baud fold. A mid-length burst (nsym 200-500, prominence
    # above the weak-line gate) then decoded at TWICE the true baud with lock ~80: confident-wrong.
    # In-range harmonics only: the fold fires and the true baud wins again.
    from signus.pipeline import analyze
    from signus.sigio import Meta
    x, tx = generate(GenParams(mod="fsk2", fs=1e6, baud=1e5, h=h, snr=20,
                               n_symbols=nsym, seed=seed))
    r = analyze(x, Meta(1e6, "iq", "f32", "le", False))
    tag = (h, nsym, seed, r.mod, round(r.baud), round(r.lock))
    assert r.family == "fsk", tag
    assert abs(r.baud - 1e5) / 1e5 < 0.02, tag
    ber = _ber(r.bits, tx, mod_order(r.mod).bit_length() - 1)
    assert not (r.lock >= 60 and ber > 0.1), (tag, ber)


@pytest.mark.parametrize("h,fc,seed", [(0.35, 1e5, 8), (0.35, 8e3, 8), (0.5, 5e4, 1)])
def test_low_h_baud_no_halfrate_fold(h, fc, seed):
    # regression: at low modulation index / low SNR the true symbol-rate line stays strongest but
    # spurious energy at baud/2 used to clear the fold gate -> baud folded to baud/2, a confident
    # WRONG decode (lock ~100, baud 50% off). The harmonic-comb guard rejects that fold.
    from signus.pipeline import analyze
    from signus.sigio import Meta
    x, tx = generate(GenParams(mod="fsk2", fs=1e6, baud=1e5, h=h, snr=15, fc=fc,
                               n_symbols=3000, seed=seed))
    r = analyze(x, Meta(1e6, "iq", "f32", "le", False))
    if r.family != "fsk":
        return
    assert abs(r.baud - 1e5) / 1e5 < 0.02, (h, fc, seed, r.baud)   # not folded to baud/2


@pytest.mark.parametrize("mod,h,nsym,snr,fc,seed", [
    ("fsk2", 0.5, 150, 12, 8e3, 10),   # half-rate fold: baud 24% off at lock 61
    ("msk", 0.5, 150, 14, 0.0, 14),    # msk baud 32% off at lock 65
    ("msk", 0.5, 170, 14, 0.0, 12),    # msk baud 11% off at lock 63
])
def test_midband_fsk_weak_line_never_confident_wrong(mod, h, nsym, snr, fc, seed):
    # A lowered weak-line floor (nsym<100) let these 100-199-symbol bursts through: a weak
    # symbol-rate line (prominence ~3) still clusters a HALF-RATE baud fold tight enough to
    # lock >=60, so they decoded confident-wrong (baud 11-32% off). The internal symbol count
    # sits at 100-160 (a fold halves it from the true count), disproving the "confident-wrong =>
    # internal nsym <= 49" premise: the count floor must stay high enough to reject a weak line
    # here. Either a clean ValueError or a correct baud -- never a confident wrong one.
    from signus.pipeline import analyze
    from signus.sigio import Meta
    x, _ = generate(GenParams(mod=mod, fs=1e6, baud=1e5, h=h, snr=snr, fc=fc,
                              n_symbols=nsym, seed=seed))
    try:
        r = analyze(x, Meta(1e6, "iq", "f32", "le", False))
    except ValueError:
        return                                     # clean reject of an unreliable weak line
    if r.family != "fsk":
        return
    assert not (r.lock >= 60 and abs(r.baud - 1e5) / 1e5 > 0.02), \
        (mod, h, nsym, snr, fc, seed, round(r.baud), round(r.lock))


@pytest.mark.parametrize("nsym", [12, 20, 40, 64, 100])
def test_short_fsk_burst_never_confident_wrong_baud(nsym):
    # a short FSK burst has too few transitions for a reliable symbol-rate line, so any baud it
    # produces is a spurious peak that still clusters tight (few points -> lock ~100) = confident
    # WRONG. The prominence/symbol-count guard rejects a weak line, so no short burst decodes
    # with a wrong baud (it either decodes correctly or raises a clean ValueError).
    from signus.pipeline import analyze
    from signus.sigio import Meta
    M = Meta(1e6, "iq", "f32", "le", False)
    for mod in ("fsk2", "fsk4", "msk"):
        for fc in (0, 8e3, 5e4):
            for seed in range(4):
                x, _ = generate(GenParams(mod=mod, fs=1e6, baud=1e5, fc=fc, snr=18,
                                          n_symbols=nsym, seed=seed))
                try:
                    r = analyze(x, M)
                except ValueError:
                    continue                   # clean reject of an unreliable burst
                if r.family == "fsk":
                    assert not (r.lock >= 60 and abs(r.baud - 1e5) / 1e5 > 0.02), \
                        (mod, fc, seed, nsym, r.baud, r.lock)


def test_bw99_degenerate_burst_no_crash():
    # latent crash: the LO-edge searchsorted was unclamped (hi was clamped) -- an all-zero /
    # denormal-power burst made the floor-subtracted PSD sum to zero and indexed one past the
    # end (raw IndexError where the caller expects a bandwidth).
    from signus.fsk import _bw99
    assert _bw99(np.zeros(4096, complex), 1e6) == 0.0


@pytest.mark.parametrize("mod,h", [("fsk2", 1.0), ("fsk4", 0.7), ("msk", 0.5)])
def test_demod_cfo(mod, h):
    """A large carrier offset is absorbed by recentering on mean(f)."""
    bps = mod_order(mod).bit_length() - 1
    p = GenParams(mod=mod, n_symbols=4000, snr=20, h=h, seed=0, fc=8000)
    x, tx = generate(p)
    r = analyze_fsk(x, p.fs)
    assert r["mod"] == mod
    assert abs(r["baud"] - p.baud) / p.baud < 0.02
    assert abs(r["h"] - h) < 0.15
    assert r["lock"] >= 60
    assert _ber(r["bits"], tx, bps) <= (0.002 if mod == "fsk4" else 0.0)


def test_fsk_gate_rejects_high_baud_psk_keeps_real_fsk():
    # recalibrated _CV_MAX: high-symbol-rate constant-modulus PSK (fs/baud~3) used to
    # misroute into the FSK demod; real FSK (down to the supported snr-10 floor) must stay in.
    from signus.fsk import fsk_gate
    from signus.gen import GenParams, generate
    for mod, baud in (("qpsk", 1e6 / 3), ("8psk", 4e5), ("pi4dqpsk", 1e6 / 3)):
        x, _ = generate(GenParams(mod=mod, fs=1e6, baud=baud, n_symbols=6000, snr=20, seed=0))
        assert not fsk_gate(x, 1e6), f"{mod} high-baud still misgated to FSK"
    for mod, h in (("fsk2", 0.7), ("fsk4", 1.0), ("msk", 0.5)):
        # worst FSK corner (snr10, high baud) must still gate True
        x, _ = generate(GenParams(mod=mod, fs=1e6, baud=2e5, snr=10, h=h, seed=0))
        assert fsk_gate(x, 1e6), f"{mod} FSK no longer detected"


@pytest.mark.parametrize("mod,nsym,seed", [("fsk2", 12, 2), ("fsk2", 12, 3),
                                           ("fsk4", 16, 1), ("msk", 18, 0)])
def test_short_fsk_burst_rejects_cleanly_not_crash(mod, nsym, seed):
    # a too-short FSK burst has no baud line / too few symbols to cluster levels; analyze_fsk
    # must raise a clean ValueError, not an IndexError (empty np.quantile) or ZeroDivisionError.
    from signus.pipeline import analyze
    from signus.sigio import Meta
    x, _ = generate(GenParams(mod=mod, fs=1e6, baud=1e5, fc=8e3, snr=18, n_symbols=nsym, seed=seed))
    with pytest.raises(ValueError):
        analyze(x, Meta(1e6, "iq", "f32", "le"))
