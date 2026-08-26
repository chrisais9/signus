"""Blind linear-chirp / CSS (LoRa) detection + characterization. A constant-envelope
signal whose instantaneous frequency ramps linearly de-chirps to a tone: the delay-
conjugate x[t]*conj(x[t-tau]) is a beat tone at mu*tau (NONZERO), unlike FSK/CW (beat at
DC), PSK/QAM (not constant-envelope), or FM-voice/noise (broadband). Characterize-only --
reported like analog/tone, NEVER demodulated: blind CSS symbol decode needs preamble
CFO/STO sync plus LoRa's reverse-engineered Gray/interleave/Hamming/whitening framing.
Calibrated on gen chirps vs every other family: chirp beat-PAR>=936, non-chirp<=264."""

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import welch

_PAR_MIN = 400.0   # delay-conjugate beat-tone peak-to-mean floor (chirp>=936, non-chirp<=264)


def _bw99(x: np.ndarray, fs: float) -> float:
    """99%-power occupied bandwidth (a chirp fills its band ~flat). Floor-subtracted so the
    noise pedestal in the channel's LPF passband does not inflate the width."""
    f, p = welch(x, fs=fs, nperseg=min(4096, x.size), return_onesided=False, detrend=False)
    f, p = np.fft.fftshift(f), np.fft.fftshift(p)
    p = np.maximum(p - np.median(p), 0.0)    # remove the noise pedestal
    cs = np.cumsum(p) / (p.sum() + 1e-30)
    lo = f[min(np.searchsorted(cs, 0.005), f.size - 1)]  # clamp BOTH edges: a degenerate PSD
    hi = f[min(np.searchsorted(cs, 0.995), f.size - 1)]  # (all-zero after floor subtraction)
    return float(abs(hi - lo))                           # otherwise indexes one past the end


def _beat(x: np.ndarray, fs: float) -> tuple[float, float]:
    """(peak-to-mean, mu[Hz/s]) of the DC-nulled delay-conjugate spectrum."""
    x = x - x.mean()
    tau = max(1, x.size // 200)
    y = x[tau:] * np.conj(x[:-tau])
    nf = 1 << int(np.ceil(np.log2(y.size)))
    spec = np.abs(np.fft.fftshift(np.fft.fft((y - y.mean()) * np.hanning(y.size), nf))) ** 2
    fr = np.fft.fftshift(np.fft.fftfreq(nf, 1 / fs))
    spec[np.abs(fr) < 0.002 * fs] = 0.0    # null DC band: FSK/CW/tone beat here, chirps do not
    k = int(np.argmax(spec))
    return float(spec[k] / (spec.mean() + 1e-30)), float(fr[k] * fs / tau)


def is_chirp(x: np.ndarray, fs: float) -> bool:
    """True when a channel is a linear chirp / CSS (LoRa/FMCW). The beat-tone PAR is the
    discriminator: a huge margin (chirp>=936 vs PSK/QAM<=94, FSK<=91, FM<=264, noise<=13)
    makes an envelope-CV pre-gate redundant AND harmful (it rejects noisy-but-clear chirps)."""
    if x.size < 256:
        return False
    return _beat(x, fs)[0] >= _PAR_MIN


_SWEEP_PM_MAX = 1.45   # instantaneous-freq histogram peak-to-mean: a band-sweep stays ~1
_SWEEP_OCC_MIN = 0.6   # ...and fills its band; FSK sits on 2-4 discrete tones (peaky, sparse)


def sweeps_band(x: np.ndarray, fs: float, bins: int = 40) -> bool:
    """True when the instantaneous frequency SWEEPS the channel (linear chirp / CSS) rather
    than hopping between a few discrete FSK tones. A linear FMCW ramp reads bimodal to
    fsk_gate AND trips is_chirp's beat test, so analyze's chirp gate needs this to tell a sweep
    from FSK: a sweep's IF histogram is flat and fully occupied; FSK's spikes at its tones.
    Calibrated on extracted channels: 0/222 FSK/MSK (snr 8-40) pass, 214/216 FMCW + 60/60 LoRa
    pass (misses only the narrowest, gentlest ramps -- which stay 'fsk', never a regression).
    BOTH the raw and a lightly-smoothed IF must look like a sweep: at moderate SNR (~14) noise
    flattens integer-h FSK's raw histogram to sweep-like pm ~1.44 (knife-edge), but smoothing
    restores its tone spikes (pm 2.5+) while a genuine ramp stays flat (pm ~1.1) -- and at LOW
    SNR smoothing can flatten FSK instead, which the raw view still catches. The conjunction
    only ever narrows 'sweep', so every calibrated chirp above keeps its label (margin >=0.3)."""
    if x.size < 256:
        return False
    x = x - x.mean()
    f0 = np.diff(np.unwrap(np.angle(x))) * fs / (2 * np.pi)
    for f in (f0, uniform_filter1d(f0, 4)):
        lo, hi = np.percentile(f, 1), np.percentile(f, 99)
        if hi - lo < 1:
            return False
        fv = f[(f >= lo) & (f <= hi)]
        h = np.histogram(fv, bins=bins)[0] / fv.size
        pm = h.max() / (h.mean() + 1e-30)           # discrete tones spike this; a sweep ~1
        occ = float((h > 0.005).mean())             # a sweep fills the band; tones occupy few bins
        if not (pm < _SWEEP_PM_MAX and occ >= _SWEEP_OCC_MIN):
            return False
    return True


def analyze_chirp(x: np.ndarray, fs: float) -> dict:
    """Characterize a chirp: {mu[Hz/s], up, bw, par, sf, rs, tsym}. A LoRa hypothesis
    (sf, symbol rate, symbol time) is filled when bw^2/|mu| snaps to 2^(7..12); otherwise
    it stays a generic linear chirp / FMCW. bw is measured on the channel, not asserted."""
    par, mu = _beat(x, fs)
    bw = _bw99(x, fs)
    sf = rs = tsym = None
    if mu != 0 and bw > 0:
        r = float(np.log2(bw * bw / abs(mu)))
        if 7 <= round(r) <= 12 and abs(r - round(r)) < 0.3:    # power-of-two chip count -> LoRa
            sf = int(round(r))
            rs = abs(mu) / bw
            tsym = 2 ** sf / bw
    return {"mu": mu, "up": bool(mu > 0), "bw": float(bw),
            "par": par, "sf": sf, "rs": rs, "tsym": tsym}
