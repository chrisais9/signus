"""Blind linear-chirp / CSS (LoRa) detection + characterization. A constant-envelope
signal whose instantaneous frequency ramps linearly de-chirps to a tone: the delay-
conjugate x[t]*conj(x[t-tau]) is a beat tone at mu*tau (NONZERO), unlike FSK/CW (beat at
DC), PSK/QAM (not constant-envelope), or FM-voice/noise (broadband). Characterize-only --
reported like analog/tone, NEVER demodulated: blind CSS symbol decode needs preamble
CFO/STO sync plus LoRa's reverse-engineered Gray/interleave/Hamming/whitening framing.
Calibrated on gen chirps vs every other family: chirp beat-PAR>=936, non-chirp<=264."""

import numpy as np
from scipy.signal import welch

_PAR_MIN = 400.0   # delay-conjugate beat-tone peak-to-mean floor (chirp>=936, non-chirp<=264)


def _bw99(x: np.ndarray, fs: float) -> float:
    """99%-power occupied bandwidth (a chirp fills its band ~flat). Floor-subtracted so the
    noise pedestal in the channel's LPF passband does not inflate the width."""
    f, p = welch(x, fs=fs, nperseg=min(4096, x.size), return_onesided=False, detrend=False)
    f, p = np.fft.fftshift(f), np.fft.fftshift(p)
    p = np.maximum(p - np.median(p), 0.0)    # remove the noise pedestal
    cs = np.cumsum(p) / (p.sum() + 1e-30)
    lo = f[np.searchsorted(cs, 0.005)]
    hi = f[min(np.searchsorted(cs, 0.995), f.size - 1)]
    return float(abs(hi - lo))


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
