"""Per-channel triage: decide whether an extracted channel is a demodulable digital
signal or something the demodulator must NOT force onto a constellation (analog FM
voice, a CW tone/spur). Envelope CV is the SNR-robust separator: RRC-shaped PSK/QAM
always sit at CV>=0.25, while FSK/FM/CW are constant-envelope (CV<=0.05)."""

import numpy as np

from .chirp import is_chirp, sweeps_band
from .fsk import fsk_gate

_CV_CE = 0.15      # constant-envelope ceiling: below it a non-FSK signal is analog/tone
_TONE_PAR = 500.0  # M-th-power peak-to-mean above which a constant-envelope signal is CW


def _carrier_par(x: np.ndarray, powers: tuple[int, ...] = (1, 2, 4, 8)) -> float:
    """Best peak-to-mean of the M-th-power magnitude spectrum. A CW tone spikes at
    p=1; analog FM and noise stay flat at every power."""
    x = x - x.mean()
    nfft = 1 << int(np.ceil(np.log2(x.size)))
    win = np.hanning(x.size)
    best = 0.0
    for p in powers:
        spec = np.abs(np.fft.fft(x ** p * win, nfft))
        best = max(best, float(spec.max() / (spec.mean() + 1e-30)))
    return best


def family(x: np.ndarray, fs: float) -> str:
    """One of 'fsk' | 'chirp' | 'linear' | 'analog' | 'tone'. 'linear'/'fsk' go to the
    demod; 'chirp'/'analog'/'tone' are reported as-is (never force-fit to a constellation)."""
    if fsk_gate(x, fs):
        # a linear FMCW chirp / CSS (LoRa) trips fsk_gate too -- its swept IF reads bimodal to
        # the gate -- and would then be force-demodulated into confident garbage FSK symbols.
        # is_chirp fires on both a chirp and (at some h/baud) real FSK, so the IF-SWEEP test
        # breaks the tie: a genuine band-sweep is characterized as chirp, real FSK stays FSK.
        if is_chirp(x, fs) and sweeps_band(x, fs):
            return "chirp"
        return "fsk"
    if is_chirp(x, fs):            # linear chirp / CSS (LoRa) -- constant-envelope, would
        return "chirp"            # otherwise fall through to 'analog' (no M-power tone)
    a = np.abs(x)
    if a.std() / (a.mean() + 1e-12) < _CV_CE:            # constant envelope, not FSK
        return "tone" if _carrier_par(x) >= _TONE_PAR else "analog"
    return "linear"
