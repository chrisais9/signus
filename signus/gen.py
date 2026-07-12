"""Synthetic signal generator — the ground-truth fixture for the test harness.
Deliberately shares no DSP code with the receiver (only constellation/level
reference data), so a receiver bug cannot be masked by a twin generator bug."""

import os
from dataclasses import asdict, dataclass, field
from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly

from .constellations import (
    DIFF_MODS,
    DIFF_PHASES,
    bit_labels,
    family,
    fsk_levels,
    ideal_points,
    mod_order,
    to_bits,
)
from .sigio import Meta, make_name, sidecar_write, write

_OS = 8         # shaping oversample (samples/symbol)
_SPAN = 10      # TX RRC span (symbols)


@dataclass
class GenParams:
    mod: str = "qpsk"
    n_symbols: int = 6000
    fs: float = 1e6
    baud: float = 1e5
    rolloff: float = 0.35
    fc: float = 0.0            # iq: carrier offset; real: passband carrier
    phase: float = 0.0
    timing: float = 0.0        # fractional-sample offset [0,1)
    snr: float = 20.0          # sample-power SNR (dB)
    fmt: str = "iq"
    dtype: str = "i16"
    endian: str = "le"
    bitrev: bool = False
    seed: int = 0
    pad: float = 0.0           # leading/trailing noise-only padding (fraction)
    drift_ppm: float = 0.0     # TX/RX sample-clock offset
    dc: complex = 0.0          # DC offset / LO leakage
    h: float = 0.5             # FSK modulation index (msk forces 0.5)
    taps: tuple = field(default_factory=tuple)  # multipath channel gains (complex)
    tap_sym: float = 1.0       # echo delay between taps, in SYMBOLS


def _tx_rrc(a: float, sps: int) -> np.ndarray:
    """TX pulse shaper — vectorized closed form, independent of the RX filter."""
    t = np.arange(-_SPAN * sps // 2, _SPAN * sps // 2 + 1) / sps
    with np.errstate(divide="ignore", invalid="ignore"):
        h = (np.sin(np.pi * t * (1 - a)) + 4 * a * t * np.cos(np.pi * t * (1 + a))) / (
            np.pi * t * (1 - (4 * a * t) ** 2))
    h[np.abs(t) < 1e-9] = 1 - a + 4 * a / np.pi
    if a > 0:
        sing = np.abs(np.abs(t) - 1 / (4 * a)) < 1e-9
        h[sing] = a / np.sqrt(2) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * a))
                                    + (1 - 2 / np.pi) * np.cos(np.pi / (4 * a)))
    return h / np.sqrt(np.sum(h**2))


def _linear(p: GenParams, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """PSK/QAM/OQPSK/differential: symbols -> RRC shaping at _OS samples/symbol."""
    m = mod_order(p.mod)
    idx = rng.integers(0, m, p.n_symbols)
    if p.mod in DIFF_MODS:  # data lives in the phase TRANSITIONS
        bits = to_bits(idx, p.mod)
        sym = np.exp(1j * np.cumsum(DIFF_PHASES[p.mod][idx]))
    else:
        bits = to_bits(bit_labels(p.mod)[idx], p.mod)
        sym = ideal_points(p.mod)[idx]
    up = np.zeros(p.n_symbols * _OS, dtype=complex)
    up[::_OS] = sym
    x = np.convolve(up, _tx_rrc(p.rolloff, _OS), mode="same")
    if p.mod == "oqpsk":  # Q rail delayed half a symbol
        x = x.real + 1j * np.concatenate([np.zeros(_OS // 2), x.imag[: -_OS // 2]])
    return x, bits


def _fsk(p: GenParams, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """CPFSK/MSK: Gray level sequence -> continuous-phase frequency modulation."""
    levels, labels = fsk_levels(p.mod)
    idx = rng.integers(0, levels.size, p.n_symbols)
    bits = to_bits(labels[idx], p.mod)
    h = 0.5 if p.mod == "msk" else p.h
    f_cps = np.repeat(levels[idx], _OS) * h / 2  # instantaneous freq, cycles/symbol
    x = np.exp(2j * np.pi * np.cumsum(f_cps) / _OS)
    return x, bits


def generate(p: GenParams) -> tuple[np.ndarray, np.ndarray]:
    """Return (samples, tx_bits). Chain: modulate -> resample -> timing/drift ->
    carrier/phase -> multipath taps -> (real) -> dc -> AWGN -> padding."""
    rng = np.random.default_rng(p.seed)
    x, bits = (_fsk if family(p.mod) == "fsk" else _linear)(p, rng)

    r = Fraction(p.fs / (p.baud * _OS)).limit_denominator(2000)
    if r != 1:
        x = resample_poly(x, r.numerator, r.denominator)
    x /= np.sqrt(np.mean(np.abs(x) ** 2))

    if p.timing or p.drift_ppm:  # fractional delay + clock drift via one regrid
        t = np.arange(x.size) * (1 + p.drift_ppm * 1e-6) + p.timing
        x = np.interp(t, np.arange(x.size), x.real) + 1j * np.interp(
            t, np.arange(x.size), x.imag)

    n = np.arange(x.size)
    x = x * np.exp(1j * (2 * np.pi * p.fc / p.fs * n + p.phase))
    if p.taps:  # multipath: echoes delayed by tap_sym symbols (=> real symbol-rate ISI)
        step = max(1, int(round(p.tap_sym * p.fs / p.baud)))
        kern = np.zeros((len(p.taps) - 1) * step + 1, dtype=complex)
        kern[::step] = p.taps
        x = np.convolve(x, kern, mode="same")
        x /= np.sqrt(np.mean(np.abs(x) ** 2))
    if p.fmt == "real":
        x = x.real.astype(np.float64) * np.sqrt(2)
    x = x + (p.dc if np.iscomplexobj(x) else complex(p.dc).real)

    nvar = np.mean(np.abs(x) ** 2) / 10 ** (p.snr / 10)
    if np.iscomplexobj(x):
        def noise(k: int) -> np.ndarray:
            return np.sqrt(nvar / 2) * (rng.standard_normal(k) + 1j * rng.standard_normal(k))
    else:
        def noise(k: int) -> np.ndarray:
            return np.sqrt(nvar) * rng.standard_normal(k)
    x = x + noise(x.size)
    if p.pad > 0:
        k = int(x.size * p.pad)
        x = np.concatenate([noise(k), x, noise(k)])
    return x, bits


def save(p: GenParams, outdir: str, label: str = "sig") -> str:
    """Write samples + `<file>.json` ground-truth sidecar; return the data path."""
    x, _ = generate(p)
    meta = Meta(p.fs, p.fmt, p.dtype, p.endian, p.bitrev)
    name = make_name(label, meta, "iq" if p.fmt == "iq" else "pcm")
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, name)
    write(path, x, meta)
    d = asdict(p)
    truth = {k: d[k] for k in ("mod", "fc", "baud", "rolloff", "snr")}
    if family(p.mod) == "fsk":
        truth["h"] = 0.5 if p.mod == "msk" else p.h
        truth.pop("rolloff")  # meaningless for CPFSK
    gen_info = {k: d[k] for k in ("seed", "n_symbols", "timing", "pad", "drift_ppm")}
    if p.taps:
        gen_info["taps"] = [[complex(t).real, complex(t).imag] for t in p.taps]
        gen_info["tap_sym"] = p.tap_sym
    sidecar_write(path, meta, truth, gen_info)
    return path
