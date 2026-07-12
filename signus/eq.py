"""Blind symbol-spaced FIR equalizer: CMA acquisition then decision-directed LMS,
to undo the multipath ISI (the generator's `taps=` channel) that a phase-only
carrier loop cannot remove."""

import numpy as np

from ._accel import cma_dd_equalize, cma_dd_equalize_fse
from .constellations import ideal_points


def equalize(symbols: np.ndarray, mod: str, n_taps: int = 11, mu_cma: float = 5e-3,
             mu_dd: float = 2e-3, warmup: int = 1500) -> np.ndarray:
    """Blind FIR equalizer for symbol-spaced samples; returns the cleaned symbols.

    Phase 1 is constant-modulus (Godard) acquisition, phase 2 refines by decisions,
    then the converged taps refilter the whole record so early symbols are clean too.
    """
    pts = ideal_points(mod)
    r2 = np.mean(np.abs(pts) ** 4) / np.mean(np.abs(pts) ** 2)  # Godard dispersion const
    x = symbols / np.sqrt(np.mean(np.abs(symbols) ** 2))        # AGC to unit power
    # The adaptive passes (every tap update depends on the previous output/decision) are
    # the one part that cannot be vectorized; they run as a numba kernel when available.
    return cma_dd_equalize(np.ascontiguousarray(x, dtype=np.complex128),
                           np.ascontiguousarray(pts, dtype=np.complex128),
                           float(r2), n_taps, mu_cma, mu_dd, warmup)


def equalize_fse(x2: np.ndarray, mod: str, n_taps: int = 75, mu_cma: float = 1e-3,
                 mu_dd: float = 8e-4, warmup: int = 4000) -> np.ndarray:
    """T/2 fractionally-spaced CMA->DD equalizer: 2 samples/symbol in, symbols out.

    Sees the unaliased spectrum, so it inverts channels whose symbol-rate folding
    puts zeros near the unit circle (e.g. a strong 1.5-symbol echo) where the
    symbol-spaced `equalize` cannot. A strong echo's inverse is LONG (0.8 gain
    needs ~70 T/2 taps for -20 dB); the clock must already be tracked, so feed it
    `timing(ym, sps, out=2)` -- fixed taps cannot follow a drifting clock.
    Calibrated on 2ray(0.8, 1.5 sym) x seeds 0-2: lock 77-80 (61 taps: 57-66)."""
    pts = ideal_points(mod)
    r2 = np.mean(np.abs(pts) ** 4) / np.mean(np.abs(pts) ** 2)  # Godard dispersion const
    x = x2 / np.sqrt(np.mean(np.abs(x2) ** 2))                  # AGC to unit power
    # Seven sequential passes (2 CMA + 4 DD + 1 tracked output); numba kernel when available.
    return cma_dd_equalize_fse(np.ascontiguousarray(x, dtype=np.complex128),
                               np.ascontiguousarray(pts, dtype=np.complex128),
                               float(r2), n_taps, mu_cma, mu_dd, warmup)
