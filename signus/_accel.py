"""Optional numba acceleration for the sequential feedback loops (adaptive equalizer
taps, decision-directed carrier) that CANNOT be vectorized -- each update depends on
the previous output, so they are the only python-loop hot spots in an otherwise
vectorized pipeline. Numba is an OPTIONAL extra (`pip install signus[fast]`): when it
is absent `njit` degrades to a no-op and the identical bodies run as plain numpy, so
results are unchanged and the numpy+scipy-only baseline still works. The kernels use
the same expressions as the pure-numpy originals, so the numba path stays numerically
equivalent (verified: byte-identical `w @ win`, argmin/abs/angle exact)."""

import os

import numpy as np

HAVE_NUMBA = False
if not os.environ.get("SIGNUS_NO_NUMBA"):  # escape hatch: force the pure-numpy path
    try:
        import numba
        njit = numba.njit(cache=True, fastmath=False)  # fastmath off: keep IEEE results
        HAVE_NUMBA = True
    except ImportError:  # numba absent: same bodies run as plain numpy (slower, identical)
        pass
if not HAVE_NUMBA:
    def njit(fn):
        return fn


@njit
def cma_dd_equalize(x, pts, r2, n_taps, mu_cma, mu_dd, warmup):
    """Symbol-spaced CMA acquisition -> decision-directed LMS -> converged refilter."""
    n = x.size
    mid = n_taps // 2
    xp = np.concatenate((np.zeros(mid, dtype=np.complex128), x,
                         np.zeros(mid, dtype=np.complex128)))
    w = np.zeros(n_taps, dtype=np.complex128)
    w[mid] = 1.0
    out = np.empty(n, dtype=np.complex128)
    warm = min(warmup, n)
    for i in range(warm):                        # Phase 1: CMA (blind, modulus-only)
        win = xp[i:i + n_taps]
        y = w @ win
        w = w - mu_cma * (y * (abs(y) ** 2 - r2)) * np.conj(win)
    for i in range(warm, n):                     # Phase 2: decision-directed LMS
        win = xp[i:i + n_taps]
        y = w @ win
        d = pts[np.argmin(np.abs(y - pts))]
        w = w - mu_dd * (y - d) * np.conj(win)
    for i in range(n):                           # refilter whole record with final taps
        win = xp[i:i + n_taps]
        y = w @ win
        d = pts[np.argmin(np.abs(y - pts))]
        w = w - mu_dd * (y - d) * np.conj(win)
        out[i] = y
    return out


@njit
def cma_dd_equalize_fse(x, pts, r2, n_taps, mu_cma, mu_dd, warmup):
    """T/2 fractionally-spaced: 2 CMA passes -> 4 DD passes -> tracked output pass."""
    nsym = x.size // 2
    mid = n_taps // 2
    xp = np.concatenate((np.zeros(mid, dtype=np.complex128), x,
                         np.zeros(n_taps, dtype=np.complex128)))
    w = np.zeros(n_taps, dtype=np.complex128)
    w[mid] = 1.0
    out = np.empty(nsym, dtype=np.complex128)
    warm = min(warmup, nsym)
    for _ in range(2):                           # CMA acquisition, 2 data-reuse passes
        for i in range(warm):
            win = xp[2 * i:2 * i + n_taps]
            y = w @ win
            w = w - mu_cma * (y * (abs(y) ** 2 - r2)) * np.conj(win)
    for _ in range(4):                           # decision-directed refinement
        for i in range(nsym):
            win = xp[2 * i:2 * i + n_taps]
            y = w @ win
            d = pts[np.argmin(np.abs(y - pts))]
            w = w - mu_dd * (y - d) * np.conj(win)
    for i in range(nsym):                        # final tracked output pass
        win = xp[2 * i:2 * i + n_taps]
        y = w @ win
        d = pts[np.argmin(np.abs(y - pts))]
        w = w - mu_dd * (y - d) * np.conj(win)
        out[i] = y
    return out


@njit
def dd_carrier(z, pts, alpha, beta):
    """Per-symbol decision-directed PI carrier loop (general PSK/QAM, not PSK Costas)."""
    out = np.empty(z.size, dtype=np.complex128)
    phase = 0.0
    freq = 0.0
    for i in range(z.size):
        y = z[i] * np.exp(-1j * phase)
        d = pts[np.argmin(np.abs(y - pts))]
        e = np.angle(y * np.conj(d))
        out[i] = y
        phase += freq + alpha * e
        freq += beta * e
    return out
