"""Convention anchor: the Gray bit-mapping is pinned to out-of-band literals.

Every other bit-level test round-trips gen.py -> demod through the SAME
constellations tables, so the suite is closed-loop on the convention itself: an
accidental convention change (e.g. swapping the per-axis Gray roles on the QAM
label formula) would pass all of them while silently changing every emitted bit.
The literals below are the external ground truth -- written down independently
from the documented convention (PSK: binary-reflected Gray over the phase/index;
square QAM: per-axis Gray with the Q axis in the high bits; 32qam: plain index),
NOT derived from the code under test. If one of these fails, the wire format
changed: that is a breaking change, never a test to "fix"."""

import numpy as np

from signus.constellations import bit_labels, demap_bits, fsk_levels

_R2 = 0.7071067811865476            # 1/sqrt(2)
_N16 = 3.1622776601683795           # sqrt(10): 16qam unit-power scale
_N64 = 6.48074069840786             # sqrt(42): 64qam unit-power scale

_LABELS = {
    "bpsk": [0, 1],
    "qpsk": [0, 1, 3, 2],
    "8psk": [0, 1, 3, 2, 6, 7, 5, 4],
    "16qam": [0, 1, 3, 2, 4, 5, 7, 6, 12, 13, 15, 14, 8, 9, 11, 10],
    "64qam": [0, 1, 3, 2, 6, 7, 5, 4, 8, 9, 11, 10, 14, 15, 13, 12,
              24, 25, 27, 26, 30, 31, 29, 28, 16, 17, 19, 18, 22, 23, 21, 20,
              48, 49, 51, 50, 54, 55, 53, 52, 56, 57, 59, 58, 62, 63, 61, 60,
              40, 41, 43, 42, 46, 47, 45, 44, 32, 33, 35, 34, 38, 39, 37, 36],
    "32qam": list(range(32)),
}


def test_bit_labels_pinned_to_literals():
    for mod, want in _LABELS.items():
        assert bit_labels(mod).tolist() == want, mod


def test_fsk_gray_levels_pinned():
    lv, lab = fsk_levels("fsk2")
    assert lv.tolist() == [-1.0, 1.0] and lab.tolist() == [0, 1]
    lv, lab = fsk_levels("msk")
    assert lv.tolist() == [-1.0, 1.0] and lab.tolist() == [0, 1]
    lv, lab = fsk_levels("fsk4")
    assert lv.tolist() == [-3.0, -1.0, 1.0, 3.0] and lab.tolist() == [0, 1, 3, 2]


# (mod, exact symbol, expected bits) -- each derived BY HAND from the convention,
# e.g. 16qam I=+3 (axis index 3 -> gray 2, low bits), Q=+3 (gray 2, high bits) -> 1010.
_GOLDEN = [
    ("bpsk", 1 + 0j, "0"), ("bpsk", -1 + 0j, "1"),
    ("qpsk", _R2 + _R2 * 1j, "00"), ("qpsk", -_R2 + _R2 * 1j, "01"),
    ("qpsk", -_R2 - _R2 * 1j, "11"), ("qpsk", _R2 - _R2 * 1j, "10"),
    ("8psk", 1 + 0j, "000"), ("8psk", _R2 + _R2 * 1j, "001"),
    ("8psk", -1 + 0j, "110"), ("8psk", -_R2 - _R2 * 1j, "111"),
    ("16qam", (3 + 3j) / _N16, "1010"), ("16qam", (-3 - 3j) / _N16, "0000"),
    ("16qam", (1 - 1j) / _N16, "0111"),
    ("64qam", (7 + 7j) / _N64, "100100"), ("64qam", (-7 - 7j) / _N64, "000000"),
    ("64qam", (1 + 3j) / _N64, "111110"),
]


def test_demap_golden_vectors():
    for mod, sym, bits in _GOLDEN:
        got = "".join(map(str, demap_bits(np.array([sym]), mod)))
        assert got == bits, (mod, sym, got, bits)
