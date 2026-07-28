"""Sample file I/O. Filename carries read-necessities ONLY (fs, iq|real, sample
type, endian, bitrev); ground truth lives in a `<filename>.json` sidecar the
analyzer never reads. SigMF sidecars (`<stem>.sigmf-meta`) are also understood."""

import json
import os
import re
from dataclasses import dataclass

import numpy as np

_FS_TOK = re.compile(r"(?:^|_)fs(\d+(?:\.\d+)?(?:e[+-]?\d+)?)(?:_|$)", re.I)
_RF_TOK = re.compile(r"(?:^|_)rf(\d+(?:\.\d+)?(?:e[+-]?\d+)?)(?:_|$)", re.I)

# token -> (numpy code, full-scale divisor, offset). u-types are offset binary
# (e.g. 8o / RTL-SDR u8: 0..255 with 128 = zero).
_DTYPES = {
    "i8": ("i1", 128.0, 0.0), "u8": ("u1", 128.0, 128.0),
    "i16": ("i2", 32768.0, 0.0), "u16": ("u2", 32768.0, 32768.0),
    "f32": ("f4", 1.0, 0.0), "f64": ("f8", 1.0, 0.0),
}
# baudline-style aliases: <bits>t = two's complement, <bits>o = offset binary
_ALIAS = {"8t": "i8", "8o": "u8", "16t": "i16", "16o": "u16", "32f": "f32", "64f": "f64"}
_BITREV = np.array([int(f"{i:08b}"[::-1], 2) for i in range(256)], dtype=np.uint8)

# SigMF core:datatype -> (dtype token, fmt, endian)
_SIGMF = {"cf32": ("f32", "iq"), "ci16": ("i16", "iq"), "ci8": ("i8", "iq"),
          "cu8": ("u8", "iq"), "rf32": ("f32", "real"), "ri16": ("i16", "real")}


@dataclass
class Meta:
    fs: float | None = None
    fmt: str | None = None      # 'iq' | 'real'
    dtype: str = "i16"          # key of _DTYPES
    endian: str = "le"          # 'le' | 'be' (floats included)
    bitrev: bool = False        # bit order reversed within each byte
    rf_center: float | None = None  # RF centre freq (Hz); None = unknown (report baseband)

    def ok(self) -> bool:
        return self.fs is not None and self.fmt in ("iq", "real") and self.dtype in _DTYPES


def parse_name(name: str) -> Meta:
    """Extract fs/fmt/dtype/endian/bitrev tokens from a filename; rest ignored."""
    stem = os.path.splitext(os.path.basename(name))[0].lower()
    m = Meta()
    if mt := _FS_TOK.search(stem):
        m.fs = float(mt.group(1))
    if rt := _RF_TOK.search(stem):
        m.rf_center = float(rt.group(1))
    toks = stem.split("_")
    m.fmt = next((t for t in toks if t in ("iq", "real")), None)
    m.dtype = next((_ALIAS.get(t, t) for t in toks if t in _DTYPES or t in _ALIAS), "i16")
    m.endian = "be" if "be" in toks else "le"
    m.bitrev = "bitrev" in toks
    return m


def parse_sigmf(path: str) -> Meta | None:
    """Read `<stem>.sigmf-meta` (core:sample_rate + core:datatype) if present."""
    try:
        with open(os.path.splitext(path)[0] + ".sigmf-meta") as fh:
            doc = json.load(fh)
        g = doc["global"]
        code = g["core:datatype"].replace("_le", "").replace("_be", "")
        dtype, fmt = _SIGMF[code]
        endian = "be" if g["core:datatype"].endswith("_be") else "le"
        fs = float(g["core:sample_rate"])
    except (OSError, KeyError, ValueError, TypeError, AttributeError):
        return None  # MANDATORY fields malformed: fall back to filename tokens
    try:
        # rf centre is OPTIONAL: a malformed value must cost only this field, never the whole
        # sidecar -- discarding valid fs/fmt/dtype here silently re-read the file with filename
        # tokens, which can lie about fs (a quiet garbage decode).
        caps = doc.get("captures") or []
        rf = caps[0].get("core:frequency") if caps else None
        rf = None if rf is None else float(rf)
    except (KeyError, ValueError, TypeError, AttributeError, IndexError):
        rf = None
    return Meta(fs, fmt, dtype, endian, rf_center=rf)


def make_name(label: str, m: Meta, ext: str) -> str:
    fs_tok = str(int(m.fs)) if m.fs == int(m.fs) else f"{m.fs:f}".rstrip("0").rstrip(".")
    extra = ("_be" if m.endian == "be" else "") + ("_bitrev" if m.bitrev else "")
    return f"{label}_fs{fs_tok}_{m.fmt}_{m.dtype}{extra}.{ext}"


def decode(raw: bytes, meta: Meta) -> np.ndarray:
    """Raw bytes -> samples: complex128 for 'iq', float64 for 'real'.
    Tolerates truncated files (drops trailing partial samples)."""
    code, scale, off = _DTYPES[meta.dtype]
    if meta.bitrev:
        raw = _BITREV[np.frombuffer(raw, dtype=np.uint8)].tobytes()
    step = np.dtype(code).itemsize
    dt = ("<" if meta.endian == "le" else ">") + code
    x = np.frombuffer(raw[: len(raw) - len(raw) % step], dtype=dt).astype(np.float64)
    x = (x - off) / scale
    if meta.fmt == "iq":
        x = x[: x.size & ~1]  # drop dangling half-sample
        return x[0::2] + 1j * x[1::2]
    return x


def read(path: str, meta: Meta | None = None) -> tuple[np.ndarray, Meta]:
    meta = meta or parse_sigmf(path) or parse_name(path)
    if not meta.ok():
        raise ValueError(f"need fs + fmt (filename tokens, SigMF, or --fs/--fmt): {path}")
    with open(path, "rb") as fh:
        x = decode(fh.read(), meta)
    if x.size < 256:
        raise ValueError(f"신호가 너무 짧습니다 ({x.size} samples): {path}")
    return x, meta


def write(path: str, x: np.ndarray, meta: Meta) -> None:
    """Quantize (ints: scaled to 99.9th-percentile full scale) and write interleaved."""
    code, scale, off = _DTYPES[meta.dtype]
    flat = np.column_stack([x.real, x.imag]).ravel() if np.iscomplexobj(x) else x.real
    if scale != 1.0:  # integer types
        peak = np.percentile(np.abs(flat), 99.9) or 1.0
        flat = np.round(flat * (scale - 1) / peak) + off
        info = np.iinfo(code)
        flat = np.clip(flat, info.min, info.max)
    out = flat.astype(("<" if meta.endian == "le" else ">") + code)
    if meta.bitrev:
        out = np.frombuffer(_BITREV[np.frombuffer(out.tobytes(), np.uint8)].tobytes(),
                            dtype=out.dtype)
    out.tofile(path)


def sidecar_write(path: str, meta: Meta, truth: dict, gen: dict) -> None:
    doc = {"fs": meta.fs, "fmt": meta.fmt, "dtype": meta.dtype,
           "endian": meta.endian, "bitrev": meta.bitrev, "truth": truth, "gen": gen}
    with open(path + ".json", "w") as fh:
        json.dump(doc, fh, indent=1)


def sidecar_read(path: str) -> dict | None:
    try:
        with open(path + ".json") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None
