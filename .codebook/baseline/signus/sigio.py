"""Sample file I/O. Filename carries read-necessities ONLY (fs, cplx|real, sample
type, endian, bitrev); ground truth lives in a `<filename>.json` sidecar the
analyzer never reads. SigMF sidecars (`<stem>.sigmf-meta`) are also understood.
Endianness falls back to SIGNUS_ENDIAN when the filename carries no .be/.le token."""

import json
import os
import re
from dataclasses import dataclass

import numpy as np

# fs/rf 토막은 옛 밑줄 형식 전용이다. 점으로도 끊길 수 있어 경계에 둘 다 넣는다 -- 새 형식은
# 샘플레이트를 접두사 없이 cplx|real 바로 뒤에 놓으므로 이 정규식에 걸리지 않는다.
_FS_TOK = re.compile(r"(?:^|[._])fs(\d+(?:\.\d+)?(?:e[+-]?\d+)?)(?:[._]|$)", re.I)
_RF_TOK = re.compile(r"(?:^|[._])rf(\d+(?:\.\d+)?(?:e[+-]?\d+)?)(?:[._]|$)", re.I)

# token -> (numpy code, full-scale divisor, offset). u-types are offset binary
# (e.g. 8o / RTL-SDR u8: 0..255 with 128 = zero).
_DTYPES = {
    "i8": ("i1", 128.0, 0.0), "u8": ("u1", 128.0, 128.0),
    "i16": ("i2", 32768.0, 0.0), "u16": ("u2", 32768.0, 32768.0),
    "f32": ("f4", 1.0, 0.0), "f64": ("f8", 1.0, 0.0),
}
# baudline-style aliases: <bits>t = two's complement, <bits>o = offset binary
_ALIAS = {"8t": "i8", "8o": "u8", "16t": "i16", "16o": "u16", "32f": "f32", "64f": "f64"}
_TOK = {v: k for k, v in _ALIAS.items()}     # 저장할 땐 baudline 표기로 되돌린다
_FMT = {"cplx": "iq", "real": "real", "iq": "iq"}   # iq = 옛 밑줄 형식의 이름
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
    """파일명이 읽기 필수 정보를 실어 나른다 (정답은 사이드카에만 있고 절대 읽지 않는다).

        새 형식  <이름>.cplx|real.<샘플레이트 정수>.<16t>[.be][.bitrev].pcm
        옛 형식  <이름>_fs<샘플레이트>_iq|real_<i16>[_be][_bitrev].iq

    샘플레이트는 새 형식에서 접두사 없이 온다 -- 그래서 '숫자처럼 생긴 토막'이 아니라
    cplx|real **바로 다음 자리**로 찾는다. 이름 자체가 숫자여도(20260801.cplx.…) 안 헷갈린다.
    확장자를 떼지 않고 통째로 쪼갠다: 새 형식은 마지막 토막(.pcm)이 포맷을 뜻하지 않고,
    옛 형식은 확장자(.iq)가 이미 다른 토막과 같은 말을 하므로 어느 쪽도 손해가 없다."""
    base = os.path.basename(name).lower()
    toks = re.split(r"[._]", base)
    # 진짜 포맷 토막 = "숫자 + 아는 제원 토막"을 데리고 다니는 **마지막** 후보. 세 오독을 막는다:
    #  · 라벨의 real/cplx/iq(rec_iq_20260801.cplx.…)가 fs/fmt 를 강탈 -- 마지막 후보 규칙이 막고,
    #  · 점 낀 샘플레이트(cap.cplx.2.4e6.… -> fs=2 Hz)와 라벨 숫자를 fs 로 발명(…_iq_20260728.raw)
    #    -- 숫자 뒤가 제원 토막(16t/be/…/pcm)이어야 한다는 관문이 막는다. 관문에 걸리면 fs 없음으로
    #    크게 죽는다: 조용한 오답 대신 시끄러운 실패가 이 저장소의 원칙이다.
    cands = [k for k, t in enumerate(toks) if t in _FMT]
    hits = [k for k in cands if k + 2 < len(toks) and toks[k + 1].isdigit()
            and (toks[k + 2] in _DTYPES or toks[k + 2] in _ALIAS
                 or toks[k + 2] in ("be", "bitrev", "pcm") or toks[k + 2].startswith("rf"))]
    i = hits[-1] if hits else (cands[0] if cands else None)
    m = Meta()
    if i is not None:                   # 'is not None' 이어야 한다 -- 0 번 토막(cplx.…)도 유효하다
        m.fmt = _FMT[toks[i]]
        if i in hits:
            m.fs = float(toks[i + 1])
    if m.fs is None and (mt := _FS_TOK.search(base)):
        m.fs = float(mt.group(1))
    if rt := _RF_TOK.search(base):
        m.rf_center = float(rt.group(1))
    tail = toks[i + 1:] if i is not None else toks   # 제원은 포맷 토막 **뒤**에만 산다 --
    m.dtype = next((_ALIAS.get(t, t) for t in tail   # 라벨의 u8/f32/be 가 제원을 오염 못 하게
                    if t in _DTYPES or t in _ALIAS), "i16")
    # 엔디안: 파일명 토큰(.be/.le)이 최우선, 없으면 SIGNUS_ENDIAN, 그것도 없으면 le.
    # 실장비 녹음기는 BE 라 장비에서는 export SIGNUS_ENDIAN=be 한 줄로 전 파일이 읽힌다
    # (2026-08-21: BE 캡처를 LE 로 읽어 균등분포 잡음으로 보이던 사고 뒤에 넣었다).
    env = os.environ.get("SIGNUS_ENDIAN", "le").lower()
    if env not in ("le", "be"):
        raise ValueError(f"SIGNUS_ENDIAN 은 le 또는 be 여야 합니다: {env!r}")
    m.endian = "be" if "be" in tail else ("le" if "le" in tail else env)
    m.bitrev = "bitrev" in tail
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


def make_name(label: str, m: Meta) -> str:
    """저장은 새 형식만: <이름>.cplx|real.<샘플레이트>.<16t>[.be][.bitrev].pcm"""
    extra = (".be" if m.endian == "be" else "") + (".bitrev" if m.bitrev else "")
    kind = "real" if m.fmt == "real" else "cplx"
    return f"{label}.{kind}.{m.fs:.0f}.{_TOK.get(m.dtype, m.dtype)}{extra}.pcm"


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
