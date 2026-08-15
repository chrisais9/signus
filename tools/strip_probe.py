"""strip.py — 캡처를 GUI 와 같은 방식(FFT 256 · Hamming · 50% 겹침 · 0~fs/2)의 흑백
스펙트로그램 띠 PNG 로 그린다 (장비 필사용, signus/ 옆에 저장).
    python3 strip.py kat          # 자기검증: strip-kat.png 생성 + 출력 줄을 표지와 대조
    python3 strip.py <캡처파일>     # <캡처파일>.png 생성 — GUI 와 나란히 비교
"""
import struct
import sys
import zlib

import numpy as np
from scipy.signal import stft

from signus import sigio
from signus.cli import check_code


def png_gray(img, path):
    img = np.clip(img, 0, 255).astype(np.uint8)
    h, w = img.shape

    def chunk(tag, data):
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c))

    raw = b"".join(b"\x00" + img[r].tobytes() for r in range(h))
    with open(path, "wb") as fh:
        fh.write(b"\x89PNG\r\n\x1a\n"
                 + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
                 + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


if len(sys.argv) < 2:
    raise SystemExit("사용법: python3 strip.py kat | python3 strip.py <캡처파일>")
arg = sys.argv[1]
if arg == "kat":
    fs, n = 10000.0, 60000
    t = np.arange(n) / fs
    rng = np.random.default_rng(7)          # 시드 고정 -- numpy 가 재현을 보장한다
    key = ((np.arange(n) // 250) % 2 == 0) & ((np.arange(n) // 5000) % 2 == 0)
    x = np.sin(2 * np.pi * 2200 * t) * key + 0.12 * np.sin(2 * np.pi * 3600 * t)
    x = x + 0.18 * rng.standard_normal(n)
    name = "strip-kat"
else:
    x, meta = sigio.read(arg)
    fs = meta.fs
    x = x.real if np.iscomplexobj(x) else x     # 실수 파형 기준 -- GUI 와 같은 입장
    name = arg
_, _, z = stft(x, fs=fs, window="hamming", nperseg=256, noverlap=128,
               return_onesided=True, boundary=None, padded=False)
db = 10 * np.log10(np.abs(z) ** 2 + 1e-12)
lo = float(np.percentile(db, 25))               # 바닥 근처를 검정으로, +35dB 를 흰색으로
img = np.clip((db - lo) / 35.0, 0, 1)[::-1]     # 아래=0Hz, 위=fs/2 (GUI 의 Y MAX)
k = max(1, img.shape[1] // 4000)                # 폭 ~4000픽셀 제한: 열 최대값 풀링이라
img = img[:, :img.shape[1] // k * k].reshape(img.shape[0], -1, k).max(2)   # 버스트는 남는다
png_gray(img * 255, name + ".png")
line = (f"strip {'kat' if arg == 'kat' else 'cap'} n{x.size} f{fs:.0f}"
        f" s{x.size / fs:.1f} px{img.shape[1]}x{img.shape[0]} lo{round(lo)}")
print(f"{line} #{check_code(line)}")
print(f"{name}.png 저장 — 이미지 뷰어로 열어 GUI 와 나란히 비교. s 값(초)이 GUI 가 보여주는"
      " 파일 길이와 다르면 샘플 해석(비트수/채널)이 다른 것이다")
