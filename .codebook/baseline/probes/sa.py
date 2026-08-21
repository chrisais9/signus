"""sa.py — 종합 관찰 프로브 (장비 필사용, signus/ 옆에 저장). 한 번에:
디버그 출력(포락선·버스트 표) + <캡처>.sa.png (스펙트로그램 + 검출 띠 + 버스트 번호 +
상위 버스트별 [PSD | x^2 | x^4 | x^8 | AM검파] 판독 행 — 바늘이 처음 서는 판이 변조).
    python3 sa.py kat            # 자기검증: 표지 기대 줄과 비교, sa-kat.png 생성
    python3 sa.py <캡처파일> [K]   # 판독할 버스트 수 K (기본 4, 에너지 상위부터)
x^m 판은 접힌 축(0~fs/2, 음수 주파수를 겹쳐 최대값) — 바늘 위치가 m·fc (mod fs) 다.
"""
import struct
import sys
import zlib

import numpy as np
from scipy.signal import stft, welch

from signus import dsp, sigio
from signus.chirp import is_chirp, sweeps_band
from signus.cli import check_code
from signus.fsk import fsk_gate

GLYPH = {"0": (14, 17, 19, 21, 25, 17, 14), "1": (4, 12, 4, 4, 4, 4, 14),
         "2": (14, 17, 1, 2, 4, 8, 31), "3": (31, 2, 4, 2, 1, 17, 14),
         "4": (2, 6, 10, 18, 31, 2, 2), "5": (31, 16, 30, 1, 1, 17, 14),
         "6": (6, 8, 16, 30, 17, 17, 14), "7": (31, 1, 2, 4, 8, 8, 8),
         "8": (14, 17, 17, 14, 17, 17, 14), "9": (14, 17, 17, 15, 1, 2, 12)}


def draw_text(img, row, col, text, s=2):
    for ch in text:
        for r, bits in enumerate(GLYPH[ch]):
            for c in range(5):
                if bits >> (4 - c) & 1:
                    img[row + r * s:row + r * s + s, col + c * s:col + c * s + s] = 1.0
        col += 6 * s


def png_gray(img, path):
    img = np.clip(img * 235, 0, 255).astype(np.uint8)
    h, w = img.shape

    def chunk(tag, data):
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c))

    raw = b"".join(b"\x00" + img[r].tobytes() for r in range(h))
    with open(path, "wb") as fh:
        fh.write(b"\x89PNG\r\n\x1a\n"
                 + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
                 + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def panel(fvals, dbvals, wp=500, hp=120, grid=()):
    """스펙트럼 곡선 한 판 — 바늘 보존을 위해 표시 칸마다 최대값 풀링."""
    cols = np.clip((fvals / fvals.max() * (wp - 1)).astype(int), 0, wp - 1)
    cur = np.full(wp, -1e9)
    np.maximum.at(cur, cols, dbvals)
    bad = cur < -1e8
    cur[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), cur[~bad])
    g_lo, g_hi = np.percentile(cur, 5) - 2, cur.max() + 2
    img = np.zeros((hp, wp))
    for gf in grid:
        img[::6, int(gf / fvals.max() * (wp - 1))] = 0.35
    rows = ((hp - 1) * (1 - np.clip((cur - g_lo) / (g_hi - g_lo), 0, 1))).astype(int)
    for cx in range(wp):
        r0, r1 = (rows[cx], rows[cx - 1]) if cx else (rows[0], rows[0])
        img[min(r0, r1):max(r0, r1) + 1, cx] = 1.0
    return img


def peak(fv, dbv, fmin=20.0):
    """판 하나의 최대 바늘: (주파수 Hz, 중앙값 대비 돌출 dB) -- 숫자로 받아칠 수 있게."""
    ok = fv >= fmin
    i = int(np.argmax(dbv[ok]))
    return round(float(fv[ok][i])), round(float(dbv[ok][i] - np.median(dbv[ok])))


def verdict(z, fs, d2, d4, d8, dam):
    """바늘 돌출로 버스트의 변조 부류를 판정 -- 문턱은 합성 배터리로 캘리브레이션
    (bpsk d2 54 / qpsk d4 38 / 16qam d4 30 / 8psk d8 29 / fsk am 17, 광대역 잡음꼴 무바늘)."""
    if is_chirp(z, fs) and sweeps_band(z, fs):
        return "chirp"                          # 처프/LoRa 류 -- 성상도 복조 대상 아님
    if fsk_gate(z, fs):
        return "fsk"
    if d2 >= 35 and d2 >= d4 + 5:
        return "bpsk"
    if d4 >= 28:
        return "qpsk"                           # qpsk 또는 사각 QAM (4차에서 무너짐)
    if d8 >= 24:
        return "8psk"
    return "lin" if dam >= 25 else "flat"       # lin=선형(차수불명) / flat=바늘 없음(잡음꼴)


def burst_row(z, fs, label):
    """한 버스트의 판독 행: [PSD | x^2 | x^4 | x^8 | AM]. x^m 은 음수 주파수를 접는다."""
    grid = tuple(fs / 2 * k / 5 for k in (1, 2, 3, 4))
    fr, praw = welch(z.real, fs=fs, nperseg=min(2048, z.size))
    ps = [panel(fr, 10 * np.log10(praw + 1e-18), grid=grid)]
    nfft, nums = 1 << 15, []
    f = np.fft.fftfreq(nfft, 1 / fs)
    half = nfft // 2
    for m in (2, 4, 8):
        mag = np.abs(np.fft.fft((z ** m) * np.hanning(z.size), nfft))
        fold = np.maximum(mag[:half], np.r_[mag[0], mag[1:half][::-1] * 0
                                            + mag[half + 1:][::-1]])
        ps.append(panel(f[:half], 20 * np.log10(fold + 1e-12), grid=grid))
        pf, pd = peak(f[:half], 20 * np.log10(fold + 1e-12))
        nums.append(f"p{m} f{pf} d{pd}")
    u = np.abs(z) ** 2
    u = u - u.mean()
    su = np.abs(np.fft.rfft(u * np.hanning(u.size), nfft))
    ps.append(panel(np.fft.rfftfreq(nfft, 1 / fs), 20 * np.log10(su + 1e-12), grid=grid))
    af, ad = peak(np.fft.rfftfreq(nfft, 1 / fs), 20 * np.log10(su + 1e-12))
    nums.append(f"am f{af} d{ad}")
    lab = verdict(z, fs, int(nums[0].split("d")[1]), int(nums[1].split("d")[1]),
                  int(nums[2].split("d")[1]), ad)
    nums.insert(0, lab)
    div = np.full((120, 4), 0.5)
    row = np.hstack(sum(([p, div] for p in ps[:-1]), []) + [ps[-1]])
    tag = np.zeros((120, 26))
    draw_text(tag, 4, 4, label)
    return np.hstack([tag, row]), " ".join(nums)


if len(sys.argv) < 2:
    raise SystemExit("사용법: python3 sa.py kat | python3 sa.py <캡처파일> [버스트수 K]")
arg = sys.argv[1]
kmax = int(sys.argv[2]) if len(sys.argv) > 2 else 4
if arg == "kat":
    fs, n = 10000.0, 80000
    rng = np.random.default_rng(21)      # 시드 고정 -- numpy 가 재현을 보장한다
    x0 = 0.18 * rng.standard_normal(n)
    t = np.arange(15000) / fs
    for s0, fc, ph in ((12000, 550.0, 2), (50000, 550.0, 4)):   # 앞=BPSK, 뒤=QPSK
        up = np.zeros(15000, complex)
        up[::34] = np.exp(2j * np.pi * rng.integers(0, ph, 442) / ph)
        sym = np.convolve(up, np.hanning(68), "same")           # 간이 펄스 성형
        sym /= np.sqrt(np.mean(np.abs(sym) ** 2))
        x0[s0:s0 + 15000] += (sym * np.exp(2j * np.pi * fc * t)).real * np.sqrt(2)
    name = "sa-kat"
else:
    x0, meta = sigio.read(arg)
    fs = meta.fs
    name = arg + ".sa"
xa = dsp.analytic(x0)
xa = xa - xa.mean()
n = xa.size
fb = dsp.find_bursts(xa, fs)
full = fb == [(0, n)]
pw = np.abs(xa) ** 2
lp = np.log10(np.maximum(np.convolve(pw, np.ones(64) / 64, "same"), 0) + 1e-20)
et = np.percentile(lp, 50)
ev = float(lp[lp >= et].mean() - lp[lp < et].mean())
print(f"n {n}  fs {fs:.0f}  길이 {n / fs:.3f}s  포락선 분리(중앙 기준) {ev:.2f} decades")
print(f"find_bursts → {'통짜 [(0,n)] = 미검출' if full else str(len(fb)) + '개'}")
order = sorted(range(len(fb)), key=lambda i: -float(pw[fb[i][0]:fb[i][1]].sum()))
pick = sorted(order[:kmax]) if not full else []
for i, (s0, e0) in enumerate(fb if not full else []):
    star = " ★판독" if i in pick else ""
    print(f"  {i + 1:>2}  샘플 [{s0:>8}, {e0:>8})  {1000 * (e0 - s0) / fs:8.1f} ms"
          f"  전력 {10 * np.log10(pw[s0:e0].mean() / (pw.mean() + 1e-30) + 1e-30):+5.1f} dB{star}")

x_disp = x0.real if np.iscomplexobj(x0) else x0
_, _, z = stft(x_disp, fs=fs, window="hamming", nperseg=256, noverlap=128,
               return_onesided=True, boundary=None, padded=False)
db = 10 * np.log10(np.abs(z) ** 2 + 1e-12)
lo = float(np.percentile(db, 25))
rg = float(max(10.0, min(35.0, np.percentile(db, 99.5) - lo)))
spec = np.clip((db - lo) / rg, 0, 1)[::-1]
kp = max(1, spec.shape[1] // 4000)
spec = spec[:, :spec.shape[1] // kp * kp].reshape(spec.shape[0], -1, kp).max(2)
pw_row = 26 + 5 * 500 + 4 * 4                   # 판독 행 폭 -- 스트립을 여기에 맞춰 늘린다
fx = max(1, pw_row // spec.shape[1])
spec = np.repeat(spec, fx, axis=1)
ncol = spec.shape[1]
mark = np.zeros((18, ncol))
lane = np.zeros(ncol)
for i, (s0, e0) in enumerate(fb if not full else []):
    c0 = (s0 // 128) // kp * fx
    c1 = min(ncol - 1, (e0 // 128) // kp * fx)
    lane[c0:c1 + 1] = 1.0
    draw_text(mark, 2, min(c0 + 2, ncol - 14 * len(str(i + 1))), str(i + 1))
rows = [mark, spec, np.full((2, ncol), 0.35), np.tile(lane, (10, 1)),
        np.full((6, ncol), 0.0)]
width = max(ncol, pw_row)
rows = [np.pad(r, ((0, 0), (0, width - r.shape[1]))) for r in rows]
for i in pick:
    r, num = burst_row(xa[fb[i][0]:fb[i][1]], fs, str(i + 1))
    ln = f"sa b{i + 1} {num}"              # 바늘 주파수·돌출 -- 받아칠 판독 숫자줄
    print(f"{ln} #{check_code(ln)}")
    rows += [np.pad(r, ((0, 0), (0, width - r.shape[1]))), np.full((6, width), 0.0)]
png_gray(np.vstack(rows), name + ".png")
line = (f"sa {'kat' if arg == 'kat' else 'cap'} n{n} f{fs:.0f} s{n / fs:.1f}"
        f" fb{0 if full else len(fb)} ev{round(100 * ev)}")
print(f"{line} #{check_code(line)}")
print(f"{name}.png 저장 — 위: 스펙트로그램+검출 띠+버스트 번호, 아래: ★버스트별"
      " [PSD|x²|x⁴|x⁸|AM] (x² 바늘=BPSK, x⁴=QPSK, x⁸=8PSK, AM 봉우리=심볼레이트)")
