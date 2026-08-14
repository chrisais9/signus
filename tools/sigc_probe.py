"""sigc.py — 실신호 특징 4줄 + find_bursts 단계 관찰 프로브 (장비 필사용, signus/ 옆에 저장).
    python3 sigc.py kat              # 자기검증 — 표지의 기대 4줄과 비교
    python3 sigc.py <캡처> [1-6]      # 요약 4줄(받아쳐 회신) / 단계 관찰(그림 또는 ASCII)
"""
import sys

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import get_window, stft

from signus import dsp, sigio
from signus.cli import check_code

try:
    import matplotlib.pyplot as plt
    plt = None if plt.get_backend().lower() == "agg" else plt   # 무화면(ssh)이면 ASCII 로
except ImportError:
    plt = None


def otsu(v, bins=128):
    h, ed = np.histogram(v, bins=bins)
    c = (ed[:-1] + ed[1:]) / 2
    w = np.cumsum(h).astype(float)
    m = np.cumsum(h * c)
    mb = m / np.where(w > 0, w, 1)
    mf = (m[-1] - m) / np.where(w[-1] - w > 0, w[-1] - w, 1)
    return float(c[int(np.argmax(w * (w[-1] - w) * (mb - mf) ** 2))])


def runs_of(mask):
    d = np.diff(mask.astype(np.int8))
    s = list(np.where(d == 1)[0] + 1)
    e = list(np.where(d == -1)[0] + 1)
    if mask[0]:
        s.insert(0, 0)
    if mask[-1]:
        e.append(mask.size)
    return list(zip(s, e, strict=True))


def pool(a, m):
    a = np.asarray(a, float)
    k = max(1, a.shape[-1] // m)
    return a[..., :(a.shape[-1] // k) * k].reshape(*a.shape[:-1], -1, k).max(-1)


def curve(y, marks):
    if plt:
        plt.plot(y)
        for la, v in marks:
            plt.axhline(v, ls="--", label=f"{la} {v:.2f}")
        plt.legend()
        return plt.show()
    y2 = pool(y, 100)
    lo = float(min(y2.min(), *(v for _, v in marks)))
    st = (float(max(y2.max(), *(v for _, v in marks))) - lo) / 18 or 1.0
    ry = np.round((y2 - lo) / st).astype(int)
    rm = {round((v - lo) / st): f"< {la} {v:.2f}" for la, v in marks}
    for r in range(18, -1, -1):
        print("".join("#" if ry[i] >= r else " " for i in range(y2.size)) + f"|{rm.get(r, '')}")


def image(img):
    if plt:
        plt.imshow(img, aspect="auto", origin="lower", interpolation="nearest")
        return plt.show()
    sm = pool(pool(img, 100).T, 32).T
    lo, hi = np.percentile(sm, 5), sm.max() + 1e-30
    for row in np.clip((sm - lo) / (hi - lo) * 9.99, 0, 9).astype(int)[::-1]:
        print("".join(" .:-=+*#%@"[i] for i in row))


arg, step = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0
if arg == "kat":
    fs, t = 1e6, np.arange(66000.0)     # 66000: 마지막 버스트가 파일 끝까지 이어진다
    rng = np.random.default_rng(7)      # 시드 고정 스트림 — numpy 가 재현을 보장한다
    x0 = 0.01 * (rng.standard_normal(66000) + 1j * rng.standard_normal(66000))
    x0 += 0.2 * np.exp(2j * np.pi * 0.29 * t)
    x0 += 0.5 * np.exp(-2j * np.pi * 0.11 * t) * ((t // 6000).astype(int) % 2 == 0)
    x0 += 0.0028 * np.exp(2j * np.pi * 0.41 * t)    # 문턱 경계에 걸친 성분들: 필사 오타가
    x0[31000:35000] += 0.0088 * np.exp(2j * np.pi * 0.35 * t[:4000])    # 상수 하나만
    x0[42400:47600] += 0.0075 * np.exp(2j * np.pi * -0.37 * t[:5200])   # 스쳐도 아래
    x0[200:260] = 0.95                              # 요약 4줄 어딘가가 바뀌게 만든다
    x0[21000] += 8.0
    x0[15000] += 2.0
else:
    x0, meta = sigio.read(arg)
    fs = meta.fs
raw = np.abs(np.asarray([x0.real, x0.imag])).max(0) if np.iscomplexobj(x0) else np.abs(x0)
cp = float((raw > 0.98).mean())
x = dsp.analytic(x0)
n, pw = x.size, np.abs(x) ** 2
rms = float(np.sqrt(pw.mean()) + 1e-30)
dc = abs(complex(x.mean())) / rms
lp = np.log10(uniform_filter1d(pw, max(64, n // 1000)) + 1e-20)
hot = lp >= (et := otsu(lp))
ev = float(lp[hot].mean() - lp[~hot].mean()) if 0 < hot.sum() < n else 0.0
ed = float(hot.mean())
sp = 10 * np.log10(pw.max() / (pw.mean() + 1e-30) + 1e-30)
nper = int(min(128, max(64, 1 << int(np.log2(max(n // 2, 64))))))
hop = nper // 2
_, _, z = stft(x, fs=fs, window=get_window("blackmanharris", nper), nperseg=nper,
               noverlap=nper - hop, return_onesided=False, boundary=None, padded=False)
P = np.abs(z) ** 2
nb, nc = P.shape
fl = np.percentile(P, 10, axis=1)
r = P / (fl[:, None] + 1e-30)
k = max(3, nb // 16)
sc = uniform_filter1d(np.log10(np.sort(r, axis=0)[-k:].mean(axis=0) + 1e-30), 3)
thr = otsu(sc, bins=64)
quiet = sc[sc < thr]
base = float(np.median(quiet)) if quiet.size else 9.99
cn = float(np.log10((np.log(nb / k) + 1) / 0.105))
lov, hiv = base + 0.25, base + 0.45   # find_bursts 의 두 문턱. 각 상수는 한 곳에만 쓰고,
above, hi_m = sc >= lov, sc >= hiv    # b 줄에 lo/hi 로 되찍는다 — 필사 오타면 그 숫자가 바뀐다
rr = [(s, e) for s, e in runs_of(above) if hi_m[s:e].any()]
segs = [pw[max(0, s * hop):min(n, (e - 1) * hop + nper)] for s, e in rr]
sk = sum(float(sg.max() / (sg.mean() + 1e-30)) > 60.0 for sg in segs)
gp_ = np.array([rr[i + 1][0] - rr[i][1] for i in range(len(rr) - 1)])
sb = round(10 * float(np.median([sc[s:e].max() for s, e in rr]) - base)) if rr else 0
g = float(np.median(fl) + 1e-30)
Ps, fls = np.fft.fftshift(P, 0), np.fft.fftshift(fl)
du = (Ps > 40 * g).mean(1)
occ = du > 0.1
kb, kcont = int(occ.sum()), int((fls > 5 * g).sum())
grp = runs_of(occ) if occ.any() else []
wd = max((e - s for s, e in grp), default=0)
p95 = np.percentile(Ps, 95, axis=1) / g
pk = int(np.argmax(p95))
pgrp = next(((s, e) for s, e in grp if s <= pk < e), (pk, pk + 1))
m2 = np.where(occ, p95, 0.0)
m2[pgrp[0]:pgrp[1]] = 0.0
q2 = int(np.argmax(m2))
qo, qd, qs = (q2 - nb // 2, round(100 * float(du[q2])),
              round(10 * np.log10(m2[q2] + 1e-30))) if m2[q2] > 0 else (999, 0, 0)
la = (f"sigc a n{n} f{fs:.0f} ev{round(100 * ev)} ed{round(100 * ed)}"
      f" sp{round(float(sp))} dc{round(100 * dc)} cp{round(1000 * cp)}"
      f" iq{int(np.iscomplexobj(x0))}")
lb = (f"sigc b c{nc} g{nb} b{round(100 * base)} cn{round(100 * cn)}"
      f" t{round(100 * thr)} m{round(100 * float(sc.max()))}"
      f" lo{round(100 * (lov - base))} hi{round(100 * (hiv - base))}")
lc = (f"sigc c r{len(rr)} dn{round(float(np.median([e - s for s, e in rr]))) if rr else 0}"
      f" gp{round(float(np.median(gp_))) if gp_.size else 0} sb{sb}"
      f" av{round(100 * float(above.mean()))} ah{round(100 * float(hi_m.mean()))}"
      f" sk{sk}")
ld = (f"sigc d kb{kb} kc{kcont} w{wd} p{pk - nb // 2} pd{round(100 * float(du[pk]))}"
      f" ps{round(10 * np.log10(p95[pk] + 1e-30))} q{qo} qd{qd} qs{qs}")
if step == 1:
    print(f"n {n}  fs {fs:.0f}  길이 {n / fs:.3f}s  형식 {'iq' if np.iscomplexobj(x0) else 'real'}")
    print(f"rms {rms:.4f}  dc {100 * dc:.1f}%  클리핑 {1000 * cp:.1f}‰  피크비 {float(sp):.0f}dB")
elif step == 2:
    curve(lp, [("otsu", et)])
    print(f"포락선 분리도 {ev:.3f} decades — 0.12 미만이면 find_bursts 는 통짜 [(0,n)] (베토)")
    print(f"포락선 듀티 {100 * ed:.0f}%  (버스트가 계단으로 보여야 정상)")
elif step == 3:
    image(np.log10(Ps + 1e-20))
    print("워터폴 절대전력 (아래=-fs/2, 위=+fs/2) — 가로로 끊기지 않는 띠 = 연속 방사체")
elif step == 4:
    image(np.log10(np.fft.fftshift(r, 0) + 1e-30))
    print("빈별 바닥 대비 비율 — find_bursts 가 보는 그림. 연속 방사체는 여기서 사라진다")
elif step == 5:
    curve(sc, [("base", base), ("hi", hiv), ("c_noise", cn)])
    print(f"열 점수 — hi 위 봉우리가 버스트 후보. 런 {len(rr)}개, base {base:.2f}, cn {cn:.2f}")
elif step == 6:
    for (s, e), sg in zip(rr, segs, strict=True):
        print(f"열 {s}-{e}  샘플 {s * hop}-{min(n, (e - 1) * hop + nper)}"
              f"  높이 {10 * (float(sc[s:e].max()) - base):.0f}dB"
              f"  피크/평균 {float(sg.max() / (sg.mean() + 1e-30)):.0f} (60 초과=스파이크 탈락)")
    print(f"런 {len(rr)}개 — 병합/가드 전 원시 후보")
else:
    for s in (la, lb, lc, ld):
        print(f"{s} #{check_code(s)}")
