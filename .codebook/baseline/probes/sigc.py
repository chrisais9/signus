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


if len(sys.argv) < 2:
    raise SystemExit("사용법: python3 sigc.py kat | python3 sigc.py <캡처파일> [단계 1-6]")
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
cx = np.iscomplexobj(x0)
raw = np.abs(np.asarray([x0.real, x0.imag])).max(0) if cx else np.abs(x0)
cp = float((raw > 0.98 * np.percentile(raw, 99.9)).mean())   # 만점(±1)이 아니라 캡처 자신의
#   99.9퍼센타일 기준: float 캡처는 만점 정규화가 안 돼 ±1 기준이면 통짜 오경보다(실측 cp972)
x = dsp.analytic(x0)
n, pw = x.size, np.abs(x) ** 2
rms = float(np.sqrt(pw.mean()) + 1e-30)
dc = abs(complex(x.mean())) / rms          # dc 는 블록 전에 재고, 그 뒤는 pipeline 과 똑같이
x = x - x.mean()                           # DC 블록한 신호로 잰다 — 안 그러면 dc 큰 캡처에서
pw = np.abs(x) ** 2                        # 프로브와 analyze 가 정반대 진단을 낸다
# 러닝합 잔차 때문에 정확히 0 인 구간(스켈치/뮤트)에서 음수가 나와 log10 이 NaN 이 되고,
# otsu 의 histogram 이 장비에서 생 트레이스백으로 죽었다 — 0 으로 눌러 막는다.
lp = np.log10(np.maximum(uniform_filter1d(pw, max(64, n // 1000)), 0) + 1e-20)
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
dlo, dhi = 0.25, 0.45              # find_bursts 의 두 문턱 여유. 상수는 한 곳에만 쓰고
lov, hiv = base + dlo, base + dhi  # b 줄에 이 숫자를 그대로 되찍는다 — 오타면 그게 드러난다
above, hi_m = sc >= lov, sc >= hiv
rr = [(s, e) for s, e in runs_of(above) if hi_m[s:e].any()]
spans = [(max(0, s * hop) + (hop if e - s >= 6 else 0),               # 6열 이상은 양끝 한 hop
          min(n, (e - 1) * hop + nper) - (hop if e - s >= 6 else 0))  # 씩 자른다 — dsp 의
         for s, e in rr]                                              # 스파이크 게이트와 같은
segs = [pw[s:e] for s, e in spans]                                    # 구간에서 재려고
sk = sum(float(sg.max() / (sg.mean() + 1e-30)) > 60.0 for sg in segs)
gp_ = np.array([rr[i + 1][0] - rr[i][1] for i in range(len(rr) - 1)])
sb = round(10 * float(np.median([sc[s:e].max() for s, e in rr]) - base)) if rr else 0
g = float(np.median(fl if cx else fl[:nb // 2]) + 1e-30)   # real 은 해석신호라 음수 반쪽이
#   비어 있다: 전 빈 중앙값을 쓰면 바닥이 0 으로 내려가 밴드 절반이 '점유'로 읽힌다(실측)
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
      f" cx{int(cx)}")
lb = (f"sigc b c{nc} g{nb} b{round(100 * base)} cn{round(100 * cn)}"
      f" t{round(100 * thr)} m{round(100 * float(sc.max()))}"
      f" dlo{round(100 * dlo)} dhi{round(100 * dhi)}")   # 뺄셈으로 되계산하면 안 된다 --
#   (base+0.45)-base 가 44.99999999999999 라, int 로 잘못 옮기면 44 로 찍혔다 (2026-08-14 실측)
lc = (f"sigc c r{len(rr)} dn{round(float(np.median([e - s for s, e in rr]))) if rr else 0}"
      f" gp{round(float(np.median(gp_))) if gp_.size else 0} sb{sb}"
      f" av{round(100 * float(above.mean()))} ah{round(100 * float(hi_m.mean()))}"
      f" sk{sk}")
ld = (f"sigc d kb{kb} kc{kcont} w{wd} p{pk - nb // 2} pd{round(100 * float(du[pk]))}"
      f" ps{round(10 * np.log10(p95[pk] + 1e-30))} q{qo} qd{qd} qs{qs}")
if step == 0:                      # 요약 4줄 — 이것만 받아쳐서 회신한다
    for s in (la, lb, lc, ld):
        print(f"{s} #{check_code(s)}")
    sys.exit()

# ─── 여기부터는 그림 단계(1~6) 전용이다. 요약 4줄만 쓸 거면 여기서 멈춰도 된다 ───

try:
    import matplotlib.pyplot as plt
    plt = None if plt.get_backend().lower() == "agg" else plt   # 무화면(ssh)이면 ASCII 로
except ImportError:
    plt = None


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
    # 열마다 최소·최대를 같이 그린다: max 만 그리면 버스트 사이의 골이 지워져 계단이 통짜
    # 블록으로 보이고, 관측자가 "평평하다(=베토)" 라고 정반대로 회신하게 된다.
    kk = max(1, y.size // 100)
    y2 = y[:y.size // kk * kk].reshape(-1, kk)
    ylo, yhi = y2.min(1), y2.max(1)
    bot = float(min(ylo.min(), *(v for _, v in marks)))
    st = (float(max(yhi.max(), *(v for _, v in marks))) - bot) / 18 or 1.0
    rl, rh = np.round((ylo - bot) / st).astype(int), np.round((yhi - bot) / st).astype(int)
    rm = {}
    for la, v in marks:
        rm.setdefault(round((v - bot) / st), []).append(f"{la} {v:.2f}")
    for r in range(18, -1, -1):
        bg = "-" if r in rm else " "          # 문턱은 가로 전체 선으로 (겹치면 라벨 합침)
        print("".join("#" if rl[i] >= r else ("+" if rh[i] >= r else bg)
                      for i in range(rl.size)) + "|" + " / ".join(rm.get(r, [])))
    print("#=이 구간 내내 그 위 · +=봉우리만 그 위 · -=문턱선")


def image(img):
    if plt:
        plt.imshow(img, aspect="auto", origin="lower", interpolation="nearest")
        return plt.show()
    sm = pool(pool(img, 100).T, 32).T
    hi = sm.max() + 1e-30
    lo = max(float(np.percentile(sm, 5)), hi - 4)   # 표시폭 4 decades 고정 — real 캡처는 빈
    #   음수 반쪽이 1e-16 이라 5퍼센타일을 쓰면 눈금이 14 decades 로 늘어나 통짜 벽이 된다
    for row in np.clip((sm - lo) / (hi - lo) * 9.99, 0, 9).astype(int)[::-1]:
        print("".join(" .:-=+*#%@"[i] for i in row))


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
    print("빈별 바닥 대비 비율 — find_bursts 가 보는 그림. 세기가 일정한 연속 방사체는 여기서"
          " 사라지지만, 세기가 흔들리면 가짜 버스트로 남는다 (d 줄 pd 로 확인)")
elif step == 5:
    curve(sc, [("base", base), ("hi", hiv), ("c_noise", cn)])
    print(f"열 점수 — hi 위 봉우리가 버스트 후보. 런 {len(rr)}개, base {base:.2f}, cn {cn:.2f}")
elif step == 6:
    for (s, e), (a0, a1), sg in zip(rr, spans, segs, strict=True):
        print(f"열 {s}-{e}  샘플 {a0}-{a1}  높이 {10 * (float(sc[s:e].max()) - base):.0f}dB"
              f"  피크/평균 {float(sg.max() / (sg.mean() + 1e-30)):.0f} (60 초과=스파이크 탈락)")
    print(f"위는 병합/가드 전 원시 후보 {len(rr)}개 — 아래가 실제로 분석에 쓰이는 답이다:")
    fb = dsp.find_bursts(x, fs)
    print(f"find_bursts → {fb[:6]}{'...' if len(fb) > 6 else ''}"
          + ("   ※ 통짜 = 버스트 미검출" if fb == [(0, n)] else f"   버스트 {len(fb)}개"))
