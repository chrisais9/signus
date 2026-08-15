#!/usr/bin/env python3
"""받아친 sigc 4줄(관찰 프로브 출력)을 검증·해석하고, 특징에 맞는 합성 데이터셋을 만든다.

맥 쪽 전용. 발행은 격리망 장비의 sigc.py(관찰 프로브, 필사 PDF)가 한다. 검출 코드 계산은
signus/cli.py 의 check_code 를 그대로 import 한다 — brief.py 와 같은 이유다.

    cat 받아친줄.txt | tools/sigc.py check          # 줄 단위 오타 검증 + find_bursts 출구 해석
    cat 받아친줄.txt | tools/sigc.py gen --out DIR  # 특징 주변 변형 생성 + 즉석 find_bursts 채점
"""
from __future__ import annotations

import argparse
import math
import os
import re
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

try:
    import numpy as np

    from signus import dsp
    from signus.cli import check_code
    from signus.gen import GenParams, generate
    from signus.sigio import Meta, make_name, write
except ModuleNotFoundError:                 # 시스템 파이썬엔 numpy 가 없다 -> venv 로 다시 실행
    _py = _ROOT / ".venv/bin/python"
    if _py.exists() and sys.executable != str(_py):
        os.execv(str(_py), [str(_py), *sys.argv])
    raise


# 코드 알파벳 안에서 화면→종이→채팅 손 전달 때 서로 오독되는 쌍. brief.py 의 o/i/l 복원과
# 같은 원리인데, 이쪽은 양쪽 다 유효한 코드 글자라 배제할 수 없어 쌍으로 시험한다.
# (실측 2026-08-14: 장비 화면의 #sy67 이 채팅에서 #syb7 로 옮겨졌다 — 6↔b)
_CONFUSE = {"6": "b", "b": "6", "5": "s", "s": "5", "7": "1", "1": "7",
            "y": "v", "v": "y", "g": "q", "q": "g"}


def parse(text: str) -> tuple[dict[str, dict[str, int]], list[str], list[str]]:
    """받아친 블록 -> ({줄: {키: 값}}, 문제 목록, 참고 목록). 줄마다 검출 코드를 대조한다."""
    doc, bad, notes = {}, [], []
    for raw in text.strip().splitlines():
        ln = re.sub(r"\s+", " ", raw).strip()   # check_code 와 같은 잣대로 공백을 접는다 --
        if not ln.startswith("sigc"):           # 안 그러면 'sigc  a' 를 "코드 없음" 이라고
            continue                            # 엉뚱하게 지목한다 (손 필사에서 흔한 오타)
        m = re.fullmatch(r"(sigc ([a-d])\b.*?)\s*#([0-9a-z]{4})", ln)
        if not m:
            bad.append(f"검출 코드(#xxxx)가 없습니다: {ln}")
            continue
        body, tag, got = m.group(1).rstrip(), m.group(2), m.group(3)
        want = check_code(body)
        if want != got:
            # 코드 한 글자가 시각 혼동쌍으로 뒤집힌 것이면 본문은 정상이다 — 그 경우까지
            # "불일치" 로 세우면 멀쩡한 회신으로 수사를 멈추게 된다.
            fix = next((i for i, ch in enumerate(got)
                        if got[:i] + _CONFUSE.get(ch, ch) + got[i + 1:] == want), None)
            if fix is not None:
                notes.append(f"sigc {tag}: 받은 코드 #{got} 의 '{got[fix]}' 는 화면의"
                             f" '{_CONFUSE[got[fix]]}' 오독 — 본문 일치 ✓ (#{want})")
            else:
                near = sum(a != b for a, b in zip(want, got, strict=True)) == 1
                hint = "코드 네 글자 쪽 오타일 공산" if near else "숫자 자릿수부터 재확인"
                bad.append(f"sigc {tag} 불일치 ✗  받은 코드 #{got}, 계산 #{want} ({hint})")
                continue
        doc[tag] = {k: int(v) for k, v in re.findall(r"([a-z]+)(-?\d+)", body[7:])}
    return doc, bad, notes


def interpret(doc: dict[str, dict[str, int]]) -> None:
    """어느 출구가 [(0,n)] 을 냈는지 — find_bursts 의 분기 순서 그대로 짚는다."""
    a, b, c, d = doc["a"], doc["b"], doc["c"], doc["d"]
    fs, nb, hop = float(a["f"]), b["g"], b["g"] // 2
    khz = fs / nb / 1000
    # 문턱은 장비가 되찍어 준 값을 그대로 쓴다 — 45/25 를 여기 박아 두면 장비 필사가 틀렸을 때
    # 자기모순 없이 조용히 틀린 해석이 나온다.
    dlo, dhi = b.get("dlo", 25), b.get("dhi", 45)
    if (dlo, dhi) != (25, 45):
        print(f"⚠ 장비의 문턱 상수가 표준(dlo25 dhi45)과 다릅니다: dlo{dlo} dhi{dhi} —"
              " 프로브의 lov/hiv 줄 필사부터 확인하세요. 아래 해석은 받은 값 기준입니다.")
    print(f"\n캡처: n={a['n']} ({a['n'] / fs:.3f}s @ fs {fs:.0f})  포락선 분리 {a['ev'] / 100:.2f}"
          f"  듀티 {a['ed']}%  피크비 {a['sp']}dB  dc {a['dc']}%  클리핑 {a['cp']}‰")
    print(f"셀: 열 {b['c']}·빈 {nb}  base {b['b'] / 100:.2f} (c_noise {b['cn'] / 100:.2f})"
          f"  최대 점수 {b['m'] / 100:.2f}")
    gp_note = " (≤7열은 상한으로만 신뢰 — 분해능 한계)" if 0 < c["gp"] <= 7 else ""
    print(f"버스트 구조: 런 {c['r']}개  길이 {c['dn']}열→보정 {max(c['dn'] - 3, 1) * hop}샘플"
          f"({1000 * max(c['dn'] - 3, 1) * hop / fs:.1f}ms)  간격 {c['gp']}열"
          f"→보정 {(c['gp'] + 3) * hop}샘플{gp_note}  점수 {c['sb']}dB  스파이크런 {c['sk']}")
    q = "" if d["q"] == 999 else \
        f"  부 방사체 {d['q'] * khz:+.1f}kHz (듀티 {d['qd']}%, {d['qs']}dB)"
    if not a.get("cx", 1) and d["kb"] >= nb // 2 - 2:
        # real 은 해석신호라 음수 반쪽이 비어 있다. 바닥을 전 빈에서 잡으면 g 가 0 으로 내려가
        # 양수 반쪽 전체가 '점유' 로 읽힌다 — 프로브의 g 줄(fl if cx else fl[:nb//2]) 필사 의심.
        print(f"주의: real 캡처인데 점유빈 {d['kb']}개(밴드 절반) — 프로브의 g 줄 필사를"
              " 먼저 확인할 것. 그게 맞다면 수신기 잡음이 버스트와 함께 게이팅된 캡처다.")
    print(f"방사체: 주 {d['p'] * khz:+.1f}kHz (듀티 {d['pd']}%, {d['ps']}dB){q}"
          f"  점유빈 {d['kb']}·연속점유 {d['kc']}·최대폭 {d['w']}빈≈{d['w'] * khz:.1f}kHz\n")
    if a["ev"] < 12:
        print(f"→ E1 포락선 베토: 분리도 {a['ev'] / 100:.2f} < 0.12 — find_bursts 는 셀 경로에"
              " 들어가기 전에 [(0,n)] 을 낸다.")
        if a["ev"] >= 11:
            print("  단 ev 11~13 은 반올림 회색지대다 (원시 분리도가 0.12 양쪽에 걸친다):"
                  " 단계 2 로 원시 곡선을 확인할 것.")
        if d["kc"] > 0:
            note = ""
            if a["dc"] >= 10:
                note = (f" — 단 dc {a['dc']}% 라 DC 빈이 연속 방사체로 읽혔을 수 있다:"
                        " 단계 3 으로 그 띠가 정말 대역 안에 있는지 볼 것")
            print(f"  원인 후보: 연속 방사체가 빈 {d['kc']}개를 계속 점유해 광대역 포락선을"
                  f" 평평하게 만든다 (합성 테스트에 없던 특징){note}.")
        if a["ed"] >= 90:
            print(f"  원인 후보: 듀티 {a['ed']}% — 신호가 거의 항상 켜져 있다.")
    elif b["c"] < 4:
        print("→ E2 열 부족: 레코드가 너무 짧아 버스트/레코드 구분 불가 — [(0,n)].")
    elif b["b"] >= 900:
        print("→ E3 quiet 열 없음: 문턱 아래 열이 하나도 없다 — [(0,n)].")
    elif b["b"] > b["cn"] + 25:
        print(f"→ E4 스펙트럼 셔플 가드: base {b['b'] / 100:.2f} > c_noise+0.25 — 연속 신호가"
              " 스펙트럼만 바꾸는 패턴으로 읽혀 [(0,n)].")
    elif b["m"] < b["b"] + dhi:
        print(f"→ E5 hi 미달: 최대 점수 {b['m'] / 100:.2f} 가 base+{dhi / 100:.2f} 를 못 넘는다"
              " — 런 없음, [(0,n)]. (초협대역 희석 또는 저SNR)")
    elif c["r"] > 0 and c["sk"] > 0.4 * c["r"]:
        # dsp 의 가드는 개수가 아니라 살아남은 샘플 질량이다 (covered < 0.6*(covered+spiked)):
        # 런 길이가 고르면 sk/r > 0.4 와 같은 말이라, 큰 버스트 하나만 죽어도 폴백이 걸린다.
        print(f"→ E6 스파이크 커버리지: 런 {c['r']}개 중 {c['sk']}개가 임펄스로 탈락 — 살아남은"
              " 질량이 60% 아래라 폴백 [(0,n)].")
    elif c["r"] * max(c["dn"] - 3, 1) * hop < 0.1 * (a["ed"] / 100) * a["n"]:
        print("→ E7 포락선-스팬 복귀: 검출된 런의 샘플 합이 포락선 고전력 질량의 10% 미만 —"
              " dsp 는 셀 결과를 버리고 포락선의 고전력 구간을 돌려준다. 그 구간이 레코드"
              " 양끝에 닿으면 [(0,n)] 과 구별되지 않는다 (닿지 않으면 (15,n) 같은 값이 뜬다).")
    elif b["m"] < b["b"] + dhi + 7:
        print(f"→ 회색지대: 점수 여유 {(b['m'] - b['b']) / 100:.2f}"
              f" ({dhi / 100:.2f}~{(dhi + 7) / 100:.2f}) — 검출돼도 커버리지 폴백이 [(0,n)] 과"
              " 구별되지 않는 구간(실측 2026-08-13). 판정 유보, 단계 5 관찰을 요청할 것.")
    else:
        print("→ 이 특징이면 HEAD find_bursts 는 버스트를 잡아야 정상 — 즉 이 줄의 특징만으로는"
              " [(0,n)] 이 설명되지 않는다. 장비에서 `sigc.py <캡처> 6` 을 돌려 find_bursts 의"
              " 실제 출력과 원시 후보를 함께 받아올 것 (병합·1열드랍은 이 4줄에 안 담긴다).")


def _emitters(doc: dict[str, dict[str, int]]) -> list[dict]:
    """특징 -> 방사체 목록. 절대 dB(ps/qs)를 광대역 SNR 로 역산 (근사 — 그리드가 오차를 덮는다)."""
    b, c, d = doc["b"], doc["c"], doc["d"]
    nb, hop = b["g"], b["g"] // 2
    fs = float(doc["a"]["f"])
    # w 도 창 번짐만큼 넓게 읽힌다(실측: 참 4빈 -> 8~10). 안 빼면 재합성이 2~4배 넓어져
    # 정작 재현하려던 "초협대역 희석" 영역을 스스로 지운다.
    w = max(1, min(d["w"] - 4, nb // 2))
    # dn/gp 고정 편향 보정 (실측 2026-08-13): STFT 창 번짐+3열 평활로 dn 은 ~+3열 크게,
    # gp 는 ~-3열 작게 읽힌다. dn+gp 는 참 주기를 ~0.5열 안에서 보존한다.
    blen = max(c["dn"] - 3, 2) * hop
    bgap = (c["gp"] + 3) * hop if c["gp"] else blen

    def em(off_bins: int, duty: int, db: int) -> dict:
        snr = db + 10 * math.log10(w * 0.035 / nb)          # p95/바닥 -> 광대역 SNR 근사
        off = np.clip(off_bins * fs / nb, -0.4 * fs, 0.4 * fs)
        return {"off": float(off), "bw": w * fs / nb, "snr": snr,
                "cont": duty >= 70, "blen": blen, "bgap": bgap}

    ems = [em(d["p"], d["pd"], d["ps"])]
    if d["q"] != 999:
        ems.append(em(d["q"], d["qd"], d["qs"]))
    if not any(not e["cont"] for e in ems) and c["r"] > 0:
        # 버스트 구조는 보이는데 위치를 못 쟀다 -> 주 방사체 반대편에 배치 (위치 미상 표기)
        e = em(-d["p"] or nb // 4, 50, d["ps"] - 6)
        e["cont"] = False
        ems.append(e)
        print("주의: 버스트 방사체 위치 미상 — 주 방사체 반대편에 배치했습니다.")
    return ems


def _build(n: int, fs: float, ems: list[dict], seed: int, cx: bool = True) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)  # 잡음 전력 1
    t = np.arange(n)
    for i, e in enumerate(ems):
        baud = min(max(e["bw"] / 1.35, fs / 1000), 0.4 * fs)
        p = GenParams(mod="qpsk", n_symbols=int(n * baud / fs) + 64, fs=fs, baud=baud,
                      snr=90.0, fc=e["off"], seed=17 + i, dtype="f32")
        s, _ = generate(p)
        s = s[:n] / np.sqrt(np.mean(np.abs(s[:n]) ** 2)) * 10 ** (e["snr"] / 20)
        if not e["cont"]:
            s = s * ((t % (e["blen"] + e["bgap"])) < e["blen"])
        x = x + s
    # 회신이 real(cx0)이면 실수 패스밴드로 낸다 — iq 로 만들면 장비가 지나온 hilbert 경로를
    # 재현하지 못해, 정작 재현하려던 실패가 합성에서 사라진다.
    return x if cx else x.real * np.sqrt(2)


def cmd_gen(doc: dict[str, dict[str, int]], outdir: Path) -> None:
    a, c, d = doc["a"], doc["c"], doc["d"]
    n, fs, cx = a["n"], float(a["f"]), bool(a.get("cx", 1))
    if c["r"] == 0 and d["w"] == 0:
        print("이 회신에는 재현할 구조가 없습니다 (런 0개·점유폭 0빈 = 잡음만/전 샘플 동일).")
        print("→ 합성해도 '재현 ✓' 가 무의미합니다. 단계 1·3 으로 읽기 설정부터 확인하세요.")
        return
    ems = _emitters(doc)
    bursty = [e for e in ems if not e["cont"]]
    cont = [e for e in ems if e["cont"]]

    def mod(db_burst: float = 0.0, db_cont: float = 0.0, gap_x: int = 1) -> list[dict]:
        out = []
        for e in ems:
            e2 = dict(e)
            e2["snr"] = e["snr"] + (db_cont if e["cont"] else db_burst)
            e2["bgap"] = e["bgap"] * gap_x
            out.append(e2)
        return out

    variants = [("v0-측정치", ems), ("v1-버스트+6dB", mod(db_burst=6)),
                ("v2-버스트-6dB", mod(db_burst=-6)), ("v5-간격2배", mod(gap_x=2))]
    if cont:
        variants.append(("v4-연속+6dB", mod(db_cont=6)))
        if bursty:      # 버스트 방사체가 없으면 '무연속' 은 v0 과 같은 것을 이름만 바꿔 다는 셈
            variants.append(("v3-무연속", [dict(e) for e in bursty]))

    outdir.mkdir(parents=True, exist_ok=True)
    hits = 0
    for i, (name, es) in enumerate(variants):
        x = _build(n, fs, es, seed=100 + i, cx=cx)
        meta = Meta(fs, "iq" if cx else "real", "f32")
        path = outdir / make_name(f"sigc-{name}", meta)
        write(str(path), x, meta)
        fb = dsp.find_bursts(x, fs)
        full = fb == [(0, n)]
        hits += full
        tag = "→ [(0, n)] 재현 ✓" if full else f"→ 버스트 {len(fb)}개 검출"
        print(f"{name}: {path.name}")
        print(f"   방사체 {len(es)}개 "
              + " · ".join(f"{'연속' if e['cont'] else '버스트'} {e['off'] / 1000:+.1f}kHz"
                           f" SNR {e['snr']:.0f}dB" for e in es))
        print(f"   find_bursts {fb[:3]}{'...' if len(fb) > 3 else ''}  {tag}")
    print(f"\n{len(variants)}개 변형 중 {hits}개가 [(0,n)] 재현 — 재현 변형을 tests/ 에 잠근다.")


# kat 필드가 어긋났을 때 다시 볼 프로브 소스 구간 (실사고 두 건으로 캘리브레이션:
# dhi44 → 인쇄 꼬리의 절단, q-15 qd56 qs56 → pgrp 조건이 항상 거짓 → (pk,pk+1) 폴백)
_KAT_HINT = {
    ("a", "n"): "kat 신호 구성 (fs, t = ... / x0 줄들)", ("a", "f"): "kat 신호 구성",
    ("a", "ev"): "포락선 구간 (lp/hot/ev 줄)", ("a", "ed"): "포락선 구간 (hot/ed 줄)",
    ("a", "sp"): "sp = 10 * np.log10(...) 줄", ("a", "dc"): "dc = abs(complex(x.mean()))/rms",
    ("a", "cp"): "cp = ... percentile(raw, 99.9) 줄", ("a", "cx"): "cx = np.iscomplexobj(x0)",
    ("b", "c"): "stft 호출 (nperseg/noverlap)", ("b", "g"): "stft 호출 / nb, nc = P.shape",
    ("b", "b"): "sc/thr/quiet/base 구간", ("b", "cn"): "cn = float(np.log10(...)) 줄",
    ("b", "t"): "thr = otsu(sc, bins=64)", ("b", "m"): "sc 구간 (uniform_filter1d ... , 3)",
    ("b", "dlo"): "dlo, dhi = 0.25, 0.45 와 b줄 인쇄 꼬리 — 뺄셈 되계산이면 절단 때 44",
    ("b", "dhi"): "dlo, dhi = 0.25, 0.45 와 b줄 인쇄 꼬리 — 뺄셈 되계산이면 절단 때 44",
    ("c", "r"): "above/hi_m/rr 줄", ("c", "dn"): "rr/median 줄", ("c", "gp"): "gp_ 줄",
    ("c", "sb"): "sb = round(10 * ...) 줄", ("c", "av"): "above 줄", ("c", "ah"): "hi_m 줄",
    ("c", "sk"): "spans/segs/sk 줄 (60.0 문턱)",
    ("d", "kb"): "du/occ 줄 (40 * g, 0.1)", ("d", "kc"): "kcont 줄 (fls > 5 * g)",
    ("d", "w"): "grp/wd 줄", ("d", "p"): "p95/pk 줄 (percentile 95)",
    ("d", "pd"): "du[pk] — du/occ 줄", ("d", "ps"): "p95[pk] — p95 줄",
    ("d", "q"): "pgrp = next(...) 줄 — 조건 s <= pk < e 와 for s, e 순서. 폴백이면 q가 주"
                " 방사체 어깨(-15류)로 샌다", ("d", "qd"): "pgrp/m2/q2 세 줄 (q 와 같은 원인)",
    ("d", "qs"): "pgrp/m2/q2 세 줄 (q 와 같은 원인)",
}


def kat_diff(doc: dict[str, dict[str, int]]) -> int:
    """kat 회신을 개발기에서 직접 돌린 기준 4줄과 필드 단위로 대조 — 어긋난 필드마다
    프로브의 어느 소스 구간을 다시 볼지 짚는다. 표지와 눈으로 견주는 일을 자동화한 것."""
    r = subprocess.run([sys.executable, str(_ROOT / "tools" / "sigc_probe.py"), "kat"],
                       capture_output=True, text=True, env={"PYTHONPATH": str(_ROOT)})
    ref, _, _ = parse(r.stdout)
    diffs = 0
    for tag in "abcd":
        for k, v in ref[tag].items():
            got = doc[tag].get(k, doc[tag].get(k))
            if got != v:
                diffs += 1
                hint = _KAT_HINT.get((tag, k), "해당 필드를 만드는 줄")
                print(f"sigc {tag} · {k}: 장비 {got} ≠ 기준 {v}   ← {hint}")
    if diffs == 0:
        print("kat 완전 일치 — 필사 이상 없음 ✓ 실캡처로 진행하세요.")
        return 0
    print(f"\n어긋난 필드 {diffs}개 — 위 구간의 필사를 확인한 뒤 kat 을 다시 돌려 주세요.")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description="sigc 프로브 회신 처리 (검증/해석/데이터셋 생성)")
    ap.add_argument("cmd", choices=("check", "gen"))
    ap.add_argument("--out", type=Path, help="gen: 합성 변형을 쓸 디렉터리")
    ap.add_argument("--kat", action="store_true",
                    help="check: kat 회신을 기준 4줄과 필드 단위로 대조해 용의 소스 줄을 짚는다")
    args = ap.parse_args()
    doc, bad, notes = parse(sys.stdin.read())
    for tag in sorted(doc):
        print(f"일치 ✓  sigc {tag}")
    for msg in notes + bad:
        print(msg)
    if bad:
        print("→ 고치기 전에 그 줄을 다시 봐주세요 (잘못 옮겨진 숫자로 수사를 시작하지 않는다).")
        return 1
    if set(doc) != {"a", "b", "c", "d"}:
        miss = [t for t in "abcd" if t not in doc]
        print(f"누락: {', '.join('sigc ' + t for t in miss)} — 4줄을 모두 받아쳐 주세요.")
        return 2
    # 키 이름까지 검증한다: 장비 f-문자열의 키 오타(실측 2026-08-14: pd{ 를 pdb{ 로 침)는
    # 검출 코드로는 정상이라 여기서 잡아야 한다 — 안 잡으면 해석기가 KeyError 로 죽는다.
    req = {"a": "n f ev ed sp dc cp", "b": "c g b cn t m", "c": "r dn gp sb av ah sk",
           "d": "kb kc w p pd ps q qd qs"}
    opt = {"a": {"cx"}, "b": {"dlo", "dhi"}, "c": set(), "d": set()}
    for tag, keys in req.items():
        missing = set(keys.split()) - set(doc[tag])
        extra = set(doc[tag]) - set(keys.split()) - opt[tag]
        if missing or extra:
            pair = next((f" — '{e}' 는 '{m}' 키의 오타로 보임 (프로브 그 줄의 f-문자열 필사"
                         " 확인)" for e in sorted(extra) for m in sorted(missing)
                        if e.startswith(m) or m.startswith(e)), "")
            print(f"sigc {tag} 키 이상: "
                  + (f"빠짐 {sorted(missing)} " if missing else "")
                  + (f"모르는 키 {sorted(extra)}" if extra else "") + pair)
            return 2
    if args.cmd == "check":
        if args.kat:
            return kat_diff(doc)
        interpret(doc)
        return 0
    if not args.out:
        print("gen 은 --out <디렉터리> 가 필요합니다.")
        return 2
    cmd_gen(doc, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
