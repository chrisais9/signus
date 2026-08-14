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


def parse(text: str) -> tuple[dict[str, dict[str, int]], list[str]]:
    """받아친 블록 -> ({줄: {키: 값}}, 문제 목록). 줄마다 검출 코드를 대조한다."""
    doc, bad = {}, []
    for raw in text.strip().splitlines():
        ln = raw.strip()
        if not ln.startswith("sigc"):
            continue
        m = re.fullmatch(r"(sigc ([a-d])\b.*?)\s*#([0-9a-z]{4})", ln)
        if not m:
            bad.append(f"검출 코드(#xxxx)가 없습니다: {ln}")
            continue
        body, tag, got = m.group(1).rstrip(), m.group(2), m.group(3)
        want = check_code(body)
        if want != got:
            near = sum(a != b for a, b in zip(want, got, strict=True)) == 1
            hint = "코드 네 글자 쪽 오타일 공산" if near else "숫자 자릿수부터 재확인"
            bad.append(f"sigc {tag} 불일치 ✗  받은 코드 #{got}, 계산 #{want} ({hint})")
            continue
        doc[tag] = {k: int(v) for k, v in re.findall(r"([a-z]+)(-?\d+)", body[7:])}
    return doc, bad


def interpret(doc: dict[str, dict[str, int]]) -> None:
    """어느 출구가 [(0,n)] 을 냈는지 — find_bursts 의 분기 순서 그대로 짚는다."""
    a, b, c, d = doc["a"], doc["b"], doc["c"], doc["d"]
    fs, nb, hop = float(a["f"]), b["g"], b["g"] // 2
    khz = fs / nb / 1000
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
            if not a.get("iq", 1) or a["dc"] >= 10:
                note = (" — 단 real 포맷이거나 dc≥10 이면 kc 는 위로 치우친다(실측):"
                        " kc 단독으로 연속 방사체를 단정하지 말 것")
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
    elif b["m"] < b["b"] + 45:
        print(f"→ E5 hi 미달: 최대 점수 {b['m'] / 100:.2f} 가 base+0.45 를 못 넘는다 — 런 없음,"
              " [(0,n)]. (초협대역 희석 또는 저SNR)")
    elif c["r"] > 0 and c["sk"] >= c["r"]:
        print("→ E6 스파이크 커버리지: 모든 런이 임펄스로 탈락 — 폴백 [(0,n)].")
    elif b["m"] < b["b"] + 52 or (c["r"] <= 1 and c["dn"] <= 20):
        print(f"→ 회색지대: 점수 여유 {(b['m'] - b['b']) / 100:.2f}(0.45~0.51) 또는 짧은 단일"
              " 런 — 검출돼도 커버리지 폴백이 [(0,n)] 과 구별되지 않는 구간(실측 2026-08-13)."
              " 판정 유보, 단계 5 관찰을 요청할 것.")
    else:
        print("→ 이 특징이면 HEAD find_bursts 는 버스트를 잡아야 정상. 장비 코드가 셀 기반"
              " 변경분(2026-08-13 발급) 이전 판일 가능성이 크다 — 어느 인쇄본까지 반영됐는지"
              " 확인이 먼저다.")


def _emitters(doc: dict[str, dict[str, int]]) -> list[dict]:
    """특징 -> 방사체 목록. 절대 dB(ps/qs)를 광대역 SNR 로 역산 (근사 — 그리드가 오차를 덮는다)."""
    b, c, d = doc["b"], doc["c"], doc["d"]
    nb, hop = b["g"], b["g"] // 2
    fs = float(doc["a"]["f"])
    w = max(1, min(d["w"], nb // 2))
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


def _build(n: int, fs: float, ems: list[dict], seed: int) -> np.ndarray:
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
    return x


def cmd_gen(doc: dict[str, dict[str, int]], outdir: Path) -> None:
    a = doc["a"]
    n, fs = a["n"], float(a["f"])
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
        variants += [("v3-무연속", [dict(e) for e in bursty] or ems),
                     ("v4-연속+6dB", mod(db_cont=6))]

    outdir.mkdir(parents=True, exist_ok=True)
    hits = 0
    for i, (name, es) in enumerate(variants):
        x = _build(n, fs, es, seed=100 + i)
        meta = Meta(fs, "iq", "f32")
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


def main() -> int:
    ap = argparse.ArgumentParser(description="sigc 프로브 회신 처리 (검증/해석/데이터셋 생성)")
    ap.add_argument("cmd", choices=("check", "gen"))
    ap.add_argument("--out", type=Path, help="gen: 합성 변형을 쓸 디렉터리")
    args = ap.parse_args()
    doc, bad = parse(sys.stdin.read())
    for tag in sorted(doc):
        print(f"일치 ✓  sigc {tag}")
    for msg in bad:
        print(msg)
    if bad:
        print("→ 고치기 전에 그 줄을 다시 봐주세요 (잘못 옮겨진 숫자로 수사를 시작하지 않는다).")
        return 1
    if set(doc) != {"a", "b", "c", "d"}:
        miss = [t for t in "abcd" if t not in doc]
        print(f"누락: {', '.join('sigc ' + t for t in miss)} — 4줄을 모두 받아쳐 주세요.")
        return 2
    if args.cmd == "check":
        interpret(doc)
        return 0
    if not args.out:
        print("gen 은 --out <디렉터리> 가 필요합니다.")
        return 2
    cmd_gen(doc, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
