"""CLI: analyze FILE / survey / serve. 합성·채점(gen/dataset/sweep)은 개발기 전용 lab.py 가
단다 — 격리망 장비에는 그 파일이 없어 명령 자체가 없다."""

import argparse
import importlib.util
import re
import sys
from zlib import crc32

from .pipeline import analyze_file, survey_file
from .sigio import sidecar_read

_DTYPE_CHOICES = ("i8", "u8", "i16", "u16", "f32", "f64")


# --- 손으로 옮기는 한 줄 -------------------------------------------------------
# 실신호 장비는 인터넷도 클립보드도 없다. 결과는 사람이 화면을 보고 받아쳐서 나온다. 그래서
# 이 블록은 반드시 여기(필사 대상)에 있어야 한다 -- tools/ 에 두면 그 도구부터 필사해야 하는
# 본말전도가 된다. 줄 끝의 검출 코드가 "옮기다 한 글자 틀림"을 잡는다.

_ALPHA = "0123456789abcdefghjkmnpqrstvwxyz"  # 0/o, 1/l/i 처럼 화면에서 헷갈리는 글자를 뺀 32자
_BRIEF_FLAGS = [("eq", lambda d: d["eq"]["applied"]),
                ("al", lambda d: d["detected"]["alias_resolved"]),
                ("fb", lambda d: d["detected"]["baud_fallback"]),
                ("amb", lambda d: d["detected"]["carrier_ambiguous"]),
                ("pre", lambda d: "preamble" in d["detected"])]


def check_code(text: str) -> str:
    """받아친 줄의 오타 검출 코드 (4글자 = 20비트). 공백은 '한 칸으로 줄이되 없애지는'
    않는다 -- 전부 지우면 'fs1000000 16qam' 과 'fs10000001 6qam'(샘플레이트 10배!) 이 같은
    코드가 되어, 값이 바뀌는 오타 32종이 조용히 통과했다. 대소문자와 줄 간격은 계속 무시한다."""
    body = re.sub(r"\s+", " ", re.sub(r"#[0-9a-z]{4}\b", "", text.lower())).strip()
    n = crc32(body.encode())
    return "".join(_ALPHA[(n >> (5 * i)) & 31] for i in (3, 2, 1, 0))


def brief(doc: dict, mode: str) -> str:
    """손으로 옮기는 한 줄 + 검출 코드. Result.to_json() 딕셔너리에서 만든다 -- 객체
    속성을 직접 포맷하면 to_json 이 이미 한 반올림과 두 번 겹쳐 장비와 맥이 갈린다."""
    head = f"sig2 {mode} fs{doc['fs']:.0f} {doc['fmt']}-{doc['dtype']}"
    if mode == "sv":
        lines = [f"{head} n{doc['n_emitters']}"]
        for i, e in enumerate(doc["emitters"][:12]):
            baud = f" bd{e['baud']:.0f}" if e.get("baud") else ""
            lock = f" lk{e['lock']:.0f}" if e.get("lock") is not None else ""
            lines.append(f"{i} fc{e['abs_fc']:.0f}{baud}{lock} {e.get('mod') or e['kind']}")
        if len(doc["emitters"]) > 12:
            lines.append(f"...{len(doc['emitters']) - 12}개 생략")
    else:
        d, q = doc["detected"], doc["quality"]
        p = [head, d["mod"], f"fc{d['fc']:.0f}", f"bd{d['baud']:.0f}", f"lk{q['lock']:.0f}"]
        if q["mer_db"] is not None:
            p.append(f"mer{q['mer_db']:.1f}")
        if d.get("h") is not None:                  # FSK 는 롤오프 대신 변조지수
            p.append(f"h{d['h']:.2f}")
        elif d["rolloff"] is not None:
            p.append(f"rl{d['rolloff']:.2f}")
        if len(doc["bursts"]) > 1:
            p.append(f"b{doc['burst_idx'] + 1}/{len(doc['bursts'])}")
        p += [f for f, get in _BRIEF_FLAGS if get(doc)]
        lines = [" ".join(p)]
    body = "\n".join(lines)
    return f"{body} #{check_code(body)}"


# --- analyze / survey / serve --------------------------------------------------

def _analyze(args: argparse.Namespace) -> int:
    r = analyze_file(args.file, args.fs, args.fmt, args.dtype,
                     "be" if args.be else None, args.bitrev or None, args.diff,
                     None if args.burst is None else args.burst - 1, rf=args.rf)
    j = r.to_json(views=False)          # 한 줄 요약도 여기서 만든다 (반올림이 한 번만 걸리게)
    d = j["detected"]
    truth = (sidecar_read(args.file) or {}).get("truth")  # display only, never detection
    if len(r.bursts) > 1:
        print(f"시간상 버스트 {len(r.bursts)}개 — 그중 {r.burst_idx + 1}번을 분석"
              " (--burst N 으로 선택)")
    print(f"변조       {d['mod']}" + (f"   (정답 {truth['mod']})" if truth else ""))
    print(f"중심주파수 {d['fc']:.1f} Hz" + (f"   (정답 {truth['fc']:.1f})" if truth else ""))
    if d["rf_hz"] is not None:
        print(f"실제 RF   {d['rf_hz'] / 1e6:.6f} MHz")
    print(f"baud       {d['baud']:.1f} Hz" + (f"   (정답 {truth['baud']:.1f})" if truth else ""))
    fsk = r.family == "fsk"
    tail = f"변조지수 h {d['h']:.2f}" if fsk else f"롤오프    {d['rolloff']:.2f}"
    mer = "" if fsk else f"   MER {r.mer_db:.1f} dB"
    print(f"{tail}   lock {r.lock:.1f}{mer}")
    if r.eq_applied:
        print("다중경로 보정(등화기) 적용"
              + (" (T/2 분수간격)" if r.eq_mode == "fse" else " (심볼간격)"))
    if r.alias_resolved:
        print("중심주파수 접힘(앨리어스) 보정: 후보 중 스펙트럼 무게중심에 맞는 것을 선택")
    if r.baud_fallback:
        print("baud 폴백: 스펙트럼에서 baud 선을 못 찾아 점유대역폭으로 추정 — baud 신뢰 낮음")
    if r.carrier_ambiguous:
        print("경고: 중심주파수가 접혀 있을 수 있음 — fs 에 비해 너무 높다"
              " (fc 를 그대로 믿지 말 것)")
    if args.save_iq:
        r.save_iq(args.save_iq)
    if args.save_symbols:
        r.save_symbols(args.save_symbols)
    if args.save_bits:
        r.save_bits(args.save_bits, args.packed)
    if args.report:
        r.save_report(args.report)
    if args.brief:                      # 사람용 출력을 대체하지 않고 맨 끝에 한 줄 더 --
        print(brief(j, "an"))   # 조기 return 이면 위 저장들이 조용히 무시된다
    return 0


def _fhz(v: float) -> str:
    a = abs(v)
    if a >= 1e6:
        return f"{v / 1e6:+.3f} MHz"
    if a >= 1e3:
        return f"{v / 1e3:+.2f} kHz"
    return f"{v:+.0f} Hz"


def _survey(args: argparse.Namespace) -> int:
    """Wideband survey: detect every emitter, demodulate the digital ones."""
    s = survey_file(args.file, args.fs, args.fmt, args.dtype,
                    "be" if args.be else None, args.bitrev or None, args.diff, rf=args.rf)
    rf0 = s.meta.rf_center
    note = f" · RF 중심 {rf0 / 1e6:.3f} MHz" if rf0 is not None else ""
    print(f"주파수상 방사체 {len(s.emitters)}개 — 시간상 버스트 수가 아님"
          f" (샘플레이트 {s.meta.fs:.0f} Hz{note})")
    print(f"{'#':>2} {('실제 RF' if rf0 is not None else '중심주파수'):>12} {'대역폭':>10}"
          f" {'분류':>7} {'변조':>7} {'baud':>11} {'lock':>5}")
    for i, e in enumerate(s.emitters):
        r = e.result
        mod = r.mod if r else "—"
        baud = f"{r.baud:.0f}" if r else "—"
        lock = f"{r.lock:.0f}" if r else "—"
        kind = {"linear": "디지털", "fsk": "FSK", "analog": "아날로그",
                "tone": "순수톤", "tooshort": "너무짧음", "error": "오류"}.get(e.kind, e.kind)
        fc = e.abs_fc if rf0 is None else rf0 + e.abs_fc
        print(f"{i:>2} {_fhz(fc):>12} {_fhz(e.detection.bw):>10} {kind:>7}"
              f" {mod:>7} {baud:>11} {lock:>5}")
    if args.report:
        import json
        with open(args.report, "w") as fh:
            json.dump(s.to_json(), fh, indent=1)
    if args.brief:
        print(brief(s.to_json(), "sv"))
    return 0


def _read_args(p: argparse.ArgumentParser) -> None:
    """Read-side sample-format overrides, shared by analyze/survey (sidecar/filename win)."""
    p.add_argument("--fs", type=float)
    p.add_argument("--fmt", choices=("iq", "real"))
    p.add_argument("--dtype", choices=_DTYPE_CHOICES)
    p.add_argument("--be", action="store_true", help="big-endian 샘플")
    p.add_argument("--bitrev", action="store_true", help="바이트 내 비트 역순")
    p.add_argument("--rf", type=float, help="RF 중심주파수 (Hz) — 실제 주파수로 보고")
    p.add_argument("--brief", action="store_true",
                   help="손으로 옮길 한 줄 + 오타 검출 코드를 끝에 덧붙임")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="signus")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="블라인드 복조 + 제원 탐지")
    a.add_argument("file")
    _read_args(a)
    a.add_argument("--save-iq")
    a.add_argument("--save-symbols")
    a.add_argument("--save-bits")
    a.add_argument("--packed", action="store_true")
    a.add_argument("--report")
    a.add_argument("--diff", action="store_true",
                   help="차동 디맵(D-BPSK/D-QPSK): 회전 모호성 없이 비트 복원")
    a.add_argument("--burst", type=int, help="분석할 버스트 번호 (1부터; 기본 최강 버스트)")

    sv = sub.add_parser("survey", help="광대역 캡처의 모든 신호 탐지 + 복조")
    sv.add_argument("file")
    _read_args(sv)
    sv.add_argument("--diff", action="store_true", help="차동 디맵")
    sv.add_argument("--report", help="JSON 리포트 저장 경로")

    s = sub.add_parser("serve", help="웹 UI 서버")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)

    # 합성 생성·채점(gen/dataset/sweep)은 개발기 전용이다. 격리망 장비에는 lab.py 를
    # 옮기지 않으므로 그 장비에서는 이 명령들이 아예 존재하지 않는다. try/except 로 삼키면
    # 개발기에서 lab 내부의 진짜 임포트 오류까지 "명령 없음"으로 둔갑하니 파일 유무만 본다.
    if importlib.util.find_spec("signus.lab"):
        from . import lab
        lab.add_commands(sub)

    args = ap.parse_args(argv)
    if args.cmd == "analyze":
        return _analyze(args)
    if args.cmd == "survey":
        return _survey(args)
    if args.cmd == "serve":
        from .server import run
        run(args.host, args.port)
        return 0
    return args.run(args)           # lab 이 단 명령 (gen / dataset / sweep)


if __name__ == "__main__":
    sys.exit(main())
