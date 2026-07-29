#!/usr/bin/env python3
"""손으로 쳐서 옮기는 결과 요약 — 한 줄 + 오타 검출 코드.

격리망 장비에서 클립보드 없이 결과를 옮겨야 한다. 그래서 필사용 코드북과 같은 원칙:
짧게, 헷갈리는 글자를 쓰지 않고, 틀리면 티가 나게.

    tools/brief.py emit <리포트.json> <an|sv> <리비전>   # 요약 한 줄 발행
    tools/brief.py check                                 # 받아친 줄을 stdin 으로 검증

검출 코드 `#xxxx` 는 나머지 글자 전부의 crc32 다. 한 글자만 틀려도 안 맞으므로,
받아친 쪽에서 `check` 를 돌리면 "숫자 하나 잘못 봤다"를 바로 잡아낸다. 공백과 대소문자는
계산에서 빼므로 줄바꿈·간격이 달라지는 건 통과시킨다 (진짜 위험한 건 숫자다).
"""
from __future__ import annotations

import json
import re
import sys
from zlib import crc32

# 0/O, 1/l/I 처럼 손글씨·화면에서 헷갈리는 글자를 뺀 32자 (Crockford base32 계열)
_ALPHA = "0123456789abcdefghjkmnpqrstvwxyz"
_FLAGS = [("eq", lambda d: d["eq"]["applied"]),          # 등화기
          ("al", lambda d: d["detected"]["alias_resolved"]),      # 반송파 앨리어스 보정
          ("fb", lambda d: d["detected"]["baud_fallback"]),       # 심볼레이트 폴백
          ("amb", lambda d: d["detected"]["carrier_ambiguous"]),  # 앨리어싱 경고
          ("pre", lambda d: "preamble" in d["detected"])]         # 프리앰블 동기


def check_code(text: str) -> str:
    """공백·대소문자·기존 검출코드를 뺀 나머지의 crc32 → 4글자."""
    body = re.sub(r"#[0-9a-z]{4}\b", "", text.lower())
    body = re.sub(r"\s+", "", body)
    n = crc32(body.encode()) & 0xFFFFF          # 20비트 → 4글자
    return "".join(_ALPHA[(n >> s) & 31] for s in (15, 10, 5, 0))


def _num(v: float, digits: int = 0) -> str:
    return f"{v:.{digits}f}"


def emit(doc: dict, mode: str, rev: str) -> str:
    head = f"sig1 {rev} {mode}"
    if mode == "sv":
        lines = [f"{head} fs{_num(doc['fs'])} n{doc['n_emitters']}"]
        for i, e in enumerate(doc["emitters"][:12]):
            what = e.get("mod") or e["kind"]
            baud = f" bd{_num(e['baud'])}" if e.get("baud") else ""
            lock = f" lk{_num(e['lock'])}" if e.get("lock") is not None else ""
            lines.append(f"{i} fc{_num(e['abs_fc'])}{baud}{lock} {what}")
        if len(doc["emitters"]) > 12:
            lines.append(f"...{len(doc['emitters']) - 12}개 생략")
    else:
        d, q = doc["detected"], doc["quality"]
        parts = [head, f"fs{_num(doc['fs'])}", d["mod"], f"fc{_num(d['fc'])}",
                 f"bd{_num(d['baud'])}", f"lk{_num(q['lock'])}"]
        if q.get("mer_db") is not None:
            parts.append(f"mer{_num(q['mer_db'], 1)}")
        if d.get("h") is not None:                       # FSK 는 롤오프 대신 변조지수
            parts.append(f"h{_num(d['h'], 2)}")
        elif d.get("rolloff") is not None:
            parts.append(f"rl{_num(d['rolloff'], 2)}")
        if len(doc.get("bursts", [])) > 1:
            parts.append(f"b{doc['burst_idx'] + 1}/{len(doc['bursts'])}")
        parts += [f for f, get in _FLAGS if get(doc)]
        lines = [" ".join(parts)]
    body = "\n".join(lines)
    return f"{body} #{check_code(body)}"


def check(text: str) -> int:
    m = re.search(r"#([0-9a-z]{4})\b", text.lower())
    if not m:
        print("검출 코드(#xxxx)가 없습니다 — 줄 끝을 빠뜨리셨나요?")
        return 2
    want, got = m.group(1), check_code(text)
    if want == got:
        print(f"일치 ✓  (#{want})")
        return 0
    print(f"불일치 ✗  받은 코드 #{want}, 내용으로 계산하면 #{got}")
    print("→ 한 글자가 다릅니다. 숫자의 자릿수부터, 그다음 검출 코드 네 글자를 다시 봐주세요.")
    return 1


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[1] == "check":
        return check(sys.stdin.read())
    if len(argv) == 5 and argv[1] == "emit":
        with open(argv[2]) as fh:
            print(emit(json.load(fh), argv[3], argv[4]))
        return 0
    print(__doc__.strip().splitlines()[2], file=sys.stderr)
    print("사용법: brief.py emit <json> <an|sv> <rev> | brief.py check", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
