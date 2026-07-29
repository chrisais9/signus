#!/usr/bin/env python3
"""받아친 한 줄을 검증한다 (맥 쪽 전용).

발행은 격리망 장비가 한다: `signus analyze <파일> --brief`. 발행기·검출기·지문 계산은 전부
`signus/cli.py` 안에 있고 — 그래야 인쇄물을 통해 그 장비로 흘러간다 — 여기서는 그걸 그대로
import 한다. 두 벌로 나눠 두면 한쪽만 고쳐졌을 때 오타가 0인데도 "불일치"가 뜬다.

    echo "sig2 fp… an fs… …" | tools/brief.py check
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))          # editable 설치 여부와 무관하게 저장소 코드를 쓴다

try:
    from signus.cli import check_code
except ModuleNotFoundError:             # 시스템 파이썬엔 numpy 가 없다 -> venv 로 다시 실행
    _py = _ROOT / ".venv/bin/python"
    if _py.exists() and sys.executable != str(_py):
        os.execv(str(_py), [str(_py), *sys.argv])
    raise


def check(text: str) -> int:
    if re.search(r"\bsig1\b", text.lower()):    # 옛 형식: 공백을 전부 지워 계산하던 판이라
        print("sig1 은 옛 형식입니다 — 그 장비 코드를 새 인쇄본으로 맞춘 뒤 다시 뽑아주세요.")
        print("(sig1 은 공백이 한 칸 밀린 오타를 못 잡았습니다. 값이 맞는지 보장할 수 없습니다.)")
        return 2
    m = re.search(r"#([0-9a-z]{4})\b", text.lower())
    if not m:
        print("검출 코드(#xxxx)가 없습니다 — 줄 끝을 빠뜨리셨나요?")
        return 2
    want, got = m.group(1), check_code(text)
    if want == got:
        print(f"일치 ✓  (#{want})")
        return 0
    # 코드 네 글자 자체를 잘못 옮겼을 수도 있다. 알파벳에 없는 o/i/l 을 0/1/1 로 되읽어 보고,
    # 해밍거리가 1이면 본문이 아니라 코드 쪽 오타일 공산이 크다(우도 약 7900:1).
    alt = want.translate(str.maketrans("oil", "011"))
    if alt == got:
        print(f"일치 ✓  (#{got} — 받은 코드의 o/i/l 을 0/1/1 로 읽었습니다)")
        return 0
    near = sum(a != b for a, b in zip(want, got, strict=True)) == 1
    print(f"불일치 ✗  받은 코드 #{want}, 내용으로 계산하면 #{got}")
    print("→ 검출 코드 네 글자를 먼저 다시 봐주세요." if near
          else "→ 숫자의 자릿수부터 다시 봐주세요 (검출 코드 자체의 오타는 아닌 듯합니다).")
    return 1


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "check":
        sys.exit(check(sys.stdin.read()))
    print("사용법: brief.py check   (받아친 줄을 stdin 으로)", file=sys.stderr)
    sys.exit(2)
