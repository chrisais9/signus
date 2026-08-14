#!/usr/bin/env python3
"""장비로 갈 일회용 프로브를 필사용 PDF 로 발급한다 — codebook 조판 재사용 (맥/보드 전용).

    .venv/bin/python tools/probe_pdf.py     # docs/signus-관찰프로브-sigc-필사용.pdf

표지: 쓰는 법 + KAT 기대 4줄(발급 시점에 프로브를 실제로 돌려 담는다 — 코드와 어긋날 수
없다) + 단계 관찰표 + 회신 양식. 본문: 코드 전량(60행/쪽, 100칸 접기, 이어짐 ↳).
발급 전에 코드북과 같은 문자 대조 검증을 하고, 불일치면 발급을 멈춘다.
"""
from __future__ import annotations

import html
import subprocess
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import codebook as cb  # noqa: E402

SRC = cb.ROOT / "tools" / "sigc_probe.py"
OUT = cb.DOCS / "signus-관찰프로브-sigc-필사용.pdf"

STEPS = [
    ("kat", "자기검증", "아래 기대 4줄과 글자까지 일치",
     "매번 먼저. 다르면 값이 다른 줄부터 스크립트 재대조"),
    ("(없음)", "요약 4줄", "kat 과 동일한 형식", "4줄을 #코드까지 그대로 받아쳐 회신"),
    ("1", "읽기 확인", "n 60000 · 길이 0.060s · 형식 iq", "n·fs 가 캡처 실제와 맞는가"),
    ("2", "광대역 포락선 (베토)", "계단 5개, 분리도 0.86",
     "계단이 보이는가, 분리도가 0.12 를 넘는가"),
    ("3", "워터폴 절대전력", "세로 버스트 5줄 + 위쪽 가로 연속 띠 1개",
     "가로로 끊기지 않는 띠(연속 방사체)가 있는가, 버스트 줄 수"),
    ("4", "빈별 바닥 대비 비율", "연속 띠는 사라지고 버스트 5줄만",
     "find_bursts 가 보는 그림 — 버스트가 여기서도 살아 있는가"),
    ("5", "열 점수 + 문턱", "봉우리 5개가 hi 위, base 1.51 ≈ c_noise 1.56",
     "봉우리가 hi 를 넘는가, base 가 c_noise 근처인가"),
    ("6", "후보 런 표", "런 5개, 높이 ~34dB, 피크/평균 하나만 60 초과",
     "런 개수·높이·스파이크 여부"),
]

EXTRA_CSS = """<style>
  .steps { margin-top: 10pt; }
  .srow { display: grid; grid-template-columns: 34pt 108pt 1fr 1fr; align-items: baseline;
          font-family: "Apple SD Gothic Neo", sans-serif; font-size: 8pt; line-height: 13.5pt;
          border-bottom: 0.3pt dotted #ddd; }
  .srow.head { font-weight: 700; border-bottom: 0.6pt solid #111; }
  .scmd { font-family: Menlo, monospace; font-size: 7.6pt; }
  .kat { font-family: Menlo, monospace; font-size: 8pt; line-height: 12pt; background: #f4f4f4;
         padding: 5pt 7pt; margin-top: 5pt; white-space: pre; }
  .usage { font-family: "Apple SD Gothic Neo", sans-serif; font-size: 8.6pt; line-height: 1.75;
           margin-top: 9pt; }
  .usage .mono { font-size: 8pt; }
</style>"""


def kat_lines() -> str:
    r = subprocess.run([sys.executable, str(SRC), "kat"], capture_output=True, text=True,
                       env={"PYTHONPATH": str(cb.ROOT)}, check=True)
    return r.stdout.strip()


def verify(src: str, rows: list[tuple[str, str]]) -> int:
    """코드북 verify 와 같은 문자 대조 — 인쇄물이 원본과 1글자라도 다르면 발급 금지."""
    src_lines = cb.split_lines(src)
    logical, cur, guides = [], None, 0
    for num, cell in rows:
        txt, g = cb.cell_text(cell)
        if num != cb.CONT_MARK:
            if cur is not None:
                logical.append((cur, guides))
            cur, guides = txt, g
        else:
            cur += txt
    if cur is not None:
        logical.append((cur, guides))
    fail = 0
    if len(logical) != len(src_lines):
        print(f"줄 수 불일치: 인쇄 {len(logical)} vs 원본 {len(src_lines)}")
        fail += 1
    for i, (body, g) in enumerate(logical):
        exp = src_lines[i].rstrip()
        lead = len(exp) - len(exp.lstrip(" "))
        if " " * lead + body != exp or g != -(-lead // cb.TAB):
            print(f"L{i + 1} 불일치\n  인쇄 {' ' * lead + body!r}\n  원본 {exp!r}")
            fail += 1
    return fail


def build() -> str:
    src = cb.read_src(SRC)
    rows = cb.codebook_rows("sigc.py", src)
    if verify(src, rows):
        raise SystemExit("검증 실패 — 발급 중단 (인쇄물이 원본과 다릅니다)")
    chunks = cb.paginate(rows)
    nlines = len(cb.split_lines(src))
    total = 1 + len(chunks)

    srows = ['<div class="srow head"><span>명령 인자</span><span>무엇을 보나</span>'
             '<span>kat 에서 보여야 하는 것</span><span>실캡처에서 관찰·회신할 것</span></div>']
    srows += [f'<div class="srow"><span class="scmd">{html.escape(a)}</span>'
              f'<span>{html.escape(b)}</span><span>{html.escape(c)}</span>'
              f'<span>{html.escape(d)}</span></div>' for a, b, c, d in STEPS]

    cover = f"""{EXTRA_CSS}<section class="page cover">
  <div class="ctitle">sigc <span class="cgray">관찰 프로브</span></div>
  <div class="csub">실신호 특징 4줄 + find_bursts 단계 관찰 — [(0, n)] 의 이유를 현장에서 본다</div>
  <div class="cmeta">코드 {nlines}줄 · 총 {total}쪽 &#160;|&#160; {date.today().isoformat()}
    &#160;·&#160; {cb.git_head()} &#160;|&#160; 의존성: 설치돼 있는 signus + numpy/scipy
    (matplotlib 이 있고 창을 띄울 수 있으면 그림 — ssh/무화면이면 자동으로 ASCII)</div>
  <div class="usage">
    ① 이 코드를 <span class="mono">signus/</span> 폴더 <b>옆에</b>
       <span class="mono">sigc.py</span> 로 저장한다.<br>
    ② <span class="mono">python3 sigc.py kat</span> — 아래 기대 4줄과 <b>글자까지</b> 비교.
       다르면 스크립트 필사 오타다 (kat 이 잡아준다).<br>
    ③ <span class="mono">python3 sigc.py &lt;캡처파일&gt;</span> — 요약 4줄을 <b>#코드까지</b>
       그대로 받아쳐 회신한다. (파일명은 analyze 때처럼 fs·iq/real 을 실어야 한다)<br>
    ④ <span class="mono">python3 sigc.py &lt;캡처파일&gt; 2</span> 처럼 단계 번호를 붙이면
       그 단계를 그림/ASCII 로 보여준다 — 본 것을 말로 회신하면 된다.<br>
    ⑤ 받은 쪽(맥)은 <span class="mono">tools/sigc.py check</span> 로 먼저 오타 검증 +
       출구 해석을 한다.</div>
  <div class="kat">{html.escape(kat_lines())}</div>
  <div class="steps">{''.join(srows)}</div>
  <div class="cnote">요약 4줄 읽는 법: a=포락선(ev 분리도x100 · ed 듀티% · sp 피크비dB ·
    dc% · cp 클리핑‰ · iq1=복소/iq0=실수. cp 는 정수형 캡처에서만 의미) &#160; b=셀 점수
    지형(c 열수 · g 빈수 · b base · cn · t 문턱 · m 최대 · lo/hi 문턱 여유, x100 — lo25 hi45
    가 아니면 그 상수를 잘못 옮긴 것) &#160; c=버스트 구조(r 런수 · dn 길이열 · gp 간격열 ·
    sb 점수dB · av/ah 문턱 위 열% · sk 스파이크런) &#160; d=방사체(kb 점유빈 · kc 연속점유빈 ·
    w 최대폭빈 · p/q 위치빈 · pd/qd 듀티% · ps/qs 절대dB, q999=부 방사체 없음).
    한 줄이 100칸을 넘으면 <span class="mono">{cb.CONT_MARK}</span> 이어짐 — 붙여서 입력.
    <b>kat 이 검증하는 것은 요약 4줄 경로뿐이다</b> — 단계 2~6 의 그림 코드(curve/image/pool)는
    kat 이 실행하지 않으므로, 그림 단계에서 처음 크래시하면 그 세 함수의 필사부터 다시 본다.</div>
</section>"""

    pages = []
    for idx, chunk in enumerate(chunks, 1):
        nums = [r[0] for r in chunk if r[0] != cb.CONT_MARK]
        span = f"L{nums[0]}-{nums[-1]}" if nums else ""
        body = "\n".join(f'<div class="ln">{html.escape(nm)}</div>'
                         f'<div class="cd">{c}</div>' for nm, c in chunk)
        pages.append(cb.page_html(
            "sigc.py", f"관찰 프로브 &#160;·&#160; {span} &#160;·&#160; {idx}/{len(chunks)}",
            f'<div class="grid">\n{body}\n</div>', idx + 1))
    return cb.SHELL.replace("%%TITLE%%", "signus 관찰 프로브 필사용") \
                   .replace("%%BODY%%", cover + "\n" + "\n".join(pages))


if __name__ == "__main__":
    cb.to_pdf(build(), OUT)
