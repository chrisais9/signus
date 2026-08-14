#!/usr/bin/env python3
"""장비로 갈 일회용 프로브를 필사용 PDF 로 발급한다 — codebook 조판 재사용 (맥/보드 전용).

    .venv/bin/python tools/probe_pdf.py     # docs/signus-관찰프로브-sigc-필사용.pdf

표지: 쓰는 법 + KAT 기대 4줄(발급 시점에 프로브를 실제로 돌려 담는다 — 코드와 어긋날 수
없다) + 단계 관찰표 + 회신 양식. 본문: 코드 전량(60행/쪽, 100칸 접기, 이어짐 ↳).
발급 전에 코드북과 같은 문자 대조 검증을 하고, 불일치면 발급을 멈춘다.
"""
from __future__ import annotations

import html
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import codebook as cb  # noqa: E402
import plotref  # noqa: E402

SRC = cb.ROOT / "tools" / "sigc_probe.py"
OUT = cb.DOCS / "signus-관찰프로브-sigc-필사용.pdf"

def steps_table(kat: str) -> list[tuple[str, str, str, str]]:
    """단계표의 '기대값' 칸은 발급 시점의 KAT 출력에서 뽑는다 — 손으로 적어 두면
    프로브를 고칠 때마다 조용히 어긋난다 (2026-08-13 리뷰가 실제로 잡은 사고)."""
    v = {}
    for ln in kat.splitlines():
        v.update({k: int(x) for k, x in re.findall(r"([a-z]+)(-?\d+)", ln.split("#")[0][7:])})
    fb = subprocess.run([sys.executable, str(SRC), "kat", "6"], capture_output=True, text=True,
                        env={"PYTHONPATH": str(cb.ROOT)}, check=True).stdout.strip().splitlines()
    return [
        ("kat", "자기검증", "아래 기대 4줄과 글자까지 일치",
         "매번 먼저. 다르면 값이 다른 줄부터 스크립트 재대조"),
        ("(없음)", "요약 4줄", "kat 과 같은 형식", "4줄을 #코드까지 그대로 받아쳐 회신"),
        ("1", "읽기 확인", f"n {v['n']} · 길이 {v['n'] / v['f']:.3f}s · 형식 iq",
         "n·fs 가 캡처 실제와 맞는가"),
        ("2", "광대역 포락선 (베토)", f"계단이 또렷하고 분리도 {v['ev'] / 100:.2f}",
         "계단(#)과 골(빈칸)이 보이는가, 분리도가 0.12 를 넘는가"),
        ("3", "워터폴 절대전력", "세로 버스트 줄 + 가로로 안 끊기는 띠 2개",
         "끊기지 않는 가로 띠(연속 방사체)가 있는가, 버스트 줄 수"),
        ("4", "빈별 바닥 대비 비율", "세기 일정한 연속 띠는 사라지고 버스트만",
         "find_bursts 가 보는 그림 — 버스트가 여기서도 살아 있는가"),
        ("5", "열 점수 + 문턱", f"봉우리가 hi 위, base {v['b'] / 100:.2f} ≈ c_noise"
         f" {v['cn'] / 100:.2f}", "봉우리가 hi 선을 넘는가, base 가 c_noise 근처인가"),
        ("6", "후보 런 + 실제 답", f"원시 후보 {v['r']}개 → {fb[-1][:38]}",
         "원시 후보가 병합·가드를 거쳐 몇 개로 남는가"),
    ]

EXTRA_CSS = """<style>
  /* 참고편 — 모든 행이 정확히 12pt 여야 60행/쪽 계산이 맞는다 */
  .refg { display: grid; grid-template-columns: 1fr; font-size: 8.5pt; line-height: 12pt;
          flex: none; }
  .refg > div { line-height: 12pt; overflow: hidden; white-space: pre; }
  /* 그림은 100칸 + 문턱 라벨까지 한 줄에 들어가야 한다 — 8.2pt 면 라벨이 잘렸다 */
  .ra { font-family: Menlo, monospace; font-size: 7.6pt; color: #111; }
  .rp, .rq, .rh1, .rh2, .rh3 { font-family: "Apple SD Gothic Neo", sans-serif; }
  .rp { font-size: 8.6pt; color: #111; }
  .rq { font-size: 8pt; color: #666; }
  .rh1 { font-size: 13pt; font-weight: 700; box-shadow: inset 0 -1px 0 #111; }
  .rh2 { font-size: 10.5pt; font-weight: 700; background: #e6e6e6; padding-left: 3pt;
         box-shadow: inset 0 1px 0 #888; }
  .rh3 { font-size: 8.8pt; font-weight: 700; color: #222; }
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

    kat = kat_lines()
    srows = ['<div class="srow head"><span>명령 인자</span><span>무엇을 보나</span>'
             '<span>kat 에서 보여야 하는 것</span><span>실캡처에서 관찰·회신할 것</span></div>']
    srows += [f'<div class="srow"><span class="scmd">{html.escape(a)}</span>'
              f'<span>{html.escape(b)}</span><span>{html.escape(c)}</span>'
              f'<span>{html.escape(d)}</span></div>' for a, b, c, d in steps_table(kat)]

    cover = f"""{EXTRA_CSS}<section class="page cover">
  <div class="ctitle">sigc <span class="cgray">관찰 프로브</span></div>
  <div class="csub">실신호 특징 4줄 + find_bursts 단계 관찰 — [(0, n)] 의 이유를 현장에서 본다</div>
  <div class="cmeta">코드 {nlines}줄 · 총 {total}쪽 &#160;|&#160; {date.today().isoformat()}
    &#160;·&#160; {cb.git_head()} &#160;|&#160; 의존성: 설치돼 있는 signus + numpy/scipy
    (matplotlib 이 있고 창을 띄울 수 있으면 그림 — ssh/무화면이면 자동으로 ASCII)</div>
  <div class="usage">
    ① 이 코드를 <span class="mono">signus/</span> 폴더 <b>옆에</b>
       <span class="mono">sigc.py</span> 로 저장한다. <b>필사는 두 토막으로 끊어도 된다</b> —
       「여기부터는 그림 단계」 주석 <b>앞</b>까지만 옮기면 ②③(요약 4줄)이 다 돌고,
       그 뒤는 ④의 그림 단계 전용이다.<br>
    ② <span class="mono">python3 sigc.py kat</span> — 아래 기대 4줄과 <b>글자까지</b> 비교.
       다르면 스크립트 필사 오타다 (kat 이 잡아준다).<br>
    ③ <span class="mono">python3 sigc.py &lt;캡처파일&gt;</span> — 요약 4줄을 <b>#코드까지</b>
       그대로 받아쳐 회신한다. (파일명은 analyze 때처럼 fs·iq/real 을 실어야 한다)<br>
    ④ <span class="mono">python3 sigc.py &lt;캡처파일&gt; 2</span> 처럼 단계 번호를 붙이면
       그 단계를 그림/ASCII 로 보여준다 — 본 것을 말로 회신하면 된다.<br>
    ⑤ 받은 쪽(맥)은 <span class="mono">tools/sigc.py check</span> 로 먼저 오타 검증 +
       출구 해석을 한다.</div>
  <div class="kat">{html.escape(kat)}</div>
  <div class="steps">{''.join(srows)}</div>
  <div class="cnote">요약 4줄 읽는 법: a=포락선(ev 분리도x100 · ed 듀티% · sp 피크비dB ·
    dc% · cp 포화 지표‰ = 캡처 자신의 상위 0.1% 레벨에 붙은 표본 비율, 깨끗하면 2~4·포화면
    900+ · cx1=복소/cx0=실수) &#160; b=셀 점수
    지형(c 열수 · g 빈수 · b base · cn · t 문턱 · m 최대 · dlo/dhi 는 base 로부터의 문턱 여유,
    x100 — dlo25 dhi45 가 아니면 그 상수를 잘못 옮긴 것) &#160; c=버스트 구조(r 런수 ·
    dn 길이열 · gp 간격열 ·
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
    pages += reference_pages(len(chunks) + 2)
    return cb.SHELL.replace("%%TITLE%%", "signus 관찰 프로브 필사용") \
                   .replace("%%BODY%%", cover + "\n" + "\n".join(pages))


def reference_pages(first_page: int) -> list[str]:
    """참고편 — 합성 신호 6종의 단계 2/4/5 실행 결과 + 읽는 법. 필사 대상이 아니다."""
    rows, flow = plotref.rows(), []
    for cls, text in rows:
        if cls == "rpage":                      # 시나리오는 새 쪽에서 시작
            while len(flow) % cb.LINES_PER_PAGE:
                flow.append(("rb", ""))
            continue
        flow.append((cls, text))
    out, chunks, title = [], cb.paginate(flow), "참고편"
    for i, chunk in enumerate(chunks):
        body = "\n".join(f'<div class="{c}">{cb.esc(t) if t else "&#160;"}</div>'
                         for c, t in chunk)
        head = next((t for c, t in chunk if c == "rh2"), None)
        title = head or (title + " (이어서)" if not title.endswith("(이어서)") else title)
        out.append(cb.page_html(
            title, f"참고편 — 그림 읽는 법 &#160;·&#160; {i + 1}/{len(chunks)}",
            f'<div class="refg">\n{body}\n</div>', first_page + i))
    return out


if __name__ == "__main__":
    cb.to_pdf(build(), OUT)
