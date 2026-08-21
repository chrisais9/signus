#!/usr/bin/env python3
"""sa.py(종합 관찰 프로브)를 필사용 PDF 로 발급한다 — codebook 조판 재사용 (맥/보드 전용).

    .venv/bin/python tools/sa_pdf.py                  # docs/signus-종합프로브-sa-필사용.pdf
    .venv/bin/python tools/sa_pdf.py --from git:REV   # 그 판 대비 변경분(바뀐 줄 녹색) PDF

표지: 쓰는 법 + KAT 기대 줄·기대 그림(발급 때 실제 실행) + 판독 규칙 표 + 회신 양식.
본문: 코드 전량. 발급 전 문자 대조 검증, 불일치면 중단. strip.py 를 대체한다.
"""
from __future__ import annotations

import base64
import html
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import codebook as cb  # noqa: E402
from probe_pdf import verify  # noqa: E402

SRC = cb.ROOT / "tools" / "sa_probe.py"
OUT = cb.DOCS / "signus-종합프로브-sa-필사용.pdf"


def kat() -> tuple[str, str, str]:
    """(요약 줄, 디버그 출력 전문, base64 PNG) — 전부 발급 시점의 실행 결과."""
    with tempfile.TemporaryDirectory() as td:
        r = subprocess.run([sys.executable, str(SRC), "kat"], capture_output=True,
                           text=True, cwd=td, env={"PYTHONPATH": str(cb.ROOT)}, check=True)
        line = next(ln for ln in r.stdout.splitlines() if ln.startswith("sa kat"))
        img = base64.b64encode((Path(td) / "sa-kat.png").read_bytes()).decode()
    return line, r.stdout.strip(), img


def build() -> str:
    src = cb.read_src(SRC)
    rows = cb.codebook_rows("sa.py", src)
    if verify(src, rows):
        raise SystemExit("검증 실패 — 발급 중단 (인쇄물이 원본과 다릅니다)")
    chunks = cb.paginate(rows)
    nlines = len(cb.split_lines(src))
    line, debug, img = kat()

    cover = f"""<style>
  .sa img {{ width: 100%; image-rendering: pixelated; border: 0.4pt solid #999;
             margin-top: 4pt; }}
  .sa .kat {{ font-family: Menlo, monospace; font-size: 7.6pt; line-height: 10.5pt;
              background: #f4f4f4; padding: 4pt 7pt; margin-top: 5pt; white-space: pre; }}
  .sa .usage {{ font-family: "Apple SD Gothic Neo", sans-serif; font-size: 8.6pt;
                line-height: 1.68; margin-top: 8pt; }}
  .sa table {{ border-collapse: collapse; font-size: 8.2pt; margin-top: 5pt;
               font-family: "Apple SD Gothic Neo", sans-serif; }}
  .sa td, .sa th {{ border: 0.4pt solid #999; padding: 1.5pt 6pt; }}
</style><section class="page cover sa">
  <div class="ctitle">sa <span class="cgray">종합 관찰 프로브</span></div>
  <div class="csub">스펙트로그램 + find_bursts 검출 + 버스트별 변조 판독(x²·x⁴·x⁸·AM)을
    실행 한 번, 그림 한 장으로</div>
  <div class="cmeta">코드 {nlines}줄 · 총 {1 + len(chunks)}쪽 &#160;|&#160;
    {date.today().isoformat()} &#160;·&#160; {cb.git_head()} &#160;|&#160;
    strip.py 를 대체한다 — strip 을 아직 안 쳤다면 이것만 치면 된다. 출력은 PNG 파일.</div>
  <div class="usage">
    ① 이 코드를 <span class="mono">signus/</span> 옆에 <span class="mono">sa.py</span> 로
       저장한다.<br>
    ② <span class="mono">python3 sa.py kat</span> — 아래 기대 출력과 <b>글자까지</b> 비교하고
       <span class="mono">sa-kat.png</span> 를 열어 기대 그림과 비교한다 (다르면 필사 오타).<br>
    ③ <span class="mono">python3 sa.py &lt;캡처파일&gt; [K]</span> —
       <span class="mono">&lt;캡처파일&gt;.sa.png</span> 생성 (에너지 상위 K개 버스트 판독,
       기본 4). 그림 위 = 버스트 번호 붙은 스펙트로그램 + 검출 띠(흰색=검출, 빈 띠=통짜
       미검출), 아래 = 버스트별 [PSD | x² | x⁴ | x⁸ | AM] 다섯 판.<br>
    ④ 회신: 버스트 표 몇 줄 + <span class="mono">sa cap ...</span> 요약 줄(#코드까지) +
       판독 행에서 바늘이 선 판 이름 한 마디.</div>
  <table><tr><th>바늘이 처음 서는 판</th><th>판정</th><th>바늘 위치</th><th>AM 봉우리</th></tr>
  <tr><td>x²</td><td>BPSK</td><td>2·fc (mod fs, 접힌 축)</td><td rowspan="3">심볼레이트</td></tr>
  <tr><td>x⁴</td><td>QPSK</td><td>4·fc</td></tr>
  <tr><td>x⁸</td><td>8PSK</td><td>8·fc</td></tr>
  <tr><td>없음</td><td>PSK 아님 (잡음꼴·FSK 등)</td><td>—</td><td>—</td></tr></table>
  <div class="kat">{html.escape(debug)}</div>
  <img src="data:image/png;base64,{img}">
  <div class="cnote">기대 그림: 위 스트립에 버스트 1(BPSK)·2(QPSK)와 검출 띠, 아래 판독
    2행 — 1행은 x² 판에 바늘, 2행은 x² 은 언덕뿐이고 x⁴ 판에 바늘. 한 줄이 100칸을 넘으면
    <span class="mono">{cb.CONT_MARK}</span> 이어짐 — 붙여서 입력.</div>
</section>"""

    pages = []
    for idx, chunk in enumerate(chunks, 1):
        nums = [r[0] for r in chunk if r[0] != cb.CONT_MARK]
        span = f"L{nums[0]}-{nums[-1]}" if nums else ""
        body = "\n".join(f'<div class="ln">{html.escape(nm)}</div>'
                         f'<div class="cd">{c}</div>' for nm, c in chunk)
        pages.append(cb.page_html(
            "sa.py", f"종합 프로브 &#160;·&#160; {span} &#160;·&#160; {idx}/{len(chunks)}",
            f'<div class="grid">\n{body}\n</div>', idx + 1))
    return cb.SHELL.replace("%%TITLE%%", "signus 종합 프로브 필사용") \
                   .replace("%%BODY%%", cover + "\n" + "\n".join(pages))


def build_diff(rev: str, note: str) -> str:
    """필사된 옛 판 대비 변경분 — 코드북 변경분과 같은 규약: 파일 통째, 새 줄은 녹색(새 줄번호
    볼드), 없어진 자리는 붉은 점선, 코드 칸에 +/- 접두사 없음(인덴트 보존)."""
    import subprocess as sp
    old = sp.run(["git", "-C", str(cb.ROOT), "show", f"{rev}:tools/sa_probe.py"],
                 capture_output=True, text=True, check=True).stdout.replace("\t", " " * cb.TAB)
    new = cb.read_src(SRC)
    rows, n_add, n_del, changed = cb.diff_rows("sa.py", old, new)
    if not rows:
        raise SystemExit("변경 없음")
    line, debug, _img = kat()
    chunks = cb.paginate(rows)
    cover = f"""<style>
  .sa .kat {{ font-family: Menlo, monospace; font-size: 7.6pt; line-height: 10.5pt;
              background: #f4f4f4; padding: 4pt 7pt; margin-top: 5pt; white-space: pre; }}
</style><section class="page cover sa">
  <div class="ctitle">sa <span class="cgray">변경분</span></div>
  <div class="csub">필사해 둔 sa.py 에서 고칠 줄만 — 녹색 줄을 새 줄번호 자리에 써 넣으면 된다</div>
  <div class="cmeta">{html.escape(rev)} &#160;→&#160; {date.today().isoformat()} ·
    {cb.git_head()} &#160;|&#160; <b>+{n_add}</b> / −{n_del} 줄 · 코드 {len(chunks)}쪽 ·
    새로 쓸 줄 {html.escape(cb.ranges(changed))}</div>
  <div class="cwhat"><div class="wt">바뀐 내용</div><div>·&#160; {html.escape(note)}</div></div>
  <div class="cnote"><span class="lg add">초록</span> = 새로 쓸 줄 (오른쪽 숫자가 새 줄번호)
    &#160;·&#160; 색 없는 줄 = 그대로 &#160;·&#160; <span class="lg cut">붉은 점선</span> =
    그 자리에서 줄이 사라짐 &#160;·&#160; <span class="mono">{cb.CONT_MARK}</span> = 윗줄에 붙는
    이어짐. 고친 뒤 <span class="mono">python3 sa.py kat</span> 이 아래와 같아야 한다.</div>
  <div class="kat">{html.escape(debug)}</div>
</section>"""
    pages = [cover]
    for pno, chunk in enumerate(chunks, 2):
        body = []
        for kind, o_no, n_no, cell in chunk:
            base = kind.split()[0]
            body.append(f'<div class="dn {kind}">{html.escape(o_no)}</div>'
                        f'<div class="dn {kind}">{html.escape(n_no)}</div>'
                        f'<div class="dm {kind}">{"+" if base == "add" else ""}</div>'
                        f'<div class="cd {kind}">{cell}</div>')
        pages.append(cb.page_html("sa.py", f"변경분 &#160;·&#160; {pno - 1}/{len(chunks)}",
                                  f'<div class="dg">\n{"".join(body)}\n</div>', pno))
    return cb.SHELL.replace("%%TITLE%%", "sa 변경분").replace("%%BODY%%", "\n".join(pages))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="base", help="git:<rev> — 필사된 옛 판")
    ap.add_argument("--note", default="", help="변경분 표지의 '바뀐 내용' 한 문장")
    a = ap.parse_args()
    if a.base:
        cb.to_pdf(build_diff(a.base.removeprefix("git:"), a.note),
                  cb.DOCS / "signus-종합프로브-sa-변경분.pdf")
    else:
        cb.to_pdf(build(), OUT)
