#!/usr/bin/env python3
"""strip.py(스펙트로그램 띠 뷰어)를 필사용 PDF 로 발급한다 — codebook 조판 재사용 (맥/보드 전용).

    .venv/bin/python tools/strip_pdf.py     # docs/signus-스트립뷰어-strip-필사용.pdf

표지: 쓰는 법 + KAT 기대 한 줄과 기대 그림(발급 때 프로브를 실제로 돌려 담는다) +
비교 기준 그림 2장(버스트가 있는 캡처 / 잡음뿐인 캡처) + 회신 양식. 본문: 코드 전량.
발급 전 codebook 과 같은 문자 대조 검증, 불일치면 중단.
"""
from __future__ import annotations

import base64
import html
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import codebook as cb  # noqa: E402
from probe_pdf import verify  # noqa: E402

sys.path.insert(0, str(cb.ROOT))
from signus.sigio import Meta, make_name, write  # noqa: E402

SRC = cb.ROOT / "tools" / "strip_probe.py"
OUT = cb.DOCS / "signus-스트립뷰어-strip-필사용.pdf"


def run_probe(arg: str, cwd: Path) -> str:
    r = subprocess.run([sys.executable, str(SRC), arg], capture_output=True, text=True,
                       cwd=cwd, env={"PYTHONPATH": str(cb.ROOT)}, check=True)
    return r.stdout.strip().splitlines()[0]


def refs() -> tuple[str, dict[str, str]]:
    """(KAT 기대 한 줄, {제목: base64 PNG}) — 전부 발급 시점의 실제 실행 결과다."""
    imgs: dict[str, str] = {}
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        kat_line = run_probe("kat", tdp)
        imgs["kat 기대 그림 — 장비의 strip-kat.png 가 이와 같아야 한다"] = \
            base64.b64encode((tdp / "strip-kat.png").read_bytes()).decode()
        rng = np.random.default_rng(3)
        n, fs = 200000, 10000.0
        t = np.arange(n) / fs
        key = ((np.arange(n) // 250) % 2 == 0) & ((np.arange(n) // 5000) % 2 == 0)
        burst = (np.sin(2 * np.pi * 2200 * t) * key
                 + 0.12 * np.sin(2 * np.pi * 3600 * t) + 0.18 * rng.standard_normal(n))
        noise = rng.standard_normal(n)
        for label, sig in (("버스트가 있는 캡처의 예 — GUI 에 신호가 보인다면 이 부류", burst),
                           ("잡음뿐인 캡처 — 지금 signus 가 읽은 결과는 이 모습을 예언한다",
                            noise)):
            meta = Meta(fs, "real", "i16")
            p = tdp / make_name("ref", meta)
            write(str(p), sig, meta)
            run_probe(p.name, tdp)
            imgs[label] = base64.b64encode((tdp / (p.name + ".png")).read_bytes()).decode()
    return kat_line, imgs


def build() -> str:
    src = cb.read_src(SRC)
    rows = cb.codebook_rows("strip.py", src)
    if verify(src, rows):
        raise SystemExit("검증 실패 — 발급 중단 (인쇄물이 원본과 다릅니다)")
    chunks = cb.paginate(rows)
    nlines = len(cb.split_lines(src))
    kat_line, imgs = refs()

    figs = "".join(
        f'<div class="fig"><div class="cap">{html.escape(k)}</div>'
        f'<img src="data:image/png;base64,{v}"></div>' for k, v in imgs.items())
    cover = f"""<style>
  .fig {{ margin-top: 7pt; }}
  .fig img {{ width: 100%; image-rendering: pixelated; border: 0.4pt solid #999; }}
  .fig .cap {{ font-family: "Apple SD Gothic Neo", sans-serif; font-size: 8.4pt;
               font-weight: 700; margin-bottom: 2pt; }}
  .kat {{ font-family: Menlo, monospace; font-size: 8pt; background: #f4f4f4;
          padding: 4pt 7pt; margin-top: 5pt; white-space: pre; }}
  .usage {{ font-family: "Apple SD Gothic Neo", sans-serif; font-size: 8.6pt;
            line-height: 1.7; margin-top: 8pt; }}
</style><section class="page cover">
  <div class="ctitle">strip <span class="cgray">스펙트로그램 띠</span></div>
  <div class="csub">GUI(FFT 256 · Hamming · 0~fs/2)와 같은 표시로 같은 파일을 그린다 —
    디코드가 같은지 눈으로 판정하는 결정 실험</div>
  <div class="cmeta">코드 {nlines}줄 · 총 {1 + len(chunks)}쪽 &#160;|&#160;
    {date.today().isoformat()} &#160;·&#160; {cb.git_head()} &#160;|&#160;
    출력은 PNG 파일 — 아무 이미지 뷰어로 연다 (matplotlib 불필요)</div>
  <div class="usage">
    ① 이 코드를 <span class="mono">signus/</span> 옆에 <span class="mono">strip.py</span> 로
       저장한다.<br>
    ② <span class="mono">python3 strip.py kat</span> — 출력 한 줄을 아래와 <b>글자까지</b>
       비교하고, 생성된 <span class="mono">strip-kat.png</span> 를 열어 아래 기대 그림과
       모양을 비교한다 (다르면 필사 오타).<br>
    ③ <span class="mono">python3 strip.py &lt;캡처파일&gt;</span> —
       <span class="mono">&lt;캡처파일&gt;.png</span> 가 생긴다. GUI 로 같은 파일을 연 화면과
       <b>나란히</b> 비교한다.<br>
    ④ 회신 세 가지: (가) strip 출력 한 줄(#코드까지) &#160;(나) PNG 가 아래 두 기준 그림 중
       어느 부류인지 한 마디 &#160;(다) GUI 가 표시하는 파일 길이(초) — 출력의
       <span class="mono">s</span> 값과 다르면 샘플 해석(비트수/채널)이 다른 것이다.</div>
  <div class="kat">{html.escape(kat_line)}</div>
  {figs}
</section>"""

    pages = []
    for idx, chunk in enumerate(chunks, 1):
        nums = [r[0] for r in chunk if r[0] != cb.CONT_MARK]
        span = f"L{nums[0]}-{nums[-1]}" if nums else ""
        body = "\n".join(f'<div class="ln">{html.escape(nm)}</div>'
                         f'<div class="cd">{c}</div>' for nm, c in chunk)
        pages.append(cb.page_html(
            "strip.py", f"띠 뷰어 &#160;·&#160; {span} &#160;·&#160; {idx}/{len(chunks)}",
            f'<div class="grid">\n{body}\n</div>', idx + 1))
    return cb.SHELL.replace("%%TITLE%%", "signus 스트립 뷰어 필사용") \
                   .replace("%%BODY%%", cover + "\n" + "\n".join(pages))


if __name__ == "__main__":
    cb.to_pdf(build(), OUT)
