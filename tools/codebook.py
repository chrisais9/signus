#!/usr/bin/env python3
"""필사용 인쇄물 발급기 — 코드북 PDF + 변경분(diff) PDF.

    python tools/codebook.py build        # 코드북 전체 발급 (docs/signus-코드-필사용.pdf)
    python tools/codebook.py diff         # 마지막 스냅샷 대비 변경분 (docs/signus-코드-변경분.pdf)
    python tools/codebook.py diff --from git:HEAD~3
    python tools/codebook.py verify       # 인쇄물 == 원본 문자 대조 (발급 전 필수)
    python tools/codebook.py snap         # "여기까지 필사했다" 기준점 갱신

지켜야 하는 규약 (docs/필사-코드북-지침.md 참조):
  · 테스트 코드는 넣지 않는다. 실행에 필요한 코드만.
  · 합성 생성기·채점(EXCLUDED 의 gen.py/lab.py)도 넣지 않는다 — 개발기 전용
    (2026-08-03 사용자 결정). 격리망 장비가 써야 하는 코드만 싣는다.
  · diff 는 코드 칸에 +/- 접두사를 붙이지 않는다 — 인덴트가 한 칸 밀리면
    필사본이 깨진다. 추가 표시는 줄번호 칸의 색·볼드와 별도 부호 칸으로만.
  · diff 에 없어진 코드 본문을 싣지 않는다. 지울 자리만 한 줄로 알린다.
  · 페이지 나눔을 직접 계산한다 (60행/쪽 고정) → 목차 쪽번호가 항상 정확.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import sys
import unicodedata
from datetime import date
from pathlib import Path

from pygments import lex
from pygments.lexers import CssLexer, HtmlLexer, JavascriptLexer, PythonLexer, TOMLLexer
from pygments.token import Token

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
STATE_DIR = ROOT / ".codebook"
BASE_DIR = STATE_DIR / "baseline"
STATE_JSON = STATE_DIR / "state.json"
CHROME_CANDS = (  # PDF 를 그리는 헤드리스 브라우저. 맥에서 뽑든 리눅스 보드에서
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",  # 뽑든 같은
    "/Applications/Chromium.app/Contents/MacOS/Chromium",            # 명령이 돌아야 한다
    "google-chrome-stable", "google-chrome", "chromium-browser", "chromium",
)

# ── 레이아웃 (A4 210x297mm, 8.5pt Menlo, 실측 용량 101칸) ──────────────────────
LINES_PER_PAGE = 60
WRAP_COLS = 100          # 코드북: 줄번호 칸 25pt
WRAP_COLS_DIFF = 96      # 변경분: 줄번호 2칸 + 부호칸 + 여백 = 44pt
CONT_MARK = "↳"
TAB = 4

SECTIONS: list[tuple[str, list[tuple[str, str]]]] = [
    ("0 · 프로젝트 설정", [
        ("pyproject.toml", "패키지 정의 · 의존성 · ruff/pytest 설정"),
        ("signus/__init__.py", "패키지 진입점"),
    ]),
    ("1 · 신호 입출력과 기준표", [
        ("signus/sigio.py", "샘플 파일 I/O — 파일명이 fs·iq/real·샘플타입을 실어 나른다"),
        ("signus/constellations.py", "성상도 · 그레이 비트 매핑 · FSK 레벨 (공용 기준표)"),
    ]),
    ("2 · 광대역 탐지와 채널 추출", [
        ("signus/spectrum.py", "스펙트럼 · 워터폴 뷰 (표시 전용, 탐지엔 미사용)"),
        ("signus/detect.py", "광대역 탐지 — 캡처 안의 모든 방사체를 주파수/시간 상자로"),
        ("signus/channelize.py", "채널 추출 — 광대역에서 방사체 하나만 기저대역으로 끌어냄"),
        ("signus/triage.py", "채널 트리아지 — 복조 가능한 디지털 신호인지 판별"),
    ]),
    ("3 · 수신 DSP 코어", [
        ("signus/dsp.py", "수신 DSP 단계 (벡터화) — 이 프로젝트의 심장"),
        ("signus/eq.py", "블라인드 심볼간격 FIR 등화기 — CMA 획득 후 판정지향 LMS"),
        ("signus/sync.py", "블라인드 반복 프리앰블 동기 (정의 불요)"),
        ("signus/classify.py", "결정층 — 변조 분류 · 회전 정렬 · 락 품질 · SNR"),
        ("signus/_accel.py", "선택적 numba 가속 (없으면 동일한 순수 numpy로 폴백)"),
    ]),
    ("4 · 변조별 수신 경로", [
        ("signus/fsk.py", "FSK/CPM 경로 — 정포락선 게이트 + 주파수 판별기 복조"),
        ("signus/chirp.py", "선형 처프 / CSS(LoRa) 블라인드 탐지 · 특성화"),
    ]),
    ("5 · 종단 파이프라인", [
        ("signus/pipeline.py", "블라인드 복조 종단 조립 — 두 계열의 진입점"),
    ]),
    ("6 · 사용자 인터페이스", [
        ("signus/cli.py", "CLI — analyze / survey / serve (합성·채점 명령은 개발기 전용)"),
        ("signus/server.py", "표준 라이브러리만 쓰는 웹 서버 + POST /api/analyze"),
    ]),
    ("7 · 웹 프론트엔드", [
        ("signus/web/index.html", "UI 구조"),
        ("signus/web/style.css", "UI 스타일"),
        ("signus/web/app.js", "업로드 · 분석 요청 · 성상도/워터폴 캔버스 렌더링"),
    ]),
]

# 개발기 전용 -- 인쇄물·격리망 장비에는 절대 싣지 않는다 (2026-08-03 사용자 결정).
# 합성·채점은 Claude 가 도는 개발기의 테스트·정합성 확인에만 쓰고, 장비가 써야 하는
# 기능은 반드시 SECTIONS 쪽 모듈에 넣는다. signus/ 에 새 파일이 생기면 두 목록 중
# 한쪽에 올려야 발급이 된다 (main 의 검사가 강제한다).
EXCLUDED = {
    "signus/gen.py",        # 합성 신호 생성기 -- 테스트 하네스의 정답지
    "signus/lab.py",        # 채점·전수조사 하네스 (gen/dataset/sweep 명령)
}

FILES = [(sec, rel, desc) for sec, items in SECTIONS for rel, desc in items]
LEXERS = {".py": PythonLexer, ".js": JavascriptLexer, ".html": HtmlLexer,
          ".css": CssLexer, ".toml": TOMLLexer}

CLS_MAP = [
    (Token.Comment, "c"), (Token.Literal.String.Doc, "sd"), (Token.Literal.String, "s"),
    (Token.Keyword, "k"), (Token.Name.Function, "nf"), (Token.Name.Class, "nf"),
    (Token.Name.Decorator, "nd"), (Token.Name.Builtin, "nb"), (Token.Name.Tag, "nt"),
    (Token.Name.Attribute, "na"), (Token.Literal.Number, "m"), (Token.Operator, "o"),
]


# ── 소스 → 줄별 토큰 ──────────────────────────────────────────────────────────
def token_class(tok) -> str:
    t = tok
    while t is not None and t is not Token:
        for base, cls in CLS_MAP:
            if t is base:
                return cls
        t = t.parent
    return ""


def cw(ch: str) -> int:
    """열 폭 — 한글은 2칸으로 센다 (실제 렌더는 더 좁아 넘칠 위험은 없다)."""
    return 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1


def read_src(path: Path) -> str:
    return path.read_text(encoding="utf-8").replace("\t", " " * TAB)


def split_lines(text: str) -> list[str]:
    out = text.split("\n")
    while out and out[-1] == "":
        out.pop()
    return out


def segments_of(text: str, suffix: str) -> list[list[tuple[str, str]]]:
    """줄별 [(css class, text), ...]. 문자를 하나도 잃지 않는다."""
    lines: list[list[tuple[str, str]]] = [[]]
    for tok, val in lex(text, LEXERS[suffix]()):
        cls = token_class(tok)
        for i, part in enumerate(val.split("\n")):
            if i:
                lines.append([])
            if part:
                lines[-1].append((cls, part))
    while lines and not lines[-1]:
        lines.pop()
    # pygments 는 파일 맨 앞의 개행들을 통째로 삼킨다 -- 그대로 두면 빈 첫 줄로 시작하는
    # 파일의 모든 줄번호가 하나씩 밀리고 diff_rows 가 끝을 넘어 IndexError 로 죽는다.
    lead = len(text) - len(text.lstrip("\n"))
    short = len(split_lines(text)) - len(lines)
    if short > 0:
        lines = [[] for _ in range(min(short, lead))] + lines
    return lines


# ── 접기 / 셀 렌더 ────────────────────────────────────────────────────────────
def wrap_line(segs, first_limit: int, cont_limit: int) -> list[list[tuple[str, str]]]:
    """폭에 맞춰 접는다. 이음매 양쪽이 공백이면 안 된다 — 행 끝/머리의 공백은
    눈에 보이지 않으므로 "붙여서 입력" 규칙이 그때만 틀려진다."""
    chars = [(cls, ch) for cls, txt in segs for ch in txt]
    rows, i, limit = [], 0, first_limit
    while True:
        w, j = 0, i
        while j < len(chars) and w + cw(chars[j][1]) <= limit:
            w += cw(chars[j][1])
            j += 1
        if j >= len(chars):
            rows.append(chars[i:])
            break
        cut = j
        while cut > i + 8 and (chars[cut - 1][1] == " " or chars[cut][1] == " "):
            cut -= 1
        if chars[cut - 1][1] == " " or chars[cut][1] == " ":
            cut = j
        rows.append(chars[i:cut])
        i, limit = cut, cont_limit
    out = []
    for row in rows:
        merged: list[tuple[str, str]] = []
        for cls, ch in row:
            if merged and merged[-1][0] == cls:
                merged[-1] = (cls, merged[-1][1] + ch)
            else:
                merged.append((cls, ch))
        out.append(merged)
    return out


def esc(s: str) -> str:
    return html.escape(s).replace(" ", "&#160;")


def render_cell(segs, lead: int, pad: int) -> str:
    """한 물리 행의 코드 셀. lead=들여쓰기 눈금 칸, pad=이어짐 행 들여쓰기."""
    out = []
    if pad:
        out.append(f'<span class="pad" style="width:{pad}ch"></span>')
    if lead:
        full, rem = divmod(lead, TAB)
        out.append('<span class="ig"></span>' * full)
        if rem:
            out.append(f'<span class="ig" style="width:{rem}ch"></span>')
    for cls, txt in segs:
        out.append(f'<span class="{cls}">{esc(txt)}</span>' if cls else esc(txt))
    return "".join(out)


def physical_rows(segs, cols: int) -> list[tuple[bool, str]]:
    """한 논리 줄 → [(첫 행인가, 셀 HTML)]. 원본 문자를 그대로 보존한다."""
    raw = "".join(t for _, t in segs).rstrip()
    if not raw:
        return [(True, "")]
    lead = len(raw) - len(raw.lstrip(" "))
    rest: list[tuple[str, str]] = []
    drop = lead
    for cls, txt in segs:
        if drop >= len(txt) and txt.strip() == "":
            drop -= len(txt)
            continue
        if drop:
            txt, drop = txt[drop:], 0
        rest.append((cls, txt))
    while rest and rest[-1][1].rstrip() == "":
        rest.pop()
    if rest:
        rest[-1] = (rest[-1][0], rest[-1][1].rstrip())
    pad = min(lead + 2, cols - 24)
    wrapped = wrap_line(rest, max(cols - lead, 24), cols - pad)
    return [(i == 0, render_cell(w, lead if i == 0 else 0, 0 if i == 0 else pad))
            for i, w in enumerate(wrapped)]


# ── 코드북 ────────────────────────────────────────────────────────────────────
def codebook_rows(rel: str, text: str) -> list[tuple[str, str]]:
    rows = []
    for n, segs in enumerate(segments_of(text, Path(rel).suffix), 1):
        for first, cell in physical_rows(segs, WRAP_COLS):
            rows.append((str(n) if first else CONT_MARK, cell))
    return rows


def paginate(rows: list, per_page: int = LINES_PER_PAGE) -> list[list]:
    return [rows[i:i + per_page] for i in range(0, len(rows), per_page)] or [[]]


def codebook_page_map() -> tuple[dict[str, int], int]:
    """{파일: 시작 쪽번호}, 총 쪽수 — PDF 없이도 계산되는 결정적 페이지네이션."""
    pmap, page = {}, 2                       # 1쪽 = 표지+목차
    for _, rel, _ in FILES:
        pmap[rel] = page
        page += len(paginate(codebook_rows(rel, read_src(ROOT / rel))))
    return pmap, page - 1


def build_codebook() -> str:
    pages, toc = [], []
    page_no = 2
    for sec, rel, desc in FILES:
        text = read_src(ROOT / rel)
        chunks = paginate(codebook_rows(rel, text))
        toc.append((sec, rel, desc, len(split_lines(text)), page_no))
        for idx, chunk in enumerate(chunks, 1):
            nums = [r[0] for r in chunk if r[0] != CONT_MARK]
            span = f"L{nums[0]}-{nums[-1]}" if nums else ""
            body = "\n".join(f'<div class="ln">{html.escape(n)}</div>'
                             f'<div class="cd">{c}</div>' for n, c in chunk)
            pages.append(page_html(
                rel, f"{sec} &#160;·&#160; {span} &#160;·&#160; {idx}/{len(chunks)}",
                f'<div class="grid">\n{body}\n</div>', page_no))
            page_no += 1

    total_lines = sum(t[3] for t in toc)
    rows_html, last_sec = [], None
    for sec, rel, desc, nlines, start in toc:
        if sec != last_sec:
            rows_html.append(f'<div class="tsec">{html.escape(sec)}</div>')
            last_sec = sec
        rows_html.append(
            f'<div class="trow"><span class="tbox">☐</span>'
            f'<span class="tf">{html.escape(rel)}</span>'
            f'<span class="td">{html.escape(desc)}</span>'
            f'<span class="tn">{nlines}줄</span>'
            f'<span class="tp">{start}쪽</span></div>')

    cover = f"""<section class="page cover">
  <div class="ctitle">signus</div>
  <div class="csub">Blind PSK/QAM/FSK/Chirp Demodulator — 필사용 코드북</div>
  <div class="cmeta">실행에 필요한 코드 전량 · {len(toc)}개 파일 · {total_lines:,}줄
    · 총 {page_no - 1}쪽 &#160;&#160;|&#160;&#160; {date.today().isoformat()}
    &#160;·&#160; {git_head()}</div>
  <div class="cnote">한 줄이 {WRAP_COLS}칸을 넘으면 왼쪽 줄번호 자리에
    <span class="mono">{CONT_MARK}</span> 표시가 붙고 아랫줄로 이어집니다 — 원래는 한 줄이니
    붙여서 입력하세요. 세로 점선은 들여쓰기 4칸 눈금입니다.</div>
  <div class="toc">{''.join(rows_html)}</div>
</section>"""
    return SHELL.replace("%%TITLE%%", "signus 필사용 코드북") \
                .replace("%%BODY%%", cover + "\n" + "\n".join(pages))


# ── 변경분 ────────────────────────────────────────────────────────────────────
def diff_rows(rel: str, old: str, new: str) -> tuple[list, int, int, list[int]]:
    """[(kind, 옛 줄번호, 새 줄번호, 셀)] — kind: ctx / add (+ ' gap').

    · 바뀐 파일은 **통째로** 싣는다. 필사본은 그 파일을 새로 쓰는 게 부분 수정보다
      안전하고, 어디를 고칠지 찾느라 훑을 일도 없다.
    · 코드 칸에는 +/- 접두사를 절대 붙이지 않는다 (인덴트 보존). 부호는 별도 칸.
    · 없어진 코드도, 그걸 설명하는 문구도 싣지 않는다 — 이음매의 붉은 점선(행을
      쓰지 않는 box-shadow)만 남긴다.
    """
    o_lines, n_lines = split_lines(old), split_lines(new)
    n_segs = segments_of(new, Path(rel).suffix) if new else []   # 옛 코드는 싣지 않는다
    sm = difflib.SequenceMatcher(None, o_lines, n_lines, autojunk=False)

    ev: list[tuple[str, int, int, str]] = []     # (kind, o_idx, n_idx, opcode)
    for t, i1, i2, j1, j2 in sm.get_opcodes():
        if t == "equal":
            ev += [("ctx", i1 + k, j1 + k, t) for k in range(i2 - i1)]
        else:
            ev += [("del", k, -1, t) for k in range(i1, i2)]
            ev += [("add", -1, k, t) for k in range(j1, j2)]
    n_del = sum(e[0] == "del" for e in ev)
    n_add = sum(e[0] == "add" for e in ev)
    if not (n_add or n_del):
        return [], 0, 0, []

    rows: list[tuple[str, str, str, str]] = []
    changed: list[int] = []
    seam = False
    for kind, oi, ni, _ in ev:
        if kind == "del":
            seam = True                          # 안내 문구 없이 이음매 점선만
            continue
        if kind == "add":
            changed.append(ni + 1)
        for first, cell in physical_rows(n_segs[ni], WRAP_COLS_DIFF):
            rows.append((kind + (" gap" if seam else ""),
                         "" if oi < 0 else (str(oi + 1) if first else CONT_MARK),
                         str(ni + 1) if first else CONT_MARK, cell))
            seam = False
    return rows, n_add, n_del, changed


def ranges(nums: list[int]) -> str:
    """[24,25,26,69] → '24-26, 69'"""
    out, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        out.append(str(nums[i]) if i == j else f"{nums[i]}-{nums[j]}")
        i = j + 1
    return ", ".join(out)


def pack(prefix: str, tokens: list[str], size: float = 7.4,
         usable: float = 540.0) -> list[str]:
    """토큰들을 12pt 한 행짜리 문장 여러 개로 나눈다 — CSS 줄바꿈에 맡기면 행 높이
    계산이 깨지므로 직접 접는다. 한글을 2칸으로 세어 넉넉하게."""
    def w(s: str) -> float:
        return sum(cw(c) for c in s) * size * 0.604

    lines, cur = [], prefix
    for t in tokens:
        cand = cur + t if cur.endswith(" ") or not cur else f"{cur} · {t}"
        if w(cand) > usable and cur.strip() not in ("", prefix.strip()):
            lines.append(cur)
            cur = "    " + t
        else:
            cur = cand
    lines.append(cur)
    return lines


def change_log(base_label: str) -> list[str]:
    """기준점 이후 필사 대상 파일을 건드린 커밋 제목 — --note 를 안 준 발급의 폴백 요약.
    커밋 문체는 알아듣기 어렵다는 지적(2026-08-02)에 따라 접두사(fix: 등)를 벗기고
    자동 커밋(chore)은 뺀다. 기본은 발급자가 --note 로 쓰는 보통 문장이다."""
    m = re.search(r"git (\S+)", base_label) or re.search(r"\(([0-9a-f]{6,40})", base_label)
    if not m:
        return []
    # 필사 대상 파일 그대로 넘긴다 — 최상위 접두사(signus)로 뭉치면 EXCLUDED 인
    # gen.py/lab.py 만 고친 커밋 제목이, 지면에 없는 변경인데도 표지 요약에 실린다
    paths = [rel for _, rel, _ in FILES]
    r = subprocess.run(["git", "-C", str(ROOT), "log", "--reverse", "--format=%s",
                        f"{m.group(1)}..HEAD", "--", *paths], capture_output=True, text=True)
    if r.returncode != 0:
        return []
    subs = [s for s in r.stdout.splitlines() if s.strip() and not s.startswith("chore")]
    return [re.sub(r"^\w+(\([^)]*\))?!?:\s*", "", s) for s in subs]


def build_diff(base: dict[str, str], base_label: str,
               notes: tuple[str, ...] | list[str] = ()) -> str | None:
    """바뀐 파일은 통째로 인쇄한다. 표지 한 장(무엇이 왜 바뀌었나 + 파일 표)을 앞에 둔다
    (2026-08-02 사용자 요청 — 종이 한 장보다 한눈에 보이는 쪽이 낫다)."""
    cb_map, cb_total = codebook_page_map()
    per_file = []
    for _, rel, _ in FILES:
        new = read_src(ROOT / rel)
        old = base.get(rel, "")           # 없으면 신규 파일 → 전부 추가
        if old == new:
            continue
        rows, n_add, n_del, changed = diff_rows(rel, old, new)
        if rows:
            per_file.append((rel, rows, n_add, n_del, not old, changed))
    gone = sorted(set(base) - {rel for _, rel, _ in FILES})
    if not per_file and not gone:
        return None

    # 요약은 --note(보통 문장)가 기본, 없으면 커밋 제목(접두사 제거)으로 폴백.
    # 표지는 격자가 아니라 자연 줄바꿈이 되므로 폭 자르기가 필요 없다.
    what = list(notes) or change_log(base_label)

    flow: list[tuple[str, str, str, str, str]] = []   # (owner, kind, o, n, cell)
    starts: dict[str, int] = {}
    for rel, rows, n_add, n_del, is_new, changed in per_file:
        pos = len(flow)
        if pos % LINES_PER_PAGE > LINES_PER_PAGE - 4:     # 파일 머리 고아 방지
            flow += [("", "blank", "", "", "")] * (LINES_PER_PAGE - pos % LINES_PER_PAGE)
        starts[rel] = len(flow) // LINES_PER_PAGE + 2     # +2: 1쪽은 표지
        chg = ranges(changed)
        flow.append((rel, "file", "", "", (
            f'{html.escape(rel)}{" (신규 파일)" if is_new else ""}'
            f' &#160;<span class="fa">+{n_add}</span>'
            + (f' <span class="fd">−{n_del}</span>' if n_del else '')
            + f' &#160;·&#160; 코드북 {cb_map.get(rel, 0)}쪽'
            + (f' &#160;·&#160; <span class="fc">새로 쓸 줄 {chg[:110]}'
               f'{"…" if len(chg) > 110 else ""}</span>' if not is_new else ''))))
        flow += [(rel, *r) for r in rows]
        flow.append((rel, "blank", "", "", ""))
    total_pages = 1 + (len(flow) + LINES_PER_PAGE - 1) // LINES_PER_PAGE

    tot_a = sum(f[2] for f in per_file)
    tot_d = sum(f[3] for f in per_file)
    frows = "".join(
        f'<div class="drow2"><span class="tbox">☐</span>'
        f'<span class="tf">{html.escape(rel)}{" (신규)" if is_new else ""}</span>'
        f'<span class="dadd">+{n_add}</span>'
        f'<span class="ddel">{f"−{n_del}" if n_del else ""}</span>'
        f'<span class="tp">{starts[rel]}쪽</span>'
        f'<span class="tn">코드북 {cb_map.get(rel, 0)}쪽</span></div>'
        for rel, _, n_add, n_del, is_new, _ in per_file)
    for rel in gone:
        frows += (f'<div class="drow2"><span class="tbox">☐</span>'
                  f'<span class="tf del">{html.escape(rel)} — 삭제됨 (필사본에서 지운다)</span>'
                  '<span></span><span></span><span></span><span></span></div>')
    cover = f"""<section class="page cover">
  <div class="ctitle">signus <span class="cgray">변경분</span></div>
  <div class="csub">바뀐 파일은 통째로 실었습니다 — 그 파일 쪽만 새로 쓰면 됩니다</div>
  <div class="cmeta">{html.escape(base_label)} &#160;→&#160; {date.today().isoformat()}
    · {git_head()} &#160;&#160;|&#160;&#160; {len(per_file)}개 파일 ·
    <b>+{tot_a}</b> / −{tot_d} 줄 · 코드 {total_pages - 1}쪽
    &#160;&#160;|&#160;&#160; 새 코드북 {cb_total}쪽</div>
  <div class="cwhat"><div class="wt">바뀐 내용</div>
    {"".join(f"<div>·&#160; {html.escape(w)}</div>" for w in what)
     or "<div>·&#160; (요약 없음 — 발급 시 --note 를 빠뜨렸다)</div>"}</div>
  <div class="toc">{frows}</div>
  <div class="cnote"><span class="lg add">초록</span> = 새로 쓸 줄 (오른쪽 숫자가 새 줄번호)
    &#160;·&#160; 색 없는 줄 = 그대로 &#160;·&#160; <span class="lg cut">붉은 점선</span> =
    그 자리에서 몇 줄 사라짐 &#160;·&#160; <span class="mono">{CONT_MARK}</span> =
    윗줄에 붙는 이어짐 (원래 한 줄이니 붙여서 입력)</div>
</section>"""

    pages = [cover]
    for pno, chunk in enumerate(paginate(flow), 2):
        owners: list[str] = []
        for r in chunk:
            if r[0] and (not owners or owners[-1] != r[0]):
                owners.append(r[0])
        owner = owners[0] if owners else ""
        started = any(r[0] == owner and r[1] == "file" for r in chunk)
        body = []
        for _, kind, o_no, n_no, cell in chunk:
            base_kind = kind.split()[0]
            if base_kind in ("h1", "h2", "h3", "fl", "file", "blank"):
                body.append(f'<div class="{kind}">{cell}</div>')
                continue
            body.append(
                f'<div class="dn {kind}">{html.escape(o_no)}</div>'
                f'<div class="dn {kind}">{html.escape(n_no)}</div>'
                f'<div class="dm {kind}">{"+" if base_kind == "add" else ""}</div>'
                f'<div class="cd {kind}">{cell}</div>')
        title = owner + ("" if started or not owner else " (이어서)")
        if len(owners) == 2:
            title += f" → {owners[1]}"
        elif len(owners) > 2:
            title += f" → {owners[-1]} (외 {len(owners) - 2}개)"
        pages.append(page_html(title or "변경분",
                               f"변경분 &#160;·&#160; {pno}/{total_pages}",
                               f'<div class="dg">\n{"".join(body)}\n</div>', pno))
    return SHELL.replace("%%TITLE%%", "signus 코드 변경분") \
                .replace("%%BODY%%", "\n".join(pages))


# ── 페이지 셸 ─────────────────────────────────────────────────────────────────
def page_html(rel: str, right: str, body: str, page_no: int) -> str:
    return f"""<section class="page">
  <header><span class="hl">{html.escape(rel)}</span><span class="hr">{right}</span></header>
  {body}
  <footer><span>signus 필사용 코드북</span><span class="pg">— {page_no} —</span></footer>
</section>"""


SHELL = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><title>%%TITLE%%</title>
<style>
  @page { size: A4 portrait; margin: 0; }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; background: #fff; }
  body { font-family: Menlo, "SF Mono", monospace; color: #111;
         -webkit-font-smoothing: antialiased; }
  .page { width: 210mm; height: 297mm; padding: 10mm 7mm 8mm 11mm;
          display: flex; flex-direction: column; overflow: hidden; background: #fff; }
  .page + .page { break-before: page; }

  header { display: flex; justify-content: space-between; align-items: baseline;
           font-size: 7.6pt; color: #555; border-bottom: 0.6pt solid #999;
           padding-bottom: 3pt; margin-bottom: 6pt; flex: none; }
  .hl { font-weight: 700; color: #000; font-size: 8.4pt; letter-spacing: -0.1pt; }
  .hr { font-family: "Apple SD Gothic Neo", Menlo, sans-serif; }
  footer { margin-top: auto; padding-top: 4pt; border-top: 0.4pt solid #ccc;
           display: flex; justify-content: space-between; flex: none;
           font-family: "Apple SD Gothic Neo", sans-serif; font-size: 7pt; color: #888; }
  .pg { font-variant-numeric: tabular-nums; color: #333; }

  /* 코드북: 줄번호 | 코드 */
  .grid { display: grid; grid-template-columns: 25pt 1fr; font-size: 8.5pt;
          line-height: 12pt; flex: none; }
  .ln { text-align: right; color: #aaa; font-size: 7.2pt; padding-right: 4pt;
        margin-right: 6pt; border-right: 0.4pt solid #ddd;
        font-variant-numeric: tabular-nums; white-space: nowrap; overflow: hidden; }
  /* 변경분: 옛줄 | 새줄 | 부호 | 코드 (부호는 코드 칸 밖 — 인덴트 불변) */
  .dg { display: grid; grid-template-columns: 15pt 15pt 10pt 1fr; font-size: 8.5pt;
        line-height: 12pt; flex: none; }
  .dn { text-align: right; font-size: 6.8pt; color: #b0b0b0; padding-right: 1.5pt;
        font-variant-numeric: tabular-nums; white-space: nowrap; overflow: hidden; }
  .dm { text-align: center; font-size: 8pt; font-weight: 700; color: #444;
        border-right: 0.4pt solid #ccc; }
  .dg > .cd { padding-left: 4pt; }
  .dn.add, .dm.add { color: #0a6b2d; font-weight: 700; background: #dff3e4; }
  .cd.add { background: #f2fbf5; }
  /* 없어진 줄: 코드도 안내문도 싣지 않는다. 이음매에 붉은 점선만 — 행을 안 쓴다 */
  .gap { box-shadow: inset 0 1.1pt 0 -0.2pt #cf8f8f; }

  /* 머리 블록 · 파일 띠 — 모두 정확히 12pt 한 행 */
  .h1, .h2, .h3, .fl, .file, .blank { grid-column: 1 / -1; line-height: 12pt;
        overflow: hidden; white-space: nowrap;
        font-family: "Apple SD Gothic Neo", sans-serif; }
  .h1 { font-size: 10pt; font-weight: 700; color: #000; }
  .h1s { font-size: 8pt; font-weight: 400; color: #777; }
  /* 줄 높이를 1pt 도 늘리면 60행/쪽 계산이 깨진다 → 괘선은 전부 inset shadow 로 */
  .h2 { font-size: 7.4pt; color: #555; box-shadow: inset 0 -1px 0 #111; }
  .h3 { font-size: 7.2pt; color: #444; }
  .fl { font-size: 7.4pt; color: #111;
        font-family: Menlo, "Apple SD Gothic Neo", monospace; }
  .fl.del { color: #961a1a; }
  .file { font-family: Menlo, monospace; font-size: 8.2pt; font-weight: 700;
          background: #e6e6e6; padding-left: 3pt; box-shadow: inset 0 1px 0 #888; }
  .fa { color: #0a6b2d; }
  .fd { color: #961a1a; }
  .fc { font-family: "Apple SD Gothic Neo", sans-serif; font-size: 7pt;
        font-weight: 400; color: #444; }

  .cd { white-space: pre; font-family: Menlo, "Apple SD Gothic Neo", monospace; }
  .ig { display: inline-block; width: 4ch; height: 1em; vertical-align: baseline;
        border-left: 0.4pt dotted #c8c8c8; }
  .cd > .ig:first-child { border-left: none; }
  .pad { display: inline-block; }

  /* 흑백 인쇄 최적화 배색 */
  .c { color: #7a7a7a; font-style: italic; }
  .sd { color: #6a6a6a; font-style: italic; }
  .s { color: #3a3a3a; }
  .k, .nf, .nt { color: #000; font-weight: 700; }
  .nd { color: #444; font-weight: 700; }
  .nb { color: #222; }
  .na { color: #444; }
  .m, .o { color: #2a2a2a; }

  /* 표지 + 목차 */
  .cover { padding: 22mm 14mm 12mm 14mm; }
  .ctitle { font-size: 34pt; font-weight: 700; letter-spacing: -1pt; }
  .csub { font-family: "Apple SD Gothic Neo", sans-serif; font-size: 11pt;
          color: #333; margin-top: 3pt; }
  .cmeta { font-family: "Apple SD Gothic Neo", sans-serif; font-size: 8.5pt;
           color: #666; margin-top: 10pt; padding-bottom: 8pt;
           border-bottom: 1pt solid #111; }
  .cnote { font-family: "Apple SD Gothic Neo", sans-serif; font-size: 8pt;
           color: #444; margin-top: 8pt; line-height: 1.6; }
  .mono { font-family: Menlo, monospace; font-weight: 700; color: #000; }
  .lg { padding: 0 3pt; font-weight: 700; }
  .lg.add { background: #dff3e4; color: #0a6b2d; }
  .lg.cut { background: #f7eaea; color: #8a3a3a; font-weight: 400; }
  .toc { margin-top: 10pt; }
  .tsec { font-family: "Apple SD Gothic Neo", sans-serif; font-size: 9pt;
          font-weight: 700; margin: 9pt 0 3pt; }
  .trow { display: grid; grid-template-columns: 12pt 130pt 1fr 34pt 30pt;
          align-items: baseline; font-size: 8pt; line-height: 14pt;
          border-bottom: 0.3pt dotted #ddd; }
  .drow { display: grid; grid-template-columns: 12pt 1fr 34pt 40pt 40pt 62pt 30pt;
          align-items: baseline; font-size: 8pt; line-height: 15pt;
          border-bottom: 0.3pt dotted #ddd; }
  .tbox { color: #999; }
  .tf { font-family: Menlo, monospace; font-size: 7.8pt; }
  .td { font-family: "Apple SD Gothic Neo", sans-serif; color: #666; font-size: 7.6pt; }
  .tn, .tp { font-family: "Apple SD Gothic Neo", sans-serif; text-align: right;
             color: #666; font-variant-numeric: tabular-nums; }
  .tp { font-weight: 700; color: #111; }
  /* 변경분 표지 */
  .cgray { color: #999; font-weight: 300; }
  .cwhat { font-family: "Apple SD Gothic Neo", sans-serif; font-size: 10pt; margin-top: 12pt;
           line-height: 1.8; color: #111; }
  .cwhat .wt { font-size: 8pt; font-weight: 700; color: #888; letter-spacing: 3pt;
               margin-bottom: 2pt; }
  .drow2 { display: grid; grid-template-columns: 14pt 1fr 42pt 42pt 46pt 74pt;
           align-items: baseline; font-size: 9pt; line-height: 19pt;
           border-bottom: 0.3pt dotted #ddd; }
  .tf.del { color: #961a1a; text-decoration: line-through; }
  .dadd { text-align: right; color: #0a6b2d; font-weight: 700;
          font-variant-numeric: tabular-nums; }
  .ddel { text-align: right; color: #961a1a; font-weight: 700;
          font-family: "Apple SD Gothic Neo", Menlo, sans-serif;
          font-variant-numeric: tabular-nums; }
  .zero { color: #c4c4c4; font-weight: 400; }
</style></head>
<body>
%%BODY%%
</body></html>
"""


# ── 스냅샷 / git ──────────────────────────────────────────────────────────────
def git_head() -> str:
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"],
                               capture_output=True, text=True).stdout.strip()
        return out + ("+수정중" if dirty else "")
    except Exception:
        return "git 없음"


def sections_at(rev: str) -> list[str]:
    """rev 시점 tools/codebook.py 의 SECTIONS 파일 목록 — 그때 필사 대상이던 파일.
    현재 FILES 만 보면 그 뒤 목록에서 뺀 파일(gen.py 제외가 그 예)이 기준점에 안 실려,
    변경분 표지의 '삭제됨 (필사본에서 지운다)' 안내가 조용히 빠진다."""
    r = subprocess.run(["git", "-C", str(ROOT), "show", f"{rev}:tools/codebook.py"],
                       capture_output=True, text=True)
    m = re.search(r"^SECTIONS.*?^\]", r.stdout, re.S | re.M) if r.returncode == 0 else None
    return re.findall(r'\("([^"]+)", "', m.group(0)) if m else []


def load_baseline(spec: str) -> tuple[dict[str, str], str]:
    if spec.startswith("git:"):
        rev = spec[4:]
        base = {}
        for rel in sorted({rel for _, rel, _ in FILES} | set(sections_at(rev))):
            r = subprocess.run(["git", "-C", str(ROOT), "show", f"{rev}:{rel}"],
                               capture_output=True, text=True)
            if r.returncode == 0:
                base[rel] = r.stdout.replace("\t", " " * TAB)
        return base, f"git {rev}"
    if not STATE_JSON.exists():
        return {}, "기준점 없음"
    st = json.loads(STATE_JSON.read_text(encoding="utf-8"))
    base = {}
    for rel in st["files"]:
        p = BASE_DIR / rel
        if p.exists():
            base[rel] = p.read_text(encoding="utf-8")
    return base, f"기준점 {st['date']} ({st.get('head', '?')})"


def snap() -> None:
    if BASE_DIR.exists():
        shutil.rmtree(BASE_DIR)
    files = {}
    for _, rel, _ in FILES:
        src = read_src(ROOT / rel)
        dst = BASE_DIR / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(src, encoding="utf-8")
        files[rel] = hashlib.sha1(src.encode()).hexdigest()[:12]
    STATE_JSON.write_text(json.dumps(
        {"date": date.today().isoformat(), "head": git_head(), "files": files},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"기준점 갱신: {len(files)}개 파일 → {STATE_JSON.relative_to(ROOT)}")


# ── PDF 출력 ──────────────────────────────────────────────────────────────────
def find_chrome() -> str:
    """헤드리스 브라우저 실행 파일. CHROME 환경변수가 최우선 — 후보에 없는 곳에 깔린 보드에서도
    `CHROME=/usr/bin/chromium python tools/codebook.py all` 로 그대로 발급된다."""
    for cand in (os.environ.get("CHROME"), *CHROME_CANDS):
        if cand and (found := shutil.which(cand)):
            return found
    raise SystemExit("헤드리스 크롬을 찾지 못했습니다 — chromium 을 설치하거나"
                     "(apt install chromium) CHROME 환경변수로 실행 파일 경로를 주세요")


def to_pdf(html_text: str, out: Path) -> None:
    tmp = STATE_DIR / (out.stem + ".html")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(html_text, encoding="utf-8")
    subprocess.run([find_chrome(), "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    "--run-all-compositor-stages-before-draw",
                    "--virtual-time-budget=20000",
                    f"--print-to-pdf={out}", tmp.as_uri()], check=True, capture_output=True)
    n = len(re.findall(rb"/Type\s*/Page[^s]", out.read_bytes()))
    print(f"{out.relative_to(ROOT)} — {n}쪽, {out.stat().st_size // 1024}KB")


# ── 검증: 인쇄물 == 원본 ──────────────────────────────────────────────────────
def cell_text(cell: str) -> tuple[str, int]:
    t = re.sub(r'<span class="ig"[^>]*></span>', "\0", cell)
    t = re.sub(r'<span class="pad"[^>]*></span>', "", t)
    t = re.sub(r"<[^>]+>", "", t)
    t = html.unescape(t).replace(" ", " ")
    return t.replace("\0", ""), t.count("\0")


def verify(base_spec: str = "baseline") -> int:
    fail = 0
    for _, rel, _ in FILES:
        src = read_src(ROOT / rel)
        src_lines = split_lines(src)
        segs = segments_of(src, Path(rel).suffix)
        if ["".join(t for _, t in ln) for ln in segs] != src_lines:
            print(f"LEX  {rel}: 렉서가 문자를 잃었다")
            fail += 1
        # 코드북 행 → 논리 줄 복원
        logical, cur, guides = [], [], 0
        for num, cell in codebook_rows(rel, src):
            txt, g = cell_text(cell)
            if num != CONT_MARK:
                if cur:
                    logical.append(("".join(cur), guides))
                cur, guides = [txt], g
            else:
                cur.append(txt)
        if cur:
            logical.append(("".join(cur), guides))
        if len(logical) != len(src_lines):
            print(f"BOOK {rel}: 줄 수 {len(logical)} != {len(src_lines)}")
            fail += 1
        for i, (body, g) in enumerate(logical):
            exp = src_lines[i].rstrip()
            lead = len(exp) - len(exp.lstrip(" "))
            if " " * lead + body != exp or g != -(-lead // TAB):
                print(f"BOOK {rel}:{i + 1}\n  got {' ' * lead + body!r} guides={g}\n"
                      f"  exp {exp!r} lead={lead}")
                fail += 1

    # 변경분: 모든 행이 원본 줄과 일치하고, 바뀐 줄이 하나도 빠지지 않았는가
    base, label = load_baseline(base_spec)
    for _, rel, _ in FILES:
        new = read_src(ROOT / rel)
        old = base.get(rel, "")
        if old == new:
            continue
        n_lines = split_lines(new)
        rows, n_add, _n_del, changed = diff_rows(rel, old, new)
        seen_add: set[int] = set()
        groups: list[tuple[str, list[str]]] = []           # (새 줄번호, 셀들)
        for kind, _o_no, n_no, cell in rows:
            if n_no == CONT_MARK:
                groups[-1][1].append(cell)
            else:
                groups.append((n_no, [cell]))
                if kind.split()[0] == "add":
                    seen_add.add(int(n_no))
        # 바뀐 파일은 통째로 실리므로, 행이 새 파일 전 줄을 순서대로 덮어야 한다
        if [int(n) for n, _ in groups] != list(range(1, len(n_lines) + 1)):
            print(f"DIFF {rel}: 파일 전체가 실리지 않았다 "
                  f"({len(groups)}행 vs {len(n_lines)}줄)")
            fail += 1
        for no, cells in groups:
            body = "".join(cell_text(c)[0] for c in cells)
            exp = n_lines[int(no) - 1].rstrip()
            lead = len(exp) - len(exp.lstrip(" "))
            if " " * lead + body != exp:
                print(f"DIFF {rel} L{no}\n  got {' ' * lead + body!r}\n  exp {exp!r}")
                fail += 1
        if len(seen_add) != n_add or sorted(seen_add) != changed:
            print(f"DIFF {rel}: 강조된 줄 {len(seen_add)} != 실제 추가 {n_add}")
            fail += 1

    print(f"\n{len(FILES)}개 파일 · 기준점 [{label}] — "
          f"{'전부 일치 ✓' if not fail else str(fail) + '건 불일치 ✗'}")
    return fail


# ── CLI ───────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description="필사용 코드북/변경분 발급")
    ap.add_argument("cmd", choices=("build", "diff", "verify", "snap", "all"))
    ap.add_argument("--from", dest="base", default="baseline",
                    help="변경분 기준점: baseline | git:<rev>")
    ap.add_argument("--snap", action="store_true", help="build 후 기준점도 갱신")
    ap.add_argument("--note", action="append", default=[],
                    help="변경분 머리의 '바뀐 내용'에 한 줄 추가 (여러 번 가능). 커밋 제목은"
                         " 자동으로 실리므로 커밋 전 변경을 설명할 때만 쓴다")
    a = ap.parse_args()
    DOCS.mkdir(exist_ok=True)

    missing = [rel for _, rel, _ in FILES if not (ROOT / rel).exists()]
    if missing:
        print("대상 파일이 없습니다 — tools/codebook.py 의 SECTIONS 를 갱신하세요:")
        for m in missing:
            print("  ", m)
        return 1
    listed = {rel for _, rel, _ in FILES}
    if listed & EXCLUDED:
        print("EXCLUDED 파일이 SECTIONS 에 들어 있습니다 — 인쇄물에 실리면 안 됩니다:")
        for rel in sorted(listed & EXCLUDED):
            print("  ", rel)
        return 1
    loose = sorted(str(p.relative_to(ROOT)) for suf in LEXERS
                   for p in ROOT.glob(f"signus/**/*{suf}")
                   if str(p.relative_to(ROOT)) not in listed | EXCLUDED)
    if loose:
        print("signus/ 에 SECTIONS 에도 EXCLUDED 에도 없는 파일이 있습니다 — 실을지 뺄지 정하세요:")
        for rel in loose:
            print("  ", rel)
        return 1

    if a.cmd != "snap":
        fails = verify(a.base)
        if a.cmd == "verify":
            return 1 if fails else 0
        if fails:
            print("검증 실패 — 발급 중단 (인쇄물이 원본과 다릅니다)")
            return 1

    if a.cmd in ("build", "all"):
        to_pdf(build_codebook(), DOCS / "signus-코드-필사용.pdf")
    if a.cmd in ("diff", "all"):
        base, label = load_baseline(a.base)
        if not base:
            print(f"기준점이 없어 변경분을 만들 수 없습니다 ({label}) — 먼저 snap 하세요")
        else:
            doc = build_diff(base, label, a.note)
            if doc is None:
                print("변경 없음 — 변경분 PDF 생략")
            else:
                to_pdf(doc, DOCS / "signus-코드-변경분.pdf")
    if a.cmd == "snap" or a.snap:
        snap()
    return 0


if __name__ == "__main__":
    sys.exit(main())
