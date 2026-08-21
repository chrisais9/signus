#!/usr/bin/env python3
"""참고편(그림 읽는 법) 생성 — probe_pdf.py 가 본문 뒤에 붙인다. 맥/보드 전용.

합성 신호 6종을 만들어 프로브의 단계 2/4/5 를 **실제로 돌려** 그 출력을 그대로 싣는다.
손으로 그린 그림이 아니라 발급 시점의 실행 결과라, 코드가 바뀌면 참고편도 같이 바뀐다.
각 시나리오의 출구(E1/E5/E6/검출)도 find_bursts 를 직접 불러 확인한 값이다.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from signus import dsp  # noqa: E402
from signus.gen import GenParams, generate  # noqa: E402
from signus.sigio import Meta, make_name, write  # noqa: E402

FS, N = 1e6, 65536
PROBE = ROOT / "probes" / "sigc.py"


def _qpsk(fc: float, baud: float, seed: int) -> np.ndarray:
    p = GenParams(mod="qpsk", n_symbols=int(N * baud / FS) + 128, fs=FS, baud=baud,
                  snr=90.0, fc=fc, seed=seed, dtype="f32")
    x, _ = generate(p)
    x = x[:N]
    return x / np.sqrt(np.mean(np.abs(x) ** 2))


def _mask(blen: int, gap: int) -> np.ndarray:
    m = np.zeros(N, bool)
    s = 0
    while s < N:
        m[s:s + blen] = True
        s += blen + gap
    return m


def _noise(seed: int, p: float = 0.01) -> np.ndarray:
    r = np.random.default_rng(seed)
    return np.sqrt(p / 2) * (r.standard_normal(N) + 1j * r.standard_normal(N))


def _impulses(x: np.ndarray) -> np.ndarray:
    for s in range(0, N, 12000):
        x[s + 3000] += 30.0
    return x


# (키, 제목, 무슨 신호인가, 단계 4 도 실을까, 이렇게 보이면 무슨 뜻인가)
SCENARIOS = [
    ("A", "기준 — 건강한 버스트열 (검출 성공)",
     "협대역 QPSK 한 개(+200 kHz, 심볼레이트 50 k)가 6000샘플 켜지고 6000샘플 꺼지기를"
     " 반복. 잡음 대비 20 dB. 다른 방사체는 없다.",
     True,
     "실캡처가 이 모양이면 find_bursts 는 버스트를 잡는다. 그런데도 통짜가 나온다면 4줄이"
     " 아니라 단계 6(실제 출력)을 봐야 한다 — 병합이나 1열 드랍이 원인이다.",
     lambda: _qpsk(2e5, 5e4, 3) * _mask(6000, 6000) + _noise(11)),
    ("B", "★ 유력 가설 — 연속 방사체가 섞인 캡처 (E1 베토)",
     "A 와 같은 버스트열(세기는 절반)에, 레코드 내내 꺼지지 않는 연속 QPSK(−150 kHz,"
     " 심볼레이트 80 k)가 함께 들어온다. 합성 시험에 없던 조합이다.",
     True,
     "d 줄에 pd100(듀티 100%)인 방사체가 있고 kc 가 0 보다 크면 이 경우다. 버스트는 단계"
     " 4·5 에 멀쩡히 살아 있는데, 광대역 포락선만 평평해져 셀 경로에 닿기도 전에 막힌다.",
     lambda: (_qpsk(-1.5e5, 8e4, 5) + 0.5 * _qpsk(2e5, 5e4, 6) * _mask(6000, 6000)
              + _noise(12))),
    ("C", "고듀티 95% — 켜진 시간이 너무 길다 (E5)",
     "A 와 같은 신호인데 12000샘플 켜지고 600샘플만 꺼진다. 빈별 바닥이 신호를 '늘 있는 것'"
     " 으로 학습해 버려 점수가 서지 않는다.",
     False,
     "a 줄 ed 가 90 이상이고 c 줄 r0(런 없음)이면 이 경우다. 듀티 90% 까지는 잡히고 95%"
     " 부터 무너지는 것이 실측이다 — 설계 한계이지 오작동이 아니다.",
     lambda: _qpsk(2e5, 5e4, 3) * _mask(12000, 600) + _noise(13)),
    ("D", "약한 협대역 — 잡히긴 하는데 여유가 6 dB (경계)",
     "5빈짜리 아주 좁은 QPSK(+100 kHz, 심볼레이트 20 k)가 잡음에 겨우 묻히지 않을 세기로"
     " 들어온다. 켜짐/꺼짐 주기는 A 와 같다.",
     False,
     "c 줄 sb 가 6 안팎이면 경계다(건강한 A 는 23). 여기서 몇 dB 만 더 낮아지면 통짜로"
     " 넘어간다. 잡히긴 해도 이 여유로는 신뢰하지 말고 단계 5 를 같이 볼 것.",
     lambda: 0.10 * _qpsk(1e5, 2e4, 8) * _mask(6000, 6000) + _noise(14)),
    ("E", "임펄스 간섭 — 모든 버스트에 스파이크 (E6)",
     "A 와 같은 버스트열의 버스트마다 한 샘플짜리 강한 임펄스(+30)가 박혀 있다. 전원 잡음이나"
     " 스위칭 잡음이 이렇게 보인다.",
     False,
     "a 줄 sp 가 30 을 넘고 c 줄 sk 가 r 과 비슷하면 이 경우다. 후보를 못 찾은 게 아니라"
     " 찾아 놓고 '임펄스가 만든 가짜' 로 판단해 버린 것이다 — 단계 6 의 피크/평균이 증거다.",
     False),
    ("F", "연속 방사체만 — 버스트가 애초에 없다 (E1, 정상)",
     "꺼지지 않는 QPSK 한 개(−100 kHz, 심볼레이트 100 k)만 있다. 켜고 꺼지는 사건이 없으므로"
     " 레코드 전체가 곧 신호다.",
     True,
     "c 줄 r0 이고 d 줄 pd100·kc 가 크면 이 경우다. 이때의 통짜는 오작동이 아니라 정답이다"
     " — 이 캡처에서 찾을 버스트가 없다.",
     lambda: _qpsk(-1e5, 1e5, 9) + _noise(16)),
]


def _build(key: str):
    for k, _t, _d, _s4, _m, fn in SCENARIOS:
        if k == key:
            if k == "E":
                return _impulses(_qpsk(2e5, 5e4, 3) * _mask(6000, 6000) + _noise(15))
            return fn()
    raise KeyError(key)


def _run(path: Path, step: str | None) -> list[str]:
    args = [sys.executable, str(PROBE), str(path)] + ([step] if step else [])
    r = subprocess.run(args, capture_output=True, text=True,
                       env={"PYTHONPATH": str(ROOT)}, check=True)
    return r.stdout.rstrip("\n").splitlines()


def _cw(s: str) -> int:
    return sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in s)


def wrap(text: str, cols: int = 104, indent: str = "") -> list[str]:
    """폭에 맞춰 직접 접는다 — CSS 줄바꿈에 맡기면 60행/쪽 계산이 깨진다."""
    out, cur = [], indent
    for word in text.split():
        cand = f"{cur} {word}" if cur.strip() else indent + word
        if _cw(cand) > cols and cur.strip():
            out.append(cur)
            cur = indent + word
        else:
            cur = cand
    if cur.strip():
        out.append(cur)
    return out


def _is_art(line: str) -> bool:
    """한글이 없으면 그림 행 — 프로브의 안내문은 전부 한글이라 이걸로 갈린다."""
    return not any("가" <= ch <= "힣" for ch in line)


def rows() -> list[tuple[str, str]]:
    """[(css class, 텍스트)] — 한 항목이 정확히 12pt 한 행."""
    out: list[tuple[str, str]] = []

    def t(cls: str, *lines: str) -> None:
        out.extend((cls, ln) for ln in lines)

    t("rh1", "참고편 — 단계별로 무엇이 보여야 하는가")
    for ln in wrap("아래 그림은 전부 이 프로브를 합성 신호에 실제로 돌린 출력이다."
                   " 장비에서 나오는 그림과 같은 코드·같은 조판이므로, 모양을 그대로"
                   " 견주면 된다. 가로축은 언제나 레코드 전체를 100칸으로 줄인 것이고,"
                   " 캡처 길이가 달라도 모양은 비교할 수 있다."):
        t("rp", ln)
    t("rb", "")
    for ln in wrap("곡선 그림(단계 2·5) 읽는 법: 한 칸은 여러 샘플을 묶은 것이라 그 구간의"
                   " 최솟값과 최댓값을 같이 그린다. #=그 구간이 내내 이 높이 위, +=봉우리만"
                   " 이 높이 위, 빈칸=아래. 가로로 이어지는 −−− 선은 문턱이고 오른쪽에 이름이"
                   " 붙는다. 버스트열이면 #기둥과 빈칸이 번갈아 나오는 '계단' 이 보여야 한다."):
        t("rp", ln)
    t("rb", "")
    for ln in wrap("워터폴 그림(단계 4) 읽는 법: 세로가 주파수(아래 −fs/2, 위 +fs/2),"
                   " 가로가 시간이다. 진하기는 그 칸이 자기 주파수의 바닥보다 몇 배 센가를"
                   " 뜻한다(연한 . 부터 진한 @ 까지). 세로로 짧게 끊긴 진한 자국=버스트,"
                   " 가로로 안 끊기고 쭉 이어지는 띠=연속 방사체다."):
        t("rp", ln)
    t("rb", "")
    for ln in wrap("각 시나리오 머리의 '요약 4줄' 은 그 신호를 프로브에 넣었을 때 나오는"
                   " 회신이다. 실캡처의 4줄과 숫자대를 견주면 어느 시나리오에 가까운지"
                   " 바로 짚인다."):
        t("rp", ln)

    with tempfile.TemporaryDirectory() as td:
        for key, title, desc, want4, mean, _fn in SCENARIOS:
            x = _build(key)
            meta = Meta(FS, "iq", "f32")
            path = Path(td) / make_name(f"ref{key}", meta)
            write(str(path), x, meta)
            xa = dsp.analytic(x)
            fb = dsp.find_bursts(xa - xa.mean(), FS)
            full = fb == [(0, N)]

            out.append(("rpage", ""))          # 시나리오는 새 쪽에서 시작
            t("rh2", f"{key}. {title}")
            for ln in wrap(desc):
                t("rp", ln)
            t("rb", "")
            t("rh3", "이 신호를 프로브에 넣으면 나오는 4줄")
            for ln in _run(path, None):
                t("ra", "  " + ln)
            t("rb", "")
            res = (f"find_bursts → 통짜 [(0, {N})] = 버스트 미검출" if full
                   else f"find_bursts → 버스트 {len(fb)}개  {str(fb[:3])[:56]}"
                        f"{'...' if len(fb) > 3 else ''}")
            t("rh3", res)
            t("rb", "")

            for step, name in (("2", "단계 2 — 광대역 포락선 (여기서 막히면 E1 베토)"),
                               *((("4", "단계 4 — 빈별 바닥 대비 비율 (find_bursts 가 보는 그림)"),)
                                 if want4 else ()),
                               ("5", "단계 5 — 열 점수와 문턱 (봉우리가 hi 를 넘어야 후보)")):
                t("rh3", name)
                lines = _run(path, step)
                for ln in lines:
                    if _is_art(ln):
                        t("ra", "  " + ln)
                for ln in lines:
                    if not _is_art(ln):
                        for w in wrap(ln, indent="  "):
                            t("rq", w)
                t("rb", "")

            t("rh3", "실캡처가 이렇게 보이면")
            for ln in wrap(mean):
                t("rp", ln)
    return out


if __name__ == "__main__":     # 단독 실행: 행 수만 확인 (조판은 probe_pdf.py 가 한다)
    rs = rows()
    print(f"{len(rs)}행 · 약 {-(-len(rs) // 60)}쪽")
