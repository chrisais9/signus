"""CLI: analyze FILE / gen / sweep (acceptance harness) / serve."""

import argparse
import re
import sys
from zlib import crc32

import numpy as np

from .constellations import (
    GEN_MODS,
    MODS,
    bit_labels,
    demap_bits,
    demap_diff_bits,
    family,
    ideal_points,
    mod_order,
    mod_symmetry,
)
from .gen import GenParams, generate, save
from .pipeline import Result, analyze, analyze_file, survey_file
from .sigio import Meta, sidecar_read

_DTYPE_CHOICES = ("i8", "u8", "i16", "u16", "f32", "f64")

_SNR = {"bpsk": 14, "qpsk": 14, "8psk": 18, "16qam": 22, "64qam": 28}
_SNR_LOW = {"bpsk": 6, "qpsk": 8, "8psk": 12, "16qam": 16, "64qam": 22}


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


# --- BER scoring (blind reception leaves rotation/conjugation/offset ambiguity) --

def ber(symbols: np.ndarray, mod: str, tx_bits: np.ndarray) -> float:
    """Min BER over conjugation, M-fold rotation, and symbol offset (found by
    correlating against the reconstructed TX symbol stream)."""
    k = mod_order(mod).bit_length() - 1
    sym = mod_symmetry(mod)
    inv = np.empty(mod_order(mod), dtype=int)
    inv[bit_labels(mod)] = np.arange(mod_order(mod))
    labels = (tx_bits.reshape(-1, k) << np.arange(k - 1, -1, -1)).sum(1)
    tx = ideal_points(mod)[inv[labels]]

    best = 1.0
    for z in (symbols, np.conj(symbols)):
        # |correlation| is rotation-invariant -> find the symbol offset first
        offs = range(-32, 33)
        c = [np.abs(np.vdot(*_overlap(z, tx, o))) for o in offs]
        o = list(offs)[int(np.argmax(c))]
        a, b = _overlap(z, tx, o)
        if a.size < 100:
            continue
        tx_b = demap_bits(b, mod)
        for r in range(sym):
            rx_b = demap_bits(a * np.exp(-2j * np.pi * r / sym), mod)
            best = min(best, float(np.mean(rx_b != tx_b)))
    return best


def _overlap(rx: np.ndarray, tx: np.ndarray, off: int, crop: int = 20) -> tuple:
    """Aligned overlapping slices of rx[i] vs tx[i+off], edges cropped."""
    i0 = max(0, -off) + crop
    i1 = min(rx.size, tx.size - off) - crop
    if i1 <= i0:
        return rx[:0], tx[:0]
    return rx[i0:i1], tx[i0 + off:i1 + off]


def ber_bits(rx: np.ndarray, tx: np.ndarray, k: int, span: int = 8, crop: int = 20) -> float:
    """BER of an absolute bit stream over whole-SYMBOL shifts. No rotation search:
    differential transitions and FSK levels carry no phase ambiguity."""
    a, b = rx[: rx.size // k * k].reshape(-1, k), tx[: tx.size // k * k].reshape(-1, k)
    best = 1.0
    for sh in range(-span, span + 1):  # sh = symbol offset of rx within tx
        i0, i1 = max(0, -sh) + crop, min(len(a), len(b) - sh) - crop
        if i1 - i0 >= 400:
            best = min(best, float(np.mean(a[i0:i1] != b[i0 + sh:i1 + sh])))
    return best


# --- sweep: the acceptance grid ----------------------------------------------

# constellation-identical to their parent (differential is a demap option, not a class)
_EXPECT = {"dbpsk": "bpsk", "dqpsk": "qpsk", "pi4dqpsk": "qpsk"}
_DIFF = {"dbpsk": "dbpsk", "dqpsk": "dqpsk", "pi4dqpsk": "dqpsk"}


def _score(r: Result, p: GenParams, tx_bits: np.ndarray, core: bool) -> tuple[bool, str]:
    want = _EXPECT.get(p.mod, p.mod)
    fsk = family(p.mod) == "fsk"
    e_fc = abs(r.fc - p.fc)
    e_bd = abs(r.baud - p.baud) / p.baud
    checks = [r.mod == want, e_bd < 0.01, r.lock >= (50 if want.endswith("qam") else 60)]
    # pi4dqpsk's 4th power rotates pi/symbol, so the carrier estimate absorbs baud/8;
    # the shifted band then also biases the occupied-bandwidth rolloff. Bits stay exact.
    skew = p.mod == "pi4dqpsk"
    if not skew:
        checks.append(e_fc < max(300, 1e-4 * p.fs))
    if not fsk and not skew and p.rolloff >= 0.15 and not p.taps:
        checks.append(abs(r.rolloff - p.rolloff) < 0.08)
    if fsk:
        checks.append(abs(r.h - (0.5 if p.mod == "msk" else p.h)) < 0.15)
    b = -1.0
    if core:
        k = mod_order(r.mod).bit_length() - 1
        if fsk:
            b = ber_bits(r.bits, tx_bits, k)
        elif p.mod in _EXPECT:              # differential: transitions, no rotation search
            b = ber_bits(demap_diff_bits(r.symbols, _DIFF[p.mod]), tx_bits[k:], k)
        else:
            b = ber(r.symbols, want, tx_bits)
        checks.append(b <= (0.002 if (p.taps or p.mod == "fsk4") else 0.0))
    extra = f"h{r.h:.2f}" if fsk else f"roll{r.rolloff:.2f}"
    msg = (f"{r.mod:>6} fc{e_fc:7.1f} baud{e_bd * 100:5.2f}% {extra} lock{r.lock:5.1f}"
           + (" EQ" if r.eq_applied else "") + (f" ber{b:.4f}" if b >= 0 else ""))
    return all(checks), msg


def _grid(tier: str, seeds: int) -> list[tuple[str, bool, GenParams]]:
    """(label, is_core, params) cases. CORE must pass 100%; STRETCH is report-only."""
    fs, cases = 1e6, []
    for mod in MODS:
        for ratio in (10.0, 7.69):
            for sd in range(seeds):
                rng = np.random.default_rng(sd)
                cases.append((f"{mod}/r{ratio:g}/s{sd}", True, GenParams(
                    mod=mod, fs=fs, baud=fs / ratio, snr=_SNR[mod], fc=0.008 * fs,
                    phase=rng.uniform(0, 2 * np.pi), timing=rng.uniform(), seed=sd)))
    # real passband .pcm: fc > baud*(1+roll)/2 and sym*fc < fs/2 must both hold
    for mod, baud, fc in (("bpsk", fs / 10, 0.1 * fs), ("qpsk", fs / 10, 0.1 * fs),
                          ("8psk", fs / 20, 0.05 * fs), ("16qam", fs / 10, 0.1 * fs)):
        cases.append((f"{mod}/real", True, GenParams(
            mod=mod, fs=fs, baud=baud, snr=_SNR[mod], fc=fc, fmt="real", seed=0)))
    cases += [("qpsk/cfo0", True, GenParams(mod="qpsk", fs=fs, baud=fs / 10, snr=14, seed=1)),
              ("qpsk/dc", True, GenParams(mod="qpsk", fs=fs, baud=fs / 10, snr=14,
                                          fc=0.008 * fs, dc=0.3 + 0.3j, seed=2)),
              ("qpsk/pad", True, GenParams(mod="qpsk", fs=fs, baud=fs / 10, snr=14,
                                           fc=0.008 * fs, pad=0.5, seed=3)),
              ("32qam", True, GenParams(mod="32qam", fs=fs, baud=fs / 10, snr=24,
                                        fc=0.008 * fs, seed=0))]
    for mod, h in (("fsk2", 0.7), ("fsk4", 0.7), ("msk", 0.5)):  # frequency family
        for sd in range(min(seeds, 2)):
            cases.append((f"{mod}/s{sd}", True, GenParams(
                mod=mod, fs=fs, baud=fs / 10, snr=18, h=h, seed=sd)))
    for mod in ("dbpsk", "dqpsk", "pi4dqpsk"):  # differential: constellation of the parent
        cases.append((f"{mod}", True, GenParams(mod=mod, fs=fs, baud=fs / 10, snr=16,
                                                fc=0.008 * fs, seed=0)))
    for mod in ("qpsk", "16qam"):  # symbol-spaced multipath -> equalizer rescue
        cases.append((f"{mod}/multipath", True, GenParams(
            mod=mod, fs=fs, baud=fs / 10, snr=18 if mod == "qpsk" else 24, fc=0.008 * fs,
            taps=(1.0, 0.45 * np.exp(1j * 0.8), 0.2j), tap_sym=1.0, seed=0)))
    # v3.1: 1.5-symbol echo (T/2 FSE rescue), low-rolloff baud fallback, deep alias
    cases += [("qpsk/2ray-fse", True, GenParams(mod="qpsk", fs=fs, baud=fs / 10, snr=18,
                                                fc=0.008 * fs, taps=(1.0, 0.8),
                                                tap_sym=1.5, seed=0)),
              ("qpsk/roll0.05", True, GenParams(mod="qpsk", fs=fs, baud=fs / 10, snr=18,
                                                rolloff=0.05, fc=0.008 * fs, seed=0)),
              ("16qam/roll0.08", True, GenParams(mod="16qam", fs=fs, baud=fs / 10, snr=24,
                                                 rolloff=0.08, fc=0.008 * fs, seed=0)),
              ("qpsk/alias", True, GenParams(mod="qpsk", fs=fs, baud=fs / 10, snr=14,
                                             fc=1.3 * fs / 8, seed=0))]
    if tier == "full":
        for mod in MODS:
            cases.append((f"{mod}/lowsnr", False, GenParams(
                mod=mod, fs=fs, baud=fs / 10, snr=_SNR_LOW[mod], fc=0.008 * fs, seed=0)))
        for roll in (0.1, 0.2, 0.5):
            cases.append((f"qpsk/roll{roll}", False, GenParams(
                mod="qpsk", fs=fs, baud=fs / 10, snr=18, rolloff=roll, fc=0.008 * fs, seed=0)))
        cases += [("qpsk/bigcfo", False, GenParams(mod="qpsk", fs=fs, baud=fs / 10, snr=18,
                                                   fc=0.03 * fs, seed=0)),
                  ("qpsk/drift", False, GenParams(mod="qpsk", fs=fs, baud=fs / 10, snr=18,
                                                  fc=0.008 * fs, drift_ppm=100, seed=0)),
                  ("16qam/sps4", False, GenParams(mod="16qam", fs=fs, baud=fs / 4, snr=24,
                                                  fc=0.008 * fs, seed=0))]
    return cases


def _sweep(args: argparse.Namespace) -> int:
    cases = _grid(args.tier, args.seeds)
    fails, tally = [], {"CORE": [0, 0], "STRETCH": [0, 0]}  # tier -> [pass, total]
    for label, core, p in cases:
        x, bits = generate(p)
        try:
            r = analyze(x, Meta(p.fs, p.fmt, p.dtype))  # scoring demaps separately
            ok, msg = _score(r, p, bits, core)
        except Exception as exc:  # a crash is a failure, not an abort
            ok, msg = False, f"CRASH {exc}"
        tier = "CORE" if core else "STRETCH"
        tally[tier][0] += ok
        tally[tier][1] += 1
        print(f"[{'PASS' if ok else 'FAIL'}] {tier:<7} {label:<16} {msg}")
        if core and not ok:
            fails.append(label)
    for tier, (good, n) in tally.items():
        if n:
            print(f"{tier}: {good}/{n} pass")
    if fails:
        print(f"CORE failures: {', '.join(fails)}")
        return 1
    return 0


# --- analyze / gen / serve -----------------------------------------------------

def _analyze(args: argparse.Namespace) -> int:
    r = analyze_file(args.file, args.fs, args.fmt, args.dtype,
                     "be" if args.be else None, args.bitrev or None, args.diff,
                     None if args.burst is None else args.burst - 1, rf=args.rf)
    j = r.to_json(views=False)          # 한 줄 요약도 여기서 만든다 (반올림이 한 번만 걸리게)
    d = j["detected"]
    truth = (sidecar_read(args.file) or {}).get("truth")  # display only, never detection
    if len(r.bursts) > 1:
        print(f"버스트 {len(r.bursts)}개 감지 — {r.burst_idx + 1}번째 분석 (--burst N 으로 선택)")
    print(f"모드      {d['mod']}" + (f"   (정답 {truth['mod']})" if truth else ""))
    print(f"반송파    {d['fc']:.1f} Hz" + (f"   (정답 {truth['fc']:.1f})" if truth else ""))
    if d["rf_hz"] is not None:
        print(f"실제 RF   {d['rf_hz'] / 1e6:.6f} MHz")
    print(f"심볼레이트 {d['baud']:.1f} Hz" + (f"   (정답 {truth['baud']:.1f})" if truth else ""))
    fsk = r.family == "fsk"
    tail = f"변조지수 h {d['h']:.2f}" if fsk else f"롤오프    {d['rolloff']:.2f}"
    mer = "" if fsk else f"   MER {r.mer_db:.1f} dB"
    print(f"{tail}   lock {r.lock:.1f}{mer}")
    if r.eq_applied:
        print("등화기 적용: 다중경로 ISI 보정됨"
              + (" (T/2 분수간격)" if r.eq_mode == "fse" else " (심볼간격)"))
    if r.alias_resolved:
        print("반송파 앨리어스 보정: 스펙트럼 중심으로 후보 선택")
    if r.baud_fallback:
        print("심볼레이트 폴백: 점유대역폭 사전정보로 재탐색")
    if r.carrier_ambiguous:
        print("경고: 반송파 앨리어싱 가능 (|sym*fc| > 0.4*fs)")
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
    print(f"{len(s.emitters)}개 신호 감지 (샘플레이트 {s.meta.fs:.0f} Hz{note})")
    print(f"{'#':>2} {('실제 RF' if rf0 is not None else '중심주파수'):>12} {'대역폭':>10}"
          f" {'분류':>7} {'변조':>7} {'심볼레이트':>11} {'lock':>5}")
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


def _gen(args: argparse.Namespace) -> int:
    if args.fmt == "real" and args.cfo <= args.baud * (1 + args.rolloff) / 2:
        print("real 포맷은 통과대역 반송파가 필요합니다: --cfo > baud*(1+rolloff)/2")
        return 1
    taps = tuple(complex(t) for t in args.taps.split(",")) if args.taps else ()
    p = GenParams(mod=args.mod, n_symbols=args.n, fs=args.fs, baud=args.baud,
                  rolloff=args.rolloff, fc=args.cfo, phase=args.phase, timing=args.timing,
                  snr=args.snr, fmt=args.fmt, dtype=args.dtype,
                  endian="be" if args.be else "le", bitrev=args.bitrev, seed=args.seed,
                  pad=args.pad, drift_ppm=args.drift_ppm, dc=complex(args.dc),
                  h=args.h, taps=taps)
    print(save(p, args.out, args.label))
    return 0


def _dataset(args: argparse.Namespace) -> int:
    """Curated variation matrix: every modulation + sample-type/endian/bitrev
    variants + impairment cases, all with ground-truth sidecars."""
    fs, out, made = 1e6, args.out, []
    for mod in GEN_MODS:  # one canonical file per modulation
        fc = 0.0 if family(mod) == "fsk" else 0.008 * fs
        h = 0.7 if mod == "fsk2" else 0.5  # fsk2 at h=0.5 IS msk; pick a distinct index
        made.append(save(GenParams(mod=mod, fs=fs, baud=fs / 10, snr=22, fc=fc, h=h,
                                   phase=0.6, timing=0.3, seed=1), out, mod))
    # sample-type variants: label must stay unique (endian/bitrev alone would collide)
    for i, (dt, be, br) in enumerate((("i8", 0, 0), ("u8", 0, 0), ("u16", 0, 0),
                                      ("f32", 0, 0), ("f64", 0, 0), ("i16", 1, 0),
                                      ("i16", 0, 1), ("u8", 1, 1))):
        made.append(save(GenParams(mod="qpsk", fs=fs, baud=fs / 10, snr=18,
                                   fc=0.008 * fs, dtype=dt, endian="be" if be else "le",
                                   bitrev=bool(br), seed=2), out, f"fmtvar{i}"))
    for mod in ("bpsk", "qpsk", "16qam"):  # real passband .pcm
        made.append(save(GenParams(mod=mod, fs=fs, baud=fs / 10, snr=18, fc=0.1 * fs,
                                   fmt="real", seed=3), out, f"pcm-{mod}"))
    impair = [("dc", {"dc": 0.3 + 0.3j}), ("pad", {"pad": 0.5}),
              ("drift", {"drift_ppm": 100}), ("lowsnr", {"snr": 8}),
              ("taps", {"taps": (1.0, 0.45 * np.exp(1j * 0.8), 0.2j)})]  # 1-symbol echoes
    for label, kw in impair:
        made.append(save(GenParams(mod="qpsk", fs=fs, baud=fs / 10, snr=kw.pop("snr", 18),
                                   fc=0.008 * fs, seed=4, **kw), out, label))
    made.append(save(GenParams(mod="fsk4", fs=fs, baud=fs / 10, snr=20, h=1.0, seed=5),
                     out, "fsk4h1"))
    if len(set(made)) != len(made):
        print("경고: 파일명 충돌 — 라벨이 겹칩니다")
        return 1
    print(f"{len(made)} files -> {out}/ (각 파일에 <이름>.json 정답 사이드카)")
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

    g = sub.add_parser("gen", help="합성 신호 + 정답 사이드카 생성")
    g.add_argument("--mod", default="qpsk", choices=GEN_MODS)
    g.add_argument("--fs", type=float, default=1e6)
    g.add_argument("--baud", type=float, default=1e5)
    g.add_argument("--snr", type=float, default=20)
    g.add_argument("--rolloff", type=float, default=0.35)
    g.add_argument("--cfo", type=float, default=0.0)
    g.add_argument("--phase", type=float, default=0.0)
    g.add_argument("--timing", type=float, default=0.0)
    g.add_argument("--pad", type=float, default=0.0)
    g.add_argument("--drift-ppm", type=float, default=0.0)
    g.add_argument("--dc", default="0")
    g.add_argument("--h", type=float, default=0.5, help="FSK 변조지수")
    g.add_argument("--taps", default="", help="멀티패스 탭, 예: 1,0,0.35j")
    g.add_argument("--fmt", default="iq", choices=("iq", "real"))
    g.add_argument("--dtype", default="i16", choices=_DTYPE_CHOICES)
    g.add_argument("--be", action="store_true")
    g.add_argument("--bitrev", action="store_true")
    g.add_argument("--n", type=int, default=6000)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--out", default="samples")
    g.add_argument("--label", default="sig")

    d = sub.add_parser("dataset", help="변조x샘플타입x장애 변형 매트릭스 일괄 생성")
    d.add_argument("--out", default="samples")

    w = sub.add_parser("sweep", help="전수조사: 그리드 생성->복조->채점")
    w.add_argument("--tier", default="core", choices=("core", "full"))
    w.add_argument("--seeds", type=int, default=3)

    s = sub.add_parser("serve", help="웹 UI 서버")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)

    args = ap.parse_args(argv)
    if args.cmd == "analyze":
        return _analyze(args)
    if args.cmd == "survey":
        return _survey(args)
    if args.cmd == "gen":
        return _gen(args)
    if args.cmd == "dataset":
        return _dataset(args)
    if args.cmd == "sweep":
        return _sweep(args)
    from .server import run
    run(args.host, args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
