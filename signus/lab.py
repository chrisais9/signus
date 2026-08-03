"""개발기 전용: 합성 신호 생성(gen/dataset)과 채점·전수조사(sweep) 하네스.

이 모듈과 gen.py 는 필사 인쇄물에 싣지 않는다 (2026-08-03 사용자 결정 — 합성·채점은
개발기의 테스트·정합성 확인에만 쓰고, 격리망 장비는 실신호만 다룬다). 장비에는 이 파일이
없고, cli.py 가 임포트 실패를 "명령 없음"으로 처리한다. 그러니 그 장비가 써야 하는 기능은
절대 여기 두지 않는다 (CLAUDE.md 의 기기 표)."""

import argparse

import numpy as np

from .cli import _DTYPE_CHOICES
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
from .pipeline import Result, analyze
from .sigio import Meta

_SNR = {"bpsk": 14, "qpsk": 14, "8psk": 18, "16qam": 22, "64qam": 28}
_SNR_LOW = {"bpsk": 6, "qpsk": 8, "8psk": 12, "16qam": 16, "64qam": 22}


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


# --- gen / dataset -------------------------------------------------------------

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


# --- cli 연결 -------------------------------------------------------------------

def add_commands(sub) -> None:
    """cli.main 의 서브파서에 개발기 전용 명령(gen/dataset/sweep)을 단다."""
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
    g.set_defaults(run=_gen)

    d = sub.add_parser("dataset", help="변조x샘플타입x장애 변형 매트릭스 일괄 생성")
    d.add_argument("--out", default="samples")
    d.set_defaults(run=_dataset)

    w = sub.add_parser("sweep", help="전수조사: 그리드 생성->복조->채점")
    w.add_argument("--tier", default="core", choices=("core", "full"))
    w.add_argument("--seeds", type=int, default=3)
    w.set_defaults(run=_sweep)
