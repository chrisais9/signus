# 셀 기반(눈 수준) find_bursts 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `find_bursts`를 워터폴 셀 기반으로 재구현해, 운용자가 스펙트로그램에서 눈으로 잡는
협대역·짧은 버스트(대역내 +12dB, 광대역 0dB급)를 자동 검출한다.

**Architecture:** STFT 셀 전력 → 빈별 저백분위 바닥(유색잡음·고듀티 강건) → 열(column)
점수(상위 k빈 초과 평균) → 점수 중앙값+MAD 상대 문턱 → 기존 검증 자산(런/병합/스파이크/
커버리지 가드/폴백)을 열 단위로 이식. 셀 전력부는 헬퍼로 분리해 2단계(survey 통합)가 재사용.

**Tech Stack:** numpy + scipy(signal.stft, ndimage)만. 스펙:
`docs/superpowers/specs/2026-08-11-cell-burst-detection-design.md`

## Global Constraints

- 의존성은 numpy + scipy 단 둘 (CLAUDE.md). numba 불가 영역 아님이지만 여기선 불필요.
- `find_bursts(x, fs) -> list[tuple[int,int]]` 시그니처·시간순·`[(0, n)]` 폴백 계약 불변.
- 기존 find_bursts 테스트 전부(현 9개) 그대로 그린이어야 한다 — 광대역 동작 보존 증명.
- ruff line-length=100, `pytest -q` 그린 유지.
- 모든 문턱은 "추정 바닥 + 통계 여유" 상대값 (스펙의 dB 원리 절).
- dsp.py는 필사 대상 — 코드가 바뀐 턴에 변경분 PDF 발급 + snap (CLAUDE.md).

---

### Task 1: `_cell_power` 헬퍼 (공유 코어)

**Files:**
- Modify: `signus/dsp.py` (imports에 `from scipy.signal import get_window, stft` 추가,
  `find_bursts` 바로 위에 헬퍼 추가)
- Test: `tests/test_dsp.py`

**Interfaces:**
- Produces: `_cell_power(x: np.ndarray, fs: float, nperseg: int = 256) ->
  tuple[np.ndarray, int, int]` — (P[fbin, col] 선형 전력, hop, nperseg). 짧은 레코드에서
  nperseg 자동 축소(최소 64), hop = nperseg // 2. Task 3이 소비.

- [ ] **Step 1: 실패 테스트 작성**

```python
def test_cell_power_shapes_and_energy():
    # 셀 전력 헬퍼: 모양이 맞고, 톤 하나가 정확히 한 빈 열(row)만 뜨겁게 한다
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(8192) + 1j * rng.standard_normal(8192)) * 0.01
    x += np.exp(2j * np.pi * 0.25 * np.arange(8192))     # fs/4 톤
    P, hop, nper = dsp._cell_power(x, 1e6)
    assert P.shape[0] == nper and hop == nper // 2
    assert P.shape[1] == 1 + (x.size - nper) // hop
    hot_bin = int(np.argmax(P.mean(axis=1)))
    assert P.mean(axis=1)[hot_bin] > 100 * np.median(P.mean(axis=1))
    # 짧은 레코드: nperseg가 자동으로 줄어 최소 1열은 나온다
    P2, hop2, nper2 = dsp._cell_power(x[:300], 1e6)
    assert nper2 <= 300 and P2.shape[1] >= 1
```

- [ ] **Step 2: RED 확인** — `pytest tests/test_dsp.py::test_cell_power_shapes_and_energy -q`
  → AttributeError(_cell_power 없음)로 실패해야 한다.

- [ ] **Step 3: 최소 구현**

```python
def _cell_power(x: np.ndarray, fs: float, nperseg: int = 256) -> tuple[np.ndarray, int, int]:
    """Waterfall cell power for detection: (P[fbin, col], hop, nperseg). nperseg
    auto-shrinks on short records; shared by find_bursts (and, later, survey)."""
    nperseg = int(min(nperseg, max(64, 1 << int(np.log2(max(x.size // 2, 64))))))
    hop = nperseg // 2
    win = get_window("blackmanharris", nperseg)
    _, _, z = stft(x, fs=fs, window=win, nperseg=nperseg, noverlap=nperseg - hop,
                   return_onesided=False, boundary=None, padded=False)
    return np.abs(z) ** 2, hop, nperseg
```

- [ ] **Step 4: GREEN 확인** — 같은 pytest 명령 PASS.
- [ ] **Step 5: 커밋** — `git add -u && git commit -m "feat(dsp): 워터폴 셀 전력 헬퍼(_cell_power) — 셀 검출 공유 코어"`

### Task 2: 사각지대 RED 테스트

**Files:**
- Test: `tests/test_dsp.py`

**Interfaces:**
- Consumes: 현행 `dsp.find_bursts`. 이 테스트는 Task 3 전까지 실패해야 한다.

- [ ] **Step 1: 실패 테스트 작성** (실측 사각: 협대역 1500샘플, 광대역 0dB, 대역내 +12dB)

```python
def test_find_bursts_sees_narrowband_bursts_like_the_eye():
    # the operator SEES these on a spectrogram (53% of cells lit, 2026-08-11 measurement):
    # narrowband qpsk bursts at wideband 0 dB are +12 dB per CELL (fs/bw processing gain).
    # wideband energy detection dilutes them to +3 dB -> invisible. cell detection must not.
    rng = np.random.default_rng(2)
    n = 65006
    x = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
    sig, _ = generate(GenParams(mod="qpsk", n_symbols=600, fs=1e6, baud=5e4,
                                snr=60, fc=2e5, seed=1))
    truth = []
    for k in range(4):
        s = 8000 + k * 14000
        b = sig[k * 1500:(k + 1) * 1500]
        x[s:s + 1500] += b / np.sqrt(np.mean(np.abs(b) ** 2))
        truth.append((s, s + 1500))
    bursts = dsp.find_bursts(x, 1e6)
    assert bursts != [(0, n)], "narrowband bursts invisible (wideband dilution)"
    hits = sum(any(bs <= s + 300 and e - 300 <= be for bs, be in bursts) for s, e in truth)
    assert hits >= 3, (hits, bursts[:6])
```

- [ ] **Step 2: RED 확인** — 현행 코드에서 `[(0, 65006)]` 폴백으로 실패하는지 실행으로 확인.
- [ ] **Step 3: 커밋(테스트만)** — `git add tests/test_dsp.py && git commit -m "test: 눈에는 보이는 협대역 버스트 사각을 RED로 잠금"`

### Task 3: find_bursts 셀 기반 재구현 (GREEN)

**Files:**
- Modify: `signus/dsp.py` — `find_bursts` 본문 교체 (`_SPIKE_PAR`·독스트링 철학 유지)

**Interfaces:**
- Consumes: Task 1 `_cell_power`.
- Produces: 동일 시그니처 `find_bursts(x, fs) -> list[tuple[int,int]]` (계약 불변).

- [ ] **Step 1: 구현** (초기 상수는 Task 5 캘리브레이션 대상 — 주석에 명시)

```python
def find_bursts(x: np.ndarray, fs: float) -> list[tuple[int, int]]:
    """Waterfall (cell-level) burst detector. Per-bin noise floors give the same
    narrowband processing gain the operator's eye gets on a spectrogram; column
    scores then drive the proven run/merge/guard machinery on the time axis.
    Returns bursts in time order, or [(0, size)] when nothing stands out."""
    n = x.size
    pw = np.abs(x) ** 2
    P, hop, nperseg = _cell_power(x, fs)
    if P.shape[1] < 4:                   # too few columns to tell bursts from record
        return [(0, n)]
    # Per-bin floor at a LOW percentile: tracks coloured noise per bin and stays on the
    # noise even when a bin is signal-occupied up to ~90% of the time (high-duty trains).
    # A record-filling signal owns its bins' floors entirely -> flat score -> [(0, n)].
    floor_f = np.percentile(P, 10, axis=1)[:, None]
    r = P / (floor_f + 1e-30)
    k = max(3, P.shape[0] // 16)
    sc = np.log10(np.sort(r, axis=0)[-k:].mean(axis=0) + 1e-30)  # column score (decades)
    # Thresholds are relative to the SCORE's own noise statistics (median + MAD), i.e.
    # false-alarm bounds, not absolute dB -- scale/gain invariant by construction.
    base = float(np.median(sc))
    mad = float(np.median(np.abs(sc - base))) * 1.4826
    hi = base + max(6 * mad, 0.25)
    lo = base + max(3 * mad, 0.12)
    dif = np.diff((sc >= lo).astype(np.int8))
    starts = list(np.where(dif == 1)[0] + 1)
    ends = list(np.where(dif == -1)[0] + 1)
    if sc[0] >= lo:
        starts.insert(0, 0)
    if sc[-1] >= lo:
        ends.append(sc.size)
    runs = [(s, e) for s, e in zip(starts, ends, strict=True) if (sc[s:e] >= hi).any()]
    if not runs:
        return [(0, n)]
    merged = [list(runs[0])]
    for s, e in runs[1:]:
        # merge across a valley that never returns to the noise base (fragmentation of a
        # marginal-level burst); a real inter-burst gap sits back down at the base.
        valley = sc[merged[-1][1]:s]
        if s - merged[-1][1] < 2 or float(np.median(valley)) > base + max(1.5 * mad, 0.06):
            merged[-1][1] = e
        else:
            merged.append([s, e])
    out, spiked = [], 0
    for c0, c1 in merged:
        if c1 - c0 < 2:                  # single-column flicker (noise / impulse smear)
            continue
        s, e = max(0, c0 * hop), min(n, (c1 - 1) * hop + nperseg)
        seg = pw[s:e]
        if float(seg.max() / (seg.mean() + 1e-30)) > _SPIKE_PAR:
            spiked += e - s
            continue
        out.append((int(s), int(e)))
    covered = sum(e - s for s, e in out)
    if out and covered < 0.6 * (covered + spiked):   # spike gate ate the candidate mass
        return [(0, n)]
    return out or [(0, n)]
```

- [ ] **Step 2: Task 2 테스트 GREEN 확인** — `pytest tests/test_dsp.py -q -k narrowband`
- [ ] **Step 3: 전체 find_bursts 테스트 실행** — `pytest tests/test_dsp.py -q` (Task 4에서 조정)
- [ ] **Step 4: 커밋** — `git commit -m "feat(dsp): find_bursts를 셀 기반으로 — 눈의 협대역 처리이득 확보"`

### Task 4: 기존 테스트 조정·정합 (동작 보존 증명)

**Files:**
- Modify: `signus/dsp.py` (버그 수정), `tests/test_dsp.py` (양자화 허용오차만 — 의도 완화 금지)

**Interfaces:**
- Consumes: Task 3의 find_bursts. 기존 9개 + 신규 2개 테스트 전부.

- [ ] **Step 1: `pytest -q` 전체 실행, 실패 목록 작성**
- [ ] **Step 2: 실패마다 — 원인 규명 후 코드 수정 우선.** 경계 ±8 같은 옛-메커니즘 상수만
  hop 양자화(±nperseg) 허용오차로 완화하되, 각 테스트의 "잠근 의도"(검출 개수·분리·거부·
  폴백)는 절대 완화하지 않는다. 의도가 깨지면 테스트가 아니라 구현을 고친다.
- [ ] **Step 3: `pytest -q` 전체 그린 + `ruff check` 통과 확인**
- [ ] **Step 4: 커밋** — `git commit -m "test(dsp): 셀 기반 경계 양자화 반영 — 잠근 의도는 전부 유지"`

### Task 5: 캘리브레이션 + 적대 검증 (워크플로)

**Files:**
- Modify: `signus/dsp.py` (상수 k·문턱 여유·병합 상수 확정), `tests/test_dsp.py` (발견 잠금)

**Interfaces:**
- Consumes: Task 4까지의 전체 그린 상태. 기존 하네스: scratchpad/fbgrid, fbadv, fbslice.

- [ ] **Step 1: Workflow 3 에이전트** — (a) 그리드 신구 F1: 기존 축 + (대역폭 {67k, 200k,
  전대역} × 대역내 SNR {6, 9, 12, 18dB}) 추가, 잡음 전용 오경보 0 확인; (b) 적대: FSK 톤 도약,
  처프(전 빈 스침 → 한 덩어리/전체가 정답), 임펄스(단발·연발·버스트 내부), 유색 잡음(스펙트럼
  ±10dB 기울기), 듀티 50~92%, 풀레코드 13변조 [(0,n)] 유지, 한계대비 파편화 재현 세트;
  (c) sweep 합불 신구 대조(HEAD 워크트리).
- [ ] **Step 2: 발견 → 원인별로 상수 보정 또는 코드 수정, 새 실패 모드는 RED→GREEN으로 잠금**
- [ ] **Step 3: 재검증 워크플로(바뀐 표면만) → CLEAN까지 반복**
- [ ] **Step 4: 커밋** — `git commit -m "feat(dsp): 셀 검출 상수 캘리브레이션 — 그리드·적대·sweep 검증"`

### Task 6: 발급·기록

**Files:**
- Modify: `docs/` (변경분 PDF, 코드북, snap), 메모리

**Interfaces:**
- Consumes: Task 5 완료 상태(전체 그린, 검증 통과).

- [ ] **Step 1: `.venv/bin/python tools/codebook.py all --snap --note "<보통 문장 요약>"`**
  (dsp.py 하나; 표지 요약은 "무엇이 왜" 사용자 관점 문장으로)
- [ ] **Step 2: 산출물 커밋** — 코드북/변경분/기준점
- [ ] **Step 3: 메모리 갱신** — signus-diag-probe/qa-findings에 셀 검출 완료·잔여 한계 기록
- [ ] **Step 4: 최종 보고** — 성능 수치(신구 F1, 사각 해소), 남긴 한계, 인쇄물 안내
