# 광대역 Survey UI (로드맵 #6) — 설계 스펙

- 날짜: 2026-07-12
- 브랜치: `wideband-frontend`
- 선행: `2026-07-11-wideband-frontend-design.md`(엔진/detect/channelize/triage), `signus survey`(CLI)
- 상태: 승인됨(브레인스토밍) → 구현 플랜 대기

## 1. 목표

웹 UI에서 **하나의 광대역 캡처**를 드롭하면, 전체 시간-주파수 **워터폴 위에 탐지된 에미터 상자**를
겹쳐 보여주고, **상자를 클릭하면 그 에미터의 상세 분석(성상도/제원)** 으로 드릴다운한다. 현재의
단일-신호 분석 UX는 그대로 보존한다.

## 2. 비목표 (YAGNI — 명시적 제외)

- 대용량 스트리밍 I/O(#3): 서버 `_MAX_BODY=256MB` 유지. 메모리-적재 캡처만 대상.
- 메시지 평문화/프레이밍(#5): AIS/DSC 디코드 없음.
- 클릭 시 재분석/재업로드: 하이브리드 원업로드로 대체(§4.2).
- 실시간·SDR·다중 캡처 비교: 범위 밖.

## 3. 제약 (하드)

- 런타임 의존성 numpy+scipy only, UI는 stdlib http.server + Canvas 2D, 외부 통신 0.
- Air-gap·손타이핑 → **라인 예산**: 증분 최소화, 가능하면 중복 제거로 상쇄.
- **회귀 0**: `analyze()` / `/api/analyze` / 기존 `render()` / CLI `survey` 리포트 포맷은 **불변**.
  247 pytest + CORE 53/53 BER=0 유지. anti-shared-bug 규칙 유지.

## 4. 아키텍처

### 4.1 서버 (`server.py`)
- `POST /api/survey` 신설. 쿼리스트링 meta 규약은 `/api/analyze`와 동일.
- meta 파싱/검증 로직을 **공유 헬퍼 `_meta_from_query(q, name)`** 로 추출(analyze·survey 공용, 중복 제거).
- 처리: `x = decode(body, meta)` → `payload = pipeline.survey_web(x, meta)` → `_json(200, payload)`.
- 예외 처리·`_MAX_BODY`·크기 가드는 기존 `do_POST`와 동일 패턴 재사용.
- `/api/analyze` 핸들러는 **손대지 않음**(헬퍼 추출로 인한 동작 변화 없음 — 값 동일).

### 4.2 직렬화 (`pipeline.py`)
새 함수 `survey_web(x, meta, diff=False) -> dict`. **기존 `survey()`/`Survey.to_json()`/`Emitter`는 불변**
(CLI·테스트 보호). 웹 전용 조립만 담당:

1. `xa = dsp.analytic(x); xa -= xa.mean()` (단일-신호 경로 보존을 위해 여기서 한 번 계산).
2. `dets = detect(xa, meta.fs)`.
3. **단일-충전 분기(현재 동작 무손실 보존):** `len(dets)==1 and dets[0].bw >= 0.5*meta.fs` 이면
   → `return {"mode": "single", "result": analyze(x, meta, diff=diff).to_json()}`.
   (채널화 없이 기존 경로 그대로 → UI가 받는 데이터가 현재와 **동일**.)
4. 그 외 → `sv = survey(x, meta, diff=diff)` 실행, 응답 조립:
   - `overview`: `{"fs": fs, "spectrum": spectrum(xa, fs), "waterfall": waterfall(xa, fs)}`.
   - `emitters`: 각 `Emitter`에 대해 `Emitter.to_detail()`(신규, §4.3) — det + 경량 상세.
   - `return {"mode": "survey", "overview": overview, "emitters": [...], "fs": fs, "fmt": meta.fmt}`.

주의(오버스펙 방지, 격리 우선): 이 경로는 (a) overview 워터폴용 `analytic`을 `survey()` 내부와, (b)
단일-충전 판정용 `detect`를 `survey()` 내부와 **중복 계산**한다. 둘 다 결정론적이라 결과는 동일하며(표시
박스는 `survey()`가 낸 `emitters[].detection`을 쓰므로 불일치 없음), 비용은 다중-에미터 경로에서만
발생(그 경로는 이미 N회 channelize+analyze로 무거움 → 한계효과 미미). `survey()`의 시그니처/반환을 바꿔
테스트·CLI를 깨뜨리는 것보다 **중복 계산을 감수**한다. 흔한 단일-충전 경로는 §4.2-3에서 `detect` 1회 후
바로 `analyze()`라 빠름. 문제가 되면 그때 최적화.

### 4.3 에미터 경량 상세 (`pipeline.Emitter.to_detail()`)
- 공통: `det`(fc, bw, t0, t1, snr_db, baud_hint), `kind`, `abs_fc`.
- 디지털(`result` 존재): `result.to_json(views=False)`의 결과에서 **constellation + detected + quality +
  bits**를 취하고, **경량 채널 스펙트럼** `spectrum(result.burst_x, fs_ch)` 1개를 붙임(개별 워터폴은 생략).
  - `to_json(views=False)`는 spectrum/waterfall을 아예 넣지 않으므로 payload가 가벼움(성상도는 6000점 캡).
- 비디지털(analog/tone/tooshort/error): 상세 없음 — det + kind만.
- **박스 기하는 프론트가 계산**(fc±bw/2, t0/t1) — 서버는 원시 det 값만 전달.

### 4.4 프론트엔드 (`web/`)
- **진입(자동통합):** 단일 파일 드롭 → `POST /api/survey`.
  - `mode==="single"` → 기존 `render(resp.result)` 그대로(현재 UX 동일).
  - `mode==="survey"` → 새 `renderSurvey(resp)`.
  - 다중 파일 배치 모드는 **그대로 공존**(각 파일은 여전히 `/api/analyze`; 배치는 단일-신호 표).
- **`surveyCard`(신규 뷰):**
  - 개요 워터폴 캔버스(`drawWaterfall` 재사용) + 그 위 **박스 오버레이 캔버스/absolute div**.
  - 박스 라벨: 디지털=`MOD·lock`, 비디지털=`kind`(아날로그/톤/…).
  - 박스 색(결정 B): 디지털=lock 색(초/노/빨, 기존 `statusOf` 재사용), analog·tone=회색, error=빨강.
  - 아래 **에미터 요약 리스트**(주파수순): abs_fc, kind, mod, baud, lock — 행 클릭 = 박스 클릭과 동일.
- **드릴다운:**
  - 디지털 → `render()` 재사용(성상도/제원/게이지/다운로드). `← Survey로` 백버튼(배치 백버튼 패턴 복제).
  - 비디지털(결정 A) → 간단 **정보 패널**: kind, abs_fc, bw, snr, 채널 스펙트럼(있으면).
- 상호작용: 박스 hover 하이라이트, 클릭 선택. 키보드 접근성은 리스트(버튼)로 확보.

## 5. 데이터 계약 (JSON)

```
POST /api/survey?name=&fs=&fmt=&dtype=&endian=&bitrev=&diff=  (body = raw samples)

mode==="single":  { "mode":"single", "result": <기존 analyze().to_json() 그대로> }

mode==="survey":  {
  "mode":"survey", "fs":Hz, "fmt":"iq|real",
  "overview": { "fs":Hz, "spectrum": <spectrum()>, "waterfall": <waterfall()> },
  "emitters": [ {
     "kind":"linear|fsk|analog|tone|tooshort|error",
     "abs_fc":Hz,
     "det": {"fc":Hz,"bw":Hz,"t0":sample,"t1":sample,"snr_db":dB,"baud_hint":Hz},
     // 디지털만:
     "detected": {...}, "quality": {...}, "constellation": {"i":[],"q":[]},
     "bits":"…", "channel_spectrum": <spectrum()> | null
  }, ... ]
}
```

## 6. 회귀·오버스펙 검토 게이트 (사용자 지시 — 매 단계 필수)

각 구현 단계 후 아래를 **매번** 확인하고, 어긋나면 즉시 축소/롤백한다:

1. `analyze()`·`/api/analyze`·`render()`·CLI `survey` 리포트 = **동작/포맷 불변**. (247 pytest + CORE 53/53
   + `signus survey --report` 스냅샷 동일.)
2. 추가 코드가 **기존 계약을 변형하지 않음**: `survey()`/`Survey.to_json`/`Emitter` 시그니처 불변,
   새 기능은 `survey_web`/`to_detail`/새 뷰로 **첨가**만.
3. **오버스펙 아님**: 비목표(§2)에 없는 것을 끌어들이지 않음. 새 파라미터·새 상태·새 캐시 도입 시
   "이게 #6에 꼭 필요한가?"를 통과해야 함.
4. 라인 예산: 순 증분이 큰가? meta 헬퍼 추출 등으로 상쇄했는가?

## 7. 테스트

- `tests/test_survey.py` 확장: `survey_web` — (a) 밴드-충전 단일 → `mode=="single"` + result가
  `analyze().to_json()`와 동일, (b) 3-에미터 픽스처 → `mode=="survey"`, emitters 길이/kind/기하 필드 존재,
  디지털 emitter에 constellation/bits 존재.
- 서버 스모크: `/api/survey` 200 + 스키마, 잘못된 meta 400.
- 회귀: 기존 247 + CORE 53/53 그대로 통과.

## 8. 결정 요약 (승인됨)

| | 결정 |
|---|---|
| 진입 | 자동통합: 단일 파일 → `/api/survey`, single/survey 분기 |
| 드릴다운 | 하이브리드 원업로드(클릭=즉시), 재분석 없음 |
| A 비디지털 | 정보 패널(성상도 대신) |
| B 박스 색 | kind별(디지털=lock색, analog·tone=회색, error=빨강) |
| C 단일-충전 | 채널화 없이 직접 `analyze()`로 현재 동작 보존 |

## 9. 파일 영향 요약

- `signus/server.py`: `_meta_from_query` 헬퍼 추출 + `/api/survey` 분기(+~15줄).
- `signus/pipeline.py`: `survey_web()` + `Emitter.to_detail()`(+~30줄). 기존 심볼 불변.
- `web/index.html`: `surveyCard` 뷰 마크업.
- `web/app.js`: `renderSurvey`/박스 오버레이/드릴다운 라우팅, 진입을 `/api/survey`로.
- `web/style.css`: 박스·리스트·정보패널 스타일.
- `tests/test_survey.py`: `survey_web` + 서버 스모크.
