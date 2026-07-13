# RF 절대 중심주파수 (로드맵 #7) — 설계 스펙

- 날짜: 2026-07-13 · 브랜치: `wideband-frontend`
- 상태: 승인 게이트 생략(사용자 "계속 진행") · §5 회귀 게이트 적용

## 1. 목표
캡처의 RF 중심주파수를 알 때, 기저대역 offset 대신 **실제 RF 주파수**를 보고한다
(예: AIS 채널 → 161.975 MHz). #6 Survey UI를 실무적으로 완성한다.

## 2. 원칙 (무회귀)
`Meta.rf_center: float | None = None`. **None이면 현재 동작 그대로** — 실제 RF는 알 때만 *추가로*
표시하고, 기존 기저대역 fc 출력/필드는 유지한다. 순수 첨가.

## 3. 출처 (우선순위 high→low)
1. 명시적 override: CLI `--rf`, 서버 `rf` 쿼리, `analyze_file/survey_file(rf=...)`.
2. SigMF: `captures[0]["core:frequency"]`.
3. 파일명 토큰 `rf<Hz>` (예: `cap_fs20e6_rf162e6_iq_ci16.dat`), fs 토큰과 동일 문법.
4. 없음 → None.

## 4. 실제 RF = `rf_center + baseband_fc`
- `sigio.Meta`: `rf_center` 필드 + `parse_name` rf 토큰 + `parse_sigmf` core:frequency.
- `pipeline.analyze_file/survey_file`: `rf` 인자 → Meta(override).
- `Result.to_json`: `doc["rf_center"]` echo + `detected["rf_hz"] = rf_center + fc`(None이면 None).
- `pipeline.survey_web`: 응답 top-level `rf_center` 추가(프론트가 에미터별 `rf_center + abs_fc` 계산).
- `server._meta`: `rf` 쿼리 파싱.
- CLI: `_read_args`에 `--rf` 추가(analyze+survey 공용). `_analyze`는 실제 RF 줄 출력,
  `_survey`는 rf_center 있으면 중심주파수 칸을 실제 RF로.
- Web(`app.js`): `parseName` rf 토큰 + `metaFromSigmf` core:frequency + `query()` rf 전송;
  실제 RF 표시 — 상세 제원(`rf_hz`), Survey 목록/상자 라벨/드릴 헤더(`rf_center + abs_fc`).

## 5. 회귀·오버스펙 게이트 (매 단계)
- rf 미지정 시 모든 출력/JSON/테스트 **불변**: 247→ 신규 테스트만 증가, CORE 53/53, `survey_web`
  single==direct analyze byte-동일 유지.
- 첨가만: 기존 시그니처는 인자 **추가**(기본 None)로만 확장, 필드는 **추가**만.
- 오버스펙 금지: rf 소스는 위 4개로 한정. UI에 새 입력 폼 신설하지 않음(파일명/SigMF/CLI로 충분).

## 6. 테스트
- `parse_name`/`parse_sigmf` rf 파싱; `analyze_file(rf=...)` → `detected.rf_hz == rf+fc`;
  `survey_web` rf_center echo; rf 미지정 시 rf_hz/rf_center=None(무회귀).
- 서버 `?rf=` 쿼리 스모크.
