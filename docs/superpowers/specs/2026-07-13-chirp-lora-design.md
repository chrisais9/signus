# 처프 / CSS (LoRa) 대응 — 설계 스펙

- 날짜: 2026-07-13 · 브랜치: `wideband-frontend`
- 근거: 리서치 wf_6abdf291-c70 (matched-dechirp μ-sweep / delay-conjugate deramp)

## 1. 목표 · 범위
LoRa 같은 선형 처프(CSS)/FMCW 신호를 **탐지 + 제원화**하고, 억지로 constellation을
씌우지 않는다(analog/tone과 동일 철학). **심볼/비트 복조는 하지 않는다** — 블라인드 CSS 디코드는
프리앰블 CFO/STO 동기 + 리버스엔지니어링된 Gray/interleave/Hamming/whitening 프레이밍이 필요하고,
brittle하며 signus의 blind·no-force-fit·numpy-only 헌장에 반한다(리서치 결론).

## 2. 판별기 (blind, numpy/scipy)
`chirp._beat`: **delay-conjugate deramp** `y = x[t]·conj(x[t-τ])`. 선형 처프의 2차 위상은
τ에 대해 선형항만 남아 y가 **μτ 주파수의 순음(NONZERO)** 이 된다. DC 밴드를 null(FSK/CW/톤은
DC에서 비트) 후 스펙트럼 peak-to-mean이 판별값.
- gen 캘리브레이션: 처프 beat-PAR ≥ 936(LoRa SF7)~42783(SF12), FMCW 7762; 비처프 ≤ 264
  (FM voice 264, PSK/QAM ≤ 94, FSK ≤ 91, CW 11, noise 13). 임계 `_PAR_MIN = 400`.
- **CV 프리게이트 없음**: beat-PAR 마진이 커서 불필요하고, 잡음 있는 처프의 포락선 CV를
  올려 오히려 놓친다(구현 중 확인).

## 3. 제원화 `chirp.analyze_chirp(x, fs)`
- μ(Hz/s) = beat 톤 주파수 · fs/τ, 방향 = sign(μ). **robust**(coherent-ish).
- bw = 채널 99%전력 폭(노이즈 pedestal 감산). SF = round(log2(bw²/|μ|)) ∈ [7,12] 로 snap될 때
  LoRa 가설 {SF, Rs=|μ|/bw, Tsym=2^SF/bw}. bw가 깨끗할 때만(격리된 채널) snap.
- 반환 {mu, up, bw, par, sf, rs, tsym}.

## 4. 통합 (첨가만)
- `triage.family`: fsk_gate 다음, CV 분기 전에 `is_chirp` → 'chirp'(안 그러면 constant-envelope라
  'analog'로 오분류). 기존 linear/fsk/analog/tone 불변.
- `pipeline`: `Emitter.info: dict|None` 추가(비디지털 제원). survey에서 kind=='chirp'이면
  `analyze_chirp`로 채운다. Emitter.to_json/to_detail가 info를 실어 UI로.
- `web`: 처프 박스 라벨(LoRa SFx / 처프), 목록 종류 '처프/LoRa', 정보 패널(처프율·방향, 그리고
  SF snap 시 대역폭·심볼레이트·심볼시간). μ·방향만 항상 표시(신뢰 가능), bw/SF는 격리 시.

## 5. 한계 (정직)
- **낮은 SNR / 좁은 상대전력**: delay-conjugate는 noise×noise로 잡음을 제곱 → 다른 에미터보다
  약한 LoRa(≈9dB in-channel)는 beat-PAR가 264(FM) 아래로 떨어져 못 잡음. 리서치의 coherent
  matched-dechirp(잡음에 선형)로 교체하면 개선 여지 — 후속.
- **미격리 채널 bw/SF**: 125kHz 처프는 1MHz 밴드에서 채널화가 decimate를 못 해(d=1) 이웃
  누설이 채널 PSD를 오염 → bw 과대추정 → SF snap 실패(일반 처프로 보고). μ·방향은 그대로 정확.
- 전체 심볼/비트 복조는 범위 밖(§1).

## 6. 테스트 · 검증
- `_lora` 픽스처(수신기와 코드 비공유) + `test_chirp_detector_flags_lora_not_others`
  (LoRa→chirp+SF9, PSK/QAM/noise→아님), `test_survey_reports_lora_as_chirp`(격리 LoRa→chirp,
  info, result=None). 브라우저: 3-emitter+강한 LoRa → '처프' 박스 + 정보 패널(처프율·방향).
- 회귀: 3-emitter 혼합의 qpsk/16qam/msk가 chirp로 오분류되지 않음. 255 pytest, CORE 53/53.
