#!/usr/bin/env bash
# 실신호 테스트 결과를 "붙여넣기 한 덩어리"로 만든다. 사람이 손으로 모으면 꼭 빠지는 것들
# (파일 크기·해시, 쓴 명령 그대로, 코드 리비전, 환경)을 기계가 챙긴다.
#
#   tools/report.sh analyze samples/capture_fs2000000_iq_i16.iq
#   tools/report.sh survey  capture.iq --rf 433.92e6
#   tools/report.sh analyze capture.iq --burst 3 > /tmp/report.txt
#
# --report JSON 원본은 270KB(성상도·비트·스펙트럼 배열) 라 붙여넣을 수 없다. 여기서는
# 그 배열들을 걷어낸 요약만 싣는다 — 진단에 쓰이는 건 detected/quality/burst 쪽이다.
set -uo pipefail
cmd=${1:-}; file=${2:-}
case "$cmd" in analyze | survey) ;; *) echo "사용법: $0 analyze|survey <파일> [옵션...]" >&2; exit 2 ;; esac
[ -f "$file" ] || { echo "파일이 없습니다: $file" >&2; exit 2; }
shift 2

cd "$(dirname "$0")/.." || exit 1
PY=$([ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)
json=$(mktemp -t signus-report)
trap 'rm -f "$json"' EXIT

echo "=== signus 실신호 리포트 ==="
echo "파일   $(basename "$file")  ·  $(du -h "$file" | cut -f1)  ·  sha256 $("$PY" -c "
import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest()[:12])" "$file")"
for sc in "$file.json" "${file%.*}.sigmf-meta"; do   # 정답 사이드카는 <파일>.json,
    [ -f "$sc" ] && echo "사이드카 $(basename "$sc") 있음 (분석기는 안 읽음 · 비교 표시용)"
done                                                # SigMF 는 <확장자뺀이름>.sigmf-meta
echo "명령   signus $cmd $file${*:+ $*}"
echo "리비전 $(git rev-parse --short HEAD 2>/dev/null || echo '?')$(
    [ -n "$(git status --porcelain 2>/dev/null)" ] && echo " +미커밋$(git status --porcelain | wc -l | tr -d ' ')개")"
echo "환경   $(uname -sr) · $("$PY" -c 'import sys,numpy,scipy;
print(f"py{sys.version_info.major}.{sys.version_info.minor} numpy{numpy.__version__} scipy{scipy.__version__}")')"

echo
echo "--- 출력 ---"
"$PY" -m signus.cli "$cmd" "$file" "$@" --report "$json"
status=$?
[ $status -eq 0 ] || echo "(종료코드 $status)"

echo
echo "--- 요약 (배열 제거) ---"
"$PY" - "$json" <<'EOF'
import json, sys
BULK = {"constellation", "bits", "spectrum", "waterfall", "views", "iq"}
try:
    doc = json.load(open(sys.argv[1]))
except Exception as exc:
    print(f"(리포트 JSON 없음: {exc})"); raise SystemExit(0)


def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if k not in BULK}
    return [strip(v) for v in o] if isinstance(o, list) else o


print(json.dumps(strip(doc), ensure_ascii=False, indent=1))
EOF

cat <<'EOF'

--- 내가 본 것 (직접 채워주세요) ---
기대:
실제:
근거:            # 왜 틀렸다고 보는지 — 알고 있는 제원, 다른 도구 결과, 성상도 모양 등
파일 전달 가능: 예 / 아니오
EOF
