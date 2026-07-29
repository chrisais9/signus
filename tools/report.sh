#!/usr/bin/env bash
# 붙여넣기가 되는 기기(맥·ssh 보드)에서 쓰는 긴 리포트. 손으로 옮겨야 하는 격리망 장비는
# 이걸 쓰지 않는다 -- 거기엔 tools/ 자체가 없다. 그쪽은 `signus analyze <파일> --brief` 다.
#
#   tools/report.sh analyze capture_fs2000000_iq_i16.iq
#   tools/report.sh survey  wide_fs20000000_iq_i16.iq --rf 433.92e6
#
# --report JSON 원본은 배열(성상도·비트·스펙트럼)로 270KB 라 못 옮긴다. 여기서는 그걸 걷어낸
# 요약만 싣는다 -- 진단에 쓰이는 건 detected/quality/burst 쪽이다.
set -uo pipefail
cmd=${1:-}; file=${2:-}
case "$cmd" in analyze | survey) ;; *) echo "사용법: $0 analyze|survey <파일> [옵션...]" >&2; exit 2 ;; esac
[ -f "$file" ] || { echo "파일이 없습니다: $file" >&2; exit 2; }
file=$(cd "$(dirname "$file")" && pwd)/$(basename "$file")   # 아래에서 저장소 루트로 cd 한다
shift 2                                                      # -- 상대경로면 여기서 어긋난다

cd "$(dirname "$0")/.." || exit 1
PY=$([ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)
json=$(mktemp -t signus-report)
trap 'rm -f "$json"' EXIT
rev=$(git rev-parse --short HEAD 2>/dev/null || echo '?')
[ -n "$(git status --porcelain 2>/dev/null)" ] && rev="$rev*"   # * = 미커밋 있음(원격에 없는 코드)

echo "=== signus 실신호 리포트 ==="
echo "파일   $(basename "$file")  ·  $(du -h "$file" | cut -f1)  ·  sha256 $("$PY" -c "
import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest()[:12])" "$file")"
for sc in "$file.json" "${file%.*}.sigmf-meta"; do   # 정답 사이드카는 <파일>.json,
    [ -f "$sc" ] && echo "사이드카 $(basename "$sc") 있음 (분석기는 안 읽음 · 비교 표시용)"
done                                                # SigMF 는 <확장자뺀이름>.sigmf-meta
echo "명령   signus $cmd $file${*:+ $*}"
echo "리비전 $rev   (지문은 signus selfcheck)"
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
