#!/usr/bin/env bash
# 실신호 테스트 결과를 옮기기 좋은 형태로 만든다.
#
#   tools/report.sh -s analyze capture_fs2000000_iq_i16.iq   # 한 줄 (손으로 칠 때)
#   tools/report.sh    analyze capture_fs2000000_iq_i16.iq   # 전체 (붙여넣을 수 있을 때)
#   tools/report.sh    survey  wide_fs20000000_iq_i16.iq --rf 433.92e6
#
# -s (--short) 는 클립보드가 없는 격리망 장비용이다. 손으로 옮겨야 하는 글자를 최소로 줄이고,
# 줄 끝에 오타 검출 코드(#xxxx)를 붙인다 — 받아친 쪽에서 tools/brief.py check 로 검증한다.
#
# 전체 모드는 --report JSON 에서 배열(성상도·비트·스펙트럼, 270KB)을 걷어낸 요약을 싣는다.
# 진단에 쓰이는 건 detected/quality/burst 쪽이다.
set -uo pipefail
short=0
case "${1:-}" in -s | --short) short=1; shift ;; esac
cmd=${1:-}; file=${2:-}
case "$cmd" in analyze | survey) ;; *) echo "사용법: $0 [-s] analyze|survey <파일> [옵션...]" >&2; exit 2 ;; esac
[ -f "$file" ] || { echo "파일이 없습니다: $file" >&2; exit 2; }
shift 2

cd "$(dirname "$0")/.." || exit 1
PY=$([ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)
json=$(mktemp -t signus-report)
err=$(mktemp -t signus-report-err)
trap 'rm -f "$json" "$err"' EXIT
rev=$(git rev-parse --short HEAD 2>/dev/null || echo '?')
[ -n "$(git status --porcelain 2>/dev/null)" ] && rev="$rev*"   # * = 미커밋 있음(원격에 없는 코드)

if [ "$short" = 1 ]; then
    # 한 줄 모드: 손으로 옮길 글자를 최소로. 실패하면 예외 마지막 줄만 (그게 진짜 결과다).
    if "$PY" -m signus.cli "$cmd" "$file" "$@" --report "$json" >/dev/null 2>"$err"; then
        "$PY" tools/brief.py emit "$json" "$([ "$cmd" = survey ] && echo sv || echo an)" "$rev"
    else
        msg=$(grep -E "Error|error:" "$err" | tail -1 | cut -c1-60)
        "$PY" -c "import sys; sys.path.insert(0, 'tools'); from brief import check_code
line = ' '.join(sys.argv[1:]); print(f'{line} #{check_code(line)}')" \
            "sig1" "$rev" "$([ "$cmd" = survey ] && echo sv || echo an)" "err" "${msg:-실패}"
    fi
    exit 0
fi

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
