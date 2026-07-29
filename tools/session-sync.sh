#!/usr/bin/env bash
# 세션 시작/끝에 origin 과 맞춘다 (맥 <-> ssh 보드 왕복 작업용). .claude/settings.json 의 훅이 부른다.
#
#   pull   세션 시작: origin 을 가져와 fast-forward 만 한다
#   push   세션 끝:   남은 변경을 WIP 로 커밋하고 밀어올린다
#
# 훅에서 도니까 절대 멈추지 않고, 애매하면 아무것도 하지 않고 알리기만 한다:
#   · 자격증명 프롬프트를 끈다 (HTTPS 원격에서 입력 대기로 세션이 멎는 걸 막는다)
#   · 갈라졌으면(diverged) 손대지 않는다 — 합치는 방식은 사람이 정할 일이다
#   · fast-forward 가 로컬 수정본을 덮을 상황이면 git 이 알아서 거절한다
set -uo pipefail
export GIT_TERMINAL_PROMPT=0 GIT_SSH_COMMAND="${GIT_SSH_COMMAND:-ssh -o BatchMode=yes}"

case "${1:-}" in pull | push) ;; *) echo "사용법: $0 pull|push" >&2; exit 2 ;; esac

cd "${CLAUDE_PROJECT_DIR:-$(dirname "$0")/..}" 2>/dev/null || exit 0
git rev-parse --git-dir >/dev/null 2>&1 || exit 0

# 훅 출력 규약: {"systemMessage": "..."} 한 줄. 메시지에 따옴표/줄바꿈을 넣지 않으므로 그대로 싣는다.
say() { printf '{"systemMessage": "[sync] %s"}\n' "$1"; exit 0; }

branch=$(git symbolic-ref --short -q HEAD) || say "detached HEAD — 동기화 건너뜀"
git remote get-url origin >/dev/null 2>&1 || say "origin 이 없어 동기화 건너뜀"

case "$1" in
pull)
    git fetch --quiet origin "$branch" 2>/dev/null || say "origin fetch 실패 (오프라인?) — 로컬 상태로 진행"
    local_sha=$(git rev-parse HEAD)
    remote_sha=$(git rev-parse "origin/$branch" 2>/dev/null) || say "origin/$branch 없음 — 첫 푸시 대기"
    [ "$local_sha" = "$remote_sha" ] && exit 0                      # 이미 최신: 조용히 넘어간다
    base=$(git merge-base HEAD "origin/$branch")
    [ "$base" = "$remote_sha" ] && say "로컬이 origin/$branch 보다 앞섬 — 세션 끝에 푸시됩니다"
    [ "$base" = "$local_sha" ] || say "origin/$branch 와 갈라졌습니다 — 직접 rebase/merge 하세요"
    n=$(git rev-list --count "HEAD..origin/$branch")
    if git merge --ff-only --quiet "origin/$branch" 2>/dev/null; then
        say "origin/$branch 에서 ${n}개 커밋 받아옴 (fast-forward)"
    fi
    say "${n}개 커밋 뒤처졌지만 작업본과 겹쳐 당기지 못했습니다 — 커밋/스태시 후 git pull 하세요"
    ;;
push)
    if [ -n "$(git status --porcelain)" ]; then
        git add -A || say "git add 실패 — 커밋하지 못했습니다"
        # 크기 제동: origin 은 공개 저장소다. 실신호 캡처는 .gitignore 로도 막지만 확장자를
        # 못 맞힌 것이 있을 수 있으니, 큰 파일이 딸려 들어가면 자동 커밋을 아예 멈추고 알린다.
        # 사람이 의도해서 올리는 큰 파일(인쇄물 PDF)은 그 턴에 직접 커밋하므로 여기 안 걸린다.
        big=$(git diff --cached --name-only | while IFS= read -r f; do
                  [ -f "$f" ] && [ "$(wc -c <"$f")" -gt 1048576 ] && echo "$f"
              done)
        [ -n "$big" ] && { git reset --quiet
            say "1MB 넘는 파일이 있어 자동 커밋을 멈췄습니다 (캡처 유출 방지): $(echo "$big" | tr '\n' ' ')"; }
        git commit --quiet -m "chore(wip): 세션 자동 커밋 ($(hostname -s), $(date '+%Y-%m-%d %H:%M'))" \
                   -m "세션 끝에 남아 있던 변경. 검증 안 된 진행 중 상태일 수 있습니다." \
            || say "커밋 실패 — 원격에 올리지 못했습니다"
    fi
    [ "$(git rev-list --count "origin/$branch..HEAD" 2>/dev/null || echo 1)" -eq 0 ] && exit 0
    git push --quiet origin "$branch" 2>/dev/null \
        || say "push 실패 (오프라인/인증?) — 커밋은 로컬에 남아 있습니다"
    say "origin/$branch 에 푸시 완료"
    ;;
esac
