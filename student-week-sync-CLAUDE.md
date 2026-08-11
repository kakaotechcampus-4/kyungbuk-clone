# 강의자료 최신화 — Claude 실행 지시서 (v4)

> **학생 사용법**: 본인 clone 폴더(`<학교>-clone`)에서 `claude` 실행 → 이 파일을 드래그 → **"이 문서대로 강의자료 최신화 해줘"**
> 다른 폴더에서 켜면 git 이 안 돕니다. Claude 가 감지해서 안내합니다.
>
> 📌 **이 파일은 매 주차 그대로 재사용할 수 있습니다.** 주차 번호는 강의자료에서 자동 판정합니다.
> 이미 최신이면 아무것도 하지 않고 끝납니다(멱등). 버리지 말고 보관하세요.

---

## 1. 임무

강사가 `main` 에 올린 새 강의자료를 학생의 **`<이름>/final`** 에 **머지**하고, 거기서 이번 주차 브랜치를 판다.
**학생이 쓴 코드는 한 줄도 잃지 않는다.**

### 대전제 (절대 깨지 않는다)

```
main ──(머지)──> <이름>/final ──(분기)──> <이름>/weekN ──(PR, base=final)──> final 에 누적
```

`<이름>/final` 은 **6주간 학생의 결과물이 쌓이는 곳**이다. 과정이 끝나면 학생은 이 브랜치를 **완성된 자기 프로젝트**로 가져간다. 그래서:

- 강의자료는 **`final` 에 머지**한다. (`weekN` 에 직접 넣지 않는다)
- `weekN` 은 **최신 `final` 에서 분기**한다.
- **히스토리를 재작성하지 않는다.** 과거 커밋과 머지된 PR 기록이 사라지면 결과물이 훼손된다.

---

## 2. 강사 배포 형식 (2가지 — 둘 다 같은 규칙으로 처리)

이전 주차 **정답코드는 별도 디렉터리(`student_parts_baseline/`)** 로 온다. **참고용**이고 학생 코드를 대체하지 않는다.

| | 배포 형태 | 학생 파일에 미치는 영향 |
|---|---|---|
| **A** | 정답을 `student_parts_baseline/` 에 **추가만**. `student_parts/` 는 그대로 | main 이 학생 파일을 안 건드림 |
| **B** | 정답을 옮기면서 `student_parts/` 를 **stub 으로 되돌림** | main 이 학생 파일을 **수정** → 충돌 급증 |

**둘 다 해소 규칙은 같다** (§3). B 여도 학생 코드는 그대로 지킨다.

⚠️ **`student_parts_baseline/` 은 `student_parts` 로 시작하지만 학생 파일이 아니다.** 경로 규칙은 반드시 **`student_parts/`(슬래시 포함)** 로 판정한다.

---

## 3. 해소 규칙 (충돌 시 — 이게 전부다)

| 충돌 파일 | 처리 |
|---|---|
| `student_parts/**` | **내 것 유지** (`--ours`). 강사 정답·stub 으로 덮지 않는다. |
| `fixed/**`, `run.sh`, 그 외 | **main 것** (`--theirs`) — 단, **내가 직접 고친 파일이면 멈추고 학생에게 물어본다** |
| 신규 파일 (`student_parts_baseline/`, 이번 주차 `weekN`) | 자동 병합 — 손댈 것 없음 |

> ⚠️ 애매하면 **지우지 말고 학생에게 보여주고 물어본다.**

---

## 4. Claude 실행 규칙

1. §5~§8 블록은 `# ktc-block:` 으로 시작한다. **그대로, 순서대로** 실행한다.
2. 블록을 쪼개지 마라 — 셸은 호출마다 새로 뜨고, 상태는 `.git/ktc-sync-env` 로만 전달된다.
3. 블록이 `exit 1` 이면 **이유를 학생에게 보여주고 멈춘다.** 다음 블록으로 넘어가지 않는다.
4. **`git rebase` / `git push --force` / `git reset --hard` 금지.**

---

## 5. 진단

```bash
# ktc-block: diagnose
set -euo pipefail
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
  echo "❌ git 저장소가 아닙니다. 본인 clone 폴더(<학교>-clone)에서 Claude Code 를 다시 켜세요."; exit 1; }
cd "$(git rev-parse --show-toplevel)"

git fetch origin --prune || { echo "❌ git fetch 실패(네트워크/인증). 해결 후 재실행."; exit 1; }

# 커밋 안 한 변경이 있으면 진행 금지 — 덮이면 reflog 에도 없어 복구 불가
if [ -n "$(git status --porcelain)" ]; then
  echo "⚠️ 커밋하지 않은 변경이 있습니다:"; git status --short
  echo "→ 학생에게 물어보고 커밋(git add -A && git commit -m 'wip') 하거나 git stash push -u 한 뒤 재실행."
  exit 1
fi

# 이름 감지: <이름>/final 이 로컬/원격에 있는가
CUR=$(git rev-parse --abbrev-ref HEAD); NAME=""
if [ "$CUR" != "HEAD" ] && [ "$CUR" != "main" ]; then
  c=${CUR%%/*}
  git rev-parse --verify -q "refs/heads/$c/final" >/dev/null && NAME="$c"
  [ -z "$NAME" ] && git rev-parse --verify -q "refs/remotes/origin/$c/final" >/dev/null && NAME="$c"
fi
if [ -z "$NAME" ]; then
  F=$( { git for-each-ref --format='%(refname:short)' refs/heads
         git for-each-ref --format='%(refname:short)' refs/remotes/origin | sed 's#^origin/##'
       } | grep '/final$' | sed 's#/final$##' | sort -u )
  [ "$(printf '%s\n' "$F" | sed '/^$/d' | wc -l | tr -d ' ')" = "1" ] && NAME=$(printf '%s\n' "$F" | sed '/^$/d')
fi
[ -n "$NAME" ] || { echo "❌ 이름 감지 실패. 학생에게 '<이름>/final 의 <이름>' 을 묻고 NAME=<이름> 을 넣어 재실행."; exit 1; }
echo "이름: $NAME"

# ⛔ 아직 머지 안 된 주차 작업이 있는가 — 있으면 그 PR 을 먼저 머지해야 한다.
#    (final 만 앞서 나가면 그 주차 PR 의 diff 가 지저분해지고, 학생이 weekN 에서 final 을 또 머지해야 한다)
UNMERGED=""
for b in $(git for-each-ref --format='%(refname:short)' "refs/remotes/origin/$NAME/week*"); do
  base="origin/$NAME/final"
  git rev-parse --verify -q "$base" >/dev/null || base="$NAME/final"
  [ "$(git rev-list --count "$base..$b" 2>/dev/null || echo 0)" -gt 0 ] && UNMERGED="$UNMERGED ${b#origin/}"
done
if [ -n "$UNMERGED" ]; then
  echo "⛔ 아직 '$NAME/final' 에 머지되지 않은 주차 작업이 있습니다:$UNMERGED"
  echo "→ 그 주차 PR(base = $NAME/final)을 **먼저 머지**한 뒤 다시 실행하세요."
  echo "   (리뷰 대기·변경요청 중이면 그것부터 끝내세요. 지금 진행하면 브랜치가 꼬입니다.)"
  exit 1
fi

# ★ 대전제: 강의자료는 final 에 머지한다
git checkout "$NAME/final"
if git rev-parse --verify -q "refs/remotes/origin/$NAME/final" >/dev/null; then
  git pull --ff-only origin "$NAME/final" || {
    echo "❌ final 최신화(ff-only) 실패 — 로컬/원격이 갈라졌습니다. rebase·force 하지 말고 학생에게 확인."; exit 1; }
fi

BEFORE=$(git rev-parse HEAD)
if git merge-base --is-ancestor origin/main HEAD; then
  echo "✅ 이미 최신입니다 (반영할 강의자료 없음)"; exit 0
fi
BACKUP="backup/pre-sync-$(date +%Y%m%d-%H%M%S)"; git branch "$BACKUP" "$BEFORE"
{ echo "NAME=$NAME"; echo "BEFORE=$BEFORE"; echo "BACKUP=$BACKUP"; } > .git/ktc-sync-env

echo "🛟 백업: $BACKUP  (되돌리기: git reset --keep $BACKUP)"
echo ""
echo "== main 에서 들어올 변경 =="
git diff --stat "$(git merge-base HEAD origin/main)" origin/main | tail -20
echo "→ 다음: §6 머지"
```

---

## 6. 머지 (부채 청산 + 강의자료 반영)

```bash
# ktc-block: merge
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
[ -f .git/ktc-sync-env ] || { echo "❌ §5 진단을 먼저."; exit 1; }
set -a; . .git/ktc-sync-env; set +a
: "${NAME:?}"; : "${BEFORE:?}"; : "${BACKUP:?}"

# ★ 파일 복사가 아니라 **머지**다. 머지해야 히스토리가 통합되고 다음 주차부터 충돌이 사라진다.
#   --no-commit: 커밋 전에 학생 파일을 지킬 기회를 갖는다(아래 ★★).
git merge --no-commit --no-ff origin/main >/dev/null 2>&1 || true

: > .git/ktc-sync-ask     # 학생에게 물어봐야 하는 파일

# 내가 직접 고친 파일인가? (내 버전이 main 히스토리의 어떤 버전과도 다르면 = 내 손으로 고친 것)
# ⚠️ 파이프라인 + pipefail 로 판정하지 말 것 — git rev-parse 의 개별 실패가 판정을 뒤집는다.
is_my_edit() {   # $1=경로 $2=내 blob ; 0 = 내가 직접 고침
  local f="$1" mine="$2" b c
  [ -n "$mine" ] || return 1
  for c in $(git rev-list origin/main); do
    b=$(git rev-parse "$c:$f" 2>/dev/null) || continue
    [ "$b" = "$mine" ] && return 1
  done
  return 0
}

for f in $(git diff --name-only --diff-filter=U); do
  case "$f" in
    student_parts/*)
      # 내 버전이 없는 충돌(내가 파일을 지웠는데 main 이 고침 등)은 임의 판단 금지 → 학생에게
      if git checkout --ours -- "$f" 2>/dev/null; then
        git add -- "$f"; echo "  내 것 유지: $f"
      else
        echo "$f" >> .git/ktc-sync-ask
        echo "  ❓ $f — 내 쪽에 이 파일이 없습니다(삭제/추가 충돌). 학생 확인 필요"
      fi ;;
    *)
      mine=$(git rev-parse "$BEFORE:$f" 2>/dev/null || echo "")
      if is_my_edit "$f" "$mine"; then
        # 내가 직접 고친 파일이다 → 임의로 main 것으로 덮으면 내 작업이 사라진다
        echo "$f" >> .git/ktc-sync-ask
        echo "  ❓ 내가 직접 고친 파일: $f — 학생 확인 필요"
      elif git checkout --theirs -- "$f" 2>/dev/null; then
        git add -- "$f"; echo "  main 것 사용: $f"
      else
        echo "$f" >> .git/ktc-sync-ask
        echo "  ❓ $f — 삭제/추가 충돌. 학생 확인 필요"
      fi ;;
  esac
done

# ★★ 기존 학생 파일은 **무조건 내 버전으로 복원**한다.
#    충돌이 안 나도(= git 이 조용히 자동 병합해도) 강사 답안이 내 코드에 섞일 수 있다.
#    신규 파일(이번 주차 과제)은 그대로 받는다.
restored=0
for f in $(git diff --cached --name-only -- student_parts/ 2>/dev/null); do
  git cat-file -e "$BEFORE:$f" 2>/dev/null || continue      # 신규 파일 → 그대로 둔다
  git checkout "$BEFORE" -- "$f"; git add -- "$f"
  restored=$((restored+1))
done
if [ "$restored" -gt 0 ]; then
  echo "  🛡️ 기존 학생 파일 $restored 개를 내 버전으로 복원 (강사 답안 혼입 방지)"
fi

if [ -s .git/ktc-sync-ask ]; then
  echo ""
  echo "❌ 학생 확인이 필요합니다 — 커밋하지 않았습니다."
  echo "→ Claude: 아래 파일마다 **내 수정본 vs 강사 최신본**을 학생에게 보여주고 어느 쪽을 남길지 물어보세요."
  while read -r f; do
    echo ""; echo "── $f (내가 고친 부분)"
    git diff "$(git merge-base "$BEFORE" origin/main)" "$BEFORE" -- "$f" | head -30
  done < .git/ktc-sync-ask
  echo ""
  echo "→ 학생이 정하면: 내 것 = git checkout --ours -- <파일> / 강사 것 = git checkout --theirs -- <파일>"
  echo "   그 뒤 git add <파일> 하고 §7 검증을 실행하세요."
  exit 1
fi

git commit --no-edit >/dev/null
echo "✅ 머지 완료 — 내 코드는 그대로, 새 강의자료 반영됨"
```

---

## 7. 검증 (통과해야 push)

```bash
# ktc-block: verify
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
set -a; . .git/ktc-sync-env; set +a
: "${BEFORE:?}"; : "${NAME:?}"; FAIL=0

# 머지가 안 끝났으면 여기서 잡는다
if [ -n "$(git diff --name-only --diff-filter=U)" ]; then
  echo "❌ 아직 해소되지 않은 충돌이 있습니다:"; git diff --name-only --diff-filter=U; FAIL=1
fi

# (1) 히스토리 재작성 금지 — 조치는 커밋을 '얹기만' 해야 한다
if git merge-base --is-ancestor "$BEFORE" HEAD; then echo "✅ 히스토리 보존"
else echo "❌ 히스토리 재작성됨(rebase/reset). 되돌리기: git reset --keep $BACKUP"; FAIL=1; fi

# (2) 충돌 마커 0
if git grep -nI -e '^<<<<<<<' -e '^>>>>>>>' -- . >/dev/null 2>&1; then
  echo "❌ 충돌 마커 남음:"; git grep -nI -e '^<<<<<<<' -- . | head; FAIL=1
else echo "✅ 충돌 마커 없음"; fi

# (3) ★ 내 코드 불변 — student_parts/ 가 머지 전(BEFORE)과 완전히 같아야 한다
if git diff --quiet "$BEFORE" HEAD -- student_parts/; then
  echo "✅ 내 코드 불변 (student_parts/ 변경 0)"
else
  echo "⚠️ student_parts/ 가 바뀌었습니다:"
  git diff --stat "$BEFORE" HEAD -- student_parts/
  echo "   ↑ 새 주차 파일이 '추가'만 됐으면 정상입니다. **기존 파일이 수정/삭제됐으면 내 코드가 덮인 것**입니다."
  if git diff --name-status "$BEFORE" HEAD -- student_parts/ | grep -qv '^A'; then
    echo "❌ 기존 파일이 수정/삭제됨 → 내 구현이 덮였습니다. git reset --keep $BACKUP 로 되돌리세요."; FAIL=1
  else
    echo "✅ 신규 파일 추가만 — 내 코드는 그대로"
  fi
fi

# (4) 부채 청산 — main 이 전부 흡수됐는가 (이게 다음 주차 충돌을 없앤다)
if git merge-base --is-ancestor origin/main HEAD; then echo "✅ main 전부 흡수됨 (부채 0)"
else echo "❌ main 이 아직 다 안 들어왔습니다 — 머지가 덜 끝났습니다."; FAIL=1; fi

[ "$FAIL" = 0 ] && echo "" && echo "✅ 검증 통과 → §8" || { echo ""; echo "❌ 검증 실패 — push 하지 마세요."; exit 1; }
```

---

## 8. push + 이번 주차 브랜치

```bash
# ktc-block: finish
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
set -a; . .git/ktc-sync-env; set +a
: "${NAME:?}"

# 이번 주차 번호는 **강의자료에서 자동 판정**한다 (student_parts/weekNN_*.py 의 최대 번호).
# 하드코딩하지 않으므로 이 문서는 4주차·5주차에도 그대로 재사용된다.
N=$(git ls-files 'student_parts/week*' | sed -n 's#^student_parts/week0*\([0-9][0-9]*\).*#\1#p' | sort -n | tail -1)
if [ -z "$N" ]; then
  echo "❌ 이번 주차 번호를 찾지 못했습니다. 학생에게 물어보고 N=<주차> 를 넣어 다시 실행하세요."; exit 1
fi
echo "이번 주차 = week$N (강의자료에서 자동 판정)"

git push origin "$NAME/final"          # --force 절대 금지
git checkout -b "$NAME/week$N" 2>/dev/null || git checkout "$NAME/week$N"
echo "✅ $NAME/week$N 에서 과제 진행 → 완료 후 PR (base = $NAME/final)"
echo "   이전 주차 정답은 student_parts_baseline/ 에 있습니다 — **참고용**. 본인 코드를 바꾸지 마세요."
```

---

## 9. 꼬였을 때

```bash
# ktc-block: recover
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
git merge --abort 2>/dev/null || true
git rebase --abort 2>/dev/null || true
echo "── 백업 브랜치 ──"; git for-each-ref --format='%(refname:short)' 'refs/heads/backup/*'
echo "── 내 작업이 살아있는 원격 브랜치 ──"; git for-each-ref --format='%(refname:short)' refs/remotes/origin | grep '/week' || true
echo "→ 되돌리기: git reset --keep <백업브랜치>   또는  git checkout <브랜치> -- <덮인 파일>"
echo "→ 정리 후 §5 부터 다시."
```

---

## 10. 예상 부작용 (Claude 는 이걸 알고 있어야 한다)

| 부작용                                        | 왜                                       | 대응                                                                |
| ------------------------------------------ | --------------------------------------- | ----------------------------------------------------------------- |
| **`final` 에 머지 커밋이 생긴다**                   | 파일 복사가 아니라 머지이므로                        | 정상이다. 이게 있어야 부채가 청산되고 다음 주차가 깨끗해진다. PR diff(base=final)에는 영향 없음   |
| **`fixed/`·`run.sh` 를 직접 고친 학생은 그 수정이 위험** | 규칙상 그 파일은 main 것을 쓰므로                   | §6 이 **자동 감지해서 멈추고 학생에게 물어본다**. 임의로 덮지 않는다                        |
| **배포 형태 B 면 거의 전원 충돌**                     | main 이 `student_parts/` 를 stub 으로 되돌리므로 | 정상. §3 규칙(`--ours`)으로 학생 코드는 그대로 지켜진다                             |
| **`student_parts_baseline/` 가 학생 repo 에 생긴다** | 강사 정답 배포 경로                             | 참고용. 실행에 영향 없음. **학생이 이걸 복사해 자기 코드에 붙이면 과정 취지가 무너진다** — 안내할 것     |
| **열린 PR 이 있으면 그 PR 의 diff 가 줄어 보일 수 있다**   | base(`final`)가 최신화되므로                   | 무해. PR 을 닫지 말 것                                                   |
| **`weekN` 브랜치를 이미 판 학생**                   | `final` 만 머지하면 `weekN` 은 낡은 상태          | 그 학생은 `weekN` 에서도 `git merge <이름>/final` 을 한 번 더 해야 한다. 학생에게 알릴 것 |

---

## 11. Claude 안전 규칙 (요약)

- **학생 코드 유실 방지 최우선.** `student_parts/` 는 **언제나 `--ours`**.
- **워킹트리가 더러우면 진행하지 않는다.**
- **검증(§7) 통과 전에 push 하지 않는다.**
- **`git rebase` / `git push --force` / `git reset --hard` 금지.** (2026-07-08 에 이걸로 학생들의 머지된 PR 기록이 사라졌다)
- **애매하면 실행하지 말고 학생에게 물어본다.**
