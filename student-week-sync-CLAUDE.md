# 강의자료 최신화 — Claude 실행 지시서

> **사용법 (3단계)**
> 1. **본인 clone repo 폴더에서 Claude Code 를 실행**합니다. (터미널에서 `cd <학교>-clone` → `claude`)
>    → git 명령이 이 폴더에서 돌아야 하므로 **가장 중요**합니다.
> 2. 다운로드한 **이 파일(`CLAUDE.md`)을 Claude Code 채팅창에 드래그**해서 넣습니다.
> 3. **"이 문서대로 강의자료 최신화 해줘"** 라고 입력하면 끝. Claude 가 내 git 상태 진단 → 안전 반영 → 검증까지 대신 해줍니다.
>
> ⚠️ 파일은 다운로드 폴더에 있어도 되지만, **Claude Code 자체는 반드시 내 clone repo 폴더(`<학교>-clone`)에서 켜야** 합니다.
> 다운로드 폴더 등 엉뚱한 곳에서 켜면 git 명령이 실패합니다 — 이 경우 Claude 가 감지해서 "clone repo 폴더에서 다시 켜세요" 라고 안내합니다.

---

## 0. Claude, 당신의 임무 (한 줄)

이 학생의 `<이름>/final` 브랜치에 **week02 강의자료를 반영**하되, **학생이 직접 짠 주차 과제 코드(`student_parts/weekNN_*.py`)는 절대 학생 버전으로 보존**한다. 진단 → 조치 → 검증까지 수행하고, 애매하면 파괴적 명령 대신 학생에게 물어본다.

---

## 1. 문제 상황 (왜 그냥 pull 하면 안 되나)

- 오늘 `main` 에 **week02 강의자료**가 올라왔다. 그런데 이번 업데이트는 **week01 학생 구현 파일(`student_parts/week01_wake_up_nana.py`)의 `# TODO` 자리를 "모범답안"으로 채워서** 함께 내려왔다.
- 학생은 바로 그 `# TODO` 자리에 **본인 구현**을 채워 `final` 에 머지해 둔 상태다.
- 따라서 `git pull origin main` / `git merge origin/main` 을 **그냥 하면**:
  - `student_parts/week01_wake_up_nana.py` 에서 **머지 충돌** 발생 (거의 전원)
  - 충돌을 잘못 풀면(=incoming/theirs 채택) **학생이 짠 week01 이 모범답안으로 덮여 사라짐** ⚠️

### 목표 상태(불변식) — 조치가 끝난 뒤 `final` 은 반드시 이래야 한다
- ✅ `student_parts/week01_wake_up_nana.py` = **내가 짠 버전** (모범답안 아님)
- ✅ `student_parts/week02_structure_natural_language_requests.py` = main 최신 (신규)
- ✅ `fixed/week_agent_registry.py`, `run.sh` = main 최신 (week2 실행 배선)

---

## 2. 먼저 확인 (Claude 가 실행)

```bash
# (0) 여기가 "내 clone repo 폴더"가 맞는지 먼저 확인 — 아니면 여기서 멈추고 학생에게 안내한다.
#     (학생이 다운로드 폴더 등에서 Claude Code 를 켰을 수 있음)
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "❌ 여기는 git 저장소가 아닙니다. 터미널에서 본인 clone repo 폴더(<학교>-clone)로 이동한 뒤 거기서 Claude Code 를 다시 켜세요."
  # → 학생에게 위 안내만 하고 이후 단계는 실행하지 않는다.
fi
git remote -v | grep -q "kakaotechcampus-4/.*-clone" \
  && echo "✅ 학교 clone repo 확인됨" \
  || echo "⚠️ origin 이 'kakaotechcampus-4/<학교>-clone' 이 아님 — 올바른 폴더인지 학생에게 확인 후 진행"

# (1) 안전용 현재 위치 기록 (무엇도 잃지 않음 — 되돌릴 기준점)
git rev-parse HEAD

# (2) 내 이름/브랜치 감지
#  - 현재 브랜치(<이름>/final 또는 <이름>/weekN)에서 <이름>을 뽑고, origin에 그 final이 있으면 신뢰한다.
#  - 못 잡으면(main/detached 등) 로컬 */final 이 "정확히 1개"일 때만 자동, 아니면 학생에게 물어본다.
CUR=$(git rev-parse --abbrev-ref HEAD)
NAME=${CUR%%/*}                                   # 예: 'gildong/week1' -> 'gildong'
if ! git rev-parse --verify -q "refs/heads/$NAME/final" >/dev/null \
   && ! git rev-parse --verify -q "refs/remotes/origin/$NAME/final" >/dev/null; then
  FINALS=$(git for-each-ref --format='%(refname:short)' refs/heads | grep '/final$')
  if [ "$(printf '%s\n' "$FINALS" | grep -c .)" = "1" ]; then
    NAME=$(printf '%s' "$FINALS" | sed 's#/final##')
  else
    NAME=""                                       # 감지 실패
  fi
fi
[ -n "$NAME" ] && echo "감지된 이름: $NAME" || echo "감지 실패 → 학생에게 '본인 브랜치 이름(<이름>/final 의 <이름>)'을 물어본다"

# (3) 내 첫 주차(작업) 브랜치 = 복구 소스
git fetch origin
WK1=$(git branch -r --format='%(refname:short)' | grep "origin/$NAME/week" | sort | head -1)
echo "복구 소스 브랜치: $WK1"      # 예: origin/gildong/week1

# (4) 지금 상태 진단
git status
git log --oneline -8
```

위 결과로 아래 **case** 중 하나를 판정한다.

---

## 3. 예상 case & 해결 (진단 후 해당 case 하나만 적용)

> ⚠️ 공통 안전 규칙: **`--force` push 금지**, 학생 확인 없이 원격을 덮어쓰는 행위 금지. 파괴적 명령(`reset --hard`) 전에는 그 대상이 "내가 만든 게 아닌 새 강의자료/머지"임을 확인한다. git 은 커밋된 작업을 지우지 않으므로(reflog·주차 브랜치 보존) 복구는 항상 가능하다.

### Case 0 — 아직 아무것도 안 함 (clean)
- 신호: `git status` = "nothing to commit, working tree clean", `git log` 에 main 머지 커밋 없음.
- 조치: 곧바로 **§4 표준 반영 절차**로 간다.

### Case A — `git pull` 이 `fatal: Need to specify how to reconcile...` 만 뱉고 끝남
- 신호: 아무 커밋도 안 생김, 트리 clean.
- 조치: 사실상 아무 일도 안 일어난 것. **§4 표준 반영 절차**로 간다.

### Case B — 머지하다 충돌로 멈춤 (진행 중)
- 신호: `git status` 에 `Unmerged paths` / `fix conflicts and run "git commit"`.
- 조치: 머지를 취소해 원상복구 후 표준 절차.
```bash
git merge --abort
```
→ 이후 **§4**.

### Case C — 머지를 커밋했지만 아직 push 안 함
- 신호: `git status` 에 `Your branch is ahead of 'origin/<이름>/final'`, `git log` 최상단에 `Merge ... origin/main` 커밋.
- 조치: 원격(아직 깨끗)으로 로컬 final 을 되돌린 뒤 표준 절차. (아직 week2 작업 전이라 안전)
```bash
git checkout "$NAME/final"
git reset --hard "origin/$NAME/final"
```
→ 이후 **§4**.

### Case D — 머지를 push 까지 함 (원격 final 도 바뀜)
- 신호: `git log origin/$NAME/final` 에 `Merge ... origin/main` 커밋이 있음.
- 조치: 먼저 **내 week01 이 살아있는지 판정**한다.
```bash
git checkout "$NAME/final" && git pull --ff-only origin "$NAME/final"
git diff --quiet "$WK1" -- student_parts/week01_wake_up_nana.py \
  && echo "OK: 내 week01 유지됨" \
  || echo "RESTORE: week01 이 모범답안으로 덮임 → 복구 필요"
```
- `OK` 면 → 이미 정상. **§4 에서 week02 파일 반영분만** 확인하고 넘어간다.
- `RESTORE` 면 → **§4-복구**로 내 week01 을 되살린 뒤 진행한다.

---

## 4. 표준 반영 절차 (clean 상태에서 — 아래 둘 중 하나)

먼저 항상:
```bash
git fetch origin
git checkout "$NAME/final"
```

### 방법 1 — selective (권장, 머지·충돌 개념 없음)
새 강의자료 파일만 골라 가져온다. **week01 은 손대지 않으므로 안전.**
```bash
git checkout origin/main -- \
  student_parts/week02_structure_natural_language_requests.py \
  fixed/week_agent_registry.py \
  run.sh
git commit -m "chore: week02 강의자료 반영 (내 구현 유지)"
git push origin "$NAME/final"
```

### 방법 2 — `-X ours` 머지 (한 줄, main 부수 변경까지 자동 반영)
충돌 나는 곳은 전부 내 것(final)으로, 나머지 main 변경은 정상 병합한다.
```bash
git merge -X ours --no-edit origin/main
git push origin "$NAME/final"
```
> 주의: `-X ours` 는 **모든 충돌을 내 쪽으로** 해결한다. 이번 주 충돌은 "week01 모범답안" 하나뿐이라 이게 정답이다. (다른 주차엔 충돌 파일이 이전주차 답안뿐인지 먼저 확인)

### 방법 3 — 주차 브랜치에서 바로 가져오기 (final 은 그대로 두는 방식)
`final` 을 직접 안 바꾸고, **이번 주 작업 브랜치 `<이름>/week2` 에** 새 강의자료를 얹는다. `week2` 는 `final` 에서 분기하므로 **내 week01 풀이가 그대로 포함**된다. 새 강의자료(week02 문제)는 main 에서 **파일만** 가져온다(머지 아님 → 충돌 없음).
```bash
git checkout "$NAME/final"
git pull origin "$NAME/final"       # 내 week01 풀이 포함된 최신 final
git checkout -b "$NAME/week2"       # ★ 'week2' 단독 금지 — 반드시 '<이름>/week2'
# 새 강의자료 파일만 얹기 (★ 로컬 'main' 아니라 'origin/main' — fetch 먼저)
git checkout origin/main -- \
  student_parts/week02_structure_natural_language_requests.py \
  fixed/week_agent_registry.py \
  run.sh
git add -A && git commit -m "chore: week02 강의자료 반영"
# 이제 이 브랜치에서 week02 과제 진행 → 다 하면 PR (base = <이름>/final)
```
> - `student_parts/week02_*.py` 는 **답안이 아니라 문제(TODO)** 다 — 이걸 구현하는 게 이번 주 과제.
> - 반드시 **3개 세트**(week02 문제 + `fixed/week_agent_registry.py` + `run.sh`). week02 파일 하나만 가져오면 `./run.sh --week2` 가 "Week 1만 포함" 으로 **실행 거부**된다.
> - 방법 1·2 와 차이: 이 방식은 `final` 에 새 강의자료가 **PR 머지 시점에** 들어간다(그 전까진 week2 브랜치에만). week02→final PR diff 에 스캐폴드 3파일이 함께 보이는 것은 정상.

### §4-복구 — (Case D 에서 week01 이 덮인 경우만)
내 실제 작업 브랜치에서 week01 을 되살린다.
```bash
git checkout "$WK1" -- student_parts/week01_wake_up_nana.py
git commit -m "restore: 내 week01 구현 복구"
git push origin "$NAME/final"
```

---

## 5. 최종 검증 (반드시 실행 — 여기서 통과해야 끝)

```bash
# (1) week01 = 내 작업 브랜치의 것인가? (모범답안으로 안 덮였는가)
git diff --quiet "$WK1" -- student_parts/week01_wake_up_nana.py \
  && echo "✅ week01 = 내 구현" || echo "❌ week01 이 다름 → §4-복구 실행"

# (2) week02 신규 파일 존재?
test -f student_parts/week02_structure_natural_language_requests.py \
  && echo "✅ week02 파일 있음" || echo "❌ week02 파일 없음 → §4 방법1 재실행"

# (3) week2 실행 배선(레지스트리/run.sh) 반영?
grep -q "week02_structure_natural_language_requests" fixed/week_agent_registry.py \
  && grep -qE '\-\-week2|week\(\[12\]\)' run.sh \
  && echo "✅ week2 실행 배선 OK" || echo "❌ registry/run.sh 미반영 → §4 방법1 재실행"

# (4) (선택) 실제 실행 확인
# ./run.sh --week2
```

세 개(또는 네 개)가 전부 ✅ 면 완료. 이후 이번 주 작업 브랜치를 판다:
```bash
# 방법 1·2(final 에 반영)로 진행한 경우 — week2 브랜치를 새로 판다.
# 방법 3 을 썼다면 이미 <이름>/week2 에 있으므로 이 단계는 건너뛴다.
git checkout -b "$NAME/week2" 2>/dev/null || git checkout "$NAME/week2"
# ★ 반드시 '<이름>/week2' (브랜치 규칙: */week* 만 허용, 'week2' 단독은 차단됨)
# 이제 week02 과제 진행 → 다 하면 PR (base = <이름>/final)
```

---

## 6. Claude 가 반드시 지킬 것 (안전 규칙)

- **학생 작업 유실 방지 최우선.** week01/기타 주차 파일은 항상 `origin/<이름>/week*` 브랜치가 원본이다 — 확신 없으면 거기서 복구.
- **`git push --force` 금지.** 원격 final 을 force 로 덮지 않는다.
- **이름/브랜치 감지가 불확실하면 실행하지 말고 학생에게 물어본다** (`<이름>/final` 의 `<이름>`).
- 진행 중 머지가 꼬였으면 새 파괴적 명령보다 **`git merge --abort` 로 원위치** 후 §4.
- 이번 지시서는 **week02(이번 주) 임시 조치**다. 다음 주차엔 파일명·세트가 달라질 수 있으니, `git show --stat <최근 base-code 동기화 커밋>` 으로 "가져올 새 파일 목록"을 다시 뽑아 §4 방법1의 파일 인자를 교체한다. (충돌 대상은 여전히 "직전 주차 답안 파일"일 가능성이 높고, 처리 원칙은 동일: **내 주차 파일은 keep-ours/복구, 새 강의자료 파일만 반영.**)
