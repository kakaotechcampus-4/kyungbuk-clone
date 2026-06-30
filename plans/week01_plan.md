# Week 1 Plan — personal_create_schedule / list / delete 구현

## 목표
`student_parts/week01_wake_up_nana.py` 안의 `@tool` 3개를 완성해
LangChain agent가 개인 일정을 생성·조회·삭제할 수 있게 한다.

---

## 구현 대상

### 1. `personal_create_schedule`
- 입력: `title`, `date`, `start_time`, `end_time`(기본 "미정"), `attendees`(기본 `[]`)
- 처리
  - `attendees=None`이면 빈 리스트로 치환
  - `_new_personal_id()`로 id 생성 (접두어 `personal_`)
  - `_now_iso()`로 `created_at` 기록
  - `session_id=current_session_scope()` 포함
  - `PERSONAL_SCHEDULES.append(schedule)`
- 반환 JSON: `{ ok, tool_name, created_schedule }`

### 2. `personal_list_schedules`
- 입력: `date_from`(없으면 None), `date_to`(없으면 None)
- 처리
  - `_current_session_schedules()`로 현재 세션 일정만 추출
  - `date_from` 있으면 `schedule["date"] >= date_from` 필터
  - `date_to` 있으면 `schedule["date"] <= date_to` 필터
  - PERSONAL_SCHEDULES 원본 수정 금지
- 반환 JSON: `{ ok, tool_name, schedules }`

### 3. `personal_delete_schedule`
- 입력: `schedule_id`
- 처리
  - 현재 세션 + schedule_id 일치하는 항목만 제거
  - `PERSONAL_SCHEDULES[:] = [...]` 방식으로 리스트 객체 유지
  - 삭제 전후 길이 차이로 `deleted` (bool) 결정
- 반환 JSON: `{ ok, tool_name, deleted, schedule_id }`

---

## 검증 체크리스트
- [ ] 일정 생성 후 trace에 `personal_create_schedule` tool_call 확인
- [ ] 반환 JSON에 `created_schedule` 키 존재
- [ ] 날짜 범위 필터(`date_from`/`date_to`) 동작 확인
- [ ] 다른 session의 일정은 조회/삭제에서 제외되는지 확인
- [ ] `./run.sh --week1` 실행 후 채팅 하네스 테스트 통과
