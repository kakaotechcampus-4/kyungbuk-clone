# Week 1 구현 계획 — `student_parts/week01_wake_up_nana.py`

## Context

Nana AI 일정 메이트의 1주차 과제다. LangChain agent가 개인 일정을 생성·조회·삭제할 수 있도록
`@tool` 3개를 구현하고, agent가 올바르게 동작하도록 system prompt도 추가한다.

---

## 구현 대상 파일

`kyungpook-clone/student_parts/week01_wake_up_nana.py`

---

## 구현 계획

### 1. `personal_create_schedule`

```python
@tool
def personal_create_schedule(title, date, start_time, end_time="미정", attendees=None):
```

- `attendees=None`이면 `[]`로 치환
- `_new_personal_id()`로 id, `_now_iso()`로 created_at 생성
- `session_id = current_session_scope()` 포함한 schedule dict 구성
- `PERSONAL_SCHEDULES.append(schedule)`
- 반환: `_json({"ok": True, "tool_name": "personal_create_schedule", "created_schedule": schedule})`

### 2. `personal_list_schedules`

```python
@tool
def personal_list_schedules(date_from=None, date_to=None):
```

- `_current_session_schedules()`로 현재 세션 일정만 추출 (원본 수정 금지)
- `date_from` 있으면 `schedule["date"] >= date_from` 필터
- `date_to` 있으면 `schedule["date"] <= date_to` 필터
- 반환: `_json({"ok": True, "tool_name": "personal_list_schedules", "schedules": filtered})`

### 3. `personal_delete_schedule`

```python
@tool
def personal_delete_schedule(schedule_id: str):
```

- 현재 세션 + `schedule_id` 일치 조건으로 새 리스트 구성
- `before = len(PERSONAL_SCHEDULES)`
- `PERSONAL_SCHEDULES[:] = [새 리스트]` — 리스트 객체 유지
- `deleted = len(PERSONAL_SCHEDULES) < before`
- 반환: `_json({"ok": deleted, "tool_name": "personal_delete_schedule", "deleted": deleted, "schedule_id": schedule_id})`

### 4. `week01_prompt_parts` — system prompt 추가

`current_app_date_iso()`는 이미 상단에 import되어 있으므로 바로 사용 가능.

```python
def week01_prompt_parts() -> list[str]:
    return [
        f"너는 개인 일정 메이트 나나다. "
        f"오늘은 {current_app_date_iso()}이다. "
        "상대적 날짜 표현(내일, 다음 주 등)은 이 날짜 기준으로 YYYY-MM-DD로 변환한다. "
        "일정 생성·조회·삭제가 필요하면 반드시 알맞은 tool을 호출한 뒤 짧게 답한다."
    ]
```

---

## 재사용하는 기존 함수

| 함수 | 위치 | 용도 |
|------|------|------|
| `_json(payload)` | 같은 파일 | tool 반환값 직렬화 |
| `_now_iso()` | 같은 파일 | created_at 값 |
| `_new_personal_id()` | 같은 파일 | personal_ 접두어 ID |
| `_current_session_schedules()` | 같은 파일 | 세션 범위 필터 |
| `current_session_scope()` | `fixed/session_scope.py` | 현재 session ID |
| `current_app_date_iso()` | `fixed/runtime_clock.py` | 오늘 날짜 (이미 import됨) |

---

## 검증 방법

### 수동 테스트 (E2E)
1. `./run.sh --week1` 실행
2. 채팅창에서 순서대로 테스트:
   - "내일 10시에 민수와 회의 잡아줘" → trace에 `personal_create_schedule` tool_call 확인, 반환 JSON에 `created_schedule` 키 존재
   - "지금 일정 보여줘" → `personal_list_schedules` tool_call 확인, `schedules` 키 존재
   - 특정 일정 ID로 "xxx 일정 삭제해줘" → `personal_delete_schedule` tool_call 확인, `deleted: true` 확인

### pytest 자동화 테스트

**파일 위치:** `tests/test_week01.py`

LLM 호출 없이 tool 함수를 직접 `.invoke()`로 호출해 검증한다.
각 테스트 전 `PERSONAL_SCHEDULES.clear()`로 상태 초기화.

**테스트 케이스 목록:**

| 테스트 함수 | 검증 내용 |
|------------|----------|
| `test_create_returns_created_schedule` | 반환 JSON에 `ok=True`, `created_schedule` 키 존재, title/date/start_time 값 일치 |
| `test_create_none_attendees_becomes_empty_list` | `attendees=None` 입력 시 `created_schedule["attendees"] == []` |
| `test_create_appends_to_store` | 생성 후 `PERSONAL_SCHEDULES` 길이가 1 증가 |
| `test_list_returns_current_session_only` | 다른 `session_id`로 직접 삽입한 일정은 list 결과에 포함되지 않음 |
| `test_list_date_from_filter` | `date_from="2026-07-10"` 이면 그 이전 날짜 일정은 제외 |
| `test_list_date_to_filter` | `date_to="2026-07-05"` 이면 그 이후 날짜 일정은 제외 |
| `test_list_does_not_mutate_store` | list 호출 전후 `PERSONAL_SCHEDULES` 길이 동일 |
| `test_delete_removes_correct_schedule` | `deleted=True`, `PERSONAL_SCHEDULES`에서 해당 ID 사라짐 |
| `test_delete_wrong_session_does_nothing` | 다른 session_id의 일정은 삭제되지 않음 (`deleted=False`) |
| `test_delete_nonexistent_id` | 없는 ID 삭제 시 `deleted=False` 반환, 리스트 변화 없음 |
| `test_list_date_range_combined` | `date_from` + `date_to` 동시 적용 시 범위 안 일정만 반환 (AND 조건) |
| `test_list_date_from_boundary_inclusive` | `date_from`과 정확히 같은 날짜 일정이 포함됨 (`>=` 경계 포함) |
| `test_list_date_to_boundary_inclusive` | `date_to`와 정확히 같은 날짜 일정이 포함됨 (`<=` 경계 포함) |
| `test_create_tool_name` | 반환 JSON의 `tool_name == "personal_create_schedule"` |
| `test_list_tool_name` | 반환 JSON의 `tool_name == "personal_list_schedules"` |
| `test_delete_tool_name` | 반환 JSON의 `tool_name == "personal_delete_schedule"` |
| `test_delete_then_list_e2e` | 삭제 후 `personal_list_schedules` 조회 시 해당 일정이 사라짐 (연동 검증) |
| `test_create_id_has_personal_prefix` | 생성된 일정 ID가 `"personal_"` 으로 시작함 |
| `test_create_has_session_id` | 생성된 일정 dict에 `session_id` 필드 존재 |
| `test_create_with_attendees` | 실제 attendees 값 전달 시 그대로 저장됨 |
| `test_list_empty_store` | 스토어가 비어있을 때 `ok=True`, `schedules=[]` 반환 |

**실행 명령:** `pytest tests/test_week01.py -v` (repo 루트에서)
