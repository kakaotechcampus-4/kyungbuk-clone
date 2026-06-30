# Week 1 구현 설명 — week01_wake_up_nana.py

## 1. `personal_create_schedule`

```python
@tool
def personal_create_schedule(title, date, start_time, end_time="미정", attendees=None) -> str:
```

### 구현 흐름

1. `attendees=None`이면 빈 리스트(`[]`)로 교체
2. schedule dict 구성
   - `id`: `_new_personal_id()` → `personal_` 접두어 + 10자리 UUID hex
   - `session_id`: `current_session_scope()` → 현재 대화 범위 식별자
   - `created_at`: `_now_iso()` → timezone 포함 ISO 시각
3. `PERSONAL_SCHEDULES.append(schedule)` 으로 전역 리스트에 추가
4. `_json({ ok, tool_name, created_schedule })` 반환

### 반환 예시

```json
{
  "ok": true,
  "tool_name": "personal_create_schedule",
  "created_schedule": {
    "id": "personal_3a1f9c2e87",
    "title": "팀 미팅",
    "date": "2026-07-01",
    "start_time": "10:00",
    "end_time": "11:00",
    "attendees": [],
    "session_id": "conv_abc123",
    "created_at": "2026-06-30T10:00:00.000000+09:00"
  }
}
```

---

## 2. `personal_list_schedules`

```python
@tool
def personal_list_schedules(date_from=None, date_to=None) -> str:
```

### 구현 흐름

1. `_current_session_schedules()` 로 현재 세션 일정만 복사본으로 꺼냄  
   → 원본 `PERSONAL_SCHEDULES` 를 직접 수정하지 않음
2. `date_from` 이 있으면 `s["date"] >= date_from` 인 것만 유지
3. `date_to` 가 있으면 `s["date"] <= date_to` 인 것만 유지  
   → YYYY-MM-DD 문자열은 사전 순 비교로 날짜 대소 비교가 정확히 동작함
4. `_json({ ok, tool_name, schedules })` 반환

### 반환 예시

```json
{
  "ok": true,
  "tool_name": "personal_list_schedules",
  "schedules": [
    {
      "id": "personal_3a1f9c2e87",
      "title": "팀 미팅",
      "date": "2026-07-01",
      ...
    }
  ]
}
```

---

## 3. `personal_delete_schedule`

```python
@tool
def personal_delete_schedule(schedule_id: str) -> str:
```

### 구현 흐름

1. `session_id = current_session_scope()` 로 현재 세션 확인
2. 삭제 전 길이 `before = len(PERSONAL_SCHEDULES)` 저장
3. `PERSONAL_SCHEDULES[:] = [...]` 슬라이스 대입으로 리스트 객체 자체는 유지하면서 내용만 교체  
   → 삭제 조건: `s["id"] == schedule_id AND _schedule_scope(s) == session_id`  
   → **다른 세션의 같은 ID는 절대 삭제되지 않음**
4. 삭제 후 길이 `after = len(PERSONAL_SCHEDULES)` 와 비교해 `deleted` 값 결정
5. `_json({ ok, tool_name, deleted })` 반환

### 반환 예시 (삭제 성공)

```json
{
  "ok": true,
  "tool_name": "personal_delete_schedule",
  "deleted": true
}
```

---

## 4. `PERSONAL_SCHEDULES[:] = ...` 를 쓰는 이유

```python
# 이렇게 하면 안 됨 — 변수가 새 리스트 객체를 가리킴
PERSONAL_SCHEDULES = [s for s in PERSONAL_SCHEDULES if ...]

# 이렇게 해야 함 — 기존 리스트 객체의 내용을 교체
PERSONAL_SCHEDULES[:] = [s for s in PERSONAL_SCHEDULES if ...]
```

`list_personal_schedule_dicts()` 등 다른 코드가 이미 같은 리스트 객체를 참조하고 있기 때문에,  
슬라이스 대입으로 **같은 객체를 유지**하면서 내용만 바꿔야 한다.

---

## 5. 세션 격리 구조

```
대화 A (session_id = "conv_001")         대화 B (session_id = "conv_002")
────────────────────────────────         ────────────────────────────────
personal_abc → session_id: conv_001      personal_xyz → session_id: conv_002
personal_def → session_id: conv_001
```

- 조회/삭제 시 `_current_session_schedules()` 또는 `_schedule_scope()` 로 범위를 먼저 필터링
- 대화 B에서 `personal_abc` ID를 삭제 요청해도 `session_id`가 달라 삭제되지 않음

---

## 6. System Prompt (`week01_prompt_parts`)

```python
return [
    f"""당신은 Nana입니다. ...
오늘 날짜: {current_app_date_iso()}
...
"""
]
```

- `current_app_date_iso()` 로 LLM이 "오늘", "내일" 같은 상대 날짜 표현을 YYYY-MM-DD로 변환할 수 있게 현재 날짜를 주입
- `CHAT_MEMORY_PROMPT` 를 포함해 이전 대화 기억 지침도 함께 전달
