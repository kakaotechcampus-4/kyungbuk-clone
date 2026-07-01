# Kanana Schedule Agent — CLAUDE.md

## 프로젝트 목적

카카오 카테캠 강의용 LangChain 실습 프로젝트입니다.
학생이 `student_parts/` 안의 `# TODO` 함수를 직접 구현하며 LangChain tool과
에이전트 실행 흐름을 배우는 것이 목표입니다.

## 구현 범위

- **구현 대상**: `student_parts/week01_wake_up_nana.py` 안의 `# TODO` 함수 3개
- **읽기 전용**: `fixed/` 디렉터리는 수정하지 않습니다 (구조 이해용 참고 코드)
- **수정 금지**: `app.py`, `run.sh`, `pyproject.toml`, `uv.lock`

## 도움 방식 (힌트 우선, 단계별 진행)

한 번에 세 함수를 모두 구현하지 않습니다.
아래 단계 순서대로 하나씩 진행하며, 각 단계마다 힌트를 먼저 주고 학생이 시도한 뒤 피드백합니다.

### personal_create_schedule — 3단계

1. **dict 설계**: 어떤 필드가 필요한지 먼저 생각하게 합니다.
   (`id`, `title`, `date`, `start_time`, `end_time`, `attendees`, `session_id`, `created_at`)
2. **저장**: `PERSONAL_SCHEDULES.append(schedule)`로 리스트에 넣는 방법을 안내합니다.
3. **반환**: `_json({"ok": True, "tool_name": ..., "created_schedule": ...})` 형태로 반환하도록 합니다.

### personal_list_schedules — 3단계

1. **범위 필터**: `_current_session_schedules()`로 현재 대화 일정만 추리는 이유를 설명합니다.
2. **날짜 필터**: `date_from` / `date_to`가 있을 때 문자열 비교(`>=`, `<=`)로 좁히는 방법을 안내합니다.
3. **반환**: `_json({"ok": True, "tool_name": ..., "schedules": [...]})` 형태로 반환하도록 합니다.

### personal_delete_schedule — 3단계

1. **대상 찾기**: `schedule_id`와 `session_id` 두 조건을 동시에 확인해야 하는 이유를 설명합니다.
2. **안전한 삭제**: `PERSONAL_SCHEDULES[:]`에 새 리스트를 대입해야 하는 이유(리스트 객체 유지)를 설명합니다.
3. **반환**: 삭제 전후 길이 비교로 `deleted` 값을 만들고 `_json(...)`으로 반환하도록 합니다.

## Week 1 핵심 제약 (구현 시 반드시 지킬 것)

- 일정은 `PERSONAL_SCHEDULES` 리스트(Python 메모리)에만 저장합니다. SQLite/AppStore 호출 없음.
- 모든 tool은 `_json(payload)` helper로 감싼 JSON **문자열**을 반환합니다.
- 각 일정 dict에는 반드시 `session_id = current_session_scope()`를 넣습니다.
- 조회/삭제 시 현재 `session_id`와 일치하는 일정만 대상으로 합니다.
- 반환 payload의 top-level 키: `created_schedule` / `schedules` / `deleted`

## 검증 방법

앱을 실행하고 상세 Trace 탭에서 확인합니다.

```bash
./run.sh --week1
```

자동화 테스트 없음 — Trace JSON이 기대한 키와 값을 가지는지 눈으로 확인합니다.
각 단계 구현 후 바로 실행해서 trace 결과가 어떻게 바뀌는지 비교하는 것을 권장합니다.

## 주차 경계

Week 1은 임시 메모리 CRUD만 다룹니다.
Week 2 이후 개념(SQLite 저장, structured output, RAG, MCP)을 Week 1 코드에 미리 추가하지 않습니다.
