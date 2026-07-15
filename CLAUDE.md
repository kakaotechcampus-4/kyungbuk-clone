# Kanana Schedule Agent — CLAUDE.md

## 프로젝트 목적

카카오 카테캠 강의용 LangChain 실습 프로젝트입니다.
학생이 `student_parts/` 안의 `# TODO` 함수를 직접 구현하며 LangChain tool과
에이전트 실행 흐름을 배우는 것이 목표입니다.

## 구현 범위

- **구현 대상 (활성 주차)**: `student_parts/week03_build_nanas_logbook.py` 안의 `# TODO` 함수들
- **읽기 전용**: `fixed/` 디렉터리(구조 이해용 참고 코드 — 특히 SQLite 접근은 `fixed/app_store.py`의
  `AppSQLiteStore` 메서드로만 하고 직접 SQL을 작성하지 않습니다),
  `student_parts_baseline/`(이전 주차 정답 코드 — 참고용이며 그대로 복사해 붙이지 않습니다)
- **수정 금지**: `app.py`, `run.sh`, `pyproject.toml`, `uv.lock`

## 완료된 주차

- **Week 1** (`student_parts/week01_wake_up_nana.py`): 임시 메모리(`PERSONAL_SCHEDULES`) CRUD — 완료
- **Week 2** (`student_parts/week02_structure_natural_language_requests.py`): 자연어 →
  `StructuredRequest` 구조화 — 완료

## Week 3 도움 방식 (힌트 우선, 단계별, 메인과제 → 추가과제 순)

`week03_build_nanas_logbook.py` 안에 이미 `[3주차 수강생 구현 가이드]` 주석(목표·과제 구성·
메인/추가 티어·함수별 설명)이 있습니다. 정답을 알려주지 말고, 아래 순서대로 하나씩 힌트를 먼저
주고 학생이 시도한 뒤 피드백합니다. 메인과제 3개를 먼저 끝낸 뒤에만 추가과제로 넘어갑니다.

### 메인과제

**save_structured_request — 3단계**
1. `@tool(args_schema=SaveStructuredRequestInput)`가 이미 입력을 검증했으므로, 함수 인자를
   다시 Pydantic으로 만들지 않고 바로 저장용 dict로 정리하는 이유를 설명합니다.
2. `None`인 필드는 저장 dict에서 제외해야 하는 이유를 설명합니다.
3. `AppSQLiteStore.save_structured_request(...)` 호출 결과를 `tool_result(...)` /
   `json_payload(...)`로 감싸 반환하도록 안내합니다.

**list_saved_requests / get_saved_request — 2단계**
1. `kind`/`date_from`/`date_to` 필터를 `store.list_saved_requests(...)`에 그대로 전달하는
   흐름을 안내합니다.
2. `get_saved_request`는 `request_id` 단건 조회이며, 결과가 없어도 예외 없이 `row=None`을
   유지해야 하는 이유를 설명합니다.

**personal_list_saved_schedules — 2단계**
1. 기본 `kind`를 `"personal_schedule"`로 좁혀야 하는 이유를 설명합니다.
2. `filters`와 `schedules` 키를 포함해 반환하도록 안내합니다.

### 추가과제 (메인과제 완료 후)

- **personal_update_saved_schedule**: `None` 필드는 "수정하지 않음"이라는 규칙, `schedule_id`를
  못 찾으면 `ok=False`, 찾으면 `updated_schedule`/`shared_sync`를 함께 반환하는 이유를 설명합니다.
- **_delete_saved_schedules / personal_delete_saved_schedules**: 삭제 조건이 전혀 없으면
  거부해야 하는 안전 규칙을 설명하고, `deleted_count`/`filters`/`deleted`를 유지하도록 안내합니다.
- **personal_create_schedule (Week 1 호환)**: `week01_personal_create_schedule` 결과를
  `structured_request_from_week01_schedule()`로 변환해 SQLite에도 이중 기록하는 흐름을 설명합니다.
- **레거시 payload 정규화** (`unwrap_legacy_payload`, `_save_input_from`,
  `save_structured_request_payload`): 가장 마지막에, 필요할 때만 다룹니다.

## Week 3 핵심 제약 (구현 시 반드시 지킬 것)

- SQLite 접근은 반드시 `fixed/app_store.py`의 `AppSQLiteStore` 메서드를 통해서만 합니다. 직접
  SQL을 작성하지 않습니다.
- 모든 tool은 `json_payload(...)` 또는 `tool_result(...)`로 감싼 JSON **문자열**을 반환합니다.
- `@tool(args_schema=...)`가 이미 입력을 검증하므로 tool 본문에서 Pydantic 모델을 다시 만들지
  않습니다.
- 수정 tool에서 `None` 필드는 "값을 바꾸지 않음"을 의미합니다.
- 삭제 tool은 조건이 전혀 없는 전체 삭제를 막는 안전장치를 반드시 거칩니다.

## 검증 방법

앱을 실행하고 상세 Trace 탭에서 확인합니다.

```bash
./run.sh --week3
```

자동화 테스트 없음 — Trace JSON이 기대한 키와 값을 가지는지 눈으로 확인합니다.

메인과제 확인 순서:
1. "내일 10시 개인 코칭 저장해줘" 입력 → trace에서 `extract_schedule_request` 다음
   `save_structured_request`가 호출되는지 확인합니다.
2. "내 일정 보여줘" 입력 → `personal_list_saved_schedules`가 호출되는지 확인합니다.
3. 앱을 재시작하거나 새 대화를 열어도 저장된 일정이 그대로 보이는지 확인합니다.

추가과제 확인 순서: 저장된 일정을 `personal_update_saved_schedule`로 수정한 뒤 목록에서 값이
바뀌었는지, `personal_delete_saved_schedules`로 삭제한 뒤 목록에서 사라졌는지 확인합니다.

## 주차 경계

Week 3는 SQLite 저장/조회/수정/삭제까지만 다룹니다.
Week 4 이후 개념(RAG 검색 도구 신설, 외부 캘린더 조율 로직 직접 구현 등)을 Week 3 코드에 미리
추가하지 않습니다. 외부 공유 저장소 동기화는 이미 `AppSQLiteStore.update_schedule` /
`delete_schedule` 내부에서 처리되므로, 학생이 별도로 MCP 동기화 코드를 작성할 필요는 없습니다.
