# Kanana Schedule Agent — CLAUDE.md

## 프로젝트 목적

카카오 카테캠 강의용 LangChain 실습 프로젝트입니다.
학생이 `student_parts/` 안의 `# TODO` 함수를 직접 구현하며 LangChain tool과
에이전트 실행 흐름을 배우는 것이 목표입니다.

## 구현 범위

- **구현 대상 (활성 주차)**: `student_parts/week05_load_kanas_past_conversations.py` 안의 `# TODO` 함수들
- **읽기 전용**: `fixed/` 디렉터리(구조 이해용 참고 코드 — 특히 SQLite 접근은 `fixed/app_store.py`의
  `AppSQLiteStore` 메서드로만 하고, 개인 참고자료 벡터 검색은 `fixed/reference_store.py`의
  `PersonalReferenceStore` 메서드로만 하고, 대화 RAG는 `fixed/conversation_rag_store.py`의
  `ConversationRAGStore` 메서드로만 하고, 외부 멤버 대화/공유 일정은 `fixed/mcp_client.py`의
  `call_local_mcp_tool_sync`(이 파일 별칭 `call_mcp_tool_sync`)와 `fixed/external_mcp.py`의
  `call_external_tool_payload`로만 합니다 — 직접 SQL이나 ChromaDB 호출, MCP 서버
  (`mcp_server/sqlite_mcp_server.py`) 수정을 새로 하지 않습니다),
  `student_parts_baseline/`(이전 주차 정답 코드 — 참고용이며 그대로 복사해 붙이지 않습니다)
- **수정 금지**: `app.py`, `run.sh`, `pyproject.toml`, `uv.lock`

## 완료된 주차

- **Week 1** (`student_parts/week01_wake_up_nana.py`): 임시 메모리(`PERSONAL_SCHEDULES`) CRUD — 완료
- **Week 2** (`student_parts/week02_structure_natural_language_requests.py`): 자연어 →
  `StructuredRequest` 구조화 — 완료
- **Week 3** (`student_parts/week03_build_nanas_logbook.py`): `StructuredRequest`를 SQLite에
  저장/조회/수정/삭제 — 완료
- **Week 4** (`student_parts/week04_retrieve_nanas_memory.py`): 개인 참고자료/SQLite 기록/대화
  RAG 검색 tool 분리 — 완료

## Week 5 도움 방식 (힌트 우선, 단계별, 메인과제 → 추가과제 순)

`week05_load_kanas_past_conversations.py` 안에 이미 `[5주차 수강생 구현 가이드]` 주석(목표·과제
구성·메인/추가 티어·함수별 설명)이 있습니다. 정답을 알려주지 말고, 아래 순서대로 하나씩 힌트를
먼저 주고 학생이 시도한 뒤 피드백합니다. 메인과제 5개를 먼저 끝낸 뒤에만 추가과제로 넘어갑니다.

### 메인과제 (권장 순서)

**1. `_personal_schedules_for_current_scope()`** — 이후 모든 tool이 의존하는 기반 helper
- `AppSQLiteStore(CONFIG.app_db_path).list_schedules(...)`(세션 필터 없이 전체 저장 일정)와
  Week 1 `PERSONAL_SCHEDULES`(현재 대화의 임시 일정)를 합쳐야 하는 이유를 설명합니다.
- `PERSONAL_SCHEDULES`는 `_schedule_scope(schedule)`로 현재 세션 범위만 남기고, SQLite에
  이미 저장된 것과 `schedule_id`/`id` 기준으로 중복 제거해야 하는 이유를 설명합니다.

**2. `search_previous_conversations` — 가장 단순, 첫 시도로 적합**
- `call_mcp_tool_sync("search_previous_conversations", {"query": ..., "member_names": ...,
  "limit": ...})`를 호출하고 결과 문자열을 그대로 반환하도록 안내합니다.
- 멤버 이름 정규화는 외부 SQLite store/MCP 경계에서 한 번만 일어나므로 wrapper에서 다시
  변환하지 않는 이유를 설명합니다.

**3. `load_conversation_messages`**
- `call_external_tool_payload("load_conversation_messages", {"conversation_id": ...})` 결과
  dict를 `json_payload(...)`로 감싸 반환하도록 안내합니다.
- sender/content/created_at 순서를 가공하지 않고 그대로 보존해야 하는 이유를 설명합니다.

**4. `list_shared_schedules`**
- `call_mcp_tool_sync("list_shared_schedules", args)` 호출과 결과 rows 전달 흐름을 안내합니다.
- 필터 없이 호출하면 외부 실습용 기본 공유 일정이 우선 반환된다는 점, 그리고 이 tool이
  Week 6 Kana 하위 agent에서도 그대로 재사용된다는 점을 설명합니다.

**5. `extract_schedules_from_history` + `_collect_member_schedules` + `collect_member_schedules`
— 가장 복잡, 마지막**
1. `extract_schedules_from_history`가 `call_mcp_tool_sync("extract_schedules_from_history",
   args)`를 호출해 외부 멤버 busy-time rows를 그대로 반환하도록 안내합니다.
2. `_collect_member_schedules`가 `_personal_schedules_for_current_scope()`의 내 일정을
   `_structured_request_from_schedule_row(row)`로 규격화한 뒤, 외부 멤버 rows와 같은
   `member_name`/`title`/`date`/`start_time`/`end_time`/`notes` 구조로 합쳐야 하는 이유를
   설명합니다. 멤버 이름/날짜 범위는 `normalize_external_member_names()` /
   `normalize_external_schedule_date_bounds()`로 정규화합니다.
3. `collect_member_schedules` tool은 이 rows에 `external_schedule_summary(rows)` 요약
   문자열도 함께 반환해야 LLM이 바쁜 시간을 자연어로 설명할 수 있다는 계약을 설명합니다.
   이 rows가 Week 6 공통 가능 시간 tool의 `busy_rows` 근거가 된다는 점도 짚어줍니다.

### 추가과제 (메인과제 완료 후)

- **`create_shared_schedule` / `delete_shared_schedule`**: 각각
  `call_mcp_tool_sync("create_shared_schedule" / "delete_shared_schedule", args)`를 호출해
  결과 payload를 그대로 전달하는 흐름을 안내합니다. `schedule_id` 또는
  `source_conversation_id`를 보존해야 나중에 수정/삭제 동기화가 가능하다는 이유를 설명합니다.
  구현하지 않기로 하면 `week05_tools()` 목록에서 이 두 tool을 빼면 된다는 점도 알려줍니다.

## Week 5 핵심 제약 (구현 시 반드시 지킬 것)

- 외부 멤버 대화/공유 일정 접근은 반드시 `call_mcp_tool_sync`(= `fixed/mcp_client.py`의
  `call_local_mcp_tool_sync`) 또는 `call_external_tool_payload`(`fixed/external_mcp.py`)를
  통해서만 합니다. `mcp_server/sqlite_mcp_server.py`를 직접 수정하거나 그 안의 SQL을
  이 파일에 새로 작성하지 않습니다.
- 내 일정 SQLite 접근은 반드시 `fixed/app_store.py`의 `AppSQLiteStore` 메서드를 통해서만
  합니다.
- 멤버 이름/날짜 범위 정규화는 `fixed/external_people_store.py`의
  `normalize_external_member_names()` / `normalize_external_schedule_date_bounds()`를
  사용하고, wrapper에서 중복으로 다시 정규화하지 않습니다.
- 모든 tool은 `json_payload(...)`로 감싼 JSON **문자열**을 반환합니다.
- `@tool(args_schema=...)`가 이미 입력을 검증하므로 tool 본문에서 Pydantic 모델을 다시 만들지
  않습니다.
- `_personal_schedules_for_current_scope()`는 SQLite 저장 일정과 현재 대화 범위의 임시
  일정만 합치고, 이미 SQLite에 저장된 일정과 중복하지 않습니다.
- `collect_member_schedules`/`extract_schedules_from_history` 결과 rows는
  `member_name`/`title`/`date`/`start_time`/`end_time`/`notes` 필드 구조를 유지합니다.

## 검증 방법

앱을 실행하고 상세 Trace 탭에서 확인합니다.

```bash
./run.sh --week5
```

자동화 테스트 없음 — Trace JSON이 기대한 키와 값을 가지는지 눈으로 확인합니다.

메인과제 확인 순서:
1. 외부 팀원 일정 조회 요청을 입력 → trace에서 `search_previous_conversations` →
   `load_conversation_messages` → `extract_schedules_from_history` 중 어떤 tool이 어떤
   순서로 호출됐는지 확인합니다.
2. `collect_member_schedules` 결과 rows에 "나"와 외부 멤버 일정이 같은 구조로 들어 있는지,
   `list_shared_schedules` 결과에 `rows`와 `schedule_summary`가 유지되는지 확인합니다.

추가과제 확인 순서: `create_shared_schedule`로 등록한 row가 `list_shared_schedules` 조회에
나타나고, `delete_shared_schedule`로 삭제되는지 확인합니다.

## 주차 경계

Week 5는 외부 SQLite/MCP 서버의 이전 대화·공유 일정을 wrapper tool로 감싸고, 내 일정과 외부
멤버 busy-time을 `collect_member_schedules`로 모으는 것까지만 다룹니다. Week 6 이후 개념(여러
사람의 공통 가능 시간 계산·최종 확정 로직, Kana 하위 agent 오케스트레이션 등)을 Week 5 코드에
미리 추가하지 않습니다. `mcp_server/sqlite_mcp_server.py`의 `@mcp.tool` 구현은 학생 구현
대상이 아니므로 직접 수정하지 않습니다.
