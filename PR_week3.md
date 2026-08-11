## 과제 목표
이번 주차 과제를 통해 무엇을 배우고자 했는지 간단히 적어요.
- 자연어 요청이나 임시 일정을 SQLite 데이터베이스(`AppSQLiteStore`)에 영구적으로 기록하고 관리하는 데이터 영속성 흐름을 익히고자 함
- `Pydantic`(`SaveStructuredRequestInput`)의 `@model_validator(mode="before")`를 활용하여 JSON 문자열, dict, 예전 trace wrapper(`payload`, `structured_request`) 등 다양한 형태의 데이터를 안전하게 검증하고 정규화하는 구조를 학습함
- LangChain `@tool`들을 연결하여 생성·조회·수정·삭제(CRUD) 요청 시 에이전트가 알맞은 도구를 호출하도록 프롬프트를 구성함

---

## 과제 위치
- 작업 브랜치 : `ganggang-0605/week3`  → 본인 통합 브랜치 `ganggang-0605/final` 로 PR
- 주요 파일 : `student_parts/week03_build_nanas_logbook.py`

---

## 구현한 기능
이번 주차 **기본 미션** 중 구현한 항목에 체크해요.
- [x] `SQLITE_MEMORY_PROMPT`, `WEEK03_TOOL_CALL_PROMPT` — 대화 종료 후에도 SQLite DB에 기록이 유지됨을 명시하고 CRUD 도구 호출 흐름을 안내하는 시스템 프롬프트 작성
- [x] `SaveStructuredRequestInput.unwrap_legacy_payload` / `_save_input_from` — JSON, dict, 모델, wrapper 입력을 정규화하여 검증하는 입력 핸들러 구현
- [x] `save_structured_request_payload` / `save_structured_request` — 검증된 구조화 요청을 SQLite DB에 저장하고 표준 JSON payload로 반환
- [x] `personal_create_schedule` — Week 1 임시 일정 생성과 동시에 Week 3 SQLite DB에도 기록하는 이중 저장 구현
- [x] `list_saved_requests`, `get_saved_request` — 종류·날짜 필터 및 ID 기반 저장 요청 목록/단건 조회 도구 구현
- [x] `personal_list_saved_schedules`, `personal_update_saved_schedule`, `personal_delete_saved_schedules` — 저장 일정 후보 목록 조회, ID 기반 일정/복사본 동시 수정, 조건/ID 기반 일정 삭제 도구 완성

---

## 도전 기능
**도전 미션**을 시도했다면 체크하고 간단히 적어요.
- [x] `_delete_saved_schedules` 삭제 안전 Guard — 삭제 조건(`schedule_ids`, `date`, `title`, `start_time` 등)이나 `delete_all=True`가 전혀 없으면 의도치 않은 전체 삭제를 방지하고 거부 처리하도록 안전망 구축

---

## 과제 회고 (KPT)
과제를 마치고 KPT 회고를 적어요.
- **Keep** : 가이드 주석과 `AppSQLiteStore` 인터페이스 명세를 정확히 분석하여 의도된 스키마에 맞춰 구현함. Pydantic validator를 통해 입력 유연성을 확보함.
- **Problem** : 다양한 입력 데이터 포맷(단순 dict, JSON 문자열, 중첩 wrapper 등)을 예외 없이 한 번에 정규화하는 과정에서 필드 매핑(`attendees` ↔ `members`, `id` ↔ `source_schedule_id`)을 주의 깊게 맞추어야 했음.
- **Try** : 다음 주차(Week 4 RAG 및 검색 기능)에서 저장된 기록(`structured_requests` 및 `schedules`)을 기반으로 더욱 똑똑하게 답변하는 기능과 자연스럽게 연계해볼 예정임.
