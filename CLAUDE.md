# Kanana Schedule Agent — CLAUDE.md

## 프로젝트 목적

카카오 카테캠 강의용 LangChain 실습 프로젝트입니다.
학생이 `student_parts/` 안의 `# TODO` 함수를 직접 구현하며 LangChain tool과
에이전트 실행 흐름을 배우는 것이 목표입니다.

## 구현 범위

- **구현 대상 (활성 주차)**: `student_parts/week04_retrieve_nanas_memory.py` 안의 `# TODO` 함수들
- **읽기 전용**: `fixed/` 디렉터리(구조 이해용 참고 코드 — 특히 SQLite 접근은 `fixed/app_store.py`의
  `AppSQLiteStore` 메서드로만 하고, 개인 참고자료 벡터 검색은 `fixed/reference_store.py`의
  `PersonalReferenceStore` 메서드로만 하고, 대화 RAG는 `fixed/conversation_rag_store.py`의
  `ConversationRAGStore` 메서드로만 합니다 — 직접 SQL이나 ChromaDB 호출을 새로 작성하지 않습니다),
  `student_parts_baseline/`(이전 주차 정답 코드 — 참고용이며 그대로 복사해 붙이지 않습니다)
- **수정 금지**: `app.py`, `run.sh`, `pyproject.toml`, `uv.lock`

## 완료된 주차

- **Week 1** (`student_parts/week01_wake_up_nana.py`): 임시 메모리(`PERSONAL_SCHEDULES`) CRUD — 완료
- **Week 2** (`student_parts/week02_structure_natural_language_requests.py`): 자연어 →
  `StructuredRequest` 구조화 — 완료
- **Week 3** (`student_parts/week03_build_nanas_logbook.py`): `StructuredRequest`를 SQLite에
  저장/조회/수정/삭제 — 완료

## Week 4 도움 방식 (힌트 우선, 단계별, 메인과제 → 추가과제 순)

`week04_retrieve_nanas_memory.py` 안에 이미 `[4주차 수강생 구현 가이드]` 주석(목표·과제 구성·
메인/추가 티어·함수별 설명)이 있습니다. 정답을 알려주지 말고, 아래 순서대로 하나씩 힌트를 먼저
주고 학생이 시도한 뒤 피드백합니다. 메인과제 3개를 먼저 끝낸 뒤에만 추가과제로 넘어갑니다.

### 메인과제

**add_personal_reference_dict / add_personal_reference — 2단계**
1. `PersonalReferenceStore.add_personal_reference(title, content, tags)`에 그대로 위임하는
   흐름을 안내하고, `tags`가 `None`이면 빈 list로 바꿔 넘겨야 하는 이유를 설명합니다.
2. 저장 결과 dict를 tool에서 `reference_backend`(store의 `backend_info()`)와 `reference` 키가
   있는 JSON payload로 감싸 반환하도록 안내합니다.

**search_personal_reference_hits / search_personal_references — 2단계**
1. `PersonalReferenceStore.search_personal_references(query, limit=top_k)` 호출 결과를
   그대로 순회하며 `id`/`content`/`distance`/`metadata`(title/tags) 구조로 정리하는 이유를
   설명합니다. store가 이미 이 필드들을 반환하므로 다시 계산하지 않습니다.
2. tool은 이 list를 top-level `{"hits": [...]}` 키로 감싸 반환해야 LLM이 근거 문서를 바로
   읽을 수 있다는 계약을 설명합니다.

**search_saved_request_rows / search_saved_requests — 2단계**
1. `AppSQLiteStore.search_saved_requests(query, limit=top_k)`를 그대로 호출하는 흐름을
   안내하고, `top_k`는 `safe_limit(...)`으로 tool 안에서 먼저 보정해야 하는 이유를 설명합니다.
2. 결과가 없으면 예외 없이 `rows=[]`를 그대로 반환해야 하는 이유와, top-level `{"rows": [...]}`
   키 계약을 설명합니다.

### 추가과제 (메인과제 완료 후)

- **search_conversation_messages_dict / search_conversation_message_rows /
  search_conversation_messages**: `ConversationRAGStore.sync_from_sqlite(sqlite_store)`로
  먼저 lazy sync한 뒤 `search(...)`하는 순서를 설명하고, `conversation_id`를 명시하지 않으면
  현재 대화를 검색 결과에서 제외해야 "방금 한 말"이 과거 발화처럼 섞이지 않는다는 이유를
  설명합니다. 반환 JSON은 `hits`/`rows`에 같은 결과를 넣고 `context`/`rag_backend`/`sync`도
  함께 둬야 하는 이유를 설명합니다.
- **search_nana_memory (호환 통합 검색)**: 개인 참고자료 hit와 SQLite 일정 chunk를 하나의
  `context` 문자열로 합쳐야 하는 이유, 그리고 `reference_backend`를 함께 노출해 어떤 backend가
  응답 근거인지 구분해야 하는 이유를 설명합니다. 가장 마지막에, 필요할 때만 다룹니다.

## Week 4 핵심 제약 (구현 시 반드시 지킬 것)

- 개인 참고자료 벡터 검색은 반드시 `fixed/reference_store.py`의 `PersonalReferenceStore`
  메서드를 통해서만 합니다. ChromaDB collection을 직접 호출하지 않습니다.
- SQLite 접근은 반드시 `fixed/app_store.py`의 `AppSQLiteStore` 메서드를 통해서만 합니다.
- 대화 RAG는 반드시 `fixed/conversation_rag_store.py`의 `ConversationRAGStore` 메서드
  (`sync_from_sqlite` / `search`)를 통해서만 합니다.
- 모든 tool은 `json_payload(...)`로 감싼 JSON **문자열**을 반환합니다.
- `@tool(args_schema=...)`가 이미 입력을 검증하므로 tool 본문에서 Pydantic 모델을 다시 만들지
  않습니다.
- `top_k`/`limit`은 이 파일의 `safe_limit(...)`으로 tool 안에서 보정합니다.
- `search_personal_references`는 `{"hits": [...]}`, `search_saved_requests`는
  `{"rows": [...]}`를 top-level 키로 반환하는 계약을 지킵니다.
- `search_conversation_messages`는 `conversation_id` 미지정 시 현재 대화를 검색에서 제외합니다.

## 검증 방법

앱을 실행하고 상세 Trace 탭에서 확인합니다.

```bash
./run.sh --week4
```

자동화 테스트 없음 — Trace JSON이 기대한 키와 값을 가지는지 눈으로 확인합니다.

메인과제 확인 순서:
1. 참고자료를 하나 추가("오전 회의는 피하고 싶다고 기억해줘" 등)한 뒤 관련 질문을 입력 →
   trace에서 `add_personal_reference` 다음 `search_personal_references` 호출과 top-level
   `hits` 키를 확인합니다.
2. 저장된 일정/할 일 관련 질문을 입력 → `search_saved_requests`가 호출되고 top-level `rows`
   키가 있는지 확인합니다.

추가과제 확인 순서: 일반 채팅에서 나눴던 이야기를 묻는 질문을 입력 →
`search_conversation_messages`가 호출되고, 방금 대화(현재 conversation_id)는 결과에서
제외되는지 확인합니다.

## 주차 경계

Week 4는 데이터 출처별 RAG 검색 tool 분리(개인 참고자료 / SQLite 저장 기록 / 대화 발화)까지만
다룹니다. Week 5 이후 개념(외부 캘린더 조율 로직 직접 구현, 새 MCP 도구 신설 등)을 Week 4 코드에
미리 추가하지 않습니다. 외부 공유 저장소 동기화는 이미 `fixed/` 내부에서 처리되므로, 학생이
별도로 MCP 동기화 코드를 작성할 필요는 없습니다.
