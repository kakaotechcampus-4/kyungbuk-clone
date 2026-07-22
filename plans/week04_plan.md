# Week 4 구현 계획 — `student_parts/week04_retrieve_nanas_memory.py` (메인과제)

## Context

Week 1~3에서 개인 일정 CRUD(1주) → 자연어 구조화(2주) → SQLite 영속 저장(3주)을 만들었다.
Week 4의 목표는 RAG를 "출처별로 분리된 검색 tool"로 만드는 것이다: 자유 형식 개인 참고자료는
ChromaDB 벡터 검색으로, Week 3가 구조화해 저장한 일정/할 일/알림은 SQLite 키워드 검색으로 찾는다.
`docs/week04.md`와 노트북(`4주차_나나가_기억을_찾아오다.ipynb`)이 보여주는 핵심 패턴(`search_rag_memory`
/ `search_sqlite_requests` 두 tool을 agent가 질문 성격에 따라 골라 호출)을, 실제 student_parts 파일에서는
`search_personal_references` / `search_saved_requests` + 참고자료 추가 tool `add_personal_reference`
3개로 구현했다. 추가과제(`search_conversation_messages`, `search_nana_memory`)는 이번 범위 밖.

## 구현 대상 파일

- `student_parts/week04_retrieve_nanas_memory.py` (메인 구현)
- `tests/test_week04.py` (신규 pytest 파일)

## 진행 방식

4개 step으로 나눠서, 각 step마다 "구현 → 테스트 추가 → pytest 실행 → 설명 → 승인 후 다음 step" 순서로 진행했다.

---

## Step 1 — `add_personal_reference` (참고자료 추가)

```python
def add_personal_reference_dict(reference_store, *, title, content, tags=None):
    reference = reference_store.add_personal_reference(title=title, content=content, tags=tags or [])
    return {"reference_backend": reference.get("backend"), "reference": reference}
```
tool은 위 helper를 호출해 `json_payload({"ok": True, "tool_name": "add_personal_reference", **결과})`을 반환한다.

## Step 2 — `search_personal_references` (참고자료 검색)

```python
def search_personal_reference_hits(reference_store, *, query, top_k=2):
    raw_hits = reference_store.search_personal_references(query, limit=top_k)
    return [
        {"id": h["id"], "content": h["content"], "distance": h["distance"],
         "metadata": {"title": h.get("title", ""), "tags": h.get("tags", "")}}
        for h in raw_hits
    ]
```
tool은 `safe_limit(top_k, default=2, maximum=20)`으로 보정 후 `{"hits": [...]}`을 반환한다.

## Step 3 — `search_saved_requests` (SQLite 저장 기록 검색)

```python
def search_saved_request_rows(sqlite_store, *, query, top_k=3):
    return sqlite_store.search_saved_requests(query, limit=top_k)
```
tool은 `safe_limit(top_k, default=3, maximum=50)`으로 보정 후 `{"rows": [...]}`을 반환한다.

## Step 4 — tool description 보강 + 시스템 프롬프트

- 세 tool의 docstring(=LangChain `@tool`이 그대로 쓰는 `description`)에 **서로 반대 tool 이름을 명시**해
  "이거 아니면 저거"를 tool 스스로 설명하게 만들었다 (`add_personal_reference` ↔ `save_structured_request`,
  `search_personal_references` ↔ `search_saved_requests`). system prompt보다 tool description이 tool
  선택에 더 직접적으로 쓰이기 때문.
- `week04_prompt_parts()`에 라우팅 규칙 한 문장을 **인라인으로** 추가했다 (모듈 상수로 빼지 않음 —
  한 곳에서만 쓰이므로 굳이 분리할 이유가 없다고 판단):
  - 자유 형식 메모/선호/참고자료 → `search_personal_references`
  - 저장 기록을 **특정 키워드/주제로 좁혀** 찾을 때 → `search_saved_requests`
  - 조건 없이 저장 기록 **전체**를 보여달라는 요청 → (Week 3부터 상속된 규칙대로) `list_saved_requests`
  - 참고자료 추가 요청 → `add_personal_reference`
  - 검색 결과가 비면 근거 없다고 답하고 지어내지 않는다
- **week04_system_prompt()에는 별도 역할/경계 문장을 추가하지 않았다** (아래 "시스템 프롬프트 경계 문장
  실험" 참고 — 실측 결과 득이 없고 오히려 사실과 안 맞았음).

### 발견 및 수정: prompt 누적 시 예문 충돌

Week 3의 `WEEK03_UNIFIED_LOOKUP_PROMPT`(week04까지 상속)는 "저번에 저장한 거 뭐 있어?" 같은 **조건 없는
전체 나열** 요청에 `list_saved_requests`를 쓰라고 지시한다. 처음 작성한 week04 라우팅 문장이 **같은 예문을
`search_saved_requests`에도 배정**해버려서 모순이 생겼었다. "전체 나열"과 "키워드 검색"을 구분하는 예문으로
바꾸고, 전체 나열 요청은 "이 지시보다 앞서 나온 지시대로 list_saved_requests를 그대로 쓴다"고 명시해서
해결했다.

### 실험: 시스템 프롬프트 경계 문장 (했다가 안 하기로 결정)

Week 1~3은 `weekN_system_prompt()`에 그 주차만의 역할/경계 문장을 추가하는 관례가 있다
(예: week03 — "이번 주 범위는 SQLite 저장/조회/수정/삭제까지이며 RAG 검색이나 외부 멤버 일정 조율은
하지 않는다"). 이 관례를 따라 week04에도 "새 일정을 생성하거나 외부 멤버와 일정을 조율하지 않는다"는
문장을 추가할지 검토했으나:

1. **모순 발견**: week04_tools()는 week1~3에서 상속한 일정 생성 tool(`personal_create_schedule`,
   `save_structured_request`)을 그대로 포함한다. week3와 달리 week4는 "이번 주엔 tool이 아예 없다"가
   아니라 "기존 tool + 검색 tool 추가" 구조라서, 이 경계 문장은 실제로 존재하는 tool을 쓰지 말라는
   거짓 지시가 된다.
2. **실측(A/B 테스트)**: 실제 앱 agent 2개(경계 문장 있음/없음)를 만들어 같은 질문 4개를 던져봤다.
   참고자료 검색, 저장기록 키워드 검색, 저장기록 전체 나열 3개 시나리오는 둘 다 동일하게 정확히 동작했고,
   "새 일정 생성" 시나리오에서도 경계 문장이 있는 쪽(B)이 **경계 문장을 사실상 무시하고 그대로 일정을
   저장**했다 (답변에 "이번 주엔 안 된다"는 언급 전혀 없음). 즉 이 문장은 득도 실도 없었다.
3. **결론**: 장식/부정확한 경계 문장을 추가하는 대신 현재 상태(`week04_system_prompt() =
   join_system_prompt(week04_prompt_parts())`, 추가 문장 없음)를 유지하기로 했다.

> A/B 테스트 스크립트는 세션 scratchpad에 있었고, 실제 앱 데이터를 건드리지 않도록 `w4.REFERENCE_STORE`/
> `w4.SQLITE_STORE`뿐 아니라 `week03_build_nanas_logbook.CONFIG`(week1~3 tool이 쓰는 `_store()`가
> 참조하는 경로)와 외부 MCP 동기화 함수까지 전부 격리해야 한다는 걸 첫 시도에서 실수로 확인했다
> (최초 시도에서 실제 DB에 테스트용 "팀 회의" row가 잠깐 들어갔다가 정리함).

## 재사용한 기존 코드

| 함수/클래스 | 위치 | 용도 |
|---|---|---|
| `PersonalReferenceStore` | `fixed/reference_store.py` | 참고자료 ChromaDB 저장/검색 |
| `AppSQLiteStore.search_saved_requests` | `fixed/app_store.py` | 저장 요청 LIKE 검색 |
| `safe_limit`, `json_payload` | 이 파일 상단 (기존 구현됨) | top_k 보정, JSON 직렬화 |
| `join_system_prompt`, `week03_prompt_parts` | week01/week03 | 프롬프트 누적 |

## 테스트 (`tests/test_week04.py`, pytest)

핵심 fixture `use_temp_stores`: `tmp_path` 기준 임시 `PersonalReferenceStore`/`AppSQLiteStore`로
`w4.REFERENCE_STORE`/`w4.SQLITE_STORE` 모듈 전역을 교체하고, `OpenAIEmbeddingFunction`의 네트워크 호출
메서드를 결정적 가짜 벡터로 monkeypatch해 네트워크 없이 실제 Chroma add/query 로직을 검증한다.

| 테스트 함수 | 검증 내용 |
|---|---|
| `test_add_personal_reference_dict_returns_backend_and_reference` | `reference_backend`/`reference` 키 존재 |
| `test_add_personal_reference_dict_defaults_tags_to_empty_list` | `tags=None` → `[]` |
| `test_add_personal_reference_dict_increments_collection_count` | Chroma `count()` +1 |
| `test_add_personal_reference_tool_payload_shape` | tool 반환 JSON의 `ok`/`tool_name`/`reference` |
| `test_add_personal_reference_tool_defaults_tags_to_empty_list` | tool 경로에서도 tags 기본값 |
| `test_safe_limit_clamps_below_minimum` / `_above_maximum` / `_falls_back_to_default_on_invalid_input` | `safe_limit` 경계값 |
| `test_search_personal_reference_hits_finds_added_reference` | hit 구조 `id/content/distance/metadata.title/tags` |
| `test_search_personal_references_tool_returns_hits_key` | top-level 키가 `hits` 하나뿐 |
| `test_search_personal_references_tool_uses_default_top_k` | top_k 기본값(2) 적용 |
| `test_search_saved_request_rows_finds_matching_row` / `_returns_empty_list_when_no_match` | 키워드 매칭/빈 결과 |
| `test_search_saved_requests_tool_returns_rows_key` / `_returns_empty_rows_when_no_match` | top-level 키가 `rows` 하나뿐 |
| `test_*_description_*` (3개) | tool description 상호 참조 (agent 라우팅 근거) |
| `test_week04_prompt_parts_mentions_all_three_tool_names` / `_includes_week03_parts` | 누적 프롬프트 내용 검증 |

**실행:** `pytest tests/test_week04.py -v` (repo 루트). 전체 스위트(`pytest`, week01~04) 90개 통과 확인.

### 수동 E2E 체크리스트

`tests/test_week04.py` 하단 주석 참고. `./run.sh --week4`로 앱을 띄워 참고자료 추가→검색,
저장기록 키워드 검색 vs 전체 나열, 근거 없음 처리 3개 시나리오를 상세 trace로 직접 확인한다
(아직 미확인 — `[ ]` 상태로 남아있음).

## 이번 주차에서 발견했지만 범위 밖이라 다음 PR에서 멘토님께 질문할 것

1. **`personal_create_schedule`(week3 호환판, `week03_build_nanas_logbook.py`)이 아직 미구현 스텁**이다
   (`# TODO`, 본문이 `...`만 있어 호출 시 `None` 반환). Week 3 "추가 과제" 항목이라 이번 계획에서 안 건드림.
2. **Week 2의 personal/group 분류 규칙**([week02:218-226](../student_parts/week02_structure_natural_language_requests.py#L218-L226))이
   "이름이 특정된 상대가 있는지"만 기준으로 삼고 있어서, "팀 회의"처럼 특정 개인 이름 없이 집단만 가리키는
   표현을 어떻게 분류할지 예문이 없다. 실제로 "팀"을 group_schedule의 참석자처럼 취급해버리는 걸 확인함.
   이 규칙은 구현 가이드에 명시된 게 아니라 이전 구현 시점에 임의로 추가한 휴리스틱으로 보임 — 멘토님께
   "집단 명사(팀/다들/동료들)는 personal/group 중 어느 쪽으로 분류해야 하는지" 질문할 예정.
