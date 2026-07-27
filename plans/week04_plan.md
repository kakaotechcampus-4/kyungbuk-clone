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

---

# Week 4 구현 계획 — 추가과제 (다음 세션에서 진행)

## Context

메인과제(위 내용)는 완성/테스트 통과 상태. 이번에 구현할 건 같은 파일(`student_parts/week04_retrieve_nanas_memory.py`)에 남아있는 **추가 과제** 2개 기능(함수 4개, 전부 TODO 스텁):

1. **`search_conversation_messages`** — 앱 SQLite에 자동으로 쌓이는 일반 채팅 대화(`conversations`/`messages` 표, `fixed/agent_runtime.py`가 매 turn마다 무조건 기록)를 ChromaDB로 lazy sync한 뒤 의미 기반으로 검색하는 agentic RAG. "방금 한 말"이 검색 결과에 과거 기록처럼 섞이지 않도록 현재 대화는 기본적으로 제외한다.
2. **`search_nana_memory`** — 이전 버전 호환용 통합 검색. 개인 참고자료(메모) hit와 SQLite 저장 일정 row를 한 번에 묶어 하나의 `context` 문자열로 반환한다. 가이드 주석(파일 87~91번째 줄)이 이 tool을 "참고 코드"로, 나머지 4개를 "학생 핵심 구현 대상"으로 명확히 구분하고 있고 `week04_tools()`(이미 완성된 코드)도 이 tool을 포함하지 않으므로, **agent에는 노출하지 않는다** (함수/tool은 만들되 `week04_tools()`는 손대지 않음).

필요한 부품(`ConversationRAGStore`, `current_session_scope`/`DEFAULT_SESSION_SCOPE`, `PersonalReferenceStore.backend_info`)은 `fixed/`에 이미 완성되어 있어 호출만 하면 된다. git history 커밋 `7266835`(나중에 되돌려짐)에 한 번 구현됐던 버전이 남아있어 반환 JSON 키 구조의 신뢰할 만한 참고 자료가 된다.

**1~3주차에 대한 영향**: 없음(확인 완료). import는 week04→03→02→01 단방향이고, `CONFIG`는 읽기 전용, 새 로직은 `conversations`/`messages`에 SELECT만 하고 별도 ChromaDB 컬렉션에만 쓴다. 유일한 실행 주의사항은 테스트 격리(아래 참고).

## 설계 결정

- **`search_nana_memory`의 `date_from`/`date_to`/`attendee`는 실제로 필터링한다.** `fixed/app_store.py`의 `structured_requests` 테이블에는 이미 `date`/`start_time`/`members_json` 컬럼이 있고, `search_saved_requests()`가 반환하는 row에도 이 값들이 그대로 들어있다. `fixed/` 코드를 건드릴 필요 없이, `search_saved_request_rows`로 넉넉히 가져온 뒤 **week04 파일(학생 코드) 안에서 Python으로 후처리 필터링**한다. 이미 파일에 있지만 안 쓰이던 `_decode_attendees(raw)` 헬퍼를 `members_json` 디코딩에 재사용한다.
- **`search_nana_memory`는 `week04_tools()`에 추가하지 않는다** (가이드 주석 + 기존 완성 코드 근거).
- **참고자료(메모) hit에는 날짜/참석자 개념이 없으므로, `date_from`/`date_to`/`attendee` 필터는 저장 기록(`saved_rows`) 쪽에만 적용**하고 참고자료 검색에는 적용하지 않는다.

## 구현 대상 함수 (pseudocode 수준)

### 1. `search_conversation_messages_dict(sqlite_store, conversation_rag_store, *, query, top_k=5, conversation_id=None) -> dict`

```python
sync = conversation_rag_store.sync_from_sqlite(sqlite_store)

exclude_conversation_id = None
if conversation_id is None:
    active = current_session_scope()
    if active != DEFAULT_SESSION_SCOPE:
        exclude_conversation_id = active

hits = conversation_rag_store.search(
    query=query, top_k=top_k,
    conversation_id=conversation_id,
    exclude_conversation_id=exclude_conversation_id,
)
return {
    "hits": hits,
    "rows": hits,
    "context": conversation_rag_store.context_from_hits(hits),
    "rag_backend": conversation_rag_store.backend_info(),
    "sync": sync,
}
```

### 2. `search_conversation_message_rows(sqlite_store, *, query, top_k=5, conversation_id=None) -> list[dict]`

`search_conversation_messages_dict(sqlite_store, CONVERSATION_RAG_STORE, ...)`를 호출해 `hits`만 반환 (모듈 전역 `CONVERSATION_RAG_STORE`를 직접 참조 — 테스트가 monkeypatch할 수 있도록 호출 시점에 읽음).

### 3. `search_conversation_messages` tool

```python
@tool(args_schema=SearchConversationMessagesInput)
def search_conversation_messages(query, top_k=5, conversation_id=None):
    """앱 SQLite 대화 목록을 대화 단위 ChromaDB RAG로 검색합니다.
    개인 참고자료는 search_personal_references, 구조화된 저장 일정/할 일/알림은
    search_saved_requests를 사용하세요."""
    limit = safe_limit(top_k, default=5, maximum=50)
    result = search_conversation_messages_dict(
        SQLITE_STORE, CONVERSATION_RAG_STORE, query=query, top_k=limit, conversation_id=conversation_id,
    )
    return json_payload(result)  # bare {hits, rows, context, rag_backend, sync} — search_personal_references/search_saved_requests와 동일한 규칙(ok/tool_name 래핑 없음)
```

### 4. `search_nana_memory` tool (실제 필터링 포함)

```python
@tool(args_schema=SearchNanaMemoryInput)
def search_nana_memory(query, date_from=None, date_to=None, attendee=None, limit=5):
    """이전 버전 호환용 통합 검색 tool입니다. 지금은 search_personal_references /
    search_saved_requests / search_conversation_messages로 출처를 나눠 찾는 것이
    표준이며, 이 tool은 예전 trace와의 호환을 위해 유지됩니다."""

    top_k = safe_limit(limit, default=5, maximum=20)
    reference_hits = search_personal_reference_hits(REFERENCE_STORE, query=query, top_k=top_k)

    # 필터링 여지를 남기기 위해 넉넉히 가져온 뒤 Python에서 후처리
    fetch_k = safe_limit(top_k * 4, default=20, maximum=50)
    candidate_rows = search_saved_request_rows(SQLITE_STORE, query=query, top_k=fetch_k)

    def _row_matches(row):
        if date_from and (not row.get("date") or row["date"] < date_from):
            return False
        if date_to and (not row.get("date") or row["date"] > date_to):
            return False
        if attendee and attendee not in _decode_attendees(row.get("members_json")):
            return False
        return True

    saved_rows = [row for row in candidate_rows if _row_matches(row)][:top_k]

    chunks = []
    for hit in reference_hits:
        metadata = hit.get("metadata") or {}
        chunks.append(f"[참고자료] {metadata.get('title', '')}: {hit.get('content', '')}".strip())
    for row in saved_rows:
        date = row.get("date") or "날짜 미정"
        start_time = row.get("start_time") or ""
        chunks.append(f"[저장기록] {row.get('title', '')} ({date} {start_time})".strip())
    context = "\n".join(chunks) if chunks else "관련된 개인 참고자료나 저장 기록을 찾지 못했습니다."

    return json_payload({
        "ok": True, "tool_name": "search_nana_memory", "query": query,
        "filters": {"date_from": date_from, "date_to": date_to, "attendee": attendee},
        "reference_backend": REFERENCE_STORE.backend_info(),
        "context": context, "chunks": chunks,
        "hits": reference_hits, "rows": saved_rows,
    })
```

`search_nana_memory`는 `week04_tools()`에 **추가하지 않는다**.

## `week04_tools()` / `week04_prompt_parts()` 업데이트

- **`week04_tools()`: 변경 없음.** 이미 `search_conversation_messages`까지 포함되어 있고 `search_nana_memory`는 의도적으로 제외.
- **`week04_prompt_parts()`: 기존 리스트에 문장 하나만 추가** (main과제와 같은 방식 — 별도 모듈 상수로 안 빼고 인라인):
  > "예전에 채팅으로 나눈 일반 대화 내용을 물으면(예: '전에 채팅으로 여행 얘기할 때 뭐라고 했었지?') search_conversation_messages를 쓴다. 이 tool은 conversation_id를 따로 지정하지 않으면 지금 하고 있는 대화는 자동으로 검색에서 제외하므로, 방금 한 말이 과거 기록처럼 섞여 나오지 않는다."
- **Docstring 상호 참조**: `search_conversation_messages`는 `search_personal_references`/`search_saved_requests`를 언급, `search_nana_memory`는 세 개 표준 tool을 모두 언급(예전 방식임을 명시). 이미 완성된 메인과제 tool들의 docstring은 건드리지 않는다.

## 진행 순서 (메인과제와 동일한 cadence: 구현 → 테스트 → pytest → 설명 → 승인 → 다음 step)

**Step 1 — 대화 검색 핵심 로직**
- `search_conversation_messages_dict`, `search_conversation_message_rows` 구현
- `tests/test_week04.py`의 `use_temp_stores` fixture 확장 (`w4.CONVERSATION_RAG_STORE`도 tmp_path 기반으로 monkeypatch)
- helper 단위 테스트 작성 후 `pytest tests/test_week04.py -k conversation_message` 실행

**Step 2 — `search_conversation_messages` tool**
- tool 구현, docstring 상호 참조 추가
- `week04_prompt_parts()`에 라우팅 문장 추가
- tool-level 테스트 추가, 전체 `pytest tests/test_week04.py` 실행

**Step 3 — `search_nana_memory` 호환 tool (실제 필터링 포함)**
- tool 구현 (`_decode_attendees` 재사용)
- `week04_tools()`에는 추가하지 않음을 재확인
- 필터링 테스트 포함해 전체 `pytest tests/test_week04.py` 실행

**Step 4 — 수동 E2E 체크리스트 추가**
- `tests/test_week04.py` 하단 체크리스트에 대화 검색/호환 tool 시나리오 추가 (`[ ]` 상태로, 실제 앱 실행 확인은 사용자 몫)

## 테스트 추가 (`tests/test_week04.py`)

### fixture 확장
```python
from fixed.conversation_rag_store import ConversationRAGStore
from fixed.session_scope import conversation_session_scope, DEFAULT_SESSION_SCOPE

# use_temp_stores 안에 추가:
test_conversation_rag_store = ConversationRAGStore(tmp_path / "chroma")
monkeypatch.setattr(w4, "CONVERSATION_RAG_STORE", test_conversation_rag_store)
```

### 새 테스트 함수 (이름 + 검증 내용)

**`search_conversation_messages_dict`**
- `test_..._syncs_and_finds_matching_conversation` — 대화 하나 생성 후 검색, `sync` 카운트/`hits`/`context`/`rag_backend` 확인
- `test_..._direct_tool_call_scope_does_not_exclude_anything` — `conversation_session_scope` 없이 호출 시 `DEFAULT_SESSION_SCOPE` sentinel이 제외 대상으로 오인되지 않음을 확인
- `test_..._excludes_current_conversation_when_real_scope_active` — 두 대화(A, B) 중 A를 `conversation_session_scope(A)`로 감싸고 호출 시 A 제외, B는 검색됨
- `test_..._explicit_conversation_id_overrides_exclusion` — 같은 상황에서 `conversation_id=A`를 명시하면 A가 다시 나옴
- `test_..._returns_empty_when_no_conversations_synced` — 빈 SQLite에서 `hits == [] and sync["total"] == 0`

**`search_conversation_message_rows`**
- `test_..._returns_only_hits_list` — dict 결과의 `hits`와 동일한 리스트 반환 확인

**`search_conversation_messages` tool**
- `test_..._tool_returns_expected_keys` — `.invoke()` 결과 키가 `{hits, rows, context, rag_backend, sync}` (ok/tool_name 래핑 없음)
- `test_..._tool_excludes_current_conversation_via_invoke` — tool 경유로도 제외 동작 확인
- `test_..._description_cross_references_...` — docstring에 다른 tool 이름 포함 확인

**`week04_prompt_parts`**
- `test_..._mentions_search_conversation_messages`

**`search_nana_memory`**
- `test_..._combines_reference_and_saved_request_chunks` — 참고자료+저장기록 둘 다 매칭될 때 `context`에 `[참고자료]`/`[저장기록]` 둘 다 포함
- `test_..._filters_saved_rows_by_date_range` — `date_from`/`date_to` 지정 시 범위 밖 저장기록이 `rows`/`context`에서 빠짐
- `test_..._filters_saved_rows_by_attendee` — `attendee` 지정 시 `members_json`에 없는 행이 빠짐 (`_decode_attendees` 재사용 확인)
- `test_..._not_exposed_in_week04_tools` — `search_nana_memory`가 `week04_tools()` 목록에 없음을 확인

## 재사용할 기존 코드

| 함수/클래스 | 위치 | 용도 |
|---|---|---|
| `ConversationRAGStore` | `fixed/conversation_rag_store.py` | 대화 lazy sync + 검색 (수정 불필요, 호출만) |
| `current_session_scope`, `DEFAULT_SESSION_SCOPE` | `fixed/session_scope.py` | 현재 대화 제외 판단 (이미 import됨) |
| `PersonalReferenceStore.backend_info` | `fixed/reference_store.py` | `search_nana_memory`의 `reference_backend` |
| `_decode_attendees` | 이 파일 상단 (기존, 지금까지 미사용) | `members_json` → list 디코딩, 드디어 사용됨 |
| `search_personal_reference_hits`, `search_saved_request_rows`, `safe_limit`, `json_payload` | 이 파일 (메인과제에서 이미 구현) | 그대로 재사용 |

## 검증 방법

- `pytest tests/test_week04.py -v` (repo 루트) — 새 테스트 포함 전체 통과 확인
- `pytest` 전체 스위트 — week01~04 기존 테스트에 회귀 없는지 확인
- 수동 E2E (앱 실행, `./run.sh --week4`): 이전 세션에서 나눈 잡담을 다른 세션에서 "전에 무슨 얘기했었지?"로 검색해 `search_conversation_messages` 호출과 현재 대화 제외를 trace에서 확인. `search_nana_memory`는 agent에 노출 안 되므로 REPL 등에서 직접 `.invoke()` 호출로 확인.

## 범위 밖 (건드리지 않음)

- Week 3 `personal_create_schedule` 미구현 스텁
- Week 2 "팀"류 집단명사 personal/group 분류 애매함
- 메인과제(`add_personal_reference`, `search_personal_references`, `search_saved_requests`)의 기존 코드/테스트
