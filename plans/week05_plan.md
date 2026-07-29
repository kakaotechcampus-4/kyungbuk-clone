# Week 5 구현 계획 — `student_parts/week05_load_kanas_past_conversations.py` (메인과제)

## Context

Week 1~4에서 개인 일정 CRUD(1주) → 자연어 구조화(2주) → SQLite 영속 저장(3주) → 출처별 RAG 검색(4주)을
만들었다. 지금까지는 전부 "나"의 데이터였다.

Week 5의 목표는 처음으로 "남"의 데이터를 다루는 것이다. 팀원(철수·영희·민준·서연·지훈·하린)의 이전 대화와
일정은 별도의 외부 SQLite에 있고, agent는 이 DB를 직접 읽지 못한다. 대신 `mcp_server/sqlite_mcp_server.py`가
stdio subprocess로 띄우는 **MCP 서버**의 6개 tool을 통해서만 접근한다. `docs/week05.md`와 노트북
(`5주차_카나가_지난대화를_불러오다.ipynb`)이 보여주는 핵심 개념은 "DB를 읽었다"가 아니라 **"MCP tool을
호출해 필요한 정보를 가져왔다는 trace"**다.

노트북은 개념 데모(3개 tool, 파이썬 리스트 fixture)이지만 실제 `student_parts` 파일은 훨씬 크다 — 실제
MCP 서버 + 7개 wrapper tool + Week 6 인계용 `collect_member_schedules`까지 포함한다. 구현 대상은
MCP 서버가 아니라, 그 서버를 감싸는 LangChain `@tool` wrapper와 병합 로직이다. `mcp_server/`,
`fixed/` 전체는 완성돼 있고 수정 대상이 아니다.

Week 6의 `kana_agent`가 `collect_member_schedules`와 `list_shared_schedules`를 그대로 재사용하므로
이 둘이 이번 주 설계의 중심이다. 단 **최종 회의 시간 선택은 Week 6 범위**이며 이번 주는 바쁜 시간을
모으는 데서 멈춘다. 추가과제(`create_shared_schedule`/`delete_shared_schedule`)는 메인과제 범위 밖.

## 구현 대상 파일

- `student_parts/week05_load_kanas_past_conversations.py` (메인 구현)
- `tests/test_week05.py` (신규 pytest 파일)

**수정 금지:** `mcp_server/`, `fixed/` 전체

## 진행 방식

5개 step으로 나눠서, 각 step마다 "구현 → 테스트 추가 → pytest 실행 → 설명 → 승인 후 다음 step" 순서로 진행한다.

---

## Step 1 — 단순 pass-through wrapper 4개

가이드 핵심 제약: *"wrapper tool은 직접 SQL이나 중복 정규화 helper를 두지 않고 store/MCP helper의
결과 JSON을 전달한다."* 가공하지 않는다.

```python
@tool(args_schema=SearchPreviousConversationsInput)
def search_previous_conversations(query, member_names=None, limit=5) -> str:
    return call_mcp_tool_sync("search_previous_conversations", {
        "query": query, "member_names": member_names, "limit": limit,
    })   # MCP 결과 문자열 그대로 반환


@tool(args_schema=LoadConversationMessagesInput)
def load_conversation_messages(conversation_id) -> str:
    payload = call_external_tool_payload("load_conversation_messages",
                                         {"conversation_id": conversation_id})
    return json_payload(payload)   # sender/content/created_at 순서·키 보존


@tool(args_schema=ExtractSchedulesFromHistoryInput)
def extract_schedules_from_history(member_names, date_from, date_to) -> str:
    return call_mcp_tool_sync("extract_schedules_from_history", {
        "member_names": member_names, "date_from": date_from, "date_to": date_to,
    })


@tool(args_schema=ListSharedSchedulesInput)
def list_shared_schedules(member_names=None, date_from=None, date_to=None,
                          source_conversation_id=None, limit=50) -> str:
    return call_mcp_tool_sync("list_shared_schedules", {
        "member_names": member_names, "date_from": date_from, "date_to": date_to,
        "source_conversation_id": source_conversation_id, "limit": limit,
    })
```

`member_names=None`은 "전체 멤버", `[]`는 "빈 결과" — MCP 서버 docstring이 명시한 구분이므로
`member_names or []` 같은 습관적 코드로 뭉개지 않는다.

`@tool` docstring도 함께 보강한다. Week 4 테스트가 `tool.description` 내용을 assert하는 데서 보이듯
이 프로젝트는 docstring을 주 라우팅 신호로 취급한다. `search_previous_conversations`(남의 대화)와
Week 4 `search_conversation_messages`(내 채팅)의 혼동을 docstring에서 직접 차단한다.

## Step 2 — `_personal_schedules_for_current_scope()`

```python
PERSONAL_SCHEDULE_FETCH_LIMIT = 200

def _personal_schedules_for_current_scope() -> list[dict[str, Any]]:
    saved_rows = AppSQLiteStore(CONFIG.app_db_path).list_schedules(
        limit=PERSONAL_SCHEDULE_FETCH_LIMIT      # 기본값 12는 조용히 잘림
    )
    saved_ids = {str(r["schedule_id"]) for r in saved_rows if r.get("schedule_id")}
    session_id = current_session_scope()
    session_rows = [
        s for s in PERSONAL_SCHEDULES
        if _schedule_scope(s) == session_id and str(s.get("id") or "") not in saved_ids
    ]
    return [*saved_rows, *session_rows]
```

중복 제거 키는 `schedule_id` ⟷ `id` 정확 일치. 휴리스틱이 아니라 실재하는 링크다 — Week 3
`personal_create_schedule`이 `source_schedule_id=temp["id"]`로 저장하고
([app_store.py:353](../fixed/app_store.py#L353)), `schedule_id = source_schedule_id or new_id("sch")`가 된다.
`(title, date, start_time)` 조합은 쓰지 않는다 — 같은 날 같은 제목의 별개 일정을 삼켜 Week 6이
실제로 바쁜 시간을 "비었다"고 제안하게 만들 수 있다.

세부 결정:
- `kind` 필터 없음 — `group_schedule`도 내 바쁜 시간이다.
- SQLite row는 scope 필터 없음 — `schedules`에 `session_id` 컬럼이 없고 대화를 넘나드는 게 Week 3의
  약속이다. 임시 row만 scope로 거른다.
- `PERSONAL_SCHEDULES` dict를 in-place 변경하지 않는다(Week 1 전역 리스트 오염 방지).

> 참고: 현재 Week 3 `personal_create_schedule`이 스텁이라 임시 row가 SQLite로 아직 안 간다. 중복
> 제거는 오늘은 무동작이지만 Week 3 추가과제 완성 시 바로 동작하도록 지금 넣는다.

## Step 3 — `_collect_member_schedules()` (이번 주 핵심)

```python
def _personal_busy_row(row) -> dict:
    """내 일정을 외부 멤버 row와 동일 구조로 맞춘다."""
    request_id = str(row.get("request_id") or "").strip()
    return {
        "member_name": PERSONAL_SHARED_MEMBER_NAME,          # "나"
        "title": str(row.get("title") or "제목 없음"),
        "date": _busy_row_date(row.get("date")),             # "T" 앞부분만
        "start_time": str(row.get("start_time") or "미정"),
        "end_time": str(row.get("end_time") or "미정"),
        "notes": "앱 저장 일정" if row.get("schedule_id") else "현재 대화 임시 일정",
        "source_conversation_id": f"app:{request_id}" if request_id else None,
        "schedule_id": str(row.get("schedule_id") or row.get("id") or ""),
    }


def _collect_member_schedules(*, member_names, date_from, date_to, personal_schedules) -> dict:
    requested = normalize_external_member_names(member_names)
    d_from, d_to = normalize_external_schedule_date_bounds(member_names, date_from, date_to)

    # "나"는 앱 SQLite가 원본. 공유 저장소에도 자동 동기화된 복사본이 있어
    # MCP에 "나"를 넘기면 같은 일정이 rows에 두 번 들어간다.
    external_members = [n for n in requested if n != PERSONAL_SHARED_MEMBER_NAME]

    external_rows, external_payload = [], {}
    if external_members:                                    # 빈 목록이면 subprocess 절약
        external_payload = json.loads(call_mcp_tool_sync(
            "extract_schedules_from_history",
            {"member_names": external_members, "date_from": d_from, "date_to": d_to}))
        external_rows = list(external_payload.get("rows") or [])

    personal_rows = [_personal_busy_row(r) for r in personal_schedules
                     if _in_date_range(r.get("date"), d_from, d_to)]

    rows = sorted([*personal_rows, *external_rows], key=_busy_row_sort_key)
    return {
        "ok": bool(external_payload.get("ok", True)),
        "tool_name": "collect_member_schedules",
        "member_names": requested,
        "external_member_names": external_members,   # "나" 제외가 trace에 보이도록
        "date_from": d_from, "date_to": d_to,
        "rows": rows,
        "personal_row_count": len(personal_rows),
        "external_row_count": len(external_rows),
        "schedule_summary": external_schedule_summary(rows),   # 합친 rows 전체 요약
    }
```

핵심 설계 판단:

1. **"나" 이중 계산 차단** — 외부 DB에 `나|개인 코칭|2026-07-16|app:req_...` 행이 실제로 7건 있다
   (Week 3 자동 동기화 결과, `data/kanana_external_people.sqlite3` 조회로 확인). `member_names`에
   "나"가 들어오면 MCP 인자에서 제거하고, 내 일정은 앱 SQLite에서만 가져온다.
2. **`schedule_summary`는 합친 rows 전체를 요약** — 가이드가 "LLM이 바쁜 시간을 자연어로 설명할 수
   있게"라고 명시한 이상, 내 일정을 뺀 요약은 목적에 어긋난다. `external_schedule_summary`는 이름만
   external이고 실제로는 `member_name/title/date/start_time/end_time/notes`만 읽는 shape-agnostic
   함수라 내 row에도 그대로 쓸 수 있다.
3. **`_structured_request_from_schedule_row`는 쓰지 않는다** — 이미 작성돼 있고 docstring이 "이
   용도"라고 주장하지만 실제로는 함정이다. `StructuredRequest`의 시간 validator가 `\d{2}:\d{2}`에
   안 맞는 값을 `None`으로 만들어 `"미정"` → `null`이 된다. `collect_member_schedules`와
   `list_shared_schedules`가 같은 일정의 `end_time`을 다르게 보고하게 된다. 함수는 남겨두되(주어진
   코드, 삭제 시 참고답안과 diff 충돌) 미사용으로 둔다.
4. **`rows` 정렬 + 날짜 없는 내 일정 제외** — Week 6이 `busy_rows`로 구간 스윕할 때 재정렬 없이 쓰도록
   `(date, start_time, member_name)` 정렬. `date=None`인 일정은 특정 기간의 바쁜 시간 근거가 될 수
   없으므로 제외.

tool 본체는 조립만 한다:

```python
@tool(args_schema=CollectMemberSchedulesInput)
def collect_member_schedules(member_names, date_from, date_to) -> str:
    return json_payload(_collect_member_schedules(
        member_names=member_names, date_from=date_from, date_to=date_to,
        personal_schedules=_personal_schedules_for_current_scope()))
```

## Step 4 — `week05_prompt_parts()`

Week 4 스타일 그대로: 긴 한글 문단 하나를 리스트에 인라인 추가(모듈 상수 아님). 담아야 할 라우팅 규칙:

1. 내 일정 → Week 3 앱 SQLite tool만. 외부 MCP tool 금지
2. 남(철수·영희·…) → 외부 MCP tool만. 앱 SQLite tool 금지
3. 남의 대화 내용 → `search_previous_conversations`, `query`는 조사 없는 짧은 핵심 명사구
   (MCP 서버가 토큰화를 안 한다고 docstring에 명시)
4. 대화 한 건 전문 → 검색 결과의 `conversation_id`로 `load_conversation_messages`
5. 남의 일정만 → `extract_schedules_from_history`
6. **나+남 조율 → `collect_member_schedules`, `member_names`에 "나" 넣지 말 것** (가장 중요한 규칙)
7. 공유 저장소 row 확인 → `list_shared_schedules`
8. 날짜는 항상 `YYYY-MM-DD`, 미지정 시 오늘(`current_app_date_iso()`)부터 2주
9. `rows`가 비면 근거 없다고 답하고 지어내지 않음
10. 최종 회의 시간은 혼자 확정하지 않음 (Week 5/6 책임 경계)

`current_app_date_iso`는 이미 import돼 있으나 미사용 상태 — 8번에서 사용한다.

## Step 5 — `tests/test_week05.py`

Week 4 규약(플랫 함수, `Tool.invoke({...})` → `json.loads`, `tool.description` assert, 파일 끝 수동
E2E 체크리스트)을 따른다.

**monkeypatch 대상 — 이 테스트의 핵심 난점:**

| 대상 | 이유 |
|---|---|
| `w5.call_mcp_tool_sync`, `w5.call_external_tool_payload` | import 시점에 바인딩된 별칭이라 원본 모듈을 패치해도 wrapper에 안 닿는다 |
| `fixed.mcp_client.*` → 예외 발생 | 안전망. 빠져나간 경로가 실제 subprocess를 띄워 배포된 DB를 오염시키는 걸 크게 실패시킨다 |
| `fixed.app_store`의 sync/delete 4개 | Week 3·4 fixture와 동일. 없으면 `save_structured_request`가 실제 외부 DB에 행을 남긴다 |
| `w5.CONFIG` → `tmp_path` app DB | `_personal_schedules_for_current_scope`가 `CONFIG.app_db_path`를 읽는다 |
| `PERSONAL_SCHEDULES.clear()`, `w5._WEEK05_AGENT = None` | autouse, 전후 양쪽 |

fixture 3개: `_reset_week05_state`(autouse), `fake_mcp`(tool별 가짜 payload + 호출 인자 기록),
`use_temp_app_db`.

## 재사용한 기존 코드

| 함수/클래스 | 위치 | 용도 |
|---|---|---|
| `call_local_mcp_tool_sync` | `fixed/mcp_client.py` | MCP tool 호출 (파일 내 `call_mcp_tool_sync` 별칭) |
| `call_external_tool_payload` | `fixed/external_mcp.py` | MCP 결과를 dict로 (`load_conversation_messages` 전용) |
| `normalize_external_member_names` | `fixed/external_people_store.py` | 멤버 이름 정규화 |
| `normalize_external_schedule_date_bounds` | `fixed/external_people_store.py` | ISO datetime → `YYYY-MM-DD` |
| `external_schedule_summary` | `fixed/external_people_store.py` | rows → 자연어 요약 |
| `AppSQLiteStore.list_schedules` | `fixed/app_store.py:480` | 내 저장 일정 조회 |
| `current_session_scope` / `_schedule_scope` | `fixed/session_scope.py` / 이 파일 | 현재 대화 범위 임시 일정 |
| `json_payload`, 입력 스키마 7개, `week05_tools()`, `build_week05_agent()` | 이 파일 (기존 작성됨) | 그대로 사용 |

## 테스트 (`tests/test_week05.py`, pytest)

| 축 | 테스트 함수 |
|---|---|
| helper 로직 | `test_personal_schedules_includes_sqlite_saved_rows`, `test_personal_schedules_overrides_default_limit_of_twelve`, `test_personal_schedules_includes_group_schedule_rows`, `test_personal_schedules_includes_current_scope_temp_rows`, `test_personal_schedules_excludes_other_scope_temp_rows`, `test_personal_schedules_dedups_temp_row_already_saved_to_sqlite`, `test_personal_schedules_keeps_temp_row_when_no_sqlite_twin`, `test_personal_schedules_does_not_mutate_global_list` |
| 병합 규약 | `test_collect_member_schedules_returns_expected_keys`, `test_collect_member_schedules_builds_me_rows_in_external_shape`, `test_collect_member_schedules_filters_my_rows_by_date_range`, `test_collect_member_schedules_drops_my_rows_without_date`, `test_collect_member_schedules_rows_sorted_by_date_then_time`, `test_collect_member_schedules_summary_covers_both_sources` |
| "나" 처리 | `test_collect_member_schedules_strips_me_from_mcp_member_names`, `test_collect_member_schedules_skips_mcp_call_when_no_external_members` |
| pass-through | `test_search_previous_conversations_passes_none_member_names_through`, `test_search_previous_conversations_returns_mcp_string_unchanged`, `test_load_conversation_messages_uses_call_external_tool_payload_and_preserves_order`, `test_extract_schedules_from_history_passes_args_unchanged`, `test_list_shared_schedules_passes_all_filters_including_none_member_names` |
| 누적 구조 | `test_week05_tools_accumulates_week04_tools_and_new_names`, `test_week05_prompt_parts_includes_week04_parts`, `test_week05_prompt_parts_mentions_routing_tool_names`, `test_week05_prompt_parts_states_me_is_excluded_from_member_names` |

**실행:** `pytest tests/test_week05.py -v` (repo 루트). 전체 스위트로 회귀 확인.

### 수동 E2E 체크리스트

`./run.sh --week5`로 앱을 띄워 상세 trace 탭에서 확인 (`[ ]` 상태, 실제 확인은 사용자 몫):

- [ ] "철수가 예전에 무슨 일정 얘기했어?" → `search_previous_conversations` 호출, `query`가 짧은 명사구인지
- [ ] 이어서 "그 대화 전체 보여줘" → `load_conversation_messages`가 해당 `conversation_id`로
- [ ] "철수랑 영희 7월 7일~17일 일정 알려줘" → `extract_schedules_from_history`
- [ ] "철수·영희랑 7월 7일~17일 사이 회의 잡으려는데 서로 언제 바빠?" → `collect_member_schedules`,
      rows에 "나"와 "철수"가 같은 키 구조로, `schedule_summary` 존재
- [ ] "공유 일정 저장소에 뭐 등록돼 있어?" → `list_shared_schedules`, `rows` + `schedule_summary` 유지
- [ ] "내 일정만 보여줘" → Week 3 tool로 라우팅되고 MCP tool 미호출

⚠️ **날짜 함정**: `current_app_date_iso()`는 OS 날짜(오늘)를 쓰는데 시드 데이터는 **2026-07-07~17**이다.
"다음 주"로 물으면 외부 일정이 0건 나온다. E2E에서는 날짜를 명시할 것.

## 검증 방법

1. `pytest tests/test_week05.py -v` (repo 루트)
2. `pytest tests/ -v` — Week 1~4 회귀 없음 확인
3. 수동 E2E — 위 체크리스트

## 범위 밖 (건드리지 않음)

- `mcp_server/`, `fixed/` 전체
- Week 3 추가과제 스텁 (`personal_create_schedule` 등)
- `create_shared_schedule` / `delete_shared_schedule` — 메인 검증 후 재판단. 미구현으로 남길 경우
  `week05_tools()` 목록에서 제거 (가이드 명시 방식)
- 공통 가능 시간 계산·최종 회의 시간 결정 — Week 6 (`fixed/schedule_decision.py`)

## 이번 주차에서 발견했지만 범위 밖이라 다음 PR에서 멘토님께 질문할 것

1. `_structured_request_from_schedule_row`가 `[메인]`으로 표시되고 docstring이 "내 일정 row를 외부
   멤버 row와 같은 구조로 맞출 때 사용"이라고 하는데, `StructuredRequest`의 시간 validator가
   `"미정"`을 `None`으로 만들어 `list_shared_schedules`와 결과가 불일치한다. 의도된 것인지, 아니면
   직접 구성이 맞는지 확인 필요.
2. `collect_member_schedules`의 `member_names`에 "나"가 들어올 때 MCP 인자에서 제외하는 처리를
   wrapper에서 하는 게 맞는지, 아니면 프롬프트로만 유도하는 게 의도인지 확인 필요.
