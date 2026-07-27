import json
from types import SimpleNamespace

import pytest

import student_parts.week04_retrieve_nanas_memory as w4
from fixed.app_store import AppSQLiteStore
from fixed.conversation_rag_store import ConversationRAGStore
from fixed.reference_store import OpenAIEmbeddingFunction, PersonalReferenceStore
from fixed.session_scope import DEFAULT_SESSION_SCOPE, conversation_session_scope
from student_parts.week04_retrieve_nanas_memory import (
    add_personal_reference,
    add_personal_reference_dict,
    safe_limit,
    search_conversation_message_rows,
    search_conversation_messages,
    search_conversation_messages_dict,
    search_nana_memory,
    search_personal_reference_hits,
    search_personal_references,
    search_saved_request_rows,
    search_saved_requests,
)


def _fake_embed(texts: list[str]) -> list[list[float]]:
    """네트워크 호출 없이 단어 겹침을 반영하는 결정적 가짜 벡터를 만듭니다."""

    dimension = 32
    vectors: list[list[float]] = []
    for text in texts:
        vector = [0.0] * dimension
        for token in str(text).split():
            vector[hash(token) % dimension] += 1.0
        vectors.append(vector)
    return vectors


@pytest.fixture
def use_temp_stores(tmp_path, monkeypatch):
    """REFERENCE_STORE/SQLITE_STORE를 tmp_path 기준 임시 store로 교체하고,
    embedding 호출이 실제 네트워크로 나가지 않도록 가짜 벡터로 대체합니다.
    week04 store들은 import 시점에 이미 생성돼 있으므로 CONFIG가 아니라
    모듈 전역(w4.REFERENCE_STORE/w4.SQLITE_STORE) 자체를 바꿔치기합니다.

    save_structured_request(group_schedule/personal_schedule)는 AppSQLiteStore 인스턴스가
    무엇이든 상관없이 fixed.app_store의 외부 MCP 동기화 함수를 무조건 호출해 실제
    data/kanana_external_people.sqlite3에 subprocess로 접근한다. test_week03.py의
    _reset_week03_state와 동일하게 여기서도 반드시 stub 처리해야 진짜 외부 DB가
    테스트로 오염되지 않는다."""

    monkeypatch.setattr("fixed.app_store.sync_personal_schedule_to_shared", lambda schedule: {"ok": True, "status": "stubbed"})
    monkeypatch.setattr("fixed.app_store.sync_group_schedule_to_shared", lambda schedule: {"ok": True, "status": "stubbed"})
    monkeypatch.setattr("fixed.app_store.delete_personal_schedule_from_shared", lambda request_id: {"ok": True, "deleted": []})
    monkeypatch.setattr("fixed.app_store.delete_group_schedule_from_shared", lambda schedule: {"ok": True, "deleted": []})

    monkeypatch.setattr(OpenAIEmbeddingFunction, "__call__", lambda self, input: _fake_embed(input))
    monkeypatch.setattr(OpenAIEmbeddingFunction, "embed_query", lambda self, input: _fake_embed(input))
    monkeypatch.setattr(OpenAIEmbeddingFunction, "embed_documents", lambda self, input: _fake_embed(input))

    test_reference_store = PersonalReferenceStore(tmp_path / "chroma")
    test_sqlite_store = AppSQLiteStore(tmp_path / "app.sqlite3")
    test_conversation_rag_store = ConversationRAGStore(tmp_path / "chroma")
    monkeypatch.setattr(w4, "REFERENCE_STORE", test_reference_store)
    monkeypatch.setattr(w4, "SQLITE_STORE", test_sqlite_store)
    monkeypatch.setattr(w4, "CONVERSATION_RAG_STORE", test_conversation_rag_store)
    return SimpleNamespace(
        reference_store=test_reference_store,
        sqlite_store=test_sqlite_store,
        conversation_rag_store=test_conversation_rag_store,
    )


def _seed_conversation(sqlite_store, *, title, messages):
    """title/messages로 conversation 하나를 만들고 conversation_id를 반환합니다."""

    conversation = sqlite_store.create_conversation(title=title)
    conversation_id = conversation["conversation_id"]
    for role, content in messages:
        sqlite_store.append_message(conversation_id, role, content)
    return conversation_id


# --- add_personal_reference_dict (메인과제) ---


def test_add_personal_reference_dict_returns_backend_and_reference(use_temp_stores):
    result = add_personal_reference_dict(
        use_temp_stores.reference_store,
        title="집중 시간",
        content="오전 10시~12시는 집중 회의 시간",
        tags=["preference"],
    )
    assert result["reference_backend"]["vector_store"] == "chromadb"
    assert result["reference"]["title"] == "집중 시간"
    assert result["reference"]["tags"] == ["preference"]


def test_add_personal_reference_dict_defaults_tags_to_empty_list(use_temp_stores):
    result = add_personal_reference_dict(
        use_temp_stores.reference_store,
        title="점심시간",
        content="점심시간에는 회의를 잡지 않는다",
        tags=None,
    )
    assert result["reference"]["tags"] == []


def test_add_personal_reference_dict_increments_collection_count(use_temp_stores):
    before = use_temp_stores.reference_store.collection.count()
    add_personal_reference_dict(
        use_temp_stores.reference_store,
        title="회의 선호",
        content="짧은 회의를 선호한다",
    )
    after = use_temp_stores.reference_store.collection.count()
    assert after == before + 1


# --- add_personal_reference tool (메인과제) ---


def test_add_personal_reference_tool_payload_shape(use_temp_stores):
    raw = add_personal_reference.invoke({"title": "휴가", "content": "여름 휴가는 8월 둘째 주"})
    result = json.loads(raw)
    assert result["ok"] is True
    assert result["tool_name"] == "add_personal_reference"
    assert result["reference"]["title"] == "휴가"
    assert "reference_backend" in result


def test_add_personal_reference_tool_defaults_tags_to_empty_list(use_temp_stores):
    raw = add_personal_reference.invoke(
        {"title": "출장", "content": "출장은 월요일에만 잡는다", "tags": None}
    )
    result = json.loads(raw)
    assert result["reference"]["tags"] == []


# --- safe_limit (공통) ---


def test_safe_limit_clamps_below_minimum():
    assert safe_limit(0) == 1
    assert safe_limit(-5) == 1


def test_safe_limit_clamps_above_maximum():
    assert safe_limit(999, maximum=20) == 20


def test_safe_limit_falls_back_to_default_on_invalid_input():
    assert safe_limit("abc", default=3) == 3
    assert safe_limit(None, default=5) == 5


# --- search_personal_reference_hits (메인과제) ---


def test_search_personal_reference_hits_finds_added_reference(use_temp_stores):
    content = "오전 10시부터 12시까지는 집중력이 높아 중요한 회의를 오전에 넣는다"
    add_personal_reference_dict(
        use_temp_stores.reference_store,
        title="집중 회의 선호",
        content=content,
        tags=["preference", "meeting"],
    )

    # 검색어를 저장한 문장과 똑같이 주면 가짜 embedding에서도 거리 0으로 최우선 매칭되어
    # seed 기본 참고자료(ref_focus 등) 존재 여부와 무관하게 안정적으로 검증할 수 있다.
    hits = search_personal_reference_hits(use_temp_stores.reference_store, query=content, top_k=1)

    assert len(hits) == 1
    hit = hits[0]
    assert hit["content"] == content
    assert hit["metadata"]["title"] == "집중 회의 선호"
    assert hit["metadata"]["tags"] == "preference,meeting"
    assert "distance" in hit


def test_search_personal_reference_hits_returns_all_references_for_blank_query(use_temp_stores):
    # 멘토 리뷰(PR #158): "내 참고자료 전부 검색해" 같은 요청에서 LLM이 query="" 로 호출하면
    # ChromaDB가 OpenAI embedding API에 빈 문자열을 넘겨 400 BadRequestError가 났다. 빈/공백 query는
    # 벡터 유사도 비교가 원천적으로 불가능하므로, embedding 호출 없이 참고자료 전체를 그대로 반환한다.
    before = search_personal_reference_hits(use_temp_stores.reference_store, query="", top_k=2)

    add_personal_reference_dict(
        use_temp_stores.reference_store, title="선호 A", content="아침에 운동하는 것을 선호한다", tags=["preference"]
    )
    add_personal_reference_dict(
        use_temp_stores.reference_store, title="선호 B", content="저녁에 산책하는 것을 선호한다", tags=["preference"]
    )

    after_blank = search_personal_reference_hits(use_temp_stores.reference_store, query="", top_k=2)
    after_whitespace = search_personal_reference_hits(use_temp_stores.reference_store, query="   ", top_k=2)

    assert len(after_blank) == len(before) + 2
    assert len(after_whitespace) == len(before) + 2
    titles = {hit["metadata"]["title"] for hit in after_blank}
    assert {"선호 A", "선호 B"} <= titles
    assert all(hit["distance"] is None for hit in after_blank)


# --- search_personal_references tool (메인과제) ---


def test_search_personal_references_tool_returns_hits_key(use_temp_stores):
    content = "여름 휴가는 8월 둘째 주에 간다"
    add_personal_reference.invoke({"title": "여름 휴가", "content": content})

    raw = search_personal_references.invoke({"query": content, "top_k": 1})
    result = json.loads(raw)

    assert list(result.keys()) == ["hits"]
    assert result["hits"][0]["content"] == content


def test_search_personal_references_tool_uses_default_top_k(use_temp_stores):
    add_personal_reference.invoke({"title": "메모", "content": "테스트용 메모"})

    raw = search_personal_references.invoke({"query": "테스트용 메모"})
    result = json.loads(raw)

    assert isinstance(result["hits"], list)
    assert len(result["hits"]) <= 2


def test_search_personal_references_tool_returns_all_hits_for_blank_query_without_error(use_temp_stores):
    add_personal_reference.invoke({"title": "메모1", "content": "테스트 메모 1"})
    add_personal_reference.invoke({"title": "메모2", "content": "테스트 메모 2"})

    raw = search_personal_references.invoke({"query": ""})
    result = json.loads(raw)

    assert list(result.keys()) == ["hits"]
    titles = {hit["metadata"]["title"] for hit in result["hits"]}
    assert {"메모1", "메모2"} <= titles


# --- search_saved_request_rows (메인과제) ---


def test_search_saved_request_rows_finds_matching_row(use_temp_stores):
    use_temp_stores.sqlite_store.save_structured_request(
        {"kind": "todo", "title": "보고서 제출", "date": "2026-07-25", "priority": "high", "reason": "마감 임박"}
    )

    rows = search_saved_request_rows(use_temp_stores.sqlite_store, query="보고서", top_k=5)

    assert len(rows) == 1
    assert rows[0]["title"] == "보고서 제출"
    assert rows[0]["kind"] == "todo"


def test_search_saved_request_rows_returns_empty_list_when_no_match(use_temp_stores):
    use_temp_stores.sqlite_store.save_structured_request({"kind": "todo", "title": "보고서 제출"})

    rows = search_saved_request_rows(use_temp_stores.sqlite_store, query="존재하지않는키워드", top_k=5)

    assert rows == []


# --- search_saved_requests tool (메인과제) ---


def test_search_saved_requests_tool_returns_rows_key(use_temp_stores):
    use_temp_stores.sqlite_store.save_structured_request(
        {"kind": "group_schedule", "title": "팀 회의", "date": "2026-07-30", "start_time": "15:00", "members": ["민수", "지아"]}
    )

    raw = search_saved_requests.invoke({"query": "팀 회의", "top_k": 5})
    result = json.loads(raw)

    assert list(result.keys()) == ["rows"]
    assert result["rows"][0]["title"] == "팀 회의"


def test_search_saved_requests_tool_returns_empty_rows_when_no_match(use_temp_stores):
    raw = search_saved_requests.invoke({"query": "아무도저장안한키워드"})
    result = json.loads(raw)

    assert result["rows"] == []


# --- search_conversation_messages_dict (추가과제 Step 1) ---


def test_search_conversation_messages_dict_syncs_and_finds_matching_conversation(use_temp_stores):
    conversation_id = _seed_conversation(
        use_temp_stores.sqlite_store,
        title="여행 얘기",
        messages=[("user", "제주도 여행 언제 갈지 얘기했었잖아"), ("assistant", "9월 초가 좋겠다고 했었어요")],
    )

    result = search_conversation_messages_dict(
        use_temp_stores.sqlite_store,
        use_temp_stores.conversation_rag_store,
        query="제주도 여행 언제 갈지 얘기했었잖아",
        top_k=5,
    )

    assert result["sync"]["total"] == 1
    assert result["sync"]["upserted"] == 1
    assert len(result["hits"]) == 1
    assert result["hits"][0]["conversation_id"] == conversation_id
    assert result["rows"] == result["hits"]
    assert "제주도" in result["context"]
    assert result["rag_backend"]["vector_store"] == "chromadb"


def test_search_conversation_messages_dict_direct_tool_call_scope_does_not_exclude_anything(use_temp_stores):
    conversation_id = _seed_conversation(
        use_temp_stores.sqlite_store,
        title="여행 얘기",
        messages=[("user", "제주도 여행 언제 갈지 얘기했었잖아")],
    )

    # current_session_scope()는 conversation_session_scope로 감싸지 않으면
    # DEFAULT_SESSION_SCOPE sentinel을 반환한다. 이 값이 실제 conversation_id로
    # 오인되어 제외되지 않아야 한다.
    result = search_conversation_messages_dict(
        use_temp_stores.sqlite_store,
        use_temp_stores.conversation_rag_store,
        query="제주도 여행 언제 갈지 얘기했었잖아",
        top_k=5,
    )

    assert [hit["conversation_id"] for hit in result["hits"]] == [conversation_id]


def test_search_conversation_messages_dict_excludes_current_conversation_when_real_scope_active(use_temp_stores):
    conversation_a = _seed_conversation(
        use_temp_stores.sqlite_store,
        title="대화 A",
        messages=[("user", "여행 준비물 뭐 챙길지 얘기했었잖아")],
    )
    _seed_conversation(
        use_temp_stores.sqlite_store,
        title="대화 B",
        messages=[("user", "여행 준비물 뭐 챙길지 얘기했었잖아")],
    )

    with conversation_session_scope(conversation_a):
        result = search_conversation_messages_dict(
            use_temp_stores.sqlite_store,
            use_temp_stores.conversation_rag_store,
            query="여행 준비물 뭐 챙길지 얘기했었잖아",
            top_k=5,
        )

    conversation_ids = [hit["conversation_id"] for hit in result["hits"]]
    assert conversation_a not in conversation_ids
    assert len(conversation_ids) == 1


def test_search_conversation_messages_dict_explicit_conversation_id_overrides_exclusion(use_temp_stores):
    conversation_a = _seed_conversation(
        use_temp_stores.sqlite_store,
        title="대화 A",
        messages=[("user", "여행 준비물 뭐 챙길지 얘기했었잖아")],
    )
    _seed_conversation(
        use_temp_stores.sqlite_store,
        title="대화 B",
        messages=[("user", "여행 준비물 뭐 챙길지 얘기했었잖아")],
    )

    with conversation_session_scope(conversation_a):
        result = search_conversation_messages_dict(
            use_temp_stores.sqlite_store,
            use_temp_stores.conversation_rag_store,
            query="여행 준비물 뭐 챙길지 얘기했었잖아",
            top_k=5,
            conversation_id=conversation_a,
        )

    assert [hit["conversation_id"] for hit in result["hits"]] == [conversation_a]


def test_search_conversation_messages_dict_returns_empty_when_no_conversations_synced(use_temp_stores):
    result = search_conversation_messages_dict(
        use_temp_stores.sqlite_store,
        use_temp_stores.conversation_rag_store,
        query="아무것도 없는 검색어",
        top_k=5,
    )

    assert result["hits"] == []
    assert result["sync"]["total"] == 0


# --- search_conversation_message_rows (추가과제 Step 1) ---


def test_search_conversation_message_rows_returns_only_hits_list(use_temp_stores):
    _seed_conversation(
        use_temp_stores.sqlite_store,
        title="여행 얘기",
        messages=[("user", "제주도 여행 언제 갈지 얘기했었잖아")],
    )

    dict_result = search_conversation_messages_dict(
        use_temp_stores.sqlite_store,
        use_temp_stores.conversation_rag_store,
        query="제주도 여행 언제 갈지 얘기했었잖아",
        top_k=5,
    )
    rows = search_conversation_message_rows(
        use_temp_stores.sqlite_store,
        query="제주도 여행 언제 갈지 얘기했었잖아",
        top_k=5,
    )

    assert rows == dict_result["hits"]


# --- search_conversation_messages tool (추가과제 Step 2) ---


def test_search_conversation_messages_tool_returns_expected_keys(use_temp_stores):
    _seed_conversation(
        use_temp_stores.sqlite_store,
        title="여행 얘기",
        messages=[("user", "제주도 여행 언제 갈지 얘기했었잖아")],
    )

    raw = search_conversation_messages.invoke({"query": "제주도 여행 언제 갈지 얘기했었잖아", "top_k": 5})
    result = json.loads(raw)

    assert set(result.keys()) == {"hits", "rows", "context", "rag_backend", "sync"}
    assert len(result["hits"]) == 1


def test_search_conversation_messages_tool_excludes_current_conversation_via_invoke(use_temp_stores):
    conversation_a = _seed_conversation(
        use_temp_stores.sqlite_store,
        title="대화 A",
        messages=[("user", "여행 준비물 뭐 챙길지 얘기했었잖아")],
    )
    _seed_conversation(
        use_temp_stores.sqlite_store,
        title="대화 B",
        messages=[("user", "여행 준비물 뭐 챙길지 얘기했었잖아")],
    )

    with conversation_session_scope(conversation_a):
        raw = search_conversation_messages.invoke(
            {"query": "여행 준비물 뭐 챙길지 얘기했었잖아", "top_k": 5}
        )
    result = json.loads(raw)

    conversation_ids = [hit["conversation_id"] for hit in result["hits"]]
    assert conversation_a not in conversation_ids
    assert len(conversation_ids) == 1


def test_search_conversation_messages_description_cross_references_other_tools():
    assert "search_personal_references" in search_conversation_messages.description
    assert "search_saved_requests" in search_conversation_messages.description


# --- tool description 상호 참조 (Step 4 — agent가 tool을 고르는 1차 근거) ---
# LangChain @tool은 함수 docstring을 그대로 tool.description으로 써서 LLM에게 전달한다.
# system prompt보다 이 description이 tool 선택에 더 직접적으로 쓰이므로, 서로 반대되는
# tool 이름을 언급해 "이거 아니면 저거"를 tool 스스로 설명하게 만든다.


def test_add_personal_reference_description_points_to_structured_save():
    assert "save_structured_request" in add_personal_reference.description


def test_add_personal_reference_description_instructs_tags_to_always_be_filled():
    # 멘토 리뷰(PR #158): tags가 채워질 법한 문장에도 LLM이 종종 생략해서, tool description에
    # "항상 채워라" + 예시를 명시적으로 추가했다. optional 필드라 LLM이 생략하기 쉬우므로
    # docstring이 이 지시를 실제로 담고 있는지 검증한다.
    description = add_personal_reference.description
    assert "tags" in description
    assert "항상" in description
    assert "preference" in description and "meeting" in description


def test_search_personal_references_description_cross_references_search_saved_requests():
    assert "search_saved_requests" in search_personal_references.description


def test_search_personal_references_description_instructs_blank_query_for_full_list():
    description = search_personal_references.description
    assert "빈 문자열" in description
    assert "전체" in description


def test_search_saved_requests_description_cross_references_search_personal_references():
    assert "search_personal_references" in search_saved_requests.description


# --- week04_prompt_parts (공통, Step 4) ---


def test_week04_prompt_parts_mentions_all_three_tool_names():
    joined = " ".join(w4.week04_prompt_parts())

    assert "search_personal_references" in joined
    assert "search_saved_requests" in joined
    assert "add_personal_reference" in joined


def test_week04_prompt_parts_mentions_search_conversation_messages():
    joined = " ".join(w4.week04_prompt_parts())

    assert "search_conversation_messages" in joined


def test_week04_prompt_parts_mentions_search_nana_memory():
    joined = " ".join(w4.week04_prompt_parts())

    assert "search_nana_memory" in joined


def test_week04_prompt_parts_includes_week03_parts():
    from student_parts.week03_build_nanas_logbook import week03_prompt_parts

    prompt_parts = w4.week04_prompt_parts()
    for part in week03_prompt_parts():
        assert part in prompt_parts


# --- search_nana_memory (추가과제 Step 3 → 이후 week04_tools()에 노출) ---


def test_search_nana_memory_combines_reference_and_saved_request_chunks(use_temp_stores):
    add_personal_reference_dict(
        use_temp_stores.reference_store,
        title="회의 선호",
        content="오전 회의를 선호한다",
    )
    use_temp_stores.sqlite_store.save_structured_request(
        {"kind": "todo", "title": "보고서 제출", "date": "2026-07-25", "priority": "high", "reason": "마감 임박"}
    )

    raw = search_nana_memory.invoke({"query": "보고서 제출"})
    result = json.loads(raw)

    assert result["ok"] is True
    assert result["tool_name"] == "search_nana_memory"
    assert any(chunk.startswith("[참고자료]") for chunk in result["chunks"])
    assert any(chunk.startswith("[저장기록]") for chunk in result["chunks"])
    assert "[참고자료]" in result["context"]
    assert "[저장기록]" in result["context"]
    assert "reference_backend" in result


def test_search_nana_memory_filters_saved_rows_by_date_range(use_temp_stores):
    use_temp_stores.sqlite_store.save_structured_request(
        {"kind": "todo", "title": "보고서 제출", "date": "2026-07-25"}
    )
    use_temp_stores.sqlite_store.save_structured_request(
        {"kind": "todo", "title": "보고서 제출 리마인드", "date": "2026-08-25"}
    )

    raw = search_nana_memory.invoke(
        {"query": "보고서 제출", "date_from": "2026-08-01", "date_to": "2026-08-31", "limit": 10}
    )
    result = json.loads(raw)

    titles = [row["title"] for row in result["rows"]]
    assert "보고서 제출 리마인드" in titles
    assert "보고서 제출" not in titles
    assert not any(chunk.startswith("[저장기록] 보고서 제출 (") for chunk in result["chunks"])


def test_search_nana_memory_filters_saved_rows_by_attendee(use_temp_stores):
    use_temp_stores.sqlite_store.save_structured_request(
        {
            "kind": "group_schedule",
            "title": "팀 회의",
            "date": "2026-07-30",
            "start_time": "15:00",
            "members": ["민수", "지아"],
        }
    )
    use_temp_stores.sqlite_store.save_structured_request(
        {
            "kind": "group_schedule",
            "title": "다른 팀 회의",
            "date": "2026-07-31",
            "start_time": "10:00",
            "members": ["철수"],
        }
    )

    raw = search_nana_memory.invoke({"query": "팀 회의", "attendee": "민수", "limit": 10})
    result = json.loads(raw)

    titles = [row["title"] for row in result["rows"]]
    assert titles == ["팀 회의"]


def test_search_nana_memory_exposed_in_week04_tools():
    tool_names = {getattr(tool, "name", "") for tool in w4.week04_tools()}
    assert "search_nana_memory" in tool_names


def test_search_nana_memory_description_cross_references_other_tools():
    description = search_nana_memory.description
    assert "search_conversation_messages" in description
    assert "list_saved_requests" in description
    assert "search_personal_references" in description
    assert "search_saved_requests" in description


# ============================================================
# 수동 E2E 테스트 체크리스트 (./run.sh --week4 또는 KANANA_ACTIVE_WEEK=4)
# ============================================================
# pytest가 검증하지 않는 "실제 LLM이 문맥만으로 알맞은 tool을 고르는지"는 앱을 직접
# 실행해 상세 trace 패널의 tool_call/tool_result를 눈으로 확인해야 한다. 확인했으면
# [ ]를 [x]로 바꾸고 날짜/결과를 적어두자.
#
# [x] 시나리오 1 — 참고자료 추가 → 참고자료 질문 → search_personal_references 호출
#     (2026-07-22 확인 완료 — build_week04_agent()로 직접 돌려본 사전 스모크 테스트에서
#     격리된 임시 store를 사용해 실제 LLM 호출로 확인함, Gradio 브라우저 조작 대신)
#     입력 1: "점심 시간에는 회의 잡지 말아달라고 참고자료로 적어줘"
#     확인: add_personal_reference tool_call 발생, 결과에 reference/reference_backend 키 존재
#     입력 2 (같은 대화): "내가 점심시간 관련해서 뭐라고 적어놨었지?"
#     확인: search_personal_references tool_call 발생 (search_saved_requests가 아님),
#           tool_result의 hits에 방금 적은 참고자료가 포함됨
#     결과: 정상 동작 확인. add_personal_reference → search_personal_references 순으로
#     호출됐고, 답변에 방금 저장한 메모 내용이 그대로 인용됨.
#
# [x] 시나리오 2 — 저장된 일정 키워드 검색 → search_saved_requests 호출
#     (2026-07-22 확인 완료, 위와 동일한 방식)
#     사전 준비: Week 3 방식으로 일정/할 일을 하나 저장해 둔다 (예: "보고서 제출 할일 저장해줘")
#     입력: "보고서 제출 관련해서 저장한 거 있어?"
#     확인: search_saved_requests tool_call 발생 (search_personal_references가 아님),
#           tool_result의 rows에 저장한 일정이 포함됨
#     비교 입력 (같은 대화 아님, 별도 확인): "내가 저장한 거 다 보여줘"
#     확인: 이건 list_saved_requests가 호출돼야 한다 (search_saved_requests가 아님) —
#           WEEK03_UNIFIED_LOOKUP_PROMPT가 "조건 없이 전체 나열" 요청을 우선 처리한다
#     결과: 정상 동작 확인. 키워드 질문은 search_saved_requests, 전체 나열 질문은
#     list_saved_requests로 정확히 분리 호출됨.
#
# [x] 시나리오 3 — 근거 없음 처리
#     (2026-07-22 확인 완료, 위와 동일한 방식)
#     입력: "내가 저장한 적 없는 것에 대해 물어봄" (예: "내가 화성 여행 계획에 대해 뭐라고 적어놨었지?")
#     확인: hits/rows가 비어 있을 때 모델이 근거 없다고 답하고 내용을 지어내지 않음
#     결과: 정상 동작 확인. 검색 결과가 없다고 정직하게 답하고 내용을 지어내지 않음.
#
# --- 추가과제 시나리오 (search_conversation_messages / search_nana_memory) ---
#
# [x] 시나리오 4 — 일반 채팅 발화 검색 + 현재 대화 제외
#     (2026-07-25 확인 완료 — AgentRuntime.run_agent()로 직접 돌려본 사전 스모크 테스트에서
#     CONFIG.app_db_path/chroma_dir/external_db_path를 임시 디렉터리로 격리하고
#     fixed.app_store의 외부 MCP 동기화 함수도 no-op으로 패치한 뒤 실제 LLM 호출로 확인함,
#     Gradio 브라우저 조작 대신. 실제 앱 데이터는 건드리지 않음.)
#     입력 1 (대화 A): "다음 주에 부산 여행 갈 계획인데 해운대 근처 숙소 좀 추천해줄래?"
#     입력 2 (대화 B, 별도 대화): "지난 달에 부산 여행 갔을 때 해운대에서 회 먹었던 게 진짜 맛있었어"
#     입력 3 (대화 A 안에서, 같은 대화): "내가 전에 채팅으로 부산 여행 관련해서 뭐라고 했었는지 찾아줄 수 있어?"
#     확인: search_conversation_messages tool_call 발생 (query="부산 여행"), tool_result의
#           hits에 대화 B의 conversation_id만 포함되고 대화 A(현재 대화)는 제외됨
#     결과: 정상 동작 확인. hits == [대화 B]였고 대화 A는 정확히 빠짐. 답변도 대화 B의
#     발화를 근거로 인용했고, 방금 turn3에서 한 말이 검색 결과에 섞이지 않음.
#
# [x] 시나리오 5 — search_nana_memory 필터링 (직접 .invoke() 확인, 함수 자체 동작)
#     (2026-07-25 확인 완료, 위와 동일한 격리 환경에서 직접 .invoke() 호출)
#     사전 준비: add_personal_reference로 참고자료 1건, save_structured_request로
#     group_schedule 2건(날짜/참석자 다르게) 저장
#     확인 1: 필터 없이 호출 → 참고자료+저장기록 chunk가 context에 함께 나옴
#     확인 2: date_from/date_to 지정 → 범위 밖 저장기록이 rows/context에서 빠짐
#     확인 3: attendee 지정 → 참석자 목록에 없는 저장기록이 빠짐
#     결과: 3개 확인 모두 통과.
#
# [x] 시나리오 6 — search_nana_memory가 week04_tools()에 노출된 뒤 agent가 실제로 호출하는지
#     (2026-07-27 확인 — 격리된 임시 환경에서 실제 앱 데이터는 안 건드리고 테스트해봄)
#     입력 (참고자료 1건 + 저장기록 1건을 미리 만들어둔 상태): "팀 회의 관련해서 내가 적어둔
#     메모랑 저장된 일정 둘 다 알려줘"를 5회 반복 실행
#     결과: 호출 여부가 왔다갔다 함 — 5회 중 2회만 search_nana_memory 호출(그마저도
#     search_saved_requests와 중복으로 같이 호출됨), 나머지 3회는 이전과 동일하게
#     search_personal_references + search_saved_requests를 따로 호출해서 답변을 종합함.
#     노출/description/프롬프트는 반영됐지만 agent의 실제 tool 선택은 안정적이지 않음.
#     비교 입력("내가 저장한 거 다 보여줘" → list_saved_requests 호출 확인)은 이번엔 안 해봄 — [ ]로 남겨둠.
#
# 참고: 이 확인 과정에서 tests/test_week04.py의 use_temp_stores fixture가 외부 MCP
# 동기화 함수(sync_group_schedule_to_shared 등)를 stub하지 않고 있어서, group_schedule을
# 저장하는 기존/신규 테스트가 실행될 때마다 실제 data/kanana_external_people.sqlite3에
# 테스트용 "팀 회의"/"다른 팀 회의" row가 leak되고 있었음을 발견했다 (test_week03.py는
# 이미 stub 처리돼 있었으나 test_week04.py에는 빠져 있던 기존 버그). fixture에 동일한
# stub 4줄을 추가해 재발을 막았고, 이미 leak된 36건은 실제 DB에서 삭제해 정리했다.
# ============================================================
