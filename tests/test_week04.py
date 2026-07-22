import json
from types import SimpleNamespace

import pytest

import student_parts.week04_retrieve_nanas_memory as w4
from fixed.app_store import AppSQLiteStore
from fixed.reference_store import OpenAIEmbeddingFunction, PersonalReferenceStore
from student_parts.week04_retrieve_nanas_memory import (
    add_personal_reference,
    add_personal_reference_dict,
    safe_limit,
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
    모듈 전역(w4.REFERENCE_STORE/w4.SQLITE_STORE) 자체를 바꿔치기합니다."""

    monkeypatch.setattr(OpenAIEmbeddingFunction, "__call__", lambda self, input: _fake_embed(input))
    monkeypatch.setattr(OpenAIEmbeddingFunction, "embed_query", lambda self, input: _fake_embed(input))
    monkeypatch.setattr(OpenAIEmbeddingFunction, "embed_documents", lambda self, input: _fake_embed(input))

    test_reference_store = PersonalReferenceStore(tmp_path / "chroma")
    test_sqlite_store = AppSQLiteStore(tmp_path / "app.sqlite3")
    monkeypatch.setattr(w4, "REFERENCE_STORE", test_reference_store)
    monkeypatch.setattr(w4, "SQLITE_STORE", test_sqlite_store)
    return SimpleNamespace(reference_store=test_reference_store, sqlite_store=test_sqlite_store)


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


# --- tool description 상호 참조 (Step 4 — agent가 tool을 고르는 1차 근거) ---
# LangChain @tool은 함수 docstring을 그대로 tool.description으로 써서 LLM에게 전달한다.
# system prompt보다 이 description이 tool 선택에 더 직접적으로 쓰이므로, 서로 반대되는
# tool 이름을 언급해 "이거 아니면 저거"를 tool 스스로 설명하게 만든다.


def test_add_personal_reference_description_points_to_structured_save():
    assert "save_structured_request" in add_personal_reference.description


def test_search_personal_references_description_cross_references_search_saved_requests():
    assert "search_saved_requests" in search_personal_references.description


def test_search_saved_requests_description_cross_references_search_personal_references():
    assert "search_personal_references" in search_saved_requests.description


# --- week04_prompt_parts (공통, Step 4) ---


def test_week04_prompt_parts_mentions_all_three_tool_names():
    joined = " ".join(w4.week04_prompt_parts())

    assert "search_personal_references" in joined
    assert "search_saved_requests" in joined
    assert "add_personal_reference" in joined


def test_week04_prompt_parts_includes_week03_parts():
    from student_parts.week03_build_nanas_logbook import week03_prompt_parts

    prompt_parts = w4.week04_prompt_parts()
    for part in week03_prompt_parts():
        assert part in prompt_parts


# ============================================================
# 수동 E2E 테스트 체크리스트 (./run.sh --week4 또는 KANANA_ACTIVE_WEEK=4) — 메인과제만
# ============================================================
# pytest가 검증하지 않는 "실제 LLM이 문맥만으로 알맞은 tool을 고르는지"는 앱을 직접
# 실행해 상세 trace 패널의 tool_call/tool_result를 눈으로 확인해야 한다. 확인했으면
# [ ]를 [x]로 바꾸고 날짜/결과를 적어두자.
#
# [ ] 시나리오 1 — 참고자료 추가 → 참고자료 질문 → search_personal_references 호출
#     입력 1: "점심 시간에는 회의 잡지 말아달라고 참고자료로 적어줘"
#     확인: add_personal_reference tool_call 발생, 결과에 reference/reference_backend 키 존재
#     입력 2 (같은 대화): "내가 점심시간 관련해서 뭐라고 적어놨었지?"
#     확인: search_personal_references tool_call 발생 (search_saved_requests가 아님),
#           tool_result의 hits에 방금 적은 참고자료가 포함됨
#
# [ ] 시나리오 2 — 저장된 일정 키워드 검색 → search_saved_requests 호출
#     사전 준비: Week 3 방식으로 일정/할 일을 하나 저장해 둔다 (예: "보고서 제출 할일 저장해줘")
#     입력: "보고서 제출 관련해서 저장한 거 있어?"
#     확인: search_saved_requests tool_call 발생 (search_personal_references가 아님),
#           tool_result의 rows에 저장한 일정이 포함됨
#     비교 입력 (같은 대화 아님, 별도 확인): "내가 저장한 거 다 보여줘"
#     확인: 이건 list_saved_requests가 호출돼야 한다 (search_saved_requests가 아님) —
#           WEEK03_UNIFIED_LOOKUP_PROMPT가 "조건 없이 전체 나열" 요청을 우선 처리한다
#
# [ ] 시나리오 3 — 근거 없음 처리
#     입력: "내가 저장한 적 없는 것에 대해 물어봄"
#     확인: hits/rows가 비어 있을 때 모델이 근거 없다고 답하고 내용을 지어내지 않음
# ============================================================
