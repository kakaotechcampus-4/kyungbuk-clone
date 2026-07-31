import json
from types import SimpleNamespace

import pytest

import student_parts.week05_load_kanas_past_conversations as w5
from fixed.app_store import AppSQLiteStore
from fixed.external_people_store import external_schedule_summary
from fixed.session_scope import conversation_session_scope
from student_parts.week01_wake_up_nana import PERSONAL_SCHEDULES
from student_parts.week04_retrieve_nanas_memory import week04_prompt_parts, week04_tools
from student_parts.week05_load_kanas_past_conversations import (
    _collect_member_schedules,
    _personal_schedules_for_current_scope,
    collect_member_schedules,
    create_shared_schedule,
    delete_shared_schedule,
    extract_schedules_from_history,
    list_shared_schedules,
    load_conversation_messages,
    search_previous_conversations,
    week05_prompt_parts,
    week05_tools,
)


@pytest.fixture(autouse=True)
def _reset_week05_state(monkeypatch):
    """Week3/4와 동일하게 앱 SQLite 저장이 실제 외부 MCP DB로 새어나가지 않도록 막고,
    이 테스트 파일이 놓친 경로가 실제 MCP subprocess를 띄우면 조용히 넘어가지 않고
    바로 실패하도록 만듭니다."""

    monkeypatch.setattr("fixed.app_store.sync_personal_schedule_to_shared", lambda schedule: {"ok": True, "status": "stubbed"})
    monkeypatch.setattr("fixed.app_store.sync_group_schedule_to_shared", lambda schedule: {"ok": True, "status": "stubbed"})
    monkeypatch.setattr("fixed.app_store.delete_personal_schedule_from_shared", lambda request_id: {"ok": True, "deleted": []})
    monkeypatch.setattr("fixed.app_store.delete_group_schedule_from_shared", lambda schedule: {"ok": True, "deleted": []})

    def _no_subprocess(*args, **kwargs):
        raise AssertionError("실제 MCP subprocess가 호출됐습니다. fake_mcp fixture로 w5.call_mcp_tool_sync를 patch하세요.")

    monkeypatch.setattr("fixed.mcp_client.call_local_mcp_tool_sync", _no_subprocess)
    monkeypatch.setattr("fixed.mcp_client.load_local_mcp_tools_sync", _no_subprocess)

    PERSONAL_SCHEDULES.clear()
    w5._WEEK05_AGENT = None
    yield
    PERSONAL_SCHEDULES.clear()
    w5._WEEK05_AGENT = None


class _FakeMcp:
    """w5.call_mcp_tool_sync / w5.call_external_tool_payload를 파일 내부 별칭 기준으로
    patch합니다(둘 다 import 시점에 바인딩돼 원본 모듈 patch로는 닿지 않습니다).
    tool 이름별 가짜 rows를 미리 등록해두면 그 이름으로 호출될 때 반환하고, 모든 호출
    (tool_name, args)을 calls에 기록합니다."""

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []
        self.rows_by_tool: dict[str, list[dict]] = {}

    def call(self, tool_name, args, db_path=None):
        self.calls.append((tool_name, args))
        rows = self.rows_by_tool.get(tool_name, [])
        payload = {"ok": True, "tool_name": tool_name, "rows": rows}
        if tool_name in {"extract_schedules_from_history", "list_shared_schedules"}:
            payload["schedule_summary"] = external_schedule_summary(rows)
        return json.dumps(payload, ensure_ascii=False)

    def call_external_payload(self, tool_name, args):
        return json.loads(self.call(tool_name, args))


@pytest.fixture
def fake_mcp(monkeypatch):
    fake = _FakeMcp()
    monkeypatch.setattr(w5, "call_mcp_tool_sync", fake.call)
    monkeypatch.setattr(w5, "call_external_tool_payload", fake.call_external_payload)
    return fake


@pytest.fixture
def use_temp_app_db(tmp_path, monkeypatch):
    """_personal_schedules_for_current_scope가 읽는 CONFIG.app_db_path를 tmp_path로 교체합니다."""

    monkeypatch.setattr(w5, "CONFIG", SimpleNamespace(app_db_path=tmp_path / "app.sqlite3", has_openai_key=True))
    return AppSQLiteStore(tmp_path / "app.sqlite3")


def _save_personal_schedule(store, *, title, date, start_time, end_time="미정", source_schedule_id=None, kind="personal_schedule"):
    return store.save_structured_request(
        {
            "kind": kind,
            "title": title,
            "date": date,
            "start_time": start_time,
            "end_time": end_time,
            "members": [],
            "source_schedule_id": source_schedule_id,
        }
    )


# --- search_previous_conversations ---


def test_search_previous_conversations_passes_none_member_names_through(fake_mcp):
    search_previous_conversations.invoke({"query": "회의", "member_names": None, "limit": 5})
    tool_name, args = fake_mcp.calls[-1]
    assert tool_name == "search_previous_conversations"
    assert args["member_names"] is None
    assert args["query"] == "회의"
    assert args["limit"] == 5


def test_search_previous_conversations_passes_empty_member_names_through(fake_mcp):
    search_previous_conversations.invoke({"query": "회의", "member_names": [], "limit": 5})
    _, args = fake_mcp.calls[-1]
    assert args["member_names"] == []


def test_search_previous_conversations_returns_mcp_string_unchanged(fake_mcp):
    fake_mcp.rows_by_tool["search_previous_conversations"] = [{"conversation_id": "ext_cs", "content": "hi"}]
    result = search_previous_conversations.invoke({"query": "회의"})
    expected = json.dumps(
        {"ok": True, "tool_name": "search_previous_conversations", "rows": fake_mcp.rows_by_tool["search_previous_conversations"]},
        ensure_ascii=False,
    )
    assert result == expected


# --- load_conversation_messages ---


def test_load_conversation_messages_uses_call_external_tool_payload_and_preserves_order(fake_mcp):
    fake_mcp.rows_by_tool["load_conversation_messages"] = [
        {"sender": "철수", "content": "안녕", "created_at": "2026-07-07T09:00:00"},
        {"sender": "나", "content": "네", "created_at": "2026-07-07T09:01:00"},
    ]
    result = json.loads(load_conversation_messages.invoke({"conversation_id": "ext_cs"}))
    assert result["rows"] == fake_mcp.rows_by_tool["load_conversation_messages"]
    assert list(result["rows"][0].keys()) == ["sender", "content", "created_at"]
    assert fake_mcp.calls[-1] == ("load_conversation_messages", {"conversation_id": "ext_cs"})


# --- extract_schedules_from_history ---


def test_extract_schedules_from_history_passes_args_unchanged(fake_mcp):
    extract_schedules_from_history.invoke({"member_names": ["철수 "], "date_from": "2026-07-07T00:00:00", "date_to": "2026-07-17"})
    tool_name, args = fake_mcp.calls[-1]
    assert tool_name == "extract_schedules_from_history"
    # wrapper는 정규화를 하지 않는다 — 원본 그대로 MCP 경계까지 전달돼야 한다.
    assert args["member_names"] == ["철수 "]
    assert args["date_from"] == "2026-07-07T00:00:00"


# --- list_shared_schedules ---


def test_list_shared_schedules_passes_all_filters_including_none_member_names(fake_mcp):
    list_shared_schedules.invoke(
        {
            "member_names": None,
            "date_from": "2026-07-07",
            "date_to": "2026-07-17",
            "source_conversation_id": "app:req_1",
            "limit": 10,
        }
    )
    tool_name, args = fake_mcp.calls[-1]
    assert tool_name == "list_shared_schedules"
    assert args == {
        "member_names": None,
        "date_from": "2026-07-07",
        "date_to": "2026-07-17",
        "source_conversation_id": "app:req_1",
        "limit": 10,
    }


# --- create_shared_schedule / delete_shared_schedule ---


def test_create_shared_schedule_passes_all_args_through(fake_mcp):
    create_shared_schedule.invoke(
        {
            "member_name": "철수",
            "title": "QA 리뷰",
            "date": "2026-07-10",
            "start_time": "10:00",
            "end_time": "11:00",
            "notes": "회의실 A",
            "source_conversation_id": "app:req_1",
            "schedule_id": "shared_sch_1",
        }
    )
    tool_name, args = fake_mcp.calls[-1]
    assert tool_name == "create_shared_schedule"
    assert args == {
        "member_name": "철수",
        "title": "QA 리뷰",
        "date": "2026-07-10",
        "start_time": "10:00",
        "end_time": "11:00",
        "notes": "회의실 A",
        "source_conversation_id": "app:req_1",
        "schedule_id": "shared_sch_1",
    }


def test_create_shared_schedule_uses_default_end_time_when_omitted(fake_mcp):
    create_shared_schedule.invoke({"member_name": "철수", "title": "QA 리뷰", "date": "2026-07-10", "start_time": "10:00"})
    _, args = fake_mcp.calls[-1]
    assert args["end_time"] == "미정"
    assert args["notes"] is None


def test_delete_shared_schedule_passes_args_through(fake_mcp):
    delete_shared_schedule.invoke({"schedule_id": "shared_sch_1", "source_conversation_id": None})
    tool_name, args = fake_mcp.calls[-1]
    assert tool_name == "delete_shared_schedule"
    assert args == {"schedule_id": "shared_sch_1", "source_conversation_id": None}


# --- _personal_schedules_for_current_scope ---


def test_personal_schedules_includes_sqlite_saved_rows(use_temp_app_db):
    _save_personal_schedule(use_temp_app_db, title="개인 코칭", date="2026-07-16", start_time="10:00")
    rows = _personal_schedules_for_current_scope()
    assert any(row.get("schedule_id") for row in rows)
    assert any(row.get("title") == "개인 코칭" for row in rows)


def test_personal_schedules_overrides_default_limit_of_twelve(use_temp_app_db):
    for i in range(15):
        _save_personal_schedule(use_temp_app_db, title=f"일정{i}", date="2026-07-07", start_time=f"{9 + i % 10}:00")
    rows = _personal_schedules_for_current_scope()
    assert len([r for r in rows if r.get("schedule_id")]) == 15


def test_personal_schedules_includes_group_schedule_rows(use_temp_app_db):
    _save_personal_schedule(use_temp_app_db, title="팀 회의", date="2026-07-08", start_time="14:00", kind="group_schedule")
    rows = _personal_schedules_for_current_scope()
    assert any(row.get("title") == "팀 회의" for row in rows)


def test_personal_schedules_includes_current_scope_temp_rows(use_temp_app_db):
    with conversation_session_scope("conv-a"):
        PERSONAL_SCHEDULES.append(
            {"id": "personal_temp1", "title": "임시 일정", "date": "2026-07-09", "start_time": "11:00",
             "end_time": "미정", "attendees": [], "session_id": "conv-a"}
        )
        rows = _personal_schedules_for_current_scope()
    assert any(row.get("id") == "personal_temp1" for row in rows)


def test_personal_schedules_excludes_other_scope_temp_rows(use_temp_app_db):
    PERSONAL_SCHEDULES.append(
        {"id": "personal_temp2", "title": "다른 대화 일정", "date": "2026-07-09", "start_time": "11:00",
         "end_time": "미정", "attendees": [], "session_id": "conv-b"}
    )
    with conversation_session_scope("conv-a"):
        rows = _personal_schedules_for_current_scope()
    assert not any(row.get("id") == "personal_temp2" for row in rows)


def test_personal_schedules_dedups_temp_row_already_saved_to_sqlite(use_temp_app_db):
    PERSONAL_SCHEDULES.append(
        {"id": "personal_dup1", "title": "중복 일정", "date": "2026-07-10", "start_time": "10:00",
         "end_time": "미정", "attendees": [], "session_id": None}
    )
    _save_personal_schedule(use_temp_app_db, title="중복 일정", date="2026-07-10", start_time="10:00", source_schedule_id="personal_dup1")
    rows = _personal_schedules_for_current_scope()
    matches = [row for row in rows if row.get("schedule_id") == "personal_dup1" or row.get("id") == "personal_dup1"]
    assert len(matches) == 1
    assert matches[0].get("schedule_id") == "personal_dup1"


def test_personal_schedules_keeps_temp_row_when_no_sqlite_twin(use_temp_app_db):
    PERSONAL_SCHEDULES.append(
        {"id": "personal_solo1", "title": "저장 안 된 일정", "date": "2026-07-11", "start_time": "09:00",
         "end_time": "미정", "attendees": [], "session_id": None}
    )
    rows = _personal_schedules_for_current_scope()
    assert any(row.get("id") == "personal_solo1" for row in rows)


def test_personal_schedules_does_not_mutate_global_list(use_temp_app_db):
    PERSONAL_SCHEDULES.append(
        {"id": "personal_immutable1", "title": "불변 확인용", "date": "2026-07-12", "start_time": "09:00",
         "end_time": "미정", "attendees": [], "session_id": None}
    )
    before = [dict(s) for s in PERSONAL_SCHEDULES]
    _personal_schedules_for_current_scope()
    assert PERSONAL_SCHEDULES == before


# --- _collect_member_schedules / collect_member_schedules tool ---


def test_collect_member_schedules_returns_expected_keys(fake_mcp):
    result = _collect_member_schedules(member_names=["철수"], date_from="2026-07-07", date_to="2026-07-17", personal_schedules=[])
    assert result["ok"] is True
    assert set(result.keys()) == {
        "ok", "tool_name", "member_names", "external_member_names", "date_from", "date_to",
        "rows", "personal_row_count", "external_row_count", "schedule_summary",
    }


def test_collect_member_schedules_builds_me_rows_in_external_shape(fake_mcp):
    personal_schedules = [
        {"schedule_id": "sch_1", "request_id": "req_1", "title": "개인 코칭", "date": "2026-07-16", "start_time": "10:00", "end_time": "미정"},
    ]
    result = _collect_member_schedules(member_names=[], date_from="2026-07-07", date_to="2026-07-17", personal_schedules=personal_schedules)
    me_rows = [row for row in result["rows"] if row["member_name"] == "나"]
    assert len(me_rows) == 1
    row = me_rows[0]
    assert set(row.keys()) == {"member_name", "title", "date", "start_time", "end_time", "notes", "source_conversation_id", "schedule_id"}
    assert row["end_time"] == "미정"  # StructuredRequest 경로를 안 타야 null이 안 됨


def test_collect_member_schedules_strips_me_from_mcp_member_names(fake_mcp):
    _collect_member_schedules(member_names=["나", "철수"], date_from="2026-07-07", date_to="2026-07-17", personal_schedules=[])
    assert fake_mcp.calls, "외부 멤버가 있으므로 MCP가 호출돼야 한다"
    _, args = fake_mcp.calls[-1]
    assert args["member_names"] == ["철수"]


def test_collect_member_schedules_skips_mcp_call_when_no_external_members(fake_mcp):
    _collect_member_schedules(member_names=["나"], date_from="2026-07-07", date_to="2026-07-17", personal_schedules=[])
    assert fake_mcp.calls == []


def test_collect_member_schedules_filters_my_rows_by_date_range(fake_mcp):
    personal_schedules = [
        {"schedule_id": "sch_in", "title": "범위 안", "date": "2026-07-10", "start_time": "10:00", "end_time": "11:00"},
        {"schedule_id": "sch_out", "title": "범위 밖", "date": "2026-08-01", "start_time": "10:00", "end_time": "11:00"},
    ]
    result = _collect_member_schedules(member_names=[], date_from="2026-07-07", date_to="2026-07-17", personal_schedules=personal_schedules)
    titles = [row["title"] for row in result["rows"]]
    assert "범위 안" in titles
    assert "범위 밖" not in titles


def test_collect_member_schedules_drops_my_rows_without_date(fake_mcp):
    personal_schedules = [
        {"schedule_id": "sch_nodate", "title": "날짜 없음", "date": None, "start_time": "10:00", "end_time": "11:00"},
    ]
    result = _collect_member_schedules(member_names=[], date_from="2026-07-07", date_to="2026-07-17", personal_schedules=personal_schedules)
    assert result["rows"] == []
    assert result["personal_row_count"] == 0


def test_collect_member_schedules_rows_sorted_by_date_then_time(fake_mcp):
    fake_mcp.rows_by_tool["extract_schedules_from_history"] = [
        {"member_name": "철수", "title": "A", "date": "2026-07-08", "start_time": "09:00", "end_time": "10:00", "notes": None},
    ]
    personal_schedules = [
        {"schedule_id": "sch_1", "title": "B", "date": "2026-07-07", "start_time": "10:00", "end_time": "11:00"},
    ]
    result = _collect_member_schedules(member_names=["철수"], date_from="2026-07-07", date_to="2026-07-17", personal_schedules=personal_schedules)
    dates = [row["date"] for row in result["rows"]]
    assert dates == sorted(dates)


def test_collect_member_schedules_summary_covers_both_sources(fake_mcp):
    fake_mcp.rows_by_tool["extract_schedules_from_history"] = [
        {"member_name": "철수", "title": "API 연동 실습", "date": "2026-07-07", "start_time": "10:00", "end_time": "11:00", "notes": None},
    ]
    personal_schedules = [
        {"schedule_id": "sch_1", "title": "개인 코칭", "date": "2026-07-16", "start_time": "10:00", "end_time": "미정"},
    ]
    result = _collect_member_schedules(member_names=["철수"], date_from="2026-07-07", date_to="2026-07-17", personal_schedules=personal_schedules)
    assert "나" in result["schedule_summary"]
    assert "철수" in result["schedule_summary"]


def test_collect_member_schedules_tool_invoke_returns_json_string(use_temp_app_db, fake_mcp):
    _save_personal_schedule(use_temp_app_db, title="개인 코칭", date="2026-07-16", start_time="10:00")
    result = json.loads(collect_member_schedules.invoke({"member_names": [], "date_from": "2026-07-07", "date_to": "2026-07-17"}))
    assert any(row["title"] == "개인 코칭" for row in result["rows"])


# --- week05_tools / week05_prompt_parts (누적 구조) ---


def _tool_name(item):
    return getattr(item, "name", getattr(item, "__name__", str(item)))


def test_week05_tools_accumulates_week04_tools_and_new_names():
    tool_names = {_tool_name(item) for item in week05_tools()}
    week04_tool_names = {_tool_name(item) for item in week04_tools()}
    assert week04_tool_names <= tool_names
    assert {
        "search_previous_conversations",
        "load_conversation_messages",
        "extract_schedules_from_history",
        "create_shared_schedule",
        "delete_shared_schedule",
        "list_shared_schedules",
        "collect_member_schedules",
    } <= tool_names


def test_week05_prompt_parts_includes_week04_parts():
    week05_joined = "\n".join(week05_prompt_parts())
    for part in week04_prompt_parts():
        assert part in week05_joined


def test_week05_prompt_parts_mentions_routing_tool_names():
    joined = "\n".join(week05_prompt_parts())
    for name in [
        "search_previous_conversations",
        "load_conversation_messages",
        "extract_schedules_from_history",
        "collect_member_schedules",
        "list_shared_schedules",
    ]:
        assert name in joined


def test_week05_prompt_parts_states_me_is_excluded_from_member_names():
    joined = "\n".join(week05_prompt_parts())
    assert "나" in joined
    assert "member_names" in joined


def test_week05_prompt_parts_states_default_date_range_without_calling_date_functions():
    joined = "\n".join(week05_prompt_parts())
    assert "14일" in joined
