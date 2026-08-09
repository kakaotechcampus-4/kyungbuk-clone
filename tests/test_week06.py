import json
from types import SimpleNamespace

import pytest

import student_parts.week05_load_kanas_past_conversations as w5
import student_parts.week06_kanamate_decides_schedule as w6
from student_parts.week01_wake_up_nana import PERSONAL_SCHEDULES
from student_parts.week04_retrieve_nanas_memory import week04_prompt_parts, week04_tools
from student_parts.week05_load_kanas_past_conversations import week05_prompt_parts
from student_parts.week06_kanamate_decides_schedule import (
    agent_tool_names,
    build_langchain_supervisor_agent,
    decide_final_slot,
    extract_langchain_trace,
    find_common_available_slots,
    find_common_available_slots_dict,
    kana_agent,
    kana_prompt_parts,
    kana_system_prompt,
    kana_tools,
    nana_agent,
    nana_prompt_parts,
    nana_system_prompt,
    supervisor_system_prompt,
    supervisor_tools,
    tool_name,
    week06_prompt_parts,
)


@pytest.fixture(autouse=True)
def _reset_week06_state(monkeypatch):
    """앱 SQLite 저장이 실제 외부 MCP DB로 새어나가지 않게 막고, 실제 MCP subprocess나 실제 LLM
    호출이 일어나면 조용히 넘어가지 않고 바로 실패시킵니다. Week 6은 모듈 전역 agent 3개를
    build 시점의 system prompt로 캐시하므로 테스트 전후로 반드시 비웁니다."""

    monkeypatch.setattr("fixed.app_store.sync_personal_schedule_to_shared", lambda s: {"ok": True, "status": "stubbed"})
    monkeypatch.setattr("fixed.app_store.sync_group_schedule_to_shared", lambda s: {"ok": True, "status": "stubbed"})
    monkeypatch.setattr("fixed.app_store.delete_personal_schedule_from_shared", lambda r: {"ok": True, "deleted": []})
    monkeypatch.setattr("fixed.app_store.delete_group_schedule_from_shared", lambda s: {"ok": True, "deleted": []})

    def _no_subprocess(*args, **kwargs):
        raise AssertionError("실제 MCP subprocess가 호출됐습니다.")

    def _no_llm(*args, **kwargs):
        raise AssertionError("실제 LLM이 호출됐습니다. fake_agents fixture를 사용하세요.")

    monkeypatch.setattr("fixed.mcp_client.call_local_mcp_tool_sync", _no_subprocess)
    monkeypatch.setattr("fixed.mcp_client.load_local_mcp_tools_sync", _no_subprocess)
    monkeypatch.setattr(w6, "chat_model", _no_llm)

    PERSONAL_SCHEDULES.clear()
    w6._NANA_SUBAGENT = w6._KANA_SUBAGENT = w6._SUPERVISOR_AGENT = None
    w5._WEEK05_AGENT = None
    yield
    PERSONAL_SCHEDULES.clear()
    w6._NANA_SUBAGENT = w6._KANA_SUBAGENT = w6._SUPERVISOR_AGENT = None
    w5._WEEK05_AGENT = None


class _FakeAgent:
    """create_agent(...)가 돌려주는 LangChain agent를 대신하는 가짜 객체입니다.
    invoke()는 미리 세팅된 메시지 목록을 그대로 돌려주고, 호출 payload를 기록합니다."""

    def __init__(self, messages):
        self._messages = messages
        self.invoked_with = None

    def invoke(self, payload):
        self.invoked_with = payload
        return {"messages": self._messages}


class _FakeAgentFactory:
    """w6.create_agent를 대신해 호출 kwargs를 기록하고 큐에 쌓인 메시지 목록을 순서대로 내줍니다."""

    def __init__(self):
        self.calls: list[dict] = []
        self.next_messages: list[list] = []
        self.agents: list[_FakeAgent] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        messages = self.next_messages.pop(0) if self.next_messages else []
        agent = _FakeAgent(messages)
        self.agents.append(agent)
        return agent


@pytest.fixture
def fake_agents(monkeypatch):
    factory = _FakeAgentFactory()
    monkeypatch.setattr(w6, "chat_model", lambda: "fake-model")
    monkeypatch.setattr(w6, "create_agent", lambda **kwargs: factory.create(**kwargs))
    return factory


def _ai_call(name, args, call_id="c1"):
    return SimpleNamespace(type="ai", content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def _tool_result(name, payload, call_id="c1"):
    return SimpleNamespace(
        type="tool", name=name, tool_call_id=call_id, tool_calls=[], content=json.dumps(payload, ensure_ascii=False)
    )


def _ai_final(text):
    return SimpleNamespace(type="ai", content=text, tool_calls=[])


# --- tool 구성 ---


def test_supervisor_tools_expose_only_two_delegates():
    assert [tool_name(t) for t in supervisor_tools()] == ["nana_agent", "kana_agent"]


def test_kana_tools_includes_extra_task_tools():
    names = [tool_name(t) for t in kana_tools()]
    assert "find_common_available_slots" in names
    assert "decide_final_slot" in names


def test_kana_tools_excludes_personal_app_tools():
    kana_names = {tool_name(t) for t in kana_tools()}
    personal_only_names = {"personal_create_schedule", "personal_list_schedules", "personal_delete_schedule"}
    assert kana_names.isdisjoint(personal_only_names)


def test_kana_tools_contains_expected_eight_names():
    assert [tool_name(t) for t in kana_tools()] == [
        "extract_schedule_request",
        "search_previous_conversations",
        "load_conversation_messages",
        "extract_schedules_from_history",
        "list_shared_schedules",
        "collect_member_schedules",
        "find_common_available_slots",
        "decide_final_slot",
    ]


def test_agent_tool_names_maps_each_agent():
    assert agent_tool_names("nana_agent") == [tool_name(t) for t in week04_tools()]
    assert agent_tool_names("kana_agent") == [tool_name(t) for t in kana_tools()]
    assert agent_tool_names("supervisor") == ["nana_agent", "kana_agent"]
    assert agent_tool_names("unknown") == []


def test_build_langchain_supervisor_agent_is_singleton(fake_agents):
    fake_agents.next_messages = [[], []]
    agent1 = build_langchain_supervisor_agent()
    agent2 = build_langchain_supervisor_agent()
    assert agent1 is agent2
    assert len(fake_agents.calls) == 1


def test_build_langchain_supervisor_agent_uses_supervisor_tools_and_prompt(fake_agents):
    fake_agents.next_messages = [[]]
    build_langchain_supervisor_agent()
    kwargs = fake_agents.calls[0]
    assert [tool_name(t) for t in kwargs["tools"]] == ["nana_agent", "kana_agent"]
    assert kwargs["system_prompt"] == supervisor_system_prompt()


# --- prompt 누적 구조 ---


def test_week06_prompt_parts_includes_week05_parts():
    week05_parts = week05_prompt_parts()
    parts = week06_prompt_parts()
    assert parts[: len(week05_parts)] == week05_parts


def test_week06_prompt_parts_overrides_week05_trailing_rule():
    week05_parts = week05_prompt_parts()
    parts = week06_prompt_parts()
    assert len(parts) == len(week05_parts) + 1
    assert "nana_agent" in parts[-1]
    assert "kana_agent" in parts[-1]


def test_week06_prompt_parts_mentions_both_delegate_tools():
    text = "\n".join(week06_prompt_parts())
    assert "nana_agent" in text
    assert "kana_agent" in text


def test_week06_prompt_parts_states_supervisor_does_not_call_business_tools():
    text = "\n".join(week06_prompt_parts())
    assert "직접" in text
    assert "supervisor" in text


def test_nana_prompt_parts_includes_week04_parts():
    week04_parts = week04_prompt_parts()
    parts = nana_prompt_parts()
    assert parts[: len(week04_parts)] == week04_parts


def test_nana_prompt_parts_declares_group_work_out_of_scope():
    text = "\n".join(nana_prompt_parts())
    assert "Kana" in text


def test_kana_prompt_parts_does_not_accumulate_other_weeks():
    kana_text = kana_system_prompt()
    for part in week04_prompt_parts():
        assert part not in kana_text
    for part in week05_prompt_parts():
        assert part not in kana_text


def test_kana_prompt_parts_is_self_contained_about_today():
    from fixed.runtime_clock import current_app_date_iso

    text = "\n".join(kana_prompt_parts())
    assert current_app_date_iso() in text
    assert "YYYY-MM-DD" in text


def test_kana_prompt_parts_states_default_date_range():
    text = "\n".join(kana_prompt_parts())
    assert "14일" in text


def test_kana_prompt_parts_excludes_me_from_member_names():
    text = "\n".join(kana_prompt_parts())
    assert "member_names" in text
    assert "'나'" in text or '"나"' in text


def test_kana_prompt_parts_mentions_only_available_tools():
    text = "\n".join(kana_prompt_parts())
    for name in [tool_name(t) for t in kana_tools()]:
        assert name in text


def test_kana_prompt_parts_declares_personal_save_out_of_scope():
    text = "\n".join(kana_prompt_parts())
    assert "Nana" in text


def test_supervisor_system_prompt_contains_week06_parts():
    for part in week06_prompt_parts():
        assert part in supervisor_system_prompt()


def test_supervisor_system_prompt_states_single_delegation_rule():
    text = supervisor_system_prompt()
    assert "하나" in text
    assert "nana_agent" in text
    assert "kana_agent" in text


# --- nana_agent ---


def test_nana_agent_builds_subagent_once_and_reuses(fake_agents):
    fake_agents.next_messages = [[_ai_final("hi")]]
    nana_agent.invoke({"query": "내 일정 보여줘"})
    nana_agent.invoke({"query": "내 일정 보여줘"})
    assert len(fake_agents.calls) == 1
    assert w6._NANA_SUBAGENT is not None


def test_nana_agent_uses_week04_tools_and_nana_system_prompt(fake_agents):
    fake_agents.next_messages = [[_ai_final("ok")]]
    nana_agent.invoke({"query": "x"})
    kwargs = fake_agents.calls[0]
    assert [tool_name(t) for t in kwargs["tools"]] == [tool_name(t) for t in week04_tools()]
    assert kwargs["system_prompt"] == nana_system_prompt()


def test_nana_agent_passes_query_as_single_user_message(fake_agents):
    fake_agents.next_messages = [[_ai_final("ok")]]
    nana_agent.invoke({"query": "안녕"})
    agent = fake_agents.agents[0]
    assert agent.invoked_with == {"messages": [{"role": "user", "content": "안녕"}]}


def test_nana_agent_returns_answer_trace_and_inner_tool_names(fake_agents):
    fake_agents.next_messages = [
        [
            _ai_call("personal_list_saved_schedules", {}),
            _tool_result("personal_list_saved_schedules", {"ok": True, "rows": []}),
            _ai_final("저장된 일정이 없습니다"),
        ]
    ]
    result = json.loads(nana_agent.invoke({"query": "내 저장된 일정 보여줘"}))
    assert result["inner_tool_names"] == ["personal_list_saved_schedules"]
    assert result["answer"] == "저장된 일정이 없습니다"
    assert result["selected_agent"] == "nana_agent"


def test_nana_agent_payload_has_no_top_level_final_slot_key(fake_agents):
    fake_agents.next_messages = [[_ai_final("ok")]]
    result = json.loads(nana_agent.invoke({"query": "x"}))
    assert "final_slot" not in result


# --- kana_agent ---


def test_kana_agent_builds_subagent_once_and_reuses(fake_agents):
    fake_agents.next_messages = [[_ai_final("hi")]]
    kana_agent.invoke({"query": "철수랑 회의 잡아줘"})
    kana_agent.invoke({"query": "철수랑 회의 잡아줘"})
    assert len(fake_agents.calls) == 1
    assert w6._KANA_SUBAGENT is not None


def test_kana_agent_uses_kana_tools_and_kana_system_prompt(fake_agents):
    fake_agents.next_messages = [[_ai_final("ok")]]
    kana_agent.invoke({"query": "x"})
    kwargs = fake_agents.calls[0]
    assert [tool_name(t) for t in kwargs["tools"]] == [tool_name(t) for t in kana_tools()]
    assert kwargs["system_prompt"] == kana_system_prompt()


def test_kana_agent_returns_answer_trace_and_inner_tool_names(fake_agents):
    fake_agents.next_messages = [
        [
            _ai_call("collect_member_schedules", {"member_names": ["철수"], "date_from": "2026-08-06", "date_to": "2026-08-13"}),
            _tool_result("collect_member_schedules", {"ok": True, "tool_name": "collect_member_schedules", "rows": []}),
            _ai_final("겹치는 바쁜 시간이 없습니다"),
        ]
    ]
    result = json.loads(kana_agent.invoke({"query": "철수랑 회의 잡아줘"}))
    assert result["inner_tool_names"] == ["collect_member_schedules"]
    assert result["answer"] == "겹치는 바쁜 시간이 없습니다"


def test_kana_agent_final_slot_payload_is_none_without_decide_final_slot(fake_agents):
    fake_agents.next_messages = [
        [
            _ai_call("collect_member_schedules", {}),
            _tool_result("collect_member_schedules", {"ok": True, "rows": []}),
            _ai_final("바쁜 시간이 없습니다"),
        ]
    ]
    result = json.loads(kana_agent.invoke({"query": "철수랑 회의 잡아줘"}))
    assert result["final_slot_payload"] is None
    assert result["final_decision_payload"] is None


def test_kana_agent_lifts_final_slot_payload_by_tool_name(fake_agents):
    decide_payload = {"final_slot": "2026-08-06 10:00-11:00", "reason": "근거", "candidates": []}
    fake_agents.next_messages = [
        [
            _ai_call("decide_final_slot", {}),
            _tool_result("decide_final_slot", decide_payload),
            _ai_final("확정했습니다"),
        ]
    ]
    result = json.loads(kana_agent.invoke({"query": "최종 확정해줘"}))
    assert result["final_slot_payload"] == decide_payload


def test_kana_agent_takes_last_decide_final_slot_result(fake_agents):
    first_payload = {"final_slot": "2026-08-06 10:00-11:00", "reason": "1차", "candidates": []}
    second_payload = {"final_slot": "2026-08-07 14:00-15:00", "reason": "2차", "candidates": []}
    fake_agents.next_messages = [
        [
            _ai_call("decide_final_slot", {}, call_id="d1"),
            _tool_result("decide_final_slot", first_payload, call_id="d1"),
            _ai_call("decide_final_slot", {}, call_id="d2"),
            _tool_result("decide_final_slot", second_payload, call_id="d2"),
            _ai_final("최종 확정"),
        ]
    ]
    result = json.loads(kana_agent.invoke({"query": "다시 골라줘"}))
    assert result["final_slot_payload"] == second_payload


def test_kana_agent_lifts_final_decision_payload(fake_agents):
    final_decision = {"title": "팀 회의", "status": "confirmed"}
    fake_agents.next_messages = [
        [
            _ai_call("propose_group_schedule", {}),
            _tool_result(
                "propose_group_schedule",
                {"ok": True, "tool_name": "propose_group_schedule", "final_decision": final_decision},
            ),
            _ai_final("확정했습니다"),
        ]
    ]
    result = json.loads(kana_agent.invoke({"query": "제안해줘"}))
    assert result["final_decision_payload"] == final_decision


def test_kana_agent_payload_has_no_top_level_final_slot_key(fake_agents):
    decide_payload = {"final_slot": "2026-08-06 10:00-11:00", "reason": "근거", "candidates": []}
    fake_agents.next_messages = [
        [
            _ai_call("decide_final_slot", {}),
            _tool_result("decide_final_slot", decide_payload),
            _ai_final("확정했습니다"),
        ]
    ]
    result = json.loads(kana_agent.invoke({"query": "최종 확정해줘"}))
    assert "final_slot" not in result
    assert result["final_slot_payload"]["final_slot"] == "2026-08-06 10:00-11:00"


# --- extract_langchain_trace 연결 (supervisor 레벨) ---


def test_extract_langchain_trace_lifts_kana_payload_end_to_end(fake_agents):
    decide_payload = {"final_slot": "2026-08-06 10:00-11:00", "reason": "근거", "candidates": []}
    fake_agents.next_messages = [
        [
            _ai_call("decide_final_slot", {}),
            _tool_result("decide_final_slot", decide_payload),
            _ai_final("확정했습니다"),
        ]
    ]
    kana_json = kana_agent.invoke({"query": "최종 확정해줘"})

    supervisor_messages = [
        _ai_call("kana_agent", {"query": "최종 확정해줘"}, call_id="s1"),
        SimpleNamespace(type="tool", name="kana_agent", tool_call_id="s1", tool_calls=[], content=kana_json),
        _ai_final("확정된 시간은 2026-08-06 10:00-11:00 입니다"),
    ]
    trace = extract_langchain_trace({"messages": supervisor_messages})
    assert trace["supervisor_selected_agent"] == "kana_agent"
    assert trace["inner_tool_names"] == ["decide_final_slot"]
    assert trace["final_slot_payload"]["final_slot"] == "2026-08-06 10:00-11:00"


def test_extract_langchain_trace_with_nana_payload_leaves_final_slot_payload_none(fake_agents):
    fake_agents.next_messages = [[_ai_final("일정이 없습니다")]]
    nana_json = nana_agent.invoke({"query": "내 일정 보여줘"})

    supervisor_messages = [
        _ai_call("nana_agent", {"query": "내 일정 보여줘"}, call_id="s1"),
        SimpleNamespace(type="tool", name="nana_agent", tool_call_id="s1", tool_calls=[], content=nana_json),
        _ai_final("일정이 없습니다"),
    ]
    trace = extract_langchain_trace({"messages": supervisor_messages})
    assert trace["supervisor_selected_agent"] == "nana_agent"
    assert trace["final_slot_payload"] is None


# --- 심화과제: find_common_available_slots / decide_final_slot ---


def test_find_common_available_slots_dict_keeps_non_overlapping_candidate_and_drops_overlapping_one():
    busy_rows = [{"member_name": "철수", "date": "2026-08-11", "start_time": "14:00", "end_time": "15:00"}]
    candidate_slots = [
        {"date": "2026-08-11", "start_time": "14:00", "end_time": "15:00", "duration_minutes": 60, "reason": "겹침"},
        {"date": "2026-08-11", "start_time": "16:00", "end_time": "17:00", "duration_minutes": 60, "reason": "안 겹침"},
    ]
    result = find_common_available_slots_dict(
        member_names=["철수"],
        date_from="2026-08-10",
        date_to="2026-08-14",
        busy_rows=busy_rows,
        candidate_slots=candidate_slots,
    )
    assert [slot["start_time"] for slot in result["candidate_slots"]] == ["16:00"]
    assert result["busy_rows"] == busy_rows


def test_find_common_available_slots_dict_normalizes_iso_datetime_bounds():
    result = find_common_available_slots_dict(
        member_names=["철수"],
        date_from="2026-08-10T00:00:00",
        date_to="2026-08-14T23:59:59",
        busy_rows=[],
        candidate_slots=[],
    )
    assert result["members"] == ["철수"]
    assert result["candidate_slots"] == []


def test_find_common_available_slots_dict_collects_busy_rows_when_missing(monkeypatch):
    collected_rows = [{"member_name": "나", "date": "2026-08-11", "start_time": "09:00", "end_time": "10:00"}]

    class _FakeCollectTool:
        def __init__(self):
            self.invoked_with = None

        def invoke(self, args):
            self.invoked_with = args
            return json.dumps({"rows": collected_rows}, ensure_ascii=False)

    fake_tool = _FakeCollectTool()
    monkeypatch.setattr(w6, "collect_member_schedules", fake_tool)

    result = w6.find_common_available_slots_dict(
        member_names=["철수"],
        date_from="2026-08-10",
        date_to="2026-08-14",
        candidate_slots=[],
    )
    assert fake_tool.invoked_with["member_names"] == ["나", "철수"]
    assert result["busy_rows"] == collected_rows


def test_find_common_available_slots_tool_matches_dict_version():
    busy_rows = [{"member_name": "철수", "date": "2026-08-11", "start_time": "14:00", "end_time": "15:00"}]
    candidate_slots = [{"date": "2026-08-11", "start_time": "16:00", "end_time": "17:00", "duration_minutes": 60, "reason": "가능"}]
    dict_result = find_common_available_slots_dict(
        member_names=["철수"], date_from="2026-08-10", date_to="2026-08-14", busy_rows=busy_rows, candidate_slots=candidate_slots
    )
    tool_result = json.loads(
        find_common_available_slots.invoke(
            {
                "member_names": ["철수"],
                "date_from": "2026-08-10",
                "date_to": "2026-08-14",
                "busy_rows": busy_rows,
                "candidate_slots": candidate_slots,
            }
        )
    )
    assert tool_result["candidate_slots"] == dict_result["candidate_slots"]


def test_decide_final_slot_tool_needs_agent_selection_without_a_choice():
    result = json.loads(
        decide_final_slot.invoke(
            {
                "candidate_slots": [{"date": "2026-08-11", "start_time": "16:00", "end_time": "17:00"}],
            }
        )
    )
    assert result["needs_agent_selection"] is True
    assert result["final_slot"] is None


def test_decide_final_slot_tool_confirms_when_index_selected():
    result = json.loads(
        decide_final_slot.invoke(
            {
                "candidate_slots": [
                    {"date": "2026-08-11", "start_time": "14:00", "end_time": "15:00"},
                    {"date": "2026-08-11", "start_time": "16:00", "end_time": "17:00"},
                ],
                "selected_index": 1,
                "reason": "철수·영희 모두 가능",
            }
        )
    )
    assert result["needs_agent_selection"] is False
    assert result["final_slot"] == "2026-08-11 16:00-17:00"
    assert result["reason"] == "철수·영희 모두 가능"
