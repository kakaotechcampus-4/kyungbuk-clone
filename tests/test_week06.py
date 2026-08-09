import json
from types import SimpleNamespace

import student_parts.week06_kanamate_decides_schedule as week06
from student_parts.week06_kanamate_decides_schedule import (
    _tool_call_names,
    agent_tool_names,
    kana_tools,
    supervisor_tools,
)


def test_tool_call_names_extracts_only_tool_call_events():
    events = [
        {"event": "tool_call", "tool_name": "kana_agent"},
        {"event": "tool_result", "tool_name": "kana_agent"},
        {"event": "tool_call", "tool_name": "nana_agent"},
    ]
    assert _tool_call_names(events) == ["kana_agent", "nana_agent"]


def test_tool_call_names_ignores_events_without_tool_name():
    assert _tool_call_names([{"event": "tool_call"}]) == []


def test_kana_tools_excludes_unimplemented_additional_tools():
    names = [tool.name for tool in kana_tools()]
    assert "find_common_available_slots" not in names
    assert "decide_final_slot" not in names


def test_supervisor_tools_only_exposes_two_delegates():
    names = [tool.name for tool in supervisor_tools()]
    assert names == ["nana_agent", "kana_agent"]


def test_agent_tool_names_supervisor_only_has_two_delegates():
    assert agent_tool_names("supervisor") == ["nana_agent", "kana_agent"]


class _FakeSubAgent:
    """create_agent(...) 대신 넣어서 실제 LLM 호출 없이 정해진 결과를 돌려주는 가짜 agent."""

    def __init__(self, result):
        self._result = result

    def invoke(self, _input):
        return self._result


def test_nana_agent_wraps_subagent_result_without_calling_real_llm(monkeypatch):
    fake_result = {
        "messages": [
            SimpleNamespace(
                type="ai",
                content="",
                tool_calls=[{"name": "personal_list_saved_schedules", "args": {}, "id": "call_1"}],
            ),
            SimpleNamespace(
                type="tool",
                name="personal_list_saved_schedules",
                content='{"ok": true, "schedules": []}',
                tool_call_id="call_1",
            ),
            SimpleNamespace(type="ai", content="오늘 등록된 일정이 없습니다.", tool_calls=[]),
        ]
    }
    monkeypatch.setattr(week06, "_NANA_SUBAGENT", _FakeSubAgent(fake_result))

    output = json.loads(week06.nana_agent.invoke({"query": "내 일정 보여줘"}))

    assert output["ok"] is True
    assert output["selected_agent"] == "nana_agent"
    assert output["answer"] == "오늘 등록된 일정이 없습니다."
    assert output["inner_tool_names"] == ["personal_list_saved_schedules"]