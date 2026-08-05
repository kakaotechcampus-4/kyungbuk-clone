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