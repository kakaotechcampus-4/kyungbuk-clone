from __future__ import annotations

from student_parts.week06_kanamate_decides_schedule import (
    _tool_call_names,
    extract_langchain_trace,
)


def test_tool_call_names_extracts_only_tool_call_events():
    events = [
        {"event": "tool_call", "tool_name": "nana_agent"},
        {"event": "tool_result", "tool_name": "nana_agent"},
        {"event": "tool_call", "tool_name": "kana_agent"},
    ]
    assert _tool_call_names(events) == ["nana_agent", "kana_agent"]


def test_extract_langchain_trace_picks_selected_agent_and_inner_tools():
    result = {
        "messages": [
            type(
                "Msg",
                (),
                {
                    "type": "ai",
                    "tool_calls": [{"name": "nana_agent", "args": {"query": "test"}, "id": "1"}],
                    "content": "",
                },
            )(),
            type(
                "Msg",
                (),
                {
                    "type": "tool",
                    "name": "nana_agent",
                    "tool_call_id": "1",
                    "content": '{"inner_tool_names": ["personal_list_saved_schedules"]}',
                },
            )(),
        ]
    }
    trace = extract_langchain_trace(result)
    assert trace["supervisor_selected_agent"] == "nana_agent"
    assert trace["inner_tool_names"] == ["personal_list_saved_schedules"]
    assert trace["final_slot_payload"] is None
    assert trace["final_decision_payload"] is None
