from __future__ import annotations

"""Week 5 수정(공지_코드업데이트.md A~E)과 Week 6 조율 계약을 고정하는 회귀 테스트입니다.

Week 4 리뷰에서 "스모크로 눈으로 봤다"를 "테스트가 대신 봐준다"로 옮기라는 과제를 받았습니다.
이번 주차 수정은 눈으로 확인하기 특히 어려운 종류입니다.
  - 그룹 일정이 busy row에 들어오는지는 SQLite에 그룹 일정을 만들어 둔 상태에서만 드러납니다.
  - 중복 제거는 같은 일정이 두 경로로 들어오는 특정 호출에서만 드러납니다.
  - 시간 미정 일정이 후보를 전멸시키는 문제는 그런 일정이 있는 날짜를 조율할 때만 드러납니다.
그래서 LLM이나 DB 없이 helper 계약만 고정해, 나중에 이 로직을 건드리면 회귀가 자동으로 잡히게 했습니다.

pytest가 이 repo 의존성에 없어서(pyproject.toml에 없음) 표준 assert만 사용합니다.
  - pytest가 있으면: uv run pytest tests
  - 없으면:        uv run python tests/test_week06_group_schedule.py
pytest를 dev 의존성으로 추가하면 uv.lock이 바뀌어 매주 강의자료 merge와 충돌할 수 있으므로
의존성을 늘리지 않고 두 방법 모두로 돌아가게 두었습니다.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from student_parts import week05_load_kanas_past_conversations as w5
from student_parts import week06_kanamate_decides_schedule as w6


# --- Week 5 (B) 일정 종류를 row에서 읽는다 ------------------------------------------------


def test_schedule_row_reads_group_kind_from_request_kind() -> None:
    """request_kind가 group_schedule이면 그룹 일정으로 읽어야 합니다."""

    request = w5._structured_request_from_schedule_row(
        {"title": "하린과 사전 미팅", "date": "2026-07-14", "start_time": "15:00", "request_kind": "group_schedule"}
    )
    assert request.kind == "group_schedule"


def test_schedule_row_without_request_kind_is_personal() -> None:
    """Week 1 임시 일정 row에는 request_kind가 없으므로 개인 일정으로 봅니다."""

    request = w5._structured_request_from_schedule_row({"title": "치과", "date": "2026-07-14"})
    assert request.kind == "personal_schedule"


# --- Week 5 (C) 종류에 따라 notes를 구분한다 ----------------------------------------------


def test_my_schedule_notes_distinguishes_personal_and_group() -> None:
    """개인/그룹, 그리고 참석자 유무까지 notes에서 구분돼야 합니다."""

    personal = w5.StructuredRequest(kind="personal_schedule", title="치과", original_text="치과")
    group = w5.StructuredRequest(
        kind="group_schedule", title="사전 미팅", members=["하린", "민준"], original_text="사전 미팅"
    )
    group_without_members = w5.StructuredRequest(kind="group_schedule", title="회의", original_text="회의")

    assert w5._my_schedule_notes(personal) == "Nana 개인 일정"
    assert w5._my_schedule_notes(group) == "Nana 그룹 일정 · 참석자: 하린, 민준"
    assert w5._my_schedule_notes(group_without_members) == "Nana 그룹 일정"


# --- Week 5 (D) 두 경로로 들어온 같은 일정을 한 번만 남긴다 --------------------------------


def test_dedupe_keeps_app_db_row_despite_different_shaping() -> None:
    """제목 소괄호와 end_time이 경로마다 다르게 다듬어져도 같은 일정으로 봐야 합니다."""

    app_db_row = {
        "member_name": "나",
        "title": "팀 회의 (온라인)",
        "date": "2026-07-14",
        "start_time": "15:00",
        "end_time": "18:00",
        "notes": "Nana 개인 일정",
    }
    shared_row = {
        "member_name": "나",
        "title": "팀 회의",
        "date": "2026-07-14",
        "start_time": "15:00",
        "end_time": "미정",
        "notes": "앱 개인 일정 자동 동기화",
    }

    rows = w5._dedupe_schedule_rows([app_db_row, shared_row])

    assert len(rows) == 1
    # 앞에 온 앱 DB row가 남아야 notes가 "Nana 개인 일정"으로 유지됩니다.
    assert rows[0]["notes"] == "Nana 개인 일정"
    assert rows[0]["title"] == "팀 회의 (온라인)"


def test_dedupe_keeps_distinct_schedules() -> None:
    """사람·날짜·시작시간·제목이 다르면 별개 일정으로 남겨야 합니다."""

    rows = w5._dedupe_schedule_rows(
        [
            {"member_name": "나", "title": "회의", "date": "2026-07-14", "start_time": "10:00"},
            {"member_name": "민준", "title": "회의", "date": "2026-07-14", "start_time": "10:00"},
            {"member_name": "나", "title": "회의", "date": "2026-07-14", "start_time": "11:00"},
            {"member_name": "나", "title": "다른 회의", "date": "2026-07-14", "start_time": "10:00"},
        ]
    )
    assert len(rows) == 4


def test_dedupe_treats_blank_start_time_as_미정() -> None:
    """start_time이 비어 있으면 공유 저장소의 "미정"과 같은 값으로 맞춰야 합니다."""

    rows = w5._dedupe_schedule_rows(
        [
            {"member_name": "나", "title": "출장", "date": "2026-07-14", "start_time": ""},
            {"member_name": "나", "title": "출장", "date": "2026-07-14", "start_time": "미정"},
        ]
    )
    assert len(rows) == 1


# --- Week 5 버그① 그룹 일정이 내 busy row에 들어온다 --------------------------------------


def test_group_schedule_becomes_my_busy_row() -> None:
    """하린과 잡은 그룹 회의가 민준과 조율할 때도 내 바쁜 시간으로 잡혀야 합니다.

    수정 전에는 kind 필터 때문에 이 row가 아예 없어서 그 시간이 "빈 시간"으로 추천됐습니다.
    member_names에 "나"만 넘겨 외부 MCP 조회는 skipped로 두고 앱 DB 경로만 검증합니다.
    """

    result = w5._collect_member_schedules(
        member_names=["나"],
        date_from="2026-07-14",
        date_to="2026-07-14",
        personal_schedules=[
            {
                "schedule_id": "s1",
                "title": "하린과 사전 미팅",
                "date": "2026-07-14",
                "start_time": "15:00",
                "end_time": "16:00",
                "attendees": ["하린"],
                "request_kind": "group_schedule",
            }
        ],
    )

    assert result["external_status"] == "skipped"
    assert len(result["rows"]) == 1
    row = result["rows"][0]
    assert row["member_name"] == "나"
    assert row["title"] == "하린과 사전 미팅"
    assert row["notes"] == "Nana 그룹 일정 · 참석자: 하린"
    assert row["time_status"] == "complete"


# --- Week 5 버그② members에 "나"가 두 번 들어가지 않는다 ---------------------------------


def test_member_names_never_repeat_me() -> None:
    """member_names에 "나"가 여러 번 들어와도 결과에는 한 번만 남아야 합니다."""

    result = w5._collect_member_schedules(
        member_names=["나", "나"],
        date_from="2026-07-14",
        date_to="2026-07-14",
        personal_schedules=[],
    )
    assert result["member_names"] == ["나"]


# --- Week 6 time_status를 시간 계산에 실제로 적용한다 -------------------------------------


def test_overlap_rows_split_by_time_status() -> None:
    """complete는 그대로, start_only는 회의 길이만큼 보정, date_only는 검증에서 빼야 합니다."""

    complete = {"date": "2026-07-14", "start_time": "10:00", "end_time": "11:00", "time_status": "complete"}
    start_only = {"date": "2026-07-15", "start_time": "14:00", "end_time": "미정", "time_status": "start_only"}
    date_only = {"date": "2026-07-16", "start_time": "미정", "end_time": "미정", "time_status": "date_only"}

    hard_rows, soft_rows = w6._busy_rows_for_overlap_check([complete, start_only, date_only], 60)

    assert hard_rows[0] == complete
    assert hard_rows[1]["end_time"] == "15:00"
    assert hard_rows[1]["overlap_end_assumed"] is True
    # 원본 row는 그대로 남아야 payload에서 실제 저장값을 확인할 수 있습니다.
    assert start_only["end_time"] == "미정"
    assert soft_rows == [date_only]


def test_overlap_rows_infer_status_when_missing() -> None:
    """time_status가 없는 외부 row도 값으로 종류를 판정해야 합니다."""

    hard_rows, soft_rows = w6._busy_rows_for_overlap_check(
        [
            {"date": "2026-07-14", "start_time": "10:00", "end_time": "11:00"},
            {"date": "2026-07-15", "start_time": "미정", "end_time": "미정"},
        ],
        60,
    )
    assert len(hard_rows) == 1
    assert len(soft_rows) == 1


def test_start_only_row_does_not_block_rest_of_day() -> None:
    """start_only row가 시작 시각부터 자정까지를 막아 오후를 통째로 지우면 안 됩니다."""

    hard_rows, _ = w6._busy_rows_for_overlap_check(
        [{"date": "2026-07-14", "start_time": "14:00", "end_time": "미정", "time_status": "start_only"}], 60
    )
    from fixed.schedule_decision import busy_rows_overlap

    # 15:00 이후는 막히지 않아야 합니다.

    assert busy_rows_overlap(hard_rows, "2026-07-14", 15 * 60, 16 * 60) == []
    # 14:00~15:00은 막혀야 합니다.
    assert busy_rows_overlap(hard_rows, "2026-07-14", 14 * 60 + 30, 15 * 60 + 30) != []


# --- Week 6 후보 검증 계약 ---------------------------------------------------------------


BUSY_ROWS = [
    {
        "member_name": "나",
        "title": "팀 회의",
        "date": "2026-07-14",
        "start_time": "10:00",
        "end_time": "11:00",
        "time_status": "complete",
    },
    {
        "member_name": "나",
        "title": "출장",
        "date": "2026-07-15",
        "start_time": "미정",
        "end_time": "미정",
        "time_status": "date_only",
    },
    {
        "member_name": "민준",
        "title": "외부 미팅",
        "date": "2026-07-16",
        "start_time": "14:00",
        "end_time": "미정",
        "time_status": "start_only",
    },
]


def _find_slots(candidate_slots: list[dict[str, object]]) -> dict[str, object]:
    return w6.find_common_available_slots_dict(
        member_names=["민준"],
        date_from="2026-07-14",
        date_to="2026-07-16",
        duration_minutes=60,
        limit=10,
        busy_rows=BUSY_ROWS,
        candidate_slots=candidate_slots,
    )


def test_candidate_overlapping_complete_row_is_rejected() -> None:
    """complete busy row와 겹치는 후보는 통과하면 안 됩니다."""

    payload = _find_slots([{"date": "2026-07-14", "start_time": "10:30", "end_time": "11:30", "duration_minutes": 60}])
    assert payload["candidate_slots"] == []


def test_candidate_overlapping_start_only_row_is_rejected() -> None:
    """start_only row의 보정된 구간과 겹치는 후보도 통과하면 안 됩니다."""

    payload = _find_slots([{"date": "2026-07-16", "start_time": "14:30", "end_time": "15:30", "duration_minutes": 60}])
    assert payload["candidate_slots"] == []


def test_candidate_after_start_only_row_is_accepted() -> None:
    """보정 구간이 끝난 뒤 시간은 후보로 통과해야 합니다."""

    payload = _find_slots([{"date": "2026-07-16", "start_time": "15:00", "end_time": "16:00", "duration_minutes": 60}])
    assert len(payload["candidate_slots"]) == 1


def test_date_only_row_does_not_erase_the_whole_day() -> None:
    """시간 미정 일정이 있는 날도 후보를 제안하고, 확인이 필요하다고 알려야 합니다.

    date_only row를 겹침 검증에 그대로 넘기면 그날 전체가 막혀 후보가 전멸합니다.
    """

    payload = _find_slots([{"date": "2026-07-15", "start_time": "09:00", "end_time": "10:00", "duration_minutes": 60}])

    assert len(payload["candidate_slots"]) == 1
    assert len(payload["time_unknown_rows"]) == 1
    assert any("미정" in note for note in payload["notes"])


def test_payload_keeps_validated_and_original_rows() -> None:
    """검증에 쓴 row와 원본 row를 함께 남겨 근거를 확인할 수 있어야 합니다."""

    payload = _find_slots([])
    assert payload["busy_rows_all"] == BUSY_ROWS
    # 검증에 쓴 busy_rows는 date_only row가 빠진 목록입니다.
    assert len(payload["busy_rows"]) == 2
    assert payload["collection_status"] == "provided"


def test_members_include_me_as_evidence() -> None:
    """내 일정도 겹침 근거이므로 members에 "나"가 포함돼야 합니다."""

    payload = _find_slots([])
    assert payload["members"][0] == "나"
    assert "민준" in payload["members"]


def test_members_dedupe_alias_and_real_name() -> None:
    """alias와 실제 이름이 함께 들어와도 members에 같은 사람이 두 번 남으면 안 됩니다.

    Week 5 리뷰에서 지적받은 규칙이라 Week 6 payload에서도 같은 계약을 지켜야 합니다.
    """

    payload = w6.find_common_available_slots_dict(
        member_names=["나", "민준", "민준"],
        date_from="2026-07-16",
        date_to="2026-07-16",
        busy_rows=[],
        candidate_slots=[],
    )
    assert payload["members"] == ["나", "민준"]


def test_empty_candidates_reports_why() -> None:
    """후보가 비면 왜 비었는지 notes로 알려 agent가 다시 고를 수 있어야 합니다."""

    payload = _find_slots([])
    assert payload["candidate_slots"] == []
    assert any("후보" in note for note in payload["notes"])


def test_collection_failure_is_reported_not_treated_as_free() -> None:
    """일정 조회가 실패하면 빈 시간으로 단정하지 말고 실패를 알려야 합니다."""

    original = w6.collect_member_schedules

    class _FailingTool:
        def invoke(self, _payload: dict[str, object]) -> str:
            raise RuntimeError("MCP 연결 실패")

    w6.collect_member_schedules = _FailingTool()
    try:
        payload = w6.find_common_available_slots_dict(
            member_names=["민준"],
            date_from="2026-07-14",
            date_to="2026-07-16",
            busy_rows=None,
            candidate_slots=[],
        )
    finally:
        w6.collect_member_schedules = original

    assert payload["collection_status"] == "failed"
    assert "MCP 연결 실패" in str(payload["collection_error"])
    assert any("조회에 실패" in note for note in payload["notes"])


def test_partial_collection_is_reported() -> None:
    """외부 조회만 실패한 부분 성공도 그대로 올려 보내야 합니다."""

    original = w6.collect_member_schedules

    class _PartialTool:
        def invoke(self, _payload: dict[str, object]) -> str:
            import json

            return json.dumps(
                {
                    "ok": True,
                    "rows": [BUSY_ROWS[0]],
                    "external_status": "failed",
                    "external_error": "외부 DB 없음",
                },
                ensure_ascii=False,
            )

    w6.collect_member_schedules = _PartialTool()
    try:
        payload = w6.find_common_available_slots_dict(
            member_names=["민준"],
            date_from="2026-07-14",
            date_to="2026-07-16",
            busy_rows=None,
            candidate_slots=[],
        )
    finally:
        w6.collect_member_schedules = original

    assert payload["collection_status"] == "partial"
    assert any("외부 멤버 일정을 가져오지 못했" in note for note in payload["notes"])


# --- Week 6 최종 시간 결정 계약 ----------------------------------------------------------


def test_decide_final_slot_keeps_needs_selection_without_choice() -> None:
    """agent가 고르지 않았으면 코드가 임의로 확정하지 않아야 합니다."""

    import json

    payload = json.loads(
        w6.decide_final_slot.invoke(
            {
                "candidate_slots": [
                    {"date": "2026-07-14", "start_time": "14:00", "end_time": "15:00", "duration_minutes": 60}
                ]
            }
        )
    )
    assert payload["final_slot"] is None
    assert payload["needs_agent_selection"] is True
    assert payload["candidates"] == ["2026-07-14 14:00-15:00"]


def test_decide_final_slot_resolves_selected_index() -> None:
    """selected_index를 넘기면 그 후보가 최종 시간이 돼야 합니다."""

    import json

    payload = json.loads(
        w6.decide_final_slot.invoke(
            {
                "candidate_slots": [
                    {"date": "2026-07-14", "start_time": "14:00", "end_time": "15:00", "duration_minutes": 60},
                    {"date": "2026-07-16", "start_time": "15:00", "end_time": "16:00", "duration_minutes": 60},
                ],
                "selected_index": 1,
                "needs_agent_selection": False,
                "reason": "민준의 외부 미팅 뒤 시간이라 이동 여유가 있습니다.",
            }
        )
    )
    assert payload["final_slot"] == "2026-07-16 15:00-16:00"
    assert payload["needs_agent_selection"] is False
    assert payload["reason"].startswith("민준")


def test_decide_final_slot_reports_out_of_range_index() -> None:
    """후보 범위를 벗어난 index는 확정하지 않고 이유를 남겨야 합니다."""

    import json

    payload = json.loads(w6.decide_final_slot.invoke({"candidate_slots": [], "selected_index": 3}))
    assert payload["final_slot"] is None
    assert payload["needs_agent_selection"] is True
    assert "범위" in payload["reason"]


# --- Week 6 위임 입구 검증 ---------------------------------------------------------------


def test_blank_query_is_rejected_at_delegation_boundary() -> None:
    """공백만 있는 query로 하위 agent를 실행하지 않아야 합니다."""

    for blank in ("", "   ", "\n"):
        try:
            w6.AgentQueryInput(query=blank)
        except Exception:
            continue
        raise AssertionError(f"빈 query가 통과했습니다: {blank!r}")


def test_valid_query_is_stripped() -> None:
    """정상 query는 앞뒤 공백만 정리해 그대로 넘겨야 합니다."""

    assert w6.AgentQueryInput(query="  민준이랑 시간 맞춰줘 ").query == "민준이랑 시간 맞춰줘"


# --- Week 6 역할 분리 계약 ---------------------------------------------------------------


def test_supervisor_only_sees_delegation_tools() -> None:
    """supervisor는 하위 agent 위임 tool 두 개만 볼 수 있어야 합니다."""

    assert w6.agent_tool_names("supervisor") == ["nana_agent", "kana_agent"]


def test_kana_has_coordination_tools_and_no_personal_save() -> None:
    """Kana는 조율 tool을 갖고, 개인 일정 저장 tool은 갖지 않아야 합니다."""

    names = w6.agent_tool_names("kana_agent")
    assert "collect_member_schedules" in names
    assert "find_common_available_slots" in names
    assert "decide_final_slot" in names
    assert not [name for name in names if name.startswith("personal_")]


def test_nana_has_personal_tools_and_no_coordination() -> None:
    """Nana는 개인 일정 tool을 갖고, 조율/외부 멤버 tool은 갖지 않아야 합니다."""

    names = w6.agent_tool_names("nana_agent")
    assert [name for name in names if name.startswith("personal_")]
    assert "collect_member_schedules" not in names
    assert "decide_final_slot" not in names


def test_kana_prompt_does_not_inherit_other_weeks() -> None:
    """Kana prompt는 다른 주차 조각을 누적하지 않아야 합니다.

    누적하면 Kana가 갖고 있지 않은 개인 일정 tool을 부르라는 지시가 섞입니다.
    """

    kana_parts = w6.kana_prompt_parts()
    week05_parts = w5.week05_prompt_parts()
    assert not [part for part in kana_parts if part in week05_parts]


def test_trace_records_both_delegations_in_order() -> None:
    """한 요청이 두 담당에 걸치면 위임 두 건이 순서대로 남아야 합니다.

    supervisor_selected_agent는 마지막 위임만 담아서 kana_agent → nana_agent 흐름에서
    첫 위임이 trace에서 사라집니다. 실제 실행에서 관찰한 경로라 계약으로 고정합니다.
    """

    class _Msg:
        type = "ai"
        content = ""
        tool_calls = [
            {"name": "kana_agent", "args": {}, "id": "1"},
            {"name": "nana_agent", "args": {}, "id": "2"},
        ]

    trace = w6.extract_langchain_trace({"messages": [_Msg()]})
    assert trace["supervisor_selected_agents"] == ["kana_agent", "nana_agent"]
    # 기존 UI 호환 키는 그대로 마지막 위임을 가리킵니다.
    assert trace["supervisor_selected_agent"] == "nana_agent"


def test_lift_slot_payloads_takes_last_decision() -> None:
    """decide_final_slot이 여러 번 호출되면 마지막 판단을 올려야 합니다."""

    events = [
        {"event": "tool_result", "tool_name": "decide_final_slot", "content": {"final_slot": None}},
        {
            "event": "tool_result",
            "tool_name": "decide_final_slot",
            "content": {"final_slot": "2026-07-16 15:00-16:00"},
        },
    ]
    final_slot_payload, _ = w6._lift_slot_payloads(events)
    assert final_slot_payload["final_slot"] == "2026-07-16 15:00-16:00"


def test_lift_slot_payloads_ignores_non_dict_content() -> None:
    """tool 결과가 JSON이 아니어도 예외 없이 넘어가야 합니다."""

    events = [{"event": "tool_result", "tool_name": "decide_final_slot", "content": "문자열 응답"}]
    assert w6._lift_slot_payloads(events) == (None, None)


def _run_all() -> int:
    tests = sorted(
        (name, value)
        for name, value in globals().items()
        if name.startswith("test_") and callable(value)
    )
    failed = 0
    for name, test in tests:
        try:
            test()
        except Exception as exc:  # noqa: BLE001 - 러너가 모든 실패를 모아 보고합니다.
            failed += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"PASS {name}")
    print(f"\n{len(tests) - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
