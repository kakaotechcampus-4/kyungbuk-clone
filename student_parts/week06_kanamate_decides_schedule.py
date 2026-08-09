from __future__ import annotations

import json
from typing import Any

from langchain.agents import create_agent
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from fixed.external_people_store import normalize_external_member_names
from fixed.langchain_trace import extract_agent_events, extract_final_text
from fixed.llm import chat_model
from fixed.runtime_clock import current_app_date_iso
from fixed.schedule_decision import (
    CommonSlotCandidate,
    decide_final_slot_payload,
    find_common_available_slots_payload,
    normalize_date_bound,
)
from student_parts.week01_wake_up_nana import join_system_prompt
from student_parts.week02_structure_natural_language_requests import extract_schedule_request
from student_parts.week04_retrieve_nanas_memory import week04_prompt_parts, week04_tools
from student_parts.week05_load_kanas_past_conversations import (
    collect_member_schedules,
    extract_schedules_from_history,
    list_shared_schedules,
    load_conversation_messages,
    search_previous_conversations,
    week05_prompt_parts,
)


_NANA_SUBAGENT: Any | None = None
_KANA_SUBAGENT: Any | None = None
_SUPERVISOR_AGENT: Any | None = None


def week06_system_prompt() -> str:
    """6주차 supervisor agent가 따르는 시스템 프롬프트입니다."""

    return supervisor_system_prompt()


def week06_prompt_parts() -> list[str]:
    """1~6주차 supervisor system prompt 조각을 누적합니다."""

    return [
        *week05_prompt_parts(),
        (
            "[Week 6 supervisor 위임]\n"
            f"오늘 날짜는 {current_app_date_iso()}이며, 하위 agent에 넘기는 날짜도 YYYY-MM-DD로 계산한다.\n"
            "너는 카나메이트 supervisor다. 앞선 주차 안내에 나온 업무 tool은 모두 하위 agent가 갖고 있고, "
            "너에게 보이는 tool은 nana_agent와 kana_agent 두 개뿐이다. 업무를 직접 처리하지 말고 위임한다.\n"
            "Nana 담당: 내 개인 일정 생성·조회·수정·삭제, 일정/할 일/알림의 구조화 저장, "
            "개인 참고자료와 앱 대화 기록 검색, 확정된 시간을 내 일정으로 저장하는 일.\n"
            "Kana 담당: 외부 멤버의 이전 대화 검색, 멤버별 일정 추출, 공유 일정 row 조회, "
            "여러 사람의 busy-time 종합, 공통 가능 시간 후보 검증과 최종 회의 시간 결정.\n"
            "요청에 다른 사람 이름이 나오거나 '같이', '회의', '일정 맞춰줘'처럼 여러 사람을 조율하려는 의도가 있으면 kana_agent로 위임한다.\n"
            "내 일정과 내 기록만으로 끝나는 요청이면 nana_agent로 위임한다.\n"
            "두 담당이 모두 필요한 요청이면 nana_agent로 내 일정을 먼저 확인한 뒤 kana_agent로 멤버 조율을 위임하고, "
            "Kana가 정한 시간을 내 일정으로 저장해야 하면 마지막에 다시 nana_agent에 저장을 위임한다.\n"
            "하위 agent에 넘기는 query에는 사람 이름, 날짜 범위, 회의 길이처럼 그 agent가 판단에 필요한 정보를 그대로 적어 준다."
        ),
    ]


def nana_prompt_parts() -> list[str]:
    """Week 6 Nana 하위 에이전트 전용 system prompt 조각입니다."""

    return [
        *week04_prompt_parts(),
        (
            "[Week 6 Nana 하위 에이전트]\n"
            f"오늘 날짜는 {current_app_date_iso()}이며, tool에 넘기는 날짜는 항상 YYYY-MM-DD로 계산한다.\n"
            "너는 supervisor가 개인 업무를 위임할 때 실행되는 하위 에이전트 Nana다. "
            "supervisor의 prompt는 보이지 않으므로 받은 query만 근거로 스스로 판단해 tool을 고른다.\n"
            "너의 담당은 내 개인 일정 생성·조회·수정·삭제, 일정/할 일/알림의 구조화 저장, "
            "개인 참고자료와 앱 대화 기록 검색이다.\n"
            "앱에 저장된 내 일정을 물으면 personal_list_saved_schedules를, 현재 대화에서 만든 임시 일정은 "
            "personal_list_schedules를 호출한다.\n"
            "이미 정해진 회의 시간을 내 일정으로 저장해 달라는 요청도 네 담당이므로 일정 생성·저장 tool로 처리한다.\n"
            "외부 멤버의 이전 대화 검색, 다른 사람의 busy-time 수집, 공통 가능 시간 결정은 Kana 담당이다. "
            "그런 요청이 오면 추측해서 답하지 말고 Kana 담당이라고 한 문장으로 알린다.\n"
            "답변은 tool 결과만 근거로 하고, 결과가 비어 있으면 찾지 못했다고 답한다."
        ),
    ]


def kana_prompt_parts() -> list[str]:
    """Week 6 Kana 하위 에이전트 전용 system prompt 조각입니다."""

    return [
        (
            "[Week 6 Kana 하위 에이전트]\n"
            f"오늘 날짜는 {current_app_date_iso()}이며, tool에 넘기는 날짜는 항상 YYYY-MM-DD로 계산한다.\n"
            "너는 supervisor가 그룹 조율 업무를 위임할 때 실행되는 하위 에이전트 Kana다. "
            "supervisor의 prompt는 보이지 않으므로 받은 query만 근거로 스스로 판단해 tool을 고른다.\n"
            "너의 담당은 외부 멤버의 이전 대화 검색, 멤버별 일정 추출, 공유 일정 row 조회, "
            "여러 사람의 busy-time 종합, 공통 가능 시간 후보 검증과 최종 회의 시간 결정이다.\n"
            "자연어 요청에서 사람·날짜·회의 길이를 정리해야 하면 extract_schedule_request를 먼저 호출한다.\n"
            "멤버의 이전 대화를 찾아야 하면 search_previous_conversations를 호출한다. 사람 이름은 member_names에 넣고 "
            "query에는 조사를 뗀 한두 단어짜리 주제 핵심어만 넣는다. 대화 원문이 필요하면 "
            "load_conversation_messages를 conversation_id와 함께 호출한다.\n"
            "멤버와 날짜 범위가 분명한 일정 조회는 extract_schedules_from_history를 호출하고, "
            "내 일정까지 함께 비교해야 하면 collect_member_schedules를 호출한다. "
            "공유 저장소에 등록된 row 자체를 확인해야 하면 list_shared_schedules를 호출한다.\n"
            "회의 시간을 잡아 달라는 요청이면 먼저 collect_member_schedules로 내 일정과 멤버 busy-time을 모으고, "
            "그 rows를 직접 읽어 어떤 일정과도 겹치지 않는 후보를 스스로 골라 find_common_available_slots의 "
            "candidate_slots로 넘긴 뒤, 이어서 decide_final_slot으로 최종 시간을 확정한다.\n"
            "두 tool은 후보나 최종 시간을 대신 계산해 주지 않는다. 후보 선정과 최종 선택은 항상 네가 하고 "
            "tool에는 고른 결과를 argument로 넘긴다.\n"
            "find_common_available_slots 결과만으로 답을 끝내지 말고 반드시 decide_final_slot까지 호출한다.\n"
            "답변은 tool 결과의 rows, candidate_slots, final_slot, reason만 근거로 하고, "
            "겹치지 않는 후보가 없으면 후보가 없다고 답하며 근거가 된 일정을 함께 알린다.\n"
            "확정된 시간을 내 일정으로 저장하거나 개인 일정을 만들고 지우는 일은 Nana 담당이므로 직접 하지 말고 "
            "Nana 담당이라고 알린다."
        ),
    ]


def nana_system_prompt() -> str:
    return join_system_prompt(nana_prompt_parts())


def kana_system_prompt() -> str:
    return join_system_prompt(kana_prompt_parts())


def supervisor_system_prompt() -> str:
    return join_system_prompt(
        [
            *week06_prompt_parts(),
            (
                "[Week 6 supervisor 실행 규칙]\n"
                "사용자 요청을 받으면 반드시 nana_agent 또는 kana_agent 중 하나를 먼저 호출한다. "
                "tool 호출 없이 바로 답하지 않는다.\n"
                "최종 답변은 위임 결과 JSON의 answer, trace, inner_tool_names, final_slot_payload, "
                "final_decision_payload만 근거로 작성하고, 거기 없는 일정·시간·사람을 지어내지 않는다.\n"
                "kana_agent 결과에 final_slot_payload가 있으면 final_slot과 reason, 비교한 후보(candidates)를 함께 알려 준다.\n"
                "final_slot이 비어 있거나 needs_agent_selection이 true면 아직 확정하지 못했다고 알리고 "
                "근거로 본 후보와 일정을 보여 준다.\n"
                "두 하위 agent가 모두 필요한 요청이면 각각 위임한 뒤 두 결과를 종합해 한 번에 답한다.\n"
                "하위 agent가 자기 담당이 아니라고 답하면 그 업무를 담당하는 다른 하위 agent로 다시 위임한다."
            ),
        ]
    )


def _tool_call_names(events: list[dict[str, Any]]) -> list[str]:
    return [event["tool_name"] for event in events if event.get("event") == "tool_call" and event.get("tool_name")]


def extract_langchain_trace(result: dict[str, Any]) -> dict[str, Any]:
    """Week 6 supervisor 실행 결과를 UI trace payload로 변환합니다."""

    events = extract_agent_events(result)
    inner_tool_names: list[str] = []
    final_slot_payload: dict[str, Any] | None = None
    final_decision_payload: dict[str, Any] | None = None
    selected_agent: str | None = None

    for event in events:
        if event.get("event") == "tool_call" and event.get("tool_name") in {"nana_agent", "kana_agent"}:
            selected_agent = event["tool_name"]
        content = event.get("content")
        if isinstance(content, dict):
            inner_tool_names.extend(content.get("inner_tool_names") or [])
            if content.get("final_slot_payload"):
                final_slot_payload = content["final_slot_payload"]
            elif "final_slot" in content:
                final_slot_payload = content
            if content.get("final_decision_payload"):
                final_decision_payload = content["final_decision_payload"]

    return {
        "events": events,
        "supervisor_selected_agent": selected_agent,
        "inner_tool_names": inner_tool_names,
        "final_slot_payload": final_slot_payload,
        "final_decision_payload": final_decision_payload,
    }


def tool_name(tool_object: Any) -> str:
    return getattr(tool_object, "name", getattr(tool_object, "__name__", str(tool_object)))


FIND_COMMON_AVAILABLE_SLOTS_DESCRIPTION = (
    "여러 사람의 busy-time을 근거로 agent가 직접 고른 공통 가능 시간 후보를 검증하고 기록하는 tool이다. "
    "이 tool은 후보를 대신 계산하지 않는다. candidate_slots를 비워 두고 호출하면 결과 candidate_slots도 비어 있다. "
    "먼저 collect_member_schedules 같은 일정 조회 tool로 busy row를 모으고, 그 rows를 직접 읽어 "
    "어떤 busy row와도 겹치지 않는 시간대를 스스로 골라 candidate_slots에 담아 넘겨야 한다. "
    "candidate_slots의 각 항목은 date('YYYY-MM-DD'), start_time('HH:MM'), end_time('HH:MM'), "
    "duration_minutes(회의 길이 분), reason(그 시간을 고른 짧은 근거)을 모두 포함한다. "
    "후보는 date_from~date_to 날짜 범위 안, workday_start~workday_end 업무 시간 안이어야 하고 "
    "duration_minutes 이상 길어야 하며, 겹침 검증을 통과하지 못한 후보는 결과에서 빠진다. "
    "busy_rows에는 앞선 tool output의 rows를 그대로 복사해 넘겨 검증 근거를 남긴다. "
    "busy_rows를 넘기지 않으면 member_names와 date_from/date_to로 일정을 다시 모아 검증한다. "
    "member_names에는 회의 대상 멤버 이름을, date_from/date_to에는 조회 날짜 범위를 YYYY-MM-DD로 넣고, "
    "llm_reason에는 후보 목록 전체를 그렇게 고른 이유를 적는다. "
    "이 tool 결과로 답변을 끝내지 말고, 검증을 통과한 candidate_slots를 가지고 이어서 decide_final_slot을 호출해 "
    "최종 시간을 확정한다."
)


DECIDE_FINAL_SLOT_DESCRIPTION = (
    "agent가 직접 고른 최종 회의 시간을 사용자에게 보여줄 결정 payload로 기록하는 tool이다. "
    "이 tool은 최종 시간을 대신 고르지 않는다. find_common_available_slots가 돌려준 후보 중 무엇을 쓸지 agent가 판단해 "
    "selected_index(candidate_slots의 0부터 시작하는 번호) 또는 selected_slot(후보 객체)을 넘겨야 한다. "
    "시간을 확정했으면 final_slot에 'YYYY-MM-DD HH:MM-HH:MM' 형식 문자열을 넣고 needs_agent_selection은 false로 둔다. "
    "아직 고르지 못했거나 겹치지 않는 후보가 없으면 final_slot은 null, needs_agent_selection은 true로 두고 "
    "왜 확정하지 못했는지 reason에 적는다. "
    "reason에는 그 시간을 고른 이유나 보류 이유를 사용자에게 그대로 보여줄 문장으로 적는다. "
    "결정 근거를 남기기 위해 candidate_slots에는 find_common_available_slots 결과의 후보 목록을, "
    "busy_rows에는 근거로 본 busy row를, member_names/date_from/date_to/duration_minutes에는 "
    "이번 조율의 대상 멤버와 날짜 범위, 회의 길이를 함께 넘긴다. "
    "반환 JSON의 final_slot, reason, candidates가 최종 답변의 근거가 된다."
)


class FindCommonAvailableSlotsInput(BaseModel):
    member_names: list[str] = Field(description="공통 가능 시간을 찾아야 하는 외부 멤버 이름 목록")
    date_from: str = Field(description="조회 시작 날짜. ISO datetime이면 날짜 부분만 사용")
    date_to: str = Field(description="조회 종료 날짜. ISO datetime이면 날짜 부분만 사용")
    duration_minutes: int = Field(default=60, ge=30, le=480, description="회의 길이(분)")
    workday_start: str = Field(default="09:00", description="허용 업무 시간 시작 HH:MM")
    workday_end: str = Field(default="18:00", description="허용 업무 시간 종료 HH:MM")
    limit: int = Field(default=5, ge=1, le=20, description="최대 후보 수")
    busy_rows: list[dict[str, Any]] | None = Field(
        default=None,
        description="앞선 일정 조회 tool output에서 복사한 busy_rows. 후보는 이 row들과 overlap/겹치면 안 됩니다.",
    )
    candidate_slots: list[CommonSlotCandidate] = Field(
        default_factory=list,
        description=(
            "LLM agent가 직접 고른 후보 목록. 각 항목은 date, start_time, end_time, "
            "duration_minutes, reason을 포함하고 busy_rows와 겹치면 안 됩니다."
        ),
    )
    llm_reason: str | None = Field(default=None, description="LLM agent가 후보 목록을 고른 전체 이유")


class DecideFinalSlotInput(BaseModel):
    candidate_slots: list[Any] = Field(default_factory=list, description="find_common_available_slots 결과의 후보 목록")
    selected_slot: Any | None = Field(default=None, description="LLM agent가 직접 고른 후보 객체")
    selected_index: int | None = Field(default=None, description="LLM agent가 직접 고른 candidate_slots index")
    final_slot: str | None = Field(
        default=None,
        description="최종 확정 시간 텍스트. 형식은 'YYYY-MM-DD HH:MM-HH:MM'. 미확정이면 null",
    )
    needs_agent_selection: bool | None = Field(
        default=None,
        description="후보 선택이 더 필요하면 true, final_slot을 확정했으면 false",
    )
    member_names: list[str] | None = Field(default=None, description="회의 대상 멤버 목록")
    date_from: str | None = Field(default=None, description="요청 날짜 범위 시작")
    date_to: str | None = Field(default=None, description="요청 날짜 범위 종료")
    duration_minutes: int = Field(default=60, description="회의 길이(분)")
    reason: str | None = Field(default=None, description="최종 선택 또는 보류에 대한 사용자-facing 설명")
    busy_rows: list[dict[str, Any]] | None = Field(default=None, description="최종 결정 근거로 남길 busy_rows")


class ProposeGroupScheduleInput(BaseModel):
    """기존 호환용 그룹 일정 제안 입력입니다."""

    title: str
    member_names: list[str]
    candidate_slots: list[CommonSlotCandidate] = Field(default_factory=list)
    selected_slot: CommonSlotCandidate | None = None
    reason: str | None = None


class AgentQueryInput(BaseModel):
    """하위 에이전트 위임 입력입니다."""

    query: str


def find_common_available_slots_dict(
    member_names: list[str],
    date_from: str,
    date_to: str,
    duration_minutes: int = 60,
    workday_start: str = "09:00",
    workday_end: str = "18:00",
    limit: int = 5,
    busy_rows: list[dict[str, Any]] | None = None,
    candidate_slots: list[dict[str, Any]] | None = None,
    llm_reason: str | None = None,
) -> dict[str, Any]:
    """멤버별 busy-time rows와 LLM이 고른 후보 payload를 검증 결과로 바꿉니다."""

    normalized_member_names = normalize_external_member_names(member_names)
    members_with_me = ["나", *[name for name in normalized_member_names if name != "나"]]
    normalized_date_from = normalize_date_bound(date_from)
    normalized_date_to = normalize_date_bound(date_to)

    rows = busy_rows if busy_rows is not None else []
    if busy_rows is None:
        collected = json.loads(
            collect_member_schedules.invoke(
                {
                    "member_names": members_with_me,
                    "date_from": normalized_date_from,
                    "date_to": normalized_date_to,
                }
            )
        )
        rows = collected.get("rows") or []

    return find_common_available_slots_payload(
        member_names=members_with_me,
        date_from=normalized_date_from,
        date_to=normalized_date_to,
        busy_rows=rows,
        duration_minutes=duration_minutes,
        workday_start=workday_start,
        workday_end=workday_end,
        limit=limit,
        candidate_slots=candidate_slots,
        llm_reason=llm_reason,
    )


@tool(description=FIND_COMMON_AVAILABLE_SLOTS_DESCRIPTION, args_schema=FindCommonAvailableSlotsInput)
def find_common_available_slots(
    member_names: list[str],
    date_from: str,
    date_to: str,
    duration_minutes: int = 60,
    workday_start: str = "09:00",
    workday_end: str = "18:00",
    limit: int = 5,
    busy_rows: list[dict[str, Any]] | None = None,
    candidate_slots: list[Any] | None = None,
    llm_reason: str | None = None,
) -> str:
    """수집된 멤버 일정에서 LLM이 직접 고른 공통 가능 후보 시간을 검증합니다."""

    payload = find_common_available_slots_dict(
        member_names=member_names,
        date_from=date_from,
        date_to=date_to,
        duration_minutes=duration_minutes,
        workday_start=workday_start,
        workday_end=workday_end,
        limit=limit,
        busy_rows=busy_rows,
        candidate_slots=candidate_slots,
        llm_reason=llm_reason,
    )
    return json.dumps(payload, ensure_ascii=False, default=str)


@tool(description=DECIDE_FINAL_SLOT_DESCRIPTION, args_schema=DecideFinalSlotInput)
def decide_final_slot(
    candidate_slots: list[Any] | None = None,
    selected_slot: Any | None = None,
    selected_index: int | None = None,
    final_slot: str | None = None,
    needs_agent_selection: bool | None = None,
    member_names: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    duration_minutes: int = 60,
    reason: str | None = None,
    busy_rows: list[dict[str, Any]] | None = None,
) -> str:
    """LLM이 직접 고른 후보/최종 시간을 course repo payload로 기록합니다."""

    payload = decide_final_slot_payload(
        candidate_slots=candidate_slots,
        selected_slot=selected_slot,
        selected_index=selected_index,
        member_names=member_names,
        date_from=date_from,
        date_to=date_to,
        duration_minutes=duration_minutes,
        final_slot=final_slot,
        needs_agent_selection=needs_agent_selection,
        reason=reason,
        busy_rows=busy_rows,
    )
    return json.dumps(payload, ensure_ascii=False, default=str)


def kana_tools() -> list[Any]:
    return [
        extract_schedule_request,
        search_previous_conversations,
        load_conversation_messages,
        extract_schedules_from_history,
        list_shared_schedules,
        collect_member_schedules,
        find_common_available_slots,
        decide_final_slot,
    ]


def supervisor_tools() -> list[Any]:
    return [nana_agent, kana_agent]


def agent_tool_names(agent_name: str) -> list[str]:
    if agent_name == "nana_agent":
        return [tool_name(item) for item in week04_tools()]
    if agent_name == "kana_agent":
        return [tool_name(item) for item in kana_tools()]
    if agent_name == "supervisor":
        return [tool_name(item) for item in supervisor_tools()]
    return []


@tool(args_schema=ProposeGroupScheduleInput)
def propose_group_schedule(
    title: str,
    member_names: list[str],
    candidate_slots: list[Any] | None = None,
    selected_slot: Any | None = None,
    reason: str | None = None,
) -> str:
    """Kana가 고른 후보 시간으로 최종 그룹 일정 결정 페이로드를 만듭니다."""

    slots = [slot.model_dump() if hasattr(slot, "model_dump") else slot for slot in candidate_slots or []]
    selected = selected_slot.model_dump() if hasattr(selected_slot, "model_dump") else selected_slot
    payload = {
        "title": title,
        "members": normalize_external_member_names(member_names),
        "selected_slot": selected,
        "status": "confirmed" if selected else "needs_manual_review",
        "reason": reason,
        "candidate_slots": slots,
    }
    return json.dumps({"ok": True, "tool_name": "propose_group_schedule", "final_decision": payload}, ensure_ascii=False)


@tool(args_schema=AgentQueryInput)
def nana_agent(query: str) -> str:
    """개인 일정과 개인 RAG 작업을 프롬프트 기반 Nana 하위 에이전트에게 위임합니다."""

    global _NANA_SUBAGENT
    if _NANA_SUBAGENT is None:
        _NANA_SUBAGENT = create_agent(
            model=chat_model(),
            tools=week04_tools(),
            system_prompt=nana_system_prompt(),
        )

    result = _NANA_SUBAGENT.invoke({"messages": [{"role": "user", "content": query}]})
    events = extract_agent_events(result)
    return json.dumps(
        {
            "selected_agent": "nana_agent",
            "answer": extract_final_text(result),
            "trace": events,
            "inner_tool_names": _tool_call_names(events),
        },
        ensure_ascii=False,
        default=str,
    )


@tool(args_schema=AgentQueryInput)
def kana_agent(query: str) -> str:
    """그룹 일정 종합 작업을 프롬프트 기반 Kana 하위 에이전트에게 위임합니다."""

    global _KANA_SUBAGENT
    if _KANA_SUBAGENT is None:
        _KANA_SUBAGENT = create_agent(
            model=chat_model(),
            tools=kana_tools(),
            system_prompt=kana_system_prompt(),
        )

    result = _KANA_SUBAGENT.invoke({"messages": [{"role": "user", "content": query}]})
    events = extract_agent_events(result)

    final_slot_payload: dict[str, Any] | None = None
    final_decision_payload: dict[str, Any] | None = None
    for event in events:
        content = event.get("content")
        if not isinstance(content, dict):
            continue
        if "final_slot" in content:
            final_slot_payload = content
        if content.get("final_decision"):
            final_decision_payload = content["final_decision"]

    return json.dumps(
        {
            "selected_agent": "kana_agent",
            "answer": extract_final_text(result),
            "trace": events,
            "inner_tool_names": _tool_call_names(events),
            "final_slot_payload": final_slot_payload,
            "final_decision_payload": final_decision_payload,
        },
        ensure_ascii=False,
        default=str,
    )


def build_langchain_supervisor_agent() -> object:
    """nana_agent와 kana_agent 위임 도구만 노출하는 LangChain v1 슈퍼바이저입니다."""

    global _SUPERVISOR_AGENT
    if _SUPERVISOR_AGENT is None:
        _SUPERVISOR_AGENT = create_agent(
            model=chat_model(),
            tools=supervisor_tools(),
            system_prompt=supervisor_system_prompt(),
        )
    return _SUPERVISOR_AGENT


def build_week_agent() -> object:
    """active-week registry가 호출하는 표준 Week agent builder입니다."""

    return build_langchain_supervisor_agent()
