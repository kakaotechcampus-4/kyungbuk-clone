from __future__ import annotations

import json
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from fixed.config import CONFIG
from fixed.langchain_trace import (
    extract_agent_events,
    extract_final_text,
    extract_langchain_trace as extract_common_langchain_trace,
)
from fixed.llm import chat_model
from fixed.runtime_clock import current_app_date_iso
from fixed.schedule_decision import (
    CommonSlotCandidate,
    decide_final_slot_payload,
    find_common_available_slots_payload,
)
from student_parts.week01_wake_up_nana import join_system_prompt
from student_parts.week04_retrieve_nanas_memory import week04_prompt_parts, week04_tools
from student_parts.week05_load_kanas_past_conversations import (
    collect_member_schedules,
    extract_schedules_from_history,
    list_shared_schedules,
    load_conversation_messages,
    search_previous_conversations,
)


_NANA_AGENT: Any | None = None
_KANA_AGENT: Any | None = None
_WEEK06_AGENT: Any | None = None


# [6주차 수강생 구현 가이드]
#
# 목표
#   Week 1-5는 tool을 계속 누적하는 단일 agent였습니다. Week 6은 구조 자체가 바뀌어,
#   supervisor가 업무 tool을 직접 호출하지 않고 nana_agent/kana_agent delegate tool만
#   호출하는 Supervisor/Sub-agent 구조를 만듭니다. Nana는 내 개인 일정, Kana는 외부
#   팀원 일정을 맡고, supervisor는 두 delegate 결과를 근거로 최종 회의 시간을 답합니다.
#
# 과제 구성
#   - 메인과제: nana_agent/kana_agent delegate tool과 이를 감싸는 supervisor를 완성하고,
#     Kana 전용 tool인 decide_final_slot으로 팀원 후보 중 최종 시간을 고릅니다.
#   - 추가 과제: find_common_available_slots로 collect_member_schedules의 busy_rows를
#     근거 삼아 공통 가능 시간을 자동 계산합니다. kana_agent_tools()에 포함되어 있어
#     collect_member_schedules -> find_common_available_slots -> decide_final_slot 흐름을
#     실제로 호출할 수 있습니다.
#
# 구현 위치와 사용할 코드
#   - 이 파일(student_parts/week06_kanamate_decides_schedule.py)의 delegate tool과
#     Kana 전용 tool을 구현합니다.
#   - Nana sub-agent의 tool 목록은 student_parts/week04_retrieve_nanas_memory.py의
#     week04_tools()를 그대로 재사용합니다. Week 6에서 개인 일정 tool을 새로 만들지 않습니다.
#   - Kana sub-agent의 MCP/공유 일정 tool은 student_parts/week05_load_kanas_past_conversations.py의
#     search_previous_conversations, load_conversation_messages, extract_schedules_from_history,
#     list_shared_schedules, collect_member_schedules를 그대로 재사용합니다.
#   - 최종 시간 결정 payload 검증은 fixed/schedule_decision.py의 decide_final_slot_payload(),
#     find_common_available_slots_payload()를 사용합니다. 이 모듈은 LangChain을 모르고
#     agent가 넘긴 인자만 검증·정규화하므로, 이 파일의 @tool 함수가 LLM 인자를 그대로 전달합니다.
#   - delegate tool의 내부 trace는 fixed/langchain_trace.py의 extract_agent_events(),
#     extract_final_text()로 뽑아 JSON 문자열로 감싸 supervisor에게 돌려줍니다.
#
# 메인과제 구현 대상
#   1. decide_final_slot
#      - candidate_slots(팀원 가능 시간 후보 문자열 목록), selected_slot(최종 선택), reason을 받습니다.
#      - decide_final_slot_payload(...)를 호출해 검증된 payload를 그대로 반환합니다.
#      - 개인 일정 busy_rows를 넘기지 않습니다. Kana는 팀원 후보 비교만 담당하고
#        내 개인 일정 충돌 여부는 판단하지 않습니다.
#
#   2. nana_agent
#      - request 문자열을 받아 Nana sub-agent(build_nana_agent())를 실행합니다.
#      - 결과에서 answer와 tool trace를 꺼내 {"agent": "nana", "answer", "trace", "inner_tool_names",
#        "final_slot_payload"} JSON으로 반환합니다. inner_tool_names/final_slot_payload는 supervisor가
#        raw trace 구조를 직접 몰라도 "어떤 tool이 호출됐는지"와 "확정된 시간"을 top-level에서 바로
#        읽을 수 있게 하는 필드입니다.
#
#   3. kana_agent
#      - request 문자열을 받아 Kana sub-agent(build_kana_agent())를 실행합니다.
#      - nana_agent와 같은 모양으로 {"agent": "kana", "answer", "trace", "inner_tool_names",
#        "final_slot_payload"} JSON을 반환합니다. decide_final_slot을 호출했다면
#        final_slot_payload에 그 결과가 그대로 담깁니다.
#
#   4. build_week06_agent() / build_week_agent()
#      - supervisor tool 목록에는 nana_agent, kana_agent만 놓습니다. personal_* tool이나
#        MCP tool을 supervisor에 직접 붙이지 않습니다.
#      - system prompt에서 "먼저 nana_agent, 그 다음 kana_agent를 호출하고, kana_agent에는
#        Nana가 확인한 개인 일정 내용을 전달하지 않는다"를 명시합니다.
#
# 추가 과제 구현 대상
#   1. find_common_available_slots
#      - member_names/date_from/date_to로 collect_member_schedules를 호출해 busy_rows(내 일정 +
#        외부 멤버 일정)를 모은 뒤 find_common_available_slots_payload(...)에 넘겨 자동으로
#        공통 가능 시간을 계산합니다.
#      - 이 tool은 collect_member_schedules의 rows를 그대로 근거로 쓰므로 내 개인 일정도
#        busy time에 포함됩니다. decide_final_slot과 달리 개인 일정까지 반영해 계산하는
#        경로이므로, Kana의 system prompt에서 이 tool을 쓸 때는 계산에는 내 일정이 반영되지만
#        답변에서 내 개인 일정의 제목/세부 내용은 언급하지 않는다고 구분해 둡니다.
#
# 책임 경계
#   Supervisor는 personal_* tool도, search_previous_conversations 같은 MCP tool도 직접 호출하지
#   않습니다. Nana는 내 개인 일정 CRUD/구조화 저장/개인 RAG만 맡고 외부 팀원 정보를 모릅니다.
#   Kana는 외부 MCP SQLite 조회와 팀원 후보 비교만 맡고, 메인과제 범위에서는 내 개인 일정을
#   직접 판단하지 않습니다. 개인 일정과 팀원 후보를 종합해 최종 답을 만드는 것은 supervisor의 몫입니다.
#
# 검증 방법
#   - 메인과제: "팀원 A/B/C와 다음 주 회의 시간을 잡아줘" 같은 요청을 build_week_agent()에 넣고,
#     supervisor trace에서 nana_agent -> kana_agent 순서로 tool_call/tool_result가 나오는지 봅니다.
#     nana_agent의 trace에 personal_list_saved_schedules 계열 호출이(새 대화에서도 SQLite 저장
#     일정을 확인해야 하므로 personal_list_schedules만으로는 부족합니다), kana_agent의 trace에
#     search_previous_conversations/extract_schedules_from_history/decide_final_slot 호출이
#     있는지 확인하고, kana_agent의 answer가 개인 일정 충돌을 단정하지 않는지 확인합니다.
#     nana_agent/kana_agent의 반환 JSON에서 inner_tool_names와 final_slot_payload가 raw trace와
#     일치하고, supervisor의 바깥 trace(delegate_payloads)에서도 그대로 유지되는지 확인합니다.
#   - 추가 과제: collect_member_schedules -> find_common_available_slots -> decide_final_slot
#     순서로 tool_call이 이어지는지, 응답에 내 개인 일정과 겹치지 않는 후보만 candidate_slots로
#     남는지 확인합니다.
#
# 함수별 동작 설명 ([메인]/[추가]/[공통]은 각 함수가 속한 과제 티어입니다)
#   - [공통] json_payload(payload)
#     tool 응답 dict를 한글이 보존되는 JSON 문자열로 바꿉니다.
#
#   - [공통] DisableParallelToolCalls
#     supervisor/kana가 delegate tool 또는 업무 tool을 한 번에 하나씩 순서대로 호출하도록
#     parallel_tool_calls를 끄는 middleware입니다. nana_agent -> kana_agent 순서,
#     search_previous_conversations -> extract_schedules_from_history -> decide_final_slot
#     순서를 trace에서 그대로 관찰할 수 있게 해 줍니다.
#
#   - [메인] DecideFinalSlotInput / NanaAgentInput / KanaAgentInput
#     각각 최종 시간 결정, Nana delegate 호출, Kana delegate 호출 tool의 입력 스키마입니다.
#
#   - [추가] FindCommonAvailableSlotsInput
#     공통 가능 시간 자동 계산 tool의 입력 스키마입니다. candidate_slots는
#     fixed/schedule_decision.py의 CommonSlotCandidate 목록입니다.
#
#   - [메인] decide_final_slot(...)
#     팀원 가능 시간 후보만 비교해 최종 회의 시간 후보를 결정하는 Kana 전용 tool입니다.
#
#   - [추가] find_common_available_slots(...)
#     collect_member_schedules의 rows를 busy_rows로 삼아 공통 가능 시간을 자동 계산하는 Kana tool입니다.
#
#   - [공통] _inner_tool_names(trace_events) / _final_slot_payload(trace_events)
#     delegate sub-agent의 raw trace에서 호출된 tool 이름 목록과 decide_final_slot 결과 payload를
#     각각 뽑아, nana_agent/kana_agent 반환 JSON의 top-level 필드로 올려 줍니다.
#
#   - [메인] nana_agent_tools() / nana_agent_system_prompt() / build_nana_agent()
#     Nana sub-agent가 쓸 tool 목록(week04_tools() 재사용), system prompt, agent builder입니다.
#
#   - [메인] nana_agent(...)
#     개인 일정 CRUD/구조화 저장/개인 RAG 검색을 Nana sub-agent에게 위임하는 delegate tool입니다.
#
#   - [메인] kana_agent_tools() / kana_agent_system_prompt() / build_kana_agent()
#     Kana sub-agent가 쓸 tool 목록(Week 5 MCP tool + find_common_available_slots + decide_final_slot),
#     system prompt, agent builder입니다.
#
#   - [메인] kana_agent(...)
#     외부 팀원의 이전 대화·공유 일정 조회와 최종 후보 결정을 Kana sub-agent에게 위임하는 delegate tool입니다.
#
#   - [공통] week06_tools() / week06_prompt_parts() / week06_system_prompt()
#     supervisor에게 nana_agent, kana_agent delegate tool만 공개하고, 호출 순서와 역할 경계를
#     설명하는 system prompt를 만듭니다.
#
#   - [공통] build_week06_agent() / build_week_agent()
#     supervisor LangChain agent를 한 번만 만들고 재사용합니다.
#
#   - [공통] extract_langchain_trace(result)
#     기본 이벤트 목록에 nana_agent/kana_agent tool_result에서 뽑은 delegate_payloads를
#     덧붙입니다. tool_result content는 extract_agent_events가 이미 JSON으로 파싱해 두므로,
#     nana/kana의 answer와 trace를 그대로 top-level에서 확인할 수 있습니다.


def json_payload(payload: dict[str, Any]) -> str:
    """도구 반환용 dict를 한글이 깨지지 않는 JSON 문자열로 변환합니다."""

    return json.dumps(payload, ensure_ascii=False)


class DisableParallelToolCalls(AgentMiddleware):
    """tool을 한 번에 하나씩 순서대로 호출하도록 강제하는 middleware입니다."""

    def wrap_model_call(self, request, handler):
        request.model_settings["parallel_tool_calls"] = False
        return handler(request)


class DecideFinalSlotInput(BaseModel):
    """팀원 가능 시간 후보 비교 및 최종 결정 입력입니다."""

    candidate_slots: list[str] = Field(description="비교 대상 팀원 가능 시간 후보 문자열 목록")
    selected_slot: str = Field(description="candidate_slots 중 최종으로 고른 시간 문자열")
    reason: str = Field(description="이 후보를 최종으로 고른 짧은 근거")
    member_names: list[str] | None = None
    date_from: str | None = None
    date_to: str | None = None


class FindCommonAvailableSlotsInput(BaseModel):
    """내 일정과 팀원 busy-time을 모두 반영한 공통 가능 시간 자동 계산 입력입니다."""

    member_names: list[str]
    date_from: str
    date_to: str
    candidate_slots: list[CommonSlotCandidate] | None = None
    duration_minutes: int = Field(default=60, ge=30, le=240)
    workday_start: str = "09:00"
    workday_end: str = "18:00"
    limit: int = Field(default=5, ge=1, le=20)
    reason: str | None = None


@tool(args_schema=DecideFinalSlotInput)
def decide_final_slot(
    candidate_slots: list[str],
    selected_slot: str,
    reason: str,
    member_names: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> str:
    """팀원 가능 시간 후보만 비교해 최종 회의 시간 후보를 결정합니다. 개인 일정 충돌은 판단하지 않습니다."""

    payload = decide_final_slot_payload(
        candidate_slots=candidate_slots,
        selected_slot=selected_slot,
        reason=reason,
        member_names=member_names,
        date_from=date_from,
        date_to=date_to,
    )
    return json_payload({"ok": True, "tool_name": "decide_final_slot", **payload})


@tool(args_schema=FindCommonAvailableSlotsInput)
def find_common_available_slots(
    member_names: list[str],
    date_from: str,
    date_to: str,
    candidate_slots: list[CommonSlotCandidate] | None = None,
    duration_minutes: int = 60,
    workday_start: str = "09:00",
    workday_end: str = "18:00",
    limit: int = 5,
    reason: str | None = None,
) -> str:
    """내 일정과 팀원 busy-time을 collect_member_schedules로 모아 공통으로 비어 있는 시간 후보를 계산합니다."""

    busy_payload = json.loads(
        collect_member_schedules.invoke({"member_names": member_names, "date_from": date_from, "date_to": date_to})
    )
    payload = find_common_available_slots_payload(
        member_names=member_names,
        date_from=date_from,
        date_to=date_to,
        busy_rows=busy_payload.get("rows", []),
        duration_minutes=duration_minutes,
        workday_start=workday_start,
        workday_end=workday_end,
        limit=limit,
        candidate_slots=candidate_slots,
        llm_reason=reason,
    )
    return json_payload({"ok": True, "tool_name": "find_common_available_slots", **payload})


def _inner_tool_names(trace_events: list[dict[str, Any]]) -> list[str]:
    """sub-agent trace에서 호출된 tool 이름만 순서대로 뽑습니다.

    supervisor가 delegate 결과의 raw trace 구조(event/tool_name/id 필드)를 직접 파싱하지
    않아도 "어떤 tool이 호출됐는지"를 top-level에서 바로 읽을 수 있게 합니다.
    """

    return [
        event["tool_name"]
        for event in trace_events
        if event.get("event") == "tool_call" and event.get("tool_name")
    ]


def _final_slot_payload(trace_events: list[dict[str, Any]]) -> dict[str, Any] | None:
    """sub-agent trace에서 decide_final_slot의 결과 payload를 찾습니다. 호출하지 않았으면 None입니다.

    supervisor가 확정된 시간을 얻으려고 raw trace를 뒤져 decide_final_slot tool_result를 직접
    찾아야 하는 방식 대신, 이미 찾아 둔 값을 top-level 필드로 바로 전달합니다.
    """

    for event in reversed(trace_events):
        if event.get("event") == "tool_result" and event.get("tool_name") == "decide_final_slot":
            content = event.get("content")
            if isinstance(content, dict):
                return content
    return None


def nana_agent_tools() -> list[Any]:
    """Nana sub-agent에 공개할 개인 일정 CRUD/구조화 저장/개인 RAG 도구 목록입니다."""

    return week04_tools()


def nana_agent_system_prompt() -> str:
    """Nana sub-agent가 따르는 system prompt입니다. Week 1-4 개인 tool 안내를 그대로 재사용합니다."""

    return join_system_prompt(
        [
            *week04_prompt_parts(),
            f"""오늘은 {current_app_date_iso()}이다.
Week 6부터 너는 supervisor가 위임한 요청만 처리하는 개인 메이트 Nana다.
어떤 회의/일정 조율 요청을 받아도 앞서 누적된 안내(personal_list_saved_schedules로 SQLite에 저장된 내 일정을
먼저 확인)를 그대로 따른 뒤에만 답한다. personal_list_schedules는 현재 대화의 임시 일정만 보여주므로
그것만으로는 새 대화에서 이전 주차에 저장한 일정을 놓칠 수 있다. personal_list_saved_schedules 확인 없이는
개인 일정에 충돌이 없다고 답하지 않는다.
외부 팀원의 일정이나 이전 대화는 네 책임이 아니므로 언급하지 않는다. tool 호출 없이 답하지 않는다.""",
        ]
    )


def build_nana_agent() -> object:
    """Nana sub-agent를 한 번만 만들고 재사용합니다."""

    if not CONFIG.has_openai_key:
        raise RuntimeError("PROXY_TOKEN이 .env에 필요합니다.")
    global _NANA_AGENT
    if _NANA_AGENT is None:
        _NANA_AGENT = create_agent(
            model=chat_model(),
            tools=nana_agent_tools(),
            system_prompt=nana_agent_system_prompt(),
            middleware=[DisableParallelToolCalls()],
        )
    return _NANA_AGENT


class NanaAgentInput(BaseModel):
    """Nana delegate 호출 입력입니다."""

    request: str = Field(description="Nana에게 위임할 개인 일정 요청 문장")


@tool(args_schema=NanaAgentInput)
def nana_agent(request: str) -> str:
    """내 개인 일정 생성/조회/삭제, 구조화 저장, 개인 RAG 검색을 Nana sub-agent에게 위임합니다."""

    agent = build_nana_agent()
    result = agent.invoke({"messages": [{"role": "user", "content": request}]})
    trace = extract_agent_events(result)
    return json_payload(
        {
            "ok": True,
            "tool_name": "nana_agent",
            "agent": "nana",
            "answer": extract_final_text(result),
            "trace": trace,
            "inner_tool_names": _inner_tool_names(trace),
            "final_slot_payload": _final_slot_payload(trace),
        }
    )


def kana_agent_tools() -> list[Any]:
    """Kana sub-agent에 공개할 MCP/공유 일정/최종 후보 결정 도구 목록입니다.

    find_common_available_slots(추가 과제)까지 포함해 collect_member_schedules ->
    find_common_available_slots -> decide_final_slot으로 이어지는 흐름을 실제로 호출할 수 있습니다.
    """

    return [
        search_previous_conversations,
        load_conversation_messages,
        extract_schedules_from_history,
        list_shared_schedules,
        collect_member_schedules,
        find_common_available_slots,
        decide_final_slot,
    ]


def kana_agent_system_prompt() -> str:
    """Kana sub-agent가 따르는 system prompt입니다."""

    return f"""오늘은 {current_app_date_iso()}이다.
너는 supervisor가 위임한 팀원 회의 조율 요청만 처리하는 그룹 메이트 Kana다.
외부 SQLite/MCP 서버에 저장된 팀원들의 이전 대화와 공유 일정을 조회해 팀원 기준 회의 시간 후보만 결정한다.
- conversation_id 없이 특정 팀원의 과거 발언이나 대화 자체를 찾아야 할 때는 search_previous_conversations를 먼저 호출하고,
  원문 전체가 필요하면 검색 결과의 conversation_id로 load_conversation_messages를 호출한다.
- 팀원별 일정/바쁜 시간 후보가 필요하면 extract_schedules_from_history를 호출한다.
- 공유 일정 저장소에 등록된 row 자체를 확인해야 하면 list_shared_schedules를 호출한다.
- 내 일정과 팀원 바쁜 시간을 함께 살펴 전체 그림이 필요하면 collect_member_schedules를 호출할 수 있다.
- 내 일정까지 포함한 공통 가능 시간을 자동으로 한 번에 계산해야 하면, collect_member_schedules로 모은
  busy_rows를 근거로 find_common_available_slots를 호출해 후보를 얻은 뒤 그 candidate_slots를
  decide_final_slot에 그대로 전달해 최종 후보를 확정한다. find_common_available_slots는 계산 과정에서
  내 일정도 busy time으로 반영하지만, 그렇다고 답변에서 내 개인 일정의 제목이나 세부 내용을 언급하거나
  "개인 일정과 충돌하지 않는다"처럼 개인 일정 자체를 사실로 확정해 설명하지 않는다 — "가능한 시간 후보와
  그 이유"만 말한다.
- collect_member_schedules 결과의 "나" 관련 row 자체로 개인 일정 충돌 여부를 직접 판단하거나
  답변에 쓰지 않는다.
- 팀원 후보를 모았으면 반드시 decide_final_slot을 호출해 candidate_slots와 selected_slot, reason을 기록한다.
- 답변에는 팀원 후보와 그 근거만 쓴다. 개인 일정 세부 내용이나 "충돌하지 않는다"는 단정을 쓰지 않는다.
  그건 Nana와 supervisor의 몫이다.
tool 호출 없이 답하지 않는다."""


def build_kana_agent() -> object:
    """Kana sub-agent를 한 번만 만들고 재사용합니다."""

    if not CONFIG.has_openai_key:
        raise RuntimeError("PROXY_TOKEN이 .env에 필요합니다.")
    global _KANA_AGENT
    if _KANA_AGENT is None:
        _KANA_AGENT = create_agent(
            model=chat_model(),
            tools=kana_agent_tools(),
            system_prompt=kana_agent_system_prompt(),
            middleware=[DisableParallelToolCalls()],
        )
    return _KANA_AGENT


class KanaAgentInput(BaseModel):
    """Kana delegate 호출 입력입니다."""

    request: str = Field(description="Kana에게 위임할 팀원 회의 조율 요청 문장")


@tool(args_schema=KanaAgentInput)
def kana_agent(request: str) -> str:
    """MCP SQLite에서 다른 사람들의 일정과 이전 대화 기록을 불러와 팀원 기준 회의 시간 후보를 Kana sub-agent에게 위임합니다."""

    agent = build_kana_agent()
    result = agent.invoke({"messages": [{"role": "user", "content": request}]})
    trace = extract_agent_events(result)
    return json_payload(
        {
            "ok": True,
            "tool_name": "kana_agent",
            "agent": "kana",
            "answer": extract_final_text(result),
            "trace": trace,
            "inner_tool_names": _inner_tool_names(trace),
            "final_slot_payload": _final_slot_payload(trace),
        }
    )


def week06_tools() -> list[Any]:
    """supervisor에게 공개하는 tool 목록입니다. 업무 tool 없이 delegate tool만 둡니다."""

    return [nana_agent, kana_agent]


def week06_prompt_parts() -> list[str]:
    """supervisor system prompt 조각입니다. 개인 tool 안내를 누적하지 않는 독립된 prompt입니다."""

    return [
        f"""오늘은 {current_app_date_iso()}이다.
너는 카나메이트 supervisor다. 개인 일정 tool이나 MCP 검색 tool을 직접 호출하지 않고
nana_agent, kana_agent delegate tool만 호출한다.
모든 회의 시간 결정 요청에서 반드시 먼저 nana_agent를 호출해 내 개인 일정 충돌을 확인하고,
그 다음 반드시 kana_agent를 호출해 팀원들의 이전 대화 기반 후보만 확인한다.
kana_agent에는 Nana가 확인한 개인 일정 내용을 전달하지 않는다.
두 delegate tool_result를 모두 받은 뒤에만 개인 일정과 팀원 후보를 종합해 최종 시간과 이유를 답한다.
tool 호출 없이 답하지 않는다."""
    ]


def week06_system_prompt() -> str:
    """supervisor가 따르는 system prompt입니다."""

    return join_system_prompt(week06_prompt_parts())


def build_week06_agent() -> object:
    """nana_agent, kana_agent delegate tool만 노출하는 supervisor LangChain agent를 만듭니다."""

    if not CONFIG.has_openai_key:
        raise RuntimeError("PROXY_TOKEN이 .env에 필요합니다.")
    global _WEEK06_AGENT
    if _WEEK06_AGENT is None:
        _WEEK06_AGENT = create_agent(
            model=chat_model(),
            tools=week06_tools(),
            system_prompt=week06_system_prompt(),
            middleware=[DisableParallelToolCalls()],
        )
    return _WEEK06_AGENT


def build_week_agent() -> object:
    """active-week registry가 호출하는 표준 Week agent builder입니다."""

    return build_week06_agent()


def _extract_delegate_payloads(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """supervisor trace 이벤트에서 nana_agent/kana_agent tool_result payload만 뽑습니다."""

    payloads: list[dict[str, Any]] = []
    for event in events:
        if event.get("event") != "tool_result":
            continue
        if event.get("tool_name") not in {"nana_agent", "kana_agent"}:
            continue
        content = event.get("content")
        if isinstance(content, dict):
            payloads.append(content)
    return payloads


def extract_langchain_trace(result: dict[str, Any]) -> dict[str, Any]:
    """supervisor trace에 nana_agent/kana_agent delegate payload를 delegate_payloads로 함께 담습니다."""

    trace = extract_common_langchain_trace(result)
    trace["delegate_payloads"] = _extract_delegate_payloads(trace.get("events", []))
    return trace
