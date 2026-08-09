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


# [6주차 수강생 구현 가이드]
#
# 목표
#   Week 6은 "모든 기능을 한 agent가 직접 처리"하지 않고 supervisor가 Nana/Kana 하위 agent로 위임하게 만듭니다.
#   Nana는 개인 일정/저장/RAG를 맡고, Kana는 외부 대화/멤버 일정/그룹 시간 결정을 맡습니다.
#   supervisor가 직접 볼 수 있는 tool은 nana_agent와 kana_agent 두 개뿐입니다.
#
# 과제 구성
#   - 메인과제: 한 agent가 모두 처리하던 구조를 supervisor + Nana/Kana 하위 agent로 나누어
#     supervisor가 요청을 알맞은 하위 agent에 위임하는 뼈대를 완성합니다.
#     세 agent의 system prompt를 직접 작성하는 것과 위임 wrapper tool 두 개 구현이 여기 들어갑니다.
#   - 추가 과제: Kana의 공통 가능 시간 후보 검증(find_common_available_slots)과
#     최종 시간 결정(decide_final_slot)까지 붙여 그룹 일정 조율을 마무리합니다.
#
# 구현 위치와 사용할 코드
#   - 이 파일(student_parts/week06_kanamate_decides_schedule.py)의 Week 6 전용 tool과 sub-agent wrapper를 구현합니다.
#   - 공통 가능 시간 검증/최종 선택 payload 생성은 fixed/schedule_decision.py의
#     find_common_available_slots_payload(), decide_final_slot_payload(), normalize_date_bound()를 사용합니다.
#   - Nana 하위 agent 도구는 student_parts/week04_retrieve_nanas_memory.py의 week04_tools()를 그대로 사용합니다.
#   - Kana 하위 agent 도구는 이 파일의 kana_tools()에서 구성하며, Week 2 extract_schedule_request와
#     Week 5 wrapper tool(search_previous_conversations, extract_schedules_from_history,
#     collect_member_schedules 등), find_common_available_slots, decide_final_slot을 포함합니다.
#   - supervisor가 볼 수 있는 도구는 supervisor_tools()의 nana_agent, kana_agent 두 개뿐입니다.
#   - nana_agent()/kana_agent()/build_langchain_supervisor_agent()는 create_agent(...)로 각각 필요한 agent를 만들고 재사용합니다.
#   - trace 정리는 fixed/langchain_trace.py의 extract_agent_events(), extract_final_text()를 사용합니다.
#
# 메인과제 구현 대상
#   1. week06_prompt_parts / nana_prompt_parts / kana_prompt_parts / supervisor_system_prompt
#      - supervisor와 Nana/Kana 하위 에이전트의 역할 분담을 prompt로 직접 정의합니다.
#      - supervisor는 직접 업무를 처리하지 않고 nana_agent 또는 kana_agent로만 위임하게 씁니다.
#      - Nana는 개인 일정/저장/RAG, Kana는 외부 멤버 일정/공통 시간 결정을 담당하게 씁니다.
#      - week06_prompt_parts는 week05_prompt_parts()를, nana_prompt_parts는 week04_prompt_parts()를 누적합니다.
#        kana_prompt_parts만 누적 없이 시작하므로 Kana 역할을 처음부터 작성해야 합니다.
#      - 하위 에이전트는 supervisor prompt를 공유하지 않으므로 각자 필요한 지시를 스스로 갖고 있어야 합니다.
#
#   2. nana_agent
#      - supervisor가 넘긴 query로 Nana 하위 agent를 이 tool 안에서 만들거나 재사용해 실행합니다.
#      - 개인 일정 조회/생성/수정/삭제 판단은 하위 agent가 prompt와 tool description을 근거로 수행합니다.
#      - 하위 agent 결과에서 answer, trace, inner_tool_names를 뽑아 JSON 문자열로 반환합니다.
#      - 개인 일정 생성/조회/수정/삭제, todo/reminder 저장, 개인 참고자료와 앱 대화 RAG는 Nana 담당입니다.
#
#   3. kana_agent
#      - supervisor가 넘긴 query로 Kana 하위 agent를 이 tool 안에서 만들거나 재사용해 실행합니다.
#      - 하위 trace를 훑어 decide_final_slot 결과를 final_slot_payload로 끌어올립니다.
#      - answer, trace, inner_tool_names, final_slot_payload, final_decision_payload를 JSON으로 반환합니다.
#      - 외부 멤버 일정 조회, 공유 일정 row 조회, 공통 가능 시간 후보 검증과 최종 시간 결정은 Kana 담당입니다.
#
# 추가 과제 구현 대상 (구현하지 않으려면 kana_tools() 목록에서 해당 tool을 제거)
#   1. FIND_COMMON_AVAILABLE_SLOTS_DESCRIPTION / DECIDE_FINAL_SLOT_DESCRIPTION
#      - Kana agent가 두 tool을 언제 어떤 argument로 호출할지 판단하는 유일한 근거가 tool description입니다.
#      - Python tool이 자동으로 최적 시간을 고르는 것이 아니라, agent가 busy_rows를 근거로 후보와 최종 시간을
#        직접 골라 argument로 넘기게 만들어야 합니다. 이 점이 description에 없으면 agent가 tool에 계산을 떠넘깁니다.
#      - candidate_slots 항목 형식(date, start_time, end_time, duration_minutes, reason)과
#        final_slot 형식('YYYY-MM-DD HH:MM-HH:MM')을 명시해 argument 형태를 고정합니다.
#
#   2. find_common_available_slots_dict / find_common_available_slots / decide_final_slot
#      - find_common_available_slots는 busy-time row를 Python 룰이나 nested LLM으로 훑지 않고,
#        Kana agent가 tool description을 읽고 직접 고른 candidate_slots payload를 검증/기록합니다.
#      - date_from/date_to에 ISO datetime이 들어오면 normalize_date_bound()로 날짜 부분만 사용합니다.
#      - busy_rows가 None이면 collect_member_schedules를 호출해 내 일정과 외부 멤버 busy-time을 모읍니다.
#      - decide_final_slot도 nested LLM을 만들지 않고 Kana agent가 넘긴 final_slot, selected_index,
#        needs_agent_selection, reason payload를 그대로 course repo JSON 계약에 맞춰 기록합니다.
#      - 반환 JSON은 course repo 기준 top-level final_slot, reason, candidates를 반드시 포함합니다.
#      - 후보 판단을 수행한 경우 members, busy_rows, candidate_slots도 함께 남겨 근거를 확인할 수 있게 합니다.
#      - selected_index나 selected_slot이 없으면 final_slot을 자동으로 고르지 말고 needs_agent_selection=True 상태를 유지합니다.
#
# 중요한 구조
#   Week 6 파일은 Week 1-5 구현을 다시 작성하지 않습니다.
#   이전 주차 tool을 import하고 kana_tools(), supervisor_tools()에서 역할별로 조립합니다.
#   prompt 함수는 메인과제 구현 대상입니다. supervisor와 Nana/Kana는 서로 다른 system prompt로 동작하므로,
#   위임 규칙과 역할 분담을 어떻게 쓰느냐가 Week 6 동작을 그대로 좌우합니다.
#   두 tool description 상수도 추가 과제 구현 대상입니다. Python 구현과 description이 서로 다른 계약을 말하면
#   agent가 잘못된 argument를 넘기므로, 두 tool을 구현할 때 description도 같은 계약으로 함께 씁니다.
#   각 tool이 받는 argument 이름과 형식은 FindCommonAvailableSlotsInput / DecideFinalSlotInput에 이미 정의되어 있으니
#   description은 그 스키마를 말로 풀어 agent가 언제 무엇을 채울지 판단하게 만드는 역할입니다.
#   find_common_available_slots/decide_final_slot의 실제 겹침 검증과 payload 정리는 fixed/schedule_decision.py가 맡습니다.
#
# Compatibility helper
#   propose_group_schedule은 기존 흐름을 위해 구현된 상태로 유지하며 kana_tools()에는 들어가지 않습니다.
#   현재 supervisor/kana_tools() 경로의 구현 대상은 prompt 함수 4개와 nana_agent, kana_agent(메인),
#   tool description 상수 2개와 find_common_available_slots_dict, find_common_available_slots,
#   decide_final_slot(추가)입니다.
#
# 검증 방법
#   - 메인과제: ./run.sh --week6을 실행하고, supervisor trace에서 nana_agent 또는 kana_agent 중
#     무엇이 선택됐는지, 개인 일정 조회에서 Nana 하위 agent trace에 personal_list_saved_schedules
#     호출이 남는지 확인합니다. 위임이 엉뚱한 agent로 가면 tool 구현이 아니라 prompt의 판단 기준을 먼저 고칩니다.
#     추가 과제를 아직 구현하지 않았다면 kana_tools()에서 find_common_available_slots와
#     decide_final_slot을 빼고 Kana prompt에서도 두 tool 언급을 지운 뒤 위임 흐름만 확인합니다.
#   - 추가 과제: 그룹 일정 요청에서 하위 trace에 search_previous_conversations,
#     extract_schedules_from_history 또는 collect_member_schedules, find_common_available_slots,
#     decide_final_slot이 이어지고 final_slot_payload가 최종 답변과 일치하는지 확인합니다.
#
# 함수별 동작 설명 ([메인]/[추가]/[공통]은 각 함수가 속한 과제 티어입니다)
#   - [메인] week06_system_prompt() / week06_prompt_parts()
#     supervisor agent의 system prompt를 만듭니다. supervisor는 직접 업무를 처리하지 않고 nana_agent 또는 kana_agent로 위임합니다.
#
#   - [메인] nana_prompt_parts() / kana_prompt_parts()
#     하위 에이전트별 역할 prompt를 만듭니다. Nana는 개인 일정/저장/RAG, Kana는 외부 멤버 일정/공통 시간 결정을 담당합니다.
#
#   - [메인] nana_system_prompt() / kana_system_prompt() / supervisor_system_prompt()
#     prompt 조각을 join_system_prompt(...)로 합쳐 실제 create_agent(...)에 넘길 system prompt 문자열을 만듭니다.
#     supervisor_system_prompt()는 누적 조각 뒤에 supervisor 실행 역할 지시를 덧붙이는 자리입니다.
#
#   - [공통] _tool_call_names(events)
#     trace event 목록에서 tool_call 이벤트의 tool_name만 뽑아 UI와 테스트가 호출 순서를 쉽게 확인하게 합니다.
#
#   - [공통] extract_langchain_trace(result)
#     supervisor 실행 결과를 events, 선택된 하위 agent, 내부 tool 이름, 최종 시간 payload가 포함된 trace dict로 정리합니다.
#
#   - [공통] tool_name(tool_object)
#     LangChain tool 객체와 일반 함수 객체에서 이름을 안전하게 읽습니다. agent_tool_names(...)에서 사용합니다.
#
#   - [추가] FIND_COMMON_AVAILABLE_SLOTS_DESCRIPTION / DECIDE_FINAL_SLOT_DESCRIPTION
#     Kana agent가 두 tool을 언제 어떤 argument로 호출할지 판단하는 근거가 되는 tool description입니다.
#     tool이 후보나 최종 시간을 대신 계산해주지 않는다는 점을 agent가 알 수 있게 써야 합니다.
#
#   - [추가] FindCommonAvailableSlotsInput / DecideFinalSlotInput
#     Kana agent가 공통 가능 시간 후보와 최종 선택을 tool argument로 넘길 때 쓰는 Pydantic 입력 스키마입니다.
#
#   - [공통] ProposeGroupScheduleInput / AgentQueryInput
#     호환용 그룹 일정 제안 tool(구현 완료)과 supervisor가 하위 agent에 query를 넘기는 wrapper tool(메인과제)의 입력 스키마입니다.
#
#   - [추가] find_common_available_slots_dict(...)
#     멤버 이름과 날짜 범위를 정규화하고, busy_rows가 없으면 collect_member_schedules를 호출해 수집합니다.
#     실제 후보 검증 payload 생성은 fixed/schedule_decision.py의 find_common_available_slots_payload(...)가 맡습니다.
#
#   - [추가] find_common_available_slots(...)
#     Kana agent가 직접 고른 candidate_slots가 busy_rows와 겹치지 않는지 검증하고 JSON 문자열로 반환하는 tool입니다.
#
#   - [추가] decide_final_slot(...)
#     Kana agent가 직접 고른 selected_index/final_slot/reason을 course repo 계약에 맞는 최종 payload로 기록합니다.
#
#   - [공통] kana_tools() / supervisor_tools() / agent_tool_names(agent_name)
#     Kana 하위 agent와 supervisor가 볼 수 있는 tool 목록을 역할별로 조립하고 이름 목록을 제공합니다.
#
#   - [공통] propose_group_schedule(...)
#     이전 실습 흐름과의 호환을 위해 남겨 둔 그룹 일정 최종 제안 helper입니다. 구현 완료 상태이고
#     kana_tools()에도 들어가지 않습니다. 현재 핵심 경로는 decide_final_slot입니다.
#
#   - [메인] nana_agent(query)
#     supervisor가 개인 업무를 위임할 때 호출하는 tool입니다. Week 4 tool을 가진 Nana 하위 agent를 실행합니다.
#
#   - [메인] kana_agent(query)
#     supervisor가 외부 멤버/그룹 조율 업무를 위임할 때 호출하는 tool입니다. Kana 하위 agent trace에서
#     final_slot_payload와 final_decision_payload를 끌어올려 supervisor가 최종 답변에 사용할 수 있게 합니다.
#
#   - [공통] build_langchain_supervisor_agent() / build_week_agent()
#     supervisor agent를 한 번만 만들고 재사용합니다. build_week_agent()는 실행기가 호출하는 표준 entry point입니다.


def week06_system_prompt() -> str:
    """6주차 supervisor agent가 따르는 시스템 프롬프트입니다."""

    return supervisor_system_prompt()


def week06_prompt_parts() -> list[str]:
    """1~6주차 supervisor system prompt 조각을 누적합니다."""

    return [
        *week05_prompt_parts(),
        "이번 주차에는 업무를 직접 처리하는 도구가 없다. 사용할 수 있는 도구는 nana_agent와 kana_agent 두 개뿐이다. "
        "앞의 지시에 나오는 personal_list_saved_schedules, search_personal_references, "
        "extract_schedules_from_history, collect_member_schedules 같은 도구는 모두 하위 에이전트가 갖고 있으니 "
        "직접 호출하려 하지 말고, 그 일을 담당하는 하위 에이전트에게 위임하시오.",
        "nana_agent는 사용자 본인의 데이터만 다룬다. 내 일정 조회·생성·수정·삭제, todo와 reminder 저장, "
        "내가 등록한 개인 참고자료 검색, 앱에 저장된 내 대화 검색이 여기에 해당한다.",
        "kana_agent는 나 이외의 사람이 관련된 일을 다룬다. 외부 멤버의 과거 대화 검색, 외부 멤버의 일정이나 "
        "바쁜 시간 조회, 공유 일정 저장소 조회, 나와 다른 사람의 시간을 맞추는 조율이 여기에 해당한다.",
        "먼저 사용자가 무엇을 하려는지를 보고, 그다음 누구의 데이터가 필요한지를 보시오. "
        "앱에 저장·수정·삭제하는 요청은 대상에 다른 사람이 포함돼 있어도 nana_agent로 위임한다. "
        "다른 사람의 일정이나 대화를 조회하거나 시간을 맞추는 요청만 kana_agent로 위임한다. "
        "예: \"민준이랑 회의하기로 한 일정 저장해줘\"는 nana_agent, "
        "\"민준이 언제 바빠?\"와 \"민준이랑 언제 만날까?\"는 kana_agent다.",
        "요청에 내 일정이 함께 언급돼도 다른 사람과 시간을 맞추는 것이 목적이면 kana_agent로 위임하시오. "
        "예: \"내 일정 보여줘\"는 nana_agent, \"내 일정이랑 민준이 일정 맞춰줘\"는 kana_agent다.",
        "요청에 어떤 동작(조회·저장·수정·삭제·조율)도, 사람 이름도, 날짜도 없어서 "
        "무엇을 하려는지 자체를 알 수 없는 경우에만 사용자에게 무엇을 원하는지 되묻고, "
        "임의로 한쪽을 고르지 마시오. "
        "예: \"일정 좀 도와줘\", \"뭐 좀 물어볼게\"가 여기에 해당한다. "
        "반대로 동작이나 사람 이름이 하나라도 있으면 되묻지 말고 위 기준으로 판단해 위임하시오. "
        "예: \"민준이\"만 나와도 다른 사람 데이터가 필요하므로 kana_agent다.",
    ]


def nana_prompt_parts() -> list[str]:
    """Week 6 Nana 하위 에이전트 전용 system prompt 조각입니다."""

    return [
        *week04_prompt_parts(),
        "너는 supervisor가 개인 업무를 위임할 때 호출되는 Nana 하위 에이전트다. "
        "사용자와 직접 대화하는 것이 아니라 위임받은 요청 하나를 처리해 결과를 돌려주는 역할이므로, "
        "인사말이나 다음 질문 유도 없이 요청한 결과만 답하시오.",
        "네가 담당하는 것은 사용자 본인의 데이터다. 내 일정 조회·생성·수정·삭제, todo와 reminder 저장, "
        "내가 등록한 개인 참고자료 검색, 앱에 저장된 내 대화 검색이 여기에 해당한다.",
        "다른 사람 이름이 나오는 요청이라도 앱에 무언가를 저장하거나 저장된 것을 조회·수정·삭제하는 일이면 "
        "네가 담당한다. \"민준이랑 회의하기로 한 일정 저장해줘\"처럼 참석자가 있는 그룹 일정을 저장하는 것도 "
        "네 담당이므로 거절하지 마시오.",
        "네가 담당하지 않는 것은 앱 밖의 데이터다. 외부 멤버의 과거 대화 검색, 외부 멤버의 일정이나 바쁜 시간 조회, "
        "공유 일정 저장소 조회, 나와 다른 사람의 시간을 맞추는 조율은 Kana 담당이다. "
        "판단 기준은 사람 이름이 나왔는지가 아니라 앱에 저장된 데이터로 처리할 수 있는 일인지다.",
        "담당이 아닌 요청을 받으면 무엇이 담당 밖인지 한 문장으로만 알리고, "
        "답변 마지막 줄에 다음 형식을 그대로 한 줄 추가하시오.\n"
        "HANDOFF: kana_agent | 담당이 아닌 이유\n"
        "이 줄은 supervisor가 다른 담당자에게 다시 넘기기 위해 읽는 표시이므로 형식을 바꾸지 말고, "
        "담당인 요청에는 절대 붙이지 마시오. 길게 설명하거나 대안을 제안하지도 마시오.",
        "담당 밖 데이터를 추측해서 답하지 마시오. 외부 멤버의 일정이나 공유 저장소 내용은 네 도구로 확인할 수 없으므로, "
        "모르는 것을 아는 것처럼 답하지 말고 담당이 아니라고만 알리시오.",
        "조회 요청에는 extract_schedule_request를 호출하지 마시오. "
        "이 도구는 저장할 내용을 구조화할 때만 쓴다. \"내 일정 보여줘\"처럼 이미 저장된 것을 확인하는 요청은 "
        "personal_list_saved_schedules로 바로 조회하시오.",
        "무엇을 할 수 있는지, 누가 담당하는지 묻기만 한 질문에는 도구를 호출해 실제로 저장·수정·삭제를 실행하지 마시오. "
        "\"누가 담당해?\", \"저장할 수 있어?\", \"이런 것도 돼?\"처럼 능력을 묻는 질문에는 말로만 답하시오. "
        "실제 저장은 사용자가 무엇을 저장할지 구체적으로 요청했을 때만 하시오.",
        "save_structured_request는 다음 세 가지가 모두 갖춰졌을 때만 호출하시오. "
        "① 사용자가 무언가를 저장·등록해달라고 요청했고 ② 저장할 일정의 제목이 있고 ③ 날짜가 있다. "
        "셋 중 하나라도 없으면 저장하지 말고 없는 항목을 사용자에게 물어보시오. "
        "특히 extract_schedule_request 결과의 kind가 \"unknown\"이거나 title이나 date가 비어 있으면 "
        "그 결과를 save_structured_request에 넘기지 마시오. "
        "빈 제목이나 빈 날짜로 저장하면 앱 DB에 쓸모없는 일정이 남는다.",
        "인사나 잡담(\"안녕\", \"고마워\", \"수고했어\")에는 도구를 호출하지 말고 인사로만 답하시오. "
        "묻지 않은 일정 목록을 먼저 나열하지 마시오.",
    ]


def kana_prompt_parts() -> list[str]:
    """Week 6 Kana 하위 에이전트 전용 system prompt 조각입니다."""

    return [
        f"너는 supervisor가 외부 멤버·그룹 조율 업무를 위임할 때 호출되는 Kana 하위 에이전트다. "
        f"오늘 날짜는 {current_app_date_iso()}이다. "
        "사용자와 직접 대화하는 것이 아니라 위임받은 요청 하나를 처리해 결과를 돌려주는 역할이므로, "
        "인사말이나 다음 질문 유도 없이 요청한 결과만 답하시오.",
        "네가 담당하는 것은 나 이외의 사람이 관련된 일이다. 외부 멤버의 과거 대화 검색, 외부 멤버의 일정이나 "
        "바쁜 시간 조회, 공유 일정 저장소 조회, 나와 다른 사람의 시간을 맞추는 조율이 여기에 해당한다.",
        "각 도구를 언제 어떤 인자로 쓰는지는 그 도구의 설명에 적혀 있으니 그 기준을 따르시오. "
        "도구 설명에 있는 날짜 범위 지침과 반환값 해석 방법도 반드시 지키시오.",
        "특정 멤버의 일정이 없다고 답하려면, 그 멤버 이름을 member_names에 넣어 실제로 조회한 결과가 "
        "비어 있을 때만 그렇게 답하시오. 조회하지 않은 멤버에 대해 일정이 없다고 단정하지 마시오.",
        "일정을 앱에 저장하거나 저장된 일정을 수정·삭제하는 일은 네 담당이 아니라 Nana 담당이다. "
        "조율해서 시간을 정한 뒤 그 일정을 저장해달라는 요청을 받으면, 정한 시간을 답에 명확히 적고 "
        "저장은 Nana 담당이라고 알린 뒤 아래 재위임 표시를 붙이시오.",
        "공유 일정 저장소에 일정을 새로 등록하거나 삭제하는 기능은 이번 주차에서 지원하지 않는다. "
        "공유 일정 조회는 list_shared_schedules로 지원하므로 조회 요청까지 거절하지 마시오.",
        "담당이 아닌 요청을 받으면 무엇이 담당 밖인지 한 문장으로만 알리고, "
        "답변 마지막 줄에 다음 형식을 그대로 한 줄 추가하시오.\n"
        "HANDOFF: nana_agent | 담당이 아닌 이유\n"
        "이 줄은 supervisor가 다른 담당자에게 다시 넘기기 위해 읽는 표시이므로 형식을 바꾸지 말고, "
        "담당인 요청에는 절대 붙이지 마시오. 길게 설명하거나 대안을 제안하지도 마시오.",
        "여러 사람이 함께 가능한 시간을 물으면, collect_member_schedules로 모은 rows를 근거로 "
        "겹치지 않는 시간대를 직접 골라 설명하시오. 근거 없이 시간을 만들어내지 말고, "
        "rows에 없는 사람의 일정을 아는 것처럼 답하지 마시오.",
        "여러 날을 한 덩어리로 묶어 \"며칠부터 며칠까지 가능\"이라고 답하지 마시오. "
        "덩어리로 묶으면 그 안에 있는 일정을 빠뜨리게 된다. "
        "rows에 일정이 있는 날짜는 하나도 빠뜨리지 말고 각각 언급한 뒤, 그 날짜를 피한 시간을 제시하시오.",
    ]


def nana_system_prompt() -> str:
    return join_system_prompt(nana_prompt_parts())


def kana_system_prompt() -> str:
    return join_system_prompt(kana_prompt_parts())


def supervisor_system_prompt() -> str:
    return join_system_prompt(
        [
            *week06_prompt_parts(),
            "너는 실행 단계에서 요청 유형을 판단할 수 있으면 반드시 nana_agent 또는 kana_agent 중 "
            "하나를 먼저 호출해야 한다. 하위 에이전트를 호출하지 않은 채 직접 답을 만들지 마시오. "
            "일정이 있는지 없는지, 언제가 가능한지를 하위 에이전트 결과 없이 추측해서 답하면 안 된다.",
            "하위 에이전트 결과의 answer를 근거로 사용자에게 답하시오. "
            "answer에 없는 일정이나 시간을 새로 만들어내지 말고, 결과가 비어 있으면 비어 있다고 그대로 전하시오.",
            "하위 에이전트 결과에 error가 들어 있으면 실행이 실패한 것이다. "
            "그 경우 일정이 없다거나 한가하다고 답하지 말고 error의 user_message를 사용자에게 전하시오. "
            "error_type과 debug_reason은 내부 확인용이므로 사용자에게 보여주지 마시오. "
            "필요하면 다시 시도해도 되는지 물어보시오.",
            "하위 에이전트 결과에 handoff_to가 채워져 있으면 그 에이전트가 담당이 아니라는 뜻이다. "
            "그때는 answer를 사용자에게 전하지 말고 handoff_to가 가리키는 에이전트를 즉시 호출해 "
            "같은 요청을 다시 위임하시오. 두 에이전트가 모두 handoff_to를 채워 돌려준 경우에만 "
            "사용자에게 지원하지 않는다고 답하시오. "
            "handoff_to가 비어 있으면 재위임하지 마시오. handled가 false여도 handoff_to가 없으면 "
            "담당이 아니라서가 아니라 실행이 실패한 것이므로, 위의 error 규칙에 따라 처리하시오. "
            "handoff_to와 handoff_reason은 내부 판단용이므로 사용자에게 그대로 보여주지 마시오.",
            "Nana와 Kana는 사용자에게 보이는 이름이므로 답변에 그대로 써도 된다. "
            "누가 어떤 일을 맡는지 묻는 질문에는 담당자 이름을 밝혀 답하시오. "
            "다만 \"에이전트\", \"tool\", \"limit\" 같은 내부 용어는 쓰지 말고, "
            "네가 하위 담당자 본인인 것처럼 말하지 마시오. "
            "\"Nana 에이전트, 즉 제가 담당합니다\"가 아니라 \"일정 저장은 Nana가 맡습니다\"처럼 답하시오. "
            "하위 답변에 limit 값이나 도구 이름 같은 내부 정보가 섞여 있으면 그대로 옮기지 말고 빼시오.",
            "위의 \"반드시 하위 에이전트를 먼저 호출하라\"는 규칙에는 예외가 하나 있다. "
            "사용자가 무엇을 하려는지 자체를 알 수 없는 요청에는 "
            "nana_agent도 kana_agent도 호출하지 말고 사용자에게 무엇을 원하는지만 되물으시오. "
            "\"일정 좀 도와줘\", \"뭐 좀 물어볼게\", \"안녕\"이 여기에 해당한다. "
            "이런 요청에 하위 에이전트를 호출하면 담당자가 빈 일정을 저장하거나 "
            "묻지 않은 목록을 나열하게 되므로 호출 자체를 하지 않아야 한다. "
            "되물을 때도 일정 정보를 지어내 답해서는 안 되고, 사용자가 답해서 유형이 정해지면 그때 위임하시오.",
        ]
    )


def _tool_call_names(events: list[dict[str, Any]]) -> list[str]:
    return [event["tool_name"] for event in events if event.get("event") == "tool_call" and event.get("tool_name")]


HANDOFF_PREFIX = "HANDOFF:"


def _split_handoff(answer: str) -> tuple[str, str | None, str | None]:
    """하위 agent 답변에서 재위임 표시를 떼어내고 (사용자용 답변, 대상, 이유)로 나눕니다.

    담당이 아닐 때 하위 agent가 마지막 줄에 `HANDOFF: nana_agent | 이유` 형태로 표시합니다.
    답변 문장을 의미로 해석해 "Nana 담당"인지 판단하면 모델이 표현을 조금만 바꿔도
    재위임이 누락되므로, 고정된 토큰 한 줄만 맞추게 했습니다.
    표시 줄은 사용자에게 보일 필요가 없으므로 answer에서 제거합니다.
    """

    lines = [line for line in (answer or "").splitlines()]
    for index in range(len(lines) - 1, -1, -1):
        stripped = lines[index].strip()
        if not stripped.startswith(HANDOFF_PREFIX):
            continue
        body = stripped[len(HANDOFF_PREFIX):].strip()
        target, _, reason = body.partition("|")
        target = target.strip()
        if target not in {"nana_agent", "kana_agent"}:
            continue
        del lines[index]
        return "\n".join(lines).strip(), target, reason.strip() or None
    return (answer or "").strip(), None, None


_RESULT_LIST_KEYS = ("rows", "schedules", "hits", "messages", "deleted", "candidates")


def _result_summary(content: Any) -> dict[str, Any] | None:
    """tool 결과에서 검증에 필요한 요약만 뽑습니다.

    사용자 데이터(제목, 시간, 본문)와 내부 식별자(schedule_id, request_id)는 남기지 않고,
    성공 여부와 개수처럼 "호출됐다"와 "정상 결과를 얻었다"를 구분할 수 있는 값만 남깁니다.
    """

    if not isinstance(content, dict):
        return None

    summary: dict[str, Any] = {}
    if "ok" in content:
        summary["ok"] = bool(content.get("ok"))
    for key in _RESULT_LIST_KEYS:
        value = content.get(key)
        if isinstance(value, list):
            summary[f"{key}_count"] = len(value)

    error = content.get("error")
    if isinstance(error, dict) and error.get("error_type"):
        summary["error_type"] = error["error_type"]
    elif isinstance(error, str) and error and "ok" not in summary:
        summary["ok"] = False

    external_lookup = content.get("external_lookup")
    if isinstance(external_lookup, dict):
        summary["external_lookup_ok"] = bool(external_lookup.get("ok"))

    return summary or None


def _trace_summary(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """하위 agent trace에서 호출 순서와 결과 요약만 남기고 나머지는 버립니다.

    이 값은 tool 반환 JSON에 담겨 supervisor의 LLM 컨텍스트로 들어갑니다.
    원본 event에는 filters, limit, schedule_id, request_id 같은 내부 값과 사용자 데이터가
    들어 있어, supervisor가 answer 대신 그 값을 읽고 답변에 섞는 일이 있었습니다.
    그렇다고 tool 이름만 남기면 그 호출이 성공했는지 실패 payload를 받았는지 확인할 수 없어,
    _result_summary로 ok/개수/error_type 같은 안전한 요약만 함께 남깁니다.
    """

    summarized: list[dict[str, Any]] = []
    for event in events:
        if not event.get("tool_name"):
            continue
        item: dict[str, Any] = {"event": event.get("event"), "tool_name": event.get("tool_name")}
        result = _result_summary(event.get("content"))
        if result:
            item["result"] = result
        summarized.append(item)
    return summarized


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
    # TODO: find_common_available_slots tool description을 자유롭게 작성하세요.
    #   - 이 Python tool이 후보를 계산하지 않는다는 점을 Kana agent에게 분명히 알려야 합니다.
    #     agent가 busy_rows를 읽고 candidate_slots를 직접 채워 넘기게 만드는 것이 핵심입니다.
    #   - candidate_slots 각 항목이 date(YYYY-MM-DD), start_time(HH:MM), end_time(HH:MM),
    #     duration_minutes, reason을 포함해야 한다는 형식을 적습니다.
    #   - 후보는 어떤 busy row와도 겹치면 안 되고, busy_rows도 앞선 tool output에서 복사해 넘기게 합니다.
    #   - 이 결과로 답변을 끝내지 말고 decide_final_slot을 이어서 호출하도록 유도합니다.
    ""
)


DECIDE_FINAL_SLOT_DESCRIPTION = (
    # TODO: decide_final_slot tool description을 자유롭게 작성하세요.
    #   - 이 Python tool이 최종 시간을 자동 선택하지 않는다는 점을 분명히 알려야 합니다.
    #     agent가 selected_index 또는 selected_slot과 final_slot을 직접 골라 넘기게 만듭니다.
    #   - final_slot 형식('YYYY-MM-DD HH:MM-HH:MM')과 needs_agent_selection, reason을 채우는 기준을 적습니다.
    #   - 아직 고르지 않았다면 final_slot은 null, needs_agent_selection은 true로 두게 합니다.
    #   - 근거 trace를 위해 candidate_slots, busy_rows, member_names, date_from/date_to도 함께 넘기게 합니다.
    ""
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

    # TODO: 멤버 이름/날짜 범위를 정규화하고, busy_rows를 수집한 뒤 후보 검증 payload를 만드세요.
    #   - normalize_external_member_names(...)로 멤버 이름을, normalize_date_bound(...)로 날짜를 정규화합니다.
    #   - busy_rows가 None이면 collect_member_schedules.invoke({...})를 호출해 rows를 채웁니다.
    #   - 검증 payload 생성은 find_common_available_slots_payload(...)에 넘깁니다. 이때 내 일정도 근거이므로
    #     member_names에는 "나"를 함께 포함합니다.
    ...


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

    # TODO: find_common_available_slots_dict(...) 결과를 JSON 문자열로 반환하세요.
    ...


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

    # TODO: Kana agent가 고른 최종 시간 정보를 course repo JSON 계약에 맞춰 기록하세요.
    #   - 직접 최종 시간을 고르지 말고 받은 인자를 그대로 decide_final_slot_payload(...)에 넘깁니다.
    #   - 결과를 JSON 문자열로 반환합니다.
    ...


def kana_tools() -> list[Any]:
    return [
        extract_schedule_request,
        search_previous_conversations,
        load_conversation_messages,
        extract_schedules_from_history,
        list_shared_schedules,
        collect_member_schedules,
        # find_common_available_slots, 추가과제
        # decide_final_slot, 추가과제
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

    try:
        result = _NANA_SUBAGENT.invoke({"messages": [{"role": "user", "content": query}]})
        events = extract_agent_events(result)
        answer = extract_final_text(result)
    except Exception as exc:
        return json.dumps(
            {
                "selected_agent": "nana_agent",
                "answer": "",
                "handled": False,
                "handoff_to": None,
                "handoff_reason": None,
                "trace": [],
                "inner_tool_names": [],
                "error": {
                    "user_message": "Nana가 요청을 처리하지 못했습니다. 잠시 후 다시 시도해 주세요.",
                    "error_type": type(exc).__name__,
                    "debug_reason": str(exc),
                },
            },
            ensure_ascii=False,
        )

    answer, handoff_to, handoff_reason = _split_handoff(answer)

    return json.dumps(
        {
            "selected_agent": "nana_agent",
            "answer": answer,
            "handled": handoff_to is None,
            "handoff_to": handoff_to,
            "handoff_reason": handoff_reason,
            "trace": _trace_summary(events),
            "inner_tool_names": _tool_call_names(events),
        },
        ensure_ascii=False,
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

    try:
        result = _KANA_SUBAGENT.invoke({"messages": [{"role": "user", "content": query}]})
        events = extract_agent_events(result)
        answer = extract_final_text(result)
    except Exception as exc:
        return json.dumps(
            {
                "selected_agent": "kana_agent",
                "answer": "",
                "handled": False,
                "handoff_to": None,
                "handoff_reason": None,
                "trace": [],
                "inner_tool_names": [],
                "final_slot_payload": None,
                "final_decision_payload": None,
                "error": {
                    "user_message": "Kana가 요청을 처리하지 못했습니다. 잠시 후 다시 시도해 주세요.",
                    "error_type": type(exc).__name__,
                    "debug_reason": str(exc),
                },
            },
            ensure_ascii=False,
        )

    final_slot_payload = None
    final_decision_payload = None
    for event in events:
        content = event.get("content")
        if isinstance(content, dict):
            if "final_slot" in content:
                final_slot_payload = content
            if content.get("final_decision"):
                final_decision_payload = content["final_decision"]

    answer, handoff_to, handoff_reason = _split_handoff(answer)

    return json.dumps(
        {
            "selected_agent": "kana_agent",
            "answer": answer,
            "handled": handoff_to is None,
            "handoff_to": handoff_to,
            "handoff_reason": handoff_reason,
            "trace": _trace_summary(events),
            "inner_tool_names": _tool_call_names(events),
            "final_slot_payload": final_slot_payload,
            "final_decision_payload": final_decision_payload,
        },
        ensure_ascii=False,
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
