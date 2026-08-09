from __future__ import annotations

import json
from typing import Any

from langchain.agents import create_agent
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from fixed.external_people_store import (
    PERSONAL_SHARED_MEMBER_NAME,
    normalize_external_member_names,
)
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


WEEK06_SUPERVISOR_PROMPT = (
    "너는 Nana와 Kana를 관리하는 supervisor다. 업무를 직접 처리하지 않고 nana_agent 또는 "
    "kana_agent에 위임한 뒤 돌아온 결과만 근거로 답한다. "
    "내 개인 일정·할 일·알림의 조회/저장/수정/삭제, 개인 참고자료와 앱 대화 기록 검색은 nana_agent 담당이다. "
    "다른 사람의 이전 대화, 외부 멤버 일정, 공유 일정 저장소 조회, 여러 사람의 공통 가능 시간과 "
    "최종 회의 시간 결정은 kana_agent 담당이다. "
    "일정 요청의 판단 기준은 '누구의 일정인가'다. 내 일정만이면 nana_agent, 나 말고 다른 사람 이름이 "
    "하나라도 나오면 내 일정이 함께 필요하든 아니든 kana_agent에 넘긴다. "
    "kana_agent만 다른 사람의 일정을 볼 수 있고, 내 일정까지 함께 모아서 판단하기 때문이다. "
    "'철수 7월 일정 알려줘'처럼 다른 사람 이름이 붙은 단순 조회도 kana_agent 담당이다. "
    "nana_agent는 내 일정만 볼 수 있어서 다른 사람 이름으로 물으면 그런 일정이 없다는 답만 돌아온다. "
    "한 요청에 두 담당이 섞여 있으면 사용자가 지금 받고 싶은 결과를 기준으로 하나만 먼저 호출하고, "
    "그 결과로 답할 수 없을 때만 다른 하나를 이어서 호출한다. "
    "하위 agent가 담당이 아니라고 답하면 같은 agent를 다시 부르지 말고 다른 agent에 위임한다. "
    "하위 agent 결과의 answer를 사용자에게 전달하고, final_slot_payload가 있으면 그 값을 근거로 삼는다. "
    "위임하지 않고 직접 답하지 않는다."
)


WEEK06_NANA_PROMPT = (
    "너는 개인 비서 Nana다. supervisor가 넘긴 요청만 처리한다. "
    "내 개인 일정·할 일·알림의 조회/저장/수정/삭제와 내 개인 참고자료·앱 대화 기록 검색이 담당이다. "
    "다른 사람의 일정, 다른 사람의 이전 대화, 여러 사람의 공통 가능 시간 조율은 담당이 아니다. "
    "담당이 아닌 요청은 추측해서 답하지 말고 Kana 담당이라고 한 줄로 알린다. "
    "가진 tool로 확인한 내용만 근거로 답하고, 조회 결과가 없으면 없다고 답한다."
)


# Kana는 다른 주차 prompt를 누적하지 않으므로, Week 5에서 MCP wrapper tool을 쓰며 알게 된 호출 규칙을
# 여기에 다시 적습니다. 단 Week 5 프롬프트에 있던 "최종 회의 시간 확정은 이번 주차 범위가 아니다"는
# 이번 주차에 Kana가 해야 하는 일이므로 가져오지 않습니다.
WEEK06_KANA_PROMPT = (
    "너는 그룹 일정 조율 담당 Kana다. supervisor가 넘긴 요청만 처리한다. "
    "다른 사람의 일정과 과거 대화는 앱 안에 없고 외부 SQLite/MCP 저장소에 있다. "
    "외부 멤버가 예전에 무슨 말을 했는지 찾을 때는 search_previous_conversations를 쓰고, "
    "찾은 conversation_id의 전체 메시지가 필요하면 load_conversation_messages로 이어서 읽는다. "
    "외부 대화 검색 query는 부분 문자열로만 일치하므로 단어를 붙이지 않고 핵심 단어 하나만 넣고, "
    "멤버를 좁히려면 query가 아니라 member_names를 쓴다. "
    "다른 사람 일정만 필요한 질문은 extract_schedules_from_history를 쓴다. "
    "공유 일정 저장소에 실제로 등록된 row를 확인할 때는 list_shared_schedules를 쓴다. "
    "'나까지 포함해서 언제 비어 있나' 같은 조율 질문은 collect_member_schedules로 내 일정과 외부 멤버 "
    "일정을 한 번에 모아 rows와 schedule_summary를 근거로 답한다. "
    "collect_member_schedules는 member_names에 무엇이 들어와도 내 일정을 항상 함께 모으므로, "
    "다른 사람 일정만 물었으면 rows의 '나' 일정은 답변 근거로 쓰지 않는다. "
    "rows의 time_unspecified가 true인 일정은 시작·종료 시각이 정해지지 않은 일정이다. "
    "그 날이 하루 종일 바쁘다고 단정하지 말고 시간이 미정이라는 점을 밝힌다. "
    "확정한 시간을 내 앱에 개인/그룹 일정으로 저장하는 일은 Nana 담당이므로, 저장이 필요하면 "
    "확정한 시간을 답변에 적고 저장은 Nana 담당이라고 알린다. "
    "내 개인 참고자료나 앱 대화 기록 검색도 Nana 담당이다. "
    "tool로 확인한 내용만 근거로 답하고, 모은 일정이 없으면 없다고 답한다."
)


# 빈 시간 계산은 Kana가 해야 하는 일인데, candidate_slots가 기본값 있는 optional 필드라서 tool
# description만으로는 계속 인자를 비우고 호출했습니다. 그래서 절차를 프롬프트 맨 뒤에 단계로 못박습니다.
WEEK06_KANA_COORDINATION_PROMPT = (
    "여러 사람의 회의 시간을 찾는 요청은 아래 순서를 끝까지 지킨다. 중간에서 멈추지 않는다.\n"
    "1단계. collect_member_schedules로 rows를 모은다.\n"
    "2단계. 빈 시간을 직접 계산한다. 이 계산은 tool이 해주지 않는다. "
    "요청 날짜 범위의 하루하루를 09:00~18:00으로 놓고, 그 날짜의 rows 시간대를 뺀 빈 구간을 구한다. "
    "그 날짜에 해당하는 row가 없으면 그 날은 09:00~18:00 전체가 비어 있다. "
    "time_unspecified가 true인 row가 있는 날은 시각을 알 수 없으니 후보를 만들지 않는다.\n"
    "3단계. 구한 빈 구간에서 요청 길이만큼 끊어 후보를 만들고, find_common_available_slots를 "
    "candidate_slots와 busy_rows를 모두 채워 호출한다. candidate_slots를 비운 호출은 잘못된 호출이다. "
    "예를 들어 7월 14일에 15:00-16:30 일정 하나만 있고 1시간 회의를 찾는다면 "
    'candidate_slots에 {"date": "2026-07-14", "start_time": "09:00", "end_time": "10:00", '
    '"duration_minutes": 60, "reason": "오전에 겹치는 일정이 없음"} 같은 항목들을 넣어 넘긴다.\n'
    "4단계. 검증된 후보 중 하나를 직접 고르고 decide_final_slot으로 최종 시간을 기록한다. "
    "후보가 없어도 이 호출은 한다.\n"
    "후보를 비워 호출해서 결과가 비었다면 그것은 시간이 없다는 뜻이 아니다. 후보를 만들어 다시 호출한다. "
    "'가능한 시간이 없다'는 답은 날짜 범위의 모든 날이 실제로 일정으로 막혀 있을 때만 한다."
)


def week06_system_prompt() -> str:
    """6주차 supervisor agent가 따르는 시스템 프롬프트입니다."""

    return supervisor_system_prompt()


def week06_prompt_parts() -> list[str]:
    """1~6주차 supervisor system prompt 조각을 누적합니다."""

    return [
        *week05_prompt_parts(),
        WEEK06_SUPERVISOR_PROMPT,
    ]


def nana_prompt_parts() -> list[str]:
    """Week 6 Nana 하위 에이전트 전용 system prompt 조각입니다."""

    return [
        *week04_prompt_parts(),
        WEEK06_NANA_PROMPT,
    ]


def kana_prompt_parts() -> list[str]:
    """Week 6 Kana 하위 에이전트 전용 system prompt 조각입니다."""

    return [
        WEEK06_KANA_PROMPT,
        f"오늘 날짜는 {current_app_date_iso()}이다.",
        WEEK06_KANA_COORDINATION_PROMPT,
    ]


def nana_system_prompt() -> str:
    return join_system_prompt(nana_prompt_parts())


def kana_system_prompt() -> str:
    return join_system_prompt(kana_prompt_parts())


def supervisor_system_prompt() -> str:
    return join_system_prompt(
        [
            *week06_prompt_parts(),
            # 누적된 이전 주차 조각에는 tool을 직접 쓰라는 지시가 남아 있습니다. supervisor에게는 그 tool이
            # 없으므로, 위임만 한다는 규칙을 가장 뒤에 한 번 더 못박습니다.
            # (join_system_prompt 헤더가 "뒤에 있는 지시를 우선한다"고 알려 줍니다.)
            (
                "위 안내에 나오는 개별 tool은 하위 agent가 가지고 있고 너에게는 없다. "
                "너는 nana_agent 또는 kana_agent 중 하나를 반드시 먼저 호출하고, 그 결과만 근거로 최종 답변을 만든다. "
                "내 일정이 아니라 다른 사람 이름이 나오는 요청은 조회든 조율이든 kana_agent에 넘긴다."
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
    "회의 시간 후보는 네가 직접 만들어서 candidate_slots에 채워 넣어야 한다. "
    "이 tool은 빈 시간을 찾아주지 않고, 네가 넘긴 후보가 정말 비어 있는지만 검증해 기록한다. "
    "candidate_slots 없이 부르면 검증할 것이 없어 빈 결과가 돌아오므로, 그 결과를 "
    "'가능한 시간이 없다'는 근거로 쓰면 안 된다. "
    "\n"
    "호출 전에 이 순서로 후보를 직접 계산한다. "
    "(1) collect_member_schedules로 받은 rows를 busy_rows에 그대로 복사한다. "
    "이미 모았다면 다시 호출하지 않는다. "
    "(2) date_from부터 date_to까지 하루씩 보면서 workday_start~workday_end(기본 09:00~18:00) 안에서 "
    "그 날짜의 busy row 시간대를 뺀 빈 구간을 찾는다. "
    "그 날짜에 해당하는 busy row가 하나도 없으면 업무 시간 전체가 비어 있다는 뜻이다. "
    "(3) 빈 구간에서 duration_minutes 길이만큼 끊어 후보를 만들고, 이른 날짜·이른 시간 순으로 "
    "최대 limit개까지 candidate_slots에 넣는다. "
    "(4) time_unspecified가 true인 row는 시각을 알 수 없으므로 그 날에는 후보를 만들지 않는다. "
    "\n"
    "candidate_slots의 각 항목은 date('YYYY-MM-DD'), start_time('HH:MM'), end_time('HH:MM'), "
    "duration_minutes(정수 분), reason(이 시간을 고른 짧은 근거)을 모두 채운다. "
    "예: busy row가 '2026-07-14 15:00-16:30' 하나뿐이고 duration_minutes가 60이면 "
    "{\"date\": \"2026-07-14\", \"start_time\": \"09:00\", \"end_time\": \"10:00\", "
    "\"duration_minutes\": 60, \"reason\": \"오전에 겹치는 일정이 없음\"}처럼 넘긴다. "
    "\n"
    "겹치거나 업무 시간을 벗어난 후보는 결과에서 조용히 빠진다. 넘긴 수보다 적게 돌아오면 "
    "어느 후보가 걸렸는지 busy_rows와 다시 비교해 후보를 고쳐 다시 호출한다. "
    "결과를 받으면 답변을 끝내지 말고 이어서 decide_final_slot을 호출해 최종 시간을 기록한다."
)


DECIDE_FINAL_SLOT_DESCRIPTION = (
    "최종 회의 시간은 네가 후보 중에서 직접 골라야 하고, 이 tool은 네 선택을 기록만 한다. "
    "find_common_available_slots를 호출했으면 후보가 있든 없든 반드시 이 tool까지 호출해 결과를 남긴다. "
    "find_common_available_slots가 검증해 돌려준 후보를 candidate_slots에 그대로 넘긴다. "
    "그중 하나를 골랐으면 selected_index(0부터 세는 번호)와 final_slot을 "
    "'YYYY-MM-DD HH:MM-HH:MM' 형식으로 채우고 needs_agent_selection을 false로 둔다. "
    "후보가 없거나 아직 고르지 못했으면 final_slot을 null, needs_agent_selection을 true로 두고 "
    "reason에 왜 확정하지 못했는지 적는다. 확정할 근거가 없을 때 아무 후보나 고르지 않는다. "
    "reason은 사용자에게 그대로 보여줄 설명이므로 어떤 일정 때문에 이 시간을 골랐는지 한 문장으로 적는다. "
    "판단 근거를 남기기 위해 member_names, date_from, date_to, duration_minutes, busy_rows도 함께 넘긴다."
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
    normalized_date_from = normalize_date_bound(date_from)
    normalized_date_to = normalize_date_bound(date_to)

    # agent가 앞선 tool output을 복사해 넘겼으면 그대로 쓰고, 넘기지 않았을 때만 다시 모읍니다.
    rows = busy_rows
    if rows is None:
        collected = json.loads(
            collect_member_schedules.invoke(
                {
                    "member_names": normalized_member_names,
                    "date_from": normalized_date_from,
                    "date_to": normalized_date_to,
                }
            )
        )
        rows = collected.get("rows", [])

    # collect_member_schedules는 member_names와 무관하게 내 일정을 함께 모읍니다. busy_rows에 내 일정이
    # 들어 있으므로 후보 판단 근거에도 "나"를 함께 남깁니다.
    return find_common_available_slots_payload(
        member_names=[
            PERSONAL_SHARED_MEMBER_NAME,
            *[name for name in normalized_member_names if name != PERSONAL_SHARED_MEMBER_NAME],
        ],
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

    # 최종 시간을 이 함수에서 고르지 않습니다. agent가 고른 값을 그대로 계약 payload로 옮깁니다.
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


def _run_subagent(agent: Any, selected_agent: str, query: str) -> dict[str, Any]:
    """하위 agent를 실행하고 supervisor가 읽을 공통 payload로 정리합니다."""

    result = agent.invoke({"messages": [{"role": "user", "content": query}]})
    events = extract_agent_events(result)
    return {
        "ok": True,
        "selected_agent": selected_agent,
        "answer": extract_final_text(result),
        "trace": {"events": events},
        "inner_tool_names": _tool_call_names(events),
    }


def _final_payloads_from_events(
    events: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Kana 하위 trace에서 최종 시간 결정 payload를 끌어올립니다.

    같은 tool이 여러 번 호출될 수 있으므로 마지막 결과를 최종으로 봅니다.
    """

    final_slot_payload: dict[str, Any] | None = None
    final_decision_payload: dict[str, Any] | None = None
    for event in events:
        content = event.get("content")
        if not isinstance(content, dict):
            continue
        # decide_final_slot의 반환 payload는 top-level에 final_slot을 담습니다.
        if "final_slot" in content:
            final_slot_payload = content
        if content.get("final_decision"):
            final_decision_payload = content["final_decision"]
    return final_slot_payload, final_decision_payload


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
    payload = _run_subagent(_NANA_SUBAGENT, "nana_agent", query)
    return json.dumps(payload, ensure_ascii=False, default=str)


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
    payload = _run_subagent(_KANA_SUBAGENT, "kana_agent", query)
    # 하위 agent 안에서 끝난 최종 시간 결정을 supervisor가 다시 계산하지 않고 근거로 쓸 수 있게 올립니다.
    final_slot_payload, final_decision_payload = _final_payloads_from_events(payload["trace"]["events"])
    payload["final_slot_payload"] = final_slot_payload
    payload["final_decision_payload"] = final_decision_payload
    return json.dumps(payload, ensure_ascii=False, default=str)


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
