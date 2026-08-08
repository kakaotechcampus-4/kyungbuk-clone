from __future__ import annotations

import json
import uuid
from contextvars import ContextVar
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
from student_parts.week03_build_nanas_logbook import (
    WRITE_OPERATION_ID,
    WRITE_TURN_INDEX,
    has_fresh_duplicate_confirmation,
)
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

    # 위임 기준은 Week 5까지 배운 라우팅 경계와 같다 — 질문의 주어가 "나"면 Nana,
    # 다른 멤버/그룹이면 Kana. supervisor는 이 판단만 하고 업무는 하위 agent가 한다.
    week06_role = (
        "Week 6부터 너는 직접 업무를 처리하지 않는 supervisor다. "
        "모든 요청은 nana_agent 또는 kana_agent 중 하나를 호출해 위임하고, "
        "그 결과의 answer만 근거로 최종 답변을 만든다. tool을 호출하지 않고 답하지 않는다. "
        "내 개인 일정의 생성/조회/수정/삭제, 할 일/알림 저장, 내 참고자료·내 지난 대화 검색은 nana_agent에 위임한다. "
        "다른 멤버(철수/하린 등)의 일정·대화 조회, 공유 일정 저장소, 여러 사람의 공통 가능 시간과 "
        "최종 회의 시간 결정은 kana_agent에 위임한다. "
        "위임은 한 번에 하나의 하위 agent만 호출하고, 결과를 읽은 뒤에 필요할 때만 다음 위임을 한다. "
        "두 agent를 동시에 호출하지 않는다. "
        "하위 agent가 담당이 아니라고 답하면 같은 query를 반복하지 말고 다른 agent에 위임한다. "
        "저장/삭제 확인 질문에 사용자가 아니라고 거절하면 그 작업을 위임하지 않고 취소됐음을 알린다. "
        "확인 질문에 사용자가 승인하면 위임 query에 '사용자가 확인했다'는 사실과 대상을 명시해 다시 위임한다. "
        "하위 agent 결과가 실패(ok=false)면 user_message만 사용자에게 전하고, "
        "error_type이나 debug_reason 같은 내부 오류 정보는 답변에 옮기지 않는다. "
        "일정 저장/수정/삭제 같은 상태 변경은 사용자가 이번 메시지에서 명시적으로 요청했을 때만 위임한다. "
        "하위 agent가 '저장은 Nana 담당'이라고 답하는 것은 사용자 요청이 아니므로, 그때는 저장하지 말고 "
        "제안된 시간을 알려주며 저장할지 사용자에게 물어본다. "
        "확정된 회의 저장을 위임할 때는 회의 제목뿐 아니라 참석자 이름 전부와 날짜/시간을 query에 그대로 담는다. "
        "다른 멤버가 참석하는 회의인데 시작 시간이 정해지지 않은 요청(예: '서연이랑 다음에 미팅 잡아줘')은 "
        "저장이 아니라 시간 조율이므로 kana_agent에 위임한다. "
        "다른 멤버가 포함된 일정을 nana_agent로 저장 위임하는 것은 날짜와 시작 시간이 모두 정해져 있을 때만이다. "
        "단, 사용자가 시간을 미정으로 둔 채 저장하겠다고 명시하면 조율이 아니라 저장 요청이므로 nana_agent에 위임한다. "
        "하위 agent는 이 대화의 이전 내용을 보지 못하므로, '그 대화', '방금 그 시간'처럼 "
        "앞선 턴을 가리키는 후속 요청을 위임할 때는 필요한 맥락(누구의 무엇인지, 날짜/시간)을 "
        "query에 함께 적어 하위 agent가 그 문장만으로 이해할 수 있게 한다. "
        "위임 query는 사용자 문장을 바꿔 쓰지 말고 원문 그대로 담는다 — 표현을 바꾸면 "
        "사용자의 의도(조율인지 저장인지)가 변형된다. 맥락 보충은 원문 뒤에 덧붙인다. "
        "단, 이전 턴에서 이미 수행된 위임을 다시 보내지 않는다 — 후속 요청(삭제/확정/저장 등)에는 "
        "이번 작업 query 하나만 위임하고, 이전 결과가 필요하면 요청 원문이 아니라 그 결과 내용"
        "(시간·참석자 등)을 맥락으로 덧붙인다. 이전 저장·조율 요청을 재전송하면 같은 작업이 중복 실행된다."
    )
    return [
        *week05_prompt_parts(),
        week06_role,
    ]


def nana_prompt_parts() -> list[str]:
    """Week 6 Nana 하위 에이전트 전용 system prompt 조각입니다."""

    nana_role = (
        "너는 supervisor에게서 개인 업무를 위임받은 Nana 하위 agent다. "
        "내 개인 일정의 생성/조회/수정/삭제, 할 일/알림 저장, 내 참고자료와 내 지난 대화 검색을 담당한다. "
        "다른 멤버의 일정 조회나 여러 사람의 회의 시간 조율을 요청받으면 처리하지 말고 "
        "'그룹 조율은 Kana 담당'이라고 짧게 답한다. "
        "다른 멤버가 참석하는 회의인데 시작 시간이 정해지지 않았다면, 요청 표현이 저장처럼 들려도 그것은 시간 조율 요청이다 — "
        "시간 미정 상태로 저장하지 말고 'Kana 담당'이라고 답한다. "
        "단, 사용자가 시간을 미정으로 두고 저장하겠다고 명시한 경우는 저장 요청이므로 그대로 저장한다. "
        "일정을 저장하기 전 같은 날짜의 기존 일정과 시간이 겹치면 바로 저장하지 말고, "
        "겹치는 일정을 알리고 그래도 저장할지 확인을 구한다. "
        "저장 결과에 duplicate_warning/overlap_warning이 있으면 저장된 것이 아니다 — 기존/겹치는 "
        "일정을 알리고 별개 일정이 맞는지(겹쳐도 되는지) 확인 질문으로 답한다. "
        "삭제 결과에 pending_confirmation이 있으면 삭제된 것이 아니다 — 대상 목록을 보여주고 "
        "모두 삭제할지 확인 질문으로 답한다. 확인 질문에 사용자가 거절하면 실행하지 않는다."
    )
    return [
        *week04_prompt_parts(),
        nana_role,
    ]


def kana_prompt_parts() -> list[str]:
    """Week 6 Kana 하위 에이전트 전용 system prompt 조각입니다."""

    # Kana는 다른 주차 prompt를 누적하지 않으므로, Week 5에서 검증으로 배운 외부 tool
    # 사용 규칙(내 이름은 '나', 기간 미지정은 전체 범위, LIKE query는 키워드 하나)을
    # 여기서 다시 명시해야 한다. 하위 agent는 supervisor prompt를 공유하지 않는다.
    today = current_app_date_iso()
    kana_role = (
        "너는 다른 멤버들의 기록과 그룹 일정 조율을 담당하는 Kana 하위 agent다. "
        f"오늘은 {today}이고, 상대 날짜는 이 날짜를 기준으로 해석한다. "
        "다른 멤버의 이전 대화는 search_previous_conversations로 찾고, 원문이 필요하면 검색 rows의 "
        "conversation_id로 load_conversation_messages를 호출한다. search의 query에는 핵심 단어 하나만 넣고, "
        "멤버의 대화 목록이 목적이면 query를 빈 문자열로 두고 member_names만 넣는다. "
        "멤버들의 일정·바쁜 시간은 extract_schedules_from_history로, 나를 포함한 병합은 collect_member_schedules로 "
        "조회한다(내 일정은 항상 포함된다). 공유 일정 저장소 조회는 list_shared_schedules를 쓰고, "
        "저장소에서 사용자 본인의 일정은 member_name '나'로 저장되어 있으므로 필터에도 '나'를 쓴다. "
        "기간을 지정하지 않은 일정 조회는 충분히 넓은 범위(2000-01-01~2099-12-31)로 한다."
    )
    kana_slot_flow = (
        "여러 사람의 회의 시간을 정하는 요청은 이 순서를 따른다: "
        "① collect_member_schedules로 나와 멤버들의 busy-time rows를 모은다. "
        "② rows 중 calculable이 true인 일정과 겹치지 않는 후보 시간을 네가 직접 골라 "
        "find_common_available_slots에 candidate_slots와 busy_rows로 넘겨 검증한다. "
        "③ 검증된 후보에서 최종 시간을 네가 직접 골라 decide_final_slot에 넘겨 기록한 뒤 그 결과로 답한다. "
        "tool은 후보나 최종 시간을 대신 골라주지 않는다 — 고르는 것은 네 역할이다. "
        "검색 결과와 tool 결과만 근거로 답하고, 기록에 없는 일정을 지어내지 않는다. "
        "너는 일정을 저장할 수 없고 저장은 사용자가 원할 때만 이루어진다. 시간을 제안한 뒤에는 "
        "'이 시간으로 저장할까요?'처럼 사용자의 확인을 구하는 문장으로 끝내고, 저장을 지시하는 문장은 쓰지 않는다."
    )
    return [kana_role, kana_slot_flow]


def nana_system_prompt() -> str:
    return join_system_prompt(nana_prompt_parts())


def kana_system_prompt() -> str:
    return join_system_prompt(kana_prompt_parts())


def supervisor_system_prompt() -> str:
    return join_system_prompt(
        [
            *week06_prompt_parts(),
            "너는 supervisor다. 반드시 nana_agent 또는 kana_agent 중 하나를 먼저 호출하고, "
            "돌아온 answer를 근거로만 답한다. Week 5까지의 검색/저장 tool 지시는 하위 agent가 "
            "따르는 것이므로 너는 그 tool들을 직접 호출할 수 없다.",
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
    delegated_agents: list[str] = []
    worked_agent: str | None = None

    for event in events:
        if event.get("event") == "tool_call" and event.get("tool_name") in {"nana_agent", "kana_agent"}:
            delegated_agents.append(event["tool_name"])
        content = event.get("content")
        if isinstance(content, dict):
            inner_tool_names.extend(content.get("inner_tool_names") or [])
            # 병렬/다중 위임에서 "마지막 호출"은 거절만 한 agent일 수 있다(앱 검증에서 재현).
            # 실제로 tool을 실행한 agent를 선택된 agent로 본다.
            if content.get("inner_tool_names") and content.get("selected_agent"):
                worked_agent = content["selected_agent"]
            if content.get("final_slot_payload"):
                final_slot_payload = content["final_slot_payload"]
            elif "final_slot" in content:
                final_slot_payload = content
            if content.get("final_decision_payload"):
                final_decision_payload = content["final_decision_payload"]

    return {
        "events": events,
        "supervisor_selected_agent": worked_agent or (delegated_agents[-1] if delegated_agents else None),
        "delegated_agents": delegated_agents,
        "inner_tool_names": inner_tool_names,
        "final_slot_payload": final_slot_payload,
        "final_decision_payload": final_decision_payload,
    }


def tool_name(tool_object: Any) -> str:
    return getattr(tool_object, "name", getattr(tool_object, "__name__", str(tool_object)))


FIND_COMMON_AVAILABLE_SLOTS_DESCRIPTION = (
    "네가 직접 고른 공통 가능 시간 후보를 검증하는 tool입니다. 이 tool은 후보를 대신 "
    "계산해 주지 않습니다 — busy_rows를 읽고 어떤 busy row와도 겹치지 않는 후보를 네가 "
    "골라 candidate_slots로 넘겨야 합니다. 각 후보는 date(YYYY-MM-DD), start_time(HH:MM), "
    "end_time(HH:MM), duration_minutes, reason을 포함합니다. busy_rows에는 앞선 "
    "collect_member_schedules 결과의 rows를 그대로 복사해 넘깁니다(안 넘기면 tool이 다시 "
    "수집합니다). 검증 결과로 답변을 끝내지 말고, 이어서 decide_final_slot을 호출해 최종 "
    "시간을 기록한 뒤 답합니다."
)


DECIDE_FINAL_SLOT_DESCRIPTION = (
    "네가 고른 최종 회의 시간을 기록하는 tool입니다. 이 tool은 최종 시간을 자동으로 "
    "선택해 주지 않습니다 — 검증된 candidate_slots에서 selected_index(또는 selected_slot)와 "
    "final_slot을 네가 직접 골라 넘겨야 합니다. final_slot 형식은 'YYYY-MM-DD HH:MM-HH:MM'이고, "
    "선택 근거를 reason에 사용자에게 보여줄 문장으로 적습니다. 아직 고를 수 없으면 "
    "final_slot을 null로 두고 needs_agent_selection을 true로 넘깁니다. 근거 trace가 남도록 "
    "candidate_slots, busy_rows, member_names, date_from, date_to도 함께 넘깁니다."
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

    normalized_members = normalize_external_member_names(member_names)
    normalized_from = normalize_date_bound(date_from)
    normalized_to = normalize_date_bound(date_to)

    # busy_rows를 안 넘긴 호출은 Week 5 collect로 직접 수집한다. Kana가 앞 단계 결과를
    # 복사해 넘기는 것이 기본 경로지만, 빠뜨려도 빈 근거로 검증하지 않게 하기 위해서다.
    # 리뷰 반영: 수집이 실패했는데 rows가 비었다는 이유로 검증이 진행되면
    # "근거를 조회하지 못함"이 "모두 한가함"으로 해석된다. 실패는 실패 payload로
    # 보존해 후보 검증으로 넘어가지 않고, 외부 조회만 실패한 부분 성공은
    # data_incomplete로 표시해 이 근거로 최종 시간을 확정하지 않게 안내한다.
    collection_warning: str | None = None
    if busy_rows is None:
        try:
            collected = json.loads(
                collect_member_schedules.invoke(
                    {"member_names": normalized_members, "date_from": normalized_from, "date_to": normalized_to}
                )
            )
        except (TypeError, ValueError):
            collected = None
        if not isinstance(collected, dict) or collected.get("ok") is not True or not isinstance(collected.get("rows"), list):
            reason = (collected or {}).get("error") if isinstance(collected, dict) else "응답을 해석할 수 없습니다"
            return {
                "ok": False,
                "error": f"busy-time 수집에 실패해 후보를 검증할 수 없습니다: {reason}",
                "retry_hint": (
                    "collect_member_schedules를 다시 호출해 busy_rows를 확보한 뒤 재시도하세요. "
                    "수집 실패를 '모두 한가함'으로 해석하면 안 됩니다."
                ),
            }
        busy_rows = collected.get("rows") or []
        if collected.get("external_error"):
            collection_warning = (
                f"외부 멤버 일정 조회가 일부 실패했습니다({collected['external_error']}). "
                "이 근거는 불완전하므로 최종 시간을 확정하지 말고 사용자에게 알리세요."
            )

    # 실제 겹침 검증과 payload 정리는 fixed가 맡는다. 내 일정도 근거이므로 "나"를 포함한다.
    payload = find_common_available_slots_payload(
        member_names=["나", *[name for name in normalized_members if name != "나"]],
        date_from=normalized_from,
        date_to=normalized_to,
        busy_rows=busy_rows,
        duration_minutes=duration_minutes,
        workday_start=workday_start,
        workday_end=workday_end,
        limit=limit,
        candidate_slots=candidate_slots,
        llm_reason=llm_reason,
    )
    if collection_warning:
        payload["data_incomplete"] = True
        payload["warning"] = collection_warning
    return payload


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

    # tool은 인자를 dict helper에 넘기는 입구 역할만 한다(이전 주차와 같은 패턴).
    result = find_common_available_slots_dict(
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
    payload = {"ok": True, "tool_name": "find_common_available_slots", **result}
    # 앱 검증에서 재현된 실패: Kana가 후보를 고르지 않고 빈 candidate_slots로 호출한 뒤
    # 빈 결과를 "가능 시간 없음"으로 해석했다(빈 날이 많은데도). 이 tool은 후보를
    # 계산하지 않으므로, 후보 없이 온 호출에는 교정 방법을 결과에 실어 보낸다
    # (Week 4 coverage / Week 5 retry_hint와 같은 결과-내-안내 방식).
    if not candidate_slots:
        payload["retry_hint"] = (
            "candidate_slots가 비어 있습니다. 이 tool은 후보를 대신 계산하지 않습니다 — "
            "busy_rows와 겹치지 않는 빈 시간대를 네가 직접 골라 candidate_slots에 채워 "
            "다시 호출하세요. busy가 없는 날은 업무 시간 내 어느 시간이든 후보가 될 수 있습니다. "
            "빈 결과를 '가능 시간 없음'으로 해석하면 안 됩니다."
        )
    elif not payload.get("candidate_slots"):
        payload["retry_hint"] = (
            "제안한 후보가 모두 busy와 겹쳐 탈락했습니다. busy_rows를 다시 읽고 겹치지 않는 "
            "다른 시간대로 후보를 골라 재호출하세요."
        )
    return json.dumps(payload, ensure_ascii=False)


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

    # 리뷰 반영: 선택 필드 사이의 정합성을 검증한다. source of truth는
    # selected_slot/selected_index이고, final_slot 문자열은 그 후보에서 코드가
    # 만들거나 일치 여부를 확인한다. 모순된 조합(선택과 다른 final_slot,
    # final_slot이 있는데 needs_agent_selection=true)은 성공으로 기록하지 않고
    # 실패 payload로 돌려보내 재호출을 유도한다.
    slots = [slot.model_dump() if hasattr(slot, "model_dump") else dict(slot) for slot in candidate_slots or []]

    def _slot_text(slot: dict[str, Any]) -> str:
        return f"{slot.get('date')} {slot.get('start_time')}-{slot.get('end_time')}"

    def _mismatch(reason: str) -> str:
        return json.dumps(
            {
                "ok": False,
                "tool_name": "decide_final_slot",
                "error": f"선택 필드가 서로 모순됩니다: {reason}",
                "retry_hint": (
                    "candidate_slots에서 고른 후보의 selected_index(또는 selected_slot)만 넘기세요. "
                    "final_slot은 그 후보에서 만들어지므로 다른 시간을 넣지 말고, "
                    "최종 시간을 확정했다면 needs_agent_selection은 false여야 합니다."
                ),
            },
            ensure_ascii=False,
        )

    chosen: dict[str, Any] | None = None
    if selected_slot is not None:
        chosen = selected_slot.model_dump() if hasattr(selected_slot, "model_dump") else dict(selected_slot)
        if slots and _slot_text(chosen) not in [_slot_text(slot) for slot in slots]:
            return _mismatch(f"selected_slot({_slot_text(chosen)})이 candidate_slots에 없습니다")
    elif selected_index is not None:
        if not (0 <= selected_index < len(slots)):
            return _mismatch(f"selected_index={selected_index}가 후보 {len(slots)}개 범위를 벗어납니다")
        chosen = slots[selected_index]

    if chosen is not None:
        expected_slot = _slot_text(chosen)
        if final_slot and final_slot.strip() != expected_slot:
            return _mismatch(f"선택한 후보는 {expected_slot}인데 final_slot은 {final_slot}입니다")
        if needs_agent_selection:
            return _mismatch("후보를 선택했는데 needs_agent_selection=true입니다")
        final_slot = expected_slot
        needs_agent_selection = False
    elif final_slot:
        return _mismatch("후보 선택(selected_index/selected_slot) 없이 final_slot만 넘어왔습니다")

    # 최종 시간을 여기서 고르지 않는다 — Kana가 고른 값을 그대로 계약 payload로 만든다.
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
    return json.dumps({"ok": True, "tool_name": "decide_final_slot", **payload}, ensure_ascii=False)


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


# 같은 대화에서 상태 변경(저장/삭제/등록)에 "성공"한 위임 query를 기억한다.
# supervisor가 후속 턴에서 이전 요청 원문을 재전송해 같은 저장이 중복 실행되는 실패가
# 프롬프트 수정 4회로도 재발해, 합의된 에스컬레이션 기준대로 코드 게이트로 승격했다.
# 읽기 전용 위임의 반복은 막지 않는다.
# 리뷰 반영 2건:
#  - 호출 여부가 아니라 tool 결과의 실제 성공 증거(ok + saved_rows/deleted_count 등)를
#    확인한 뒤에만 기억한다. 실패한 쓰기는 실행이 아니므로 같은 query 재시도를 막지 않는다.
#  - "사용자가 지금 같은 문장을 다시 요청"한 정상 반복과 "과거 요청의 잘못된 재전송"을
#    구분한다. 현재 사용자 메시지(턴마다 runner가 기록)와 위임 query가 겹치면 사용자의
#    현재 의사이므로 허용하고, 겹치지 않는 과거 query의 재등장만 차단한다.
_WRITE_TOOL_NAMES = {
    "save_structured_request", "personal_create_schedule", "personal_update_saved_schedule",
    "personal_delete_saved_schedules", "create_shared_schedule", "delete_shared_schedule",
    "add_personal_reference",
}
# 쓰기 tool별 "상태가 실제로 바뀌었다"는 결과 증거 필드.
_WRITE_EVIDENCE_KEYS = ("saved_rows", "deleted_count", "updated_schedule", "created_schedule", "shared_schedule", "reference")
# 대화별로 "성공한 쓰기 query → 그 쓰기를 수행한 operation_id"를 기록한다.
# operation_id는 사용자 턴마다 runner(코드)가 발급한다 — LLM에게 ID 발급을 맡기면
# 다시 프롬프트 의존이 되므로, 멘토 리뷰의 operation 단위 구분을 코드 발급으로 구현했다.
_EXECUTED_WRITE_QUERIES: dict[str, dict[str, str]] = {}
_LEDGER_LOADED = False


def _ledger_path() -> Any:
    from fixed.config import CONFIG

    return CONFIG.app_db_path.parent / "week06_write_ledger.json"


def _load_ledger() -> None:
    """앱 재시작 후에도 replay 차단이 유지되도록 장부를 파일에서 되살립니다.

    메모리 장부만 있으면 재시작 뒤 '저장 후 삭제된 내용'의 replay(부활)를 내용
    검사가 못 잡는다(DB에 없으니 새 저장으로 보임). 성공한 쓰기 query를
    write-through로 영속시켜 이 구멍을 막는다.
    """

    global _LEDGER_LOADED
    if _LEDGER_LOADED:
        return
    _LEDGER_LOADED = True
    try:
        stored = json.loads(_ledger_path().read_text(encoding="utf-8"))
        if isinstance(stored, dict):
            for conversation, queries in stored.items():
                if isinstance(queries, dict):
                    _EXECUTED_WRITE_QUERIES.setdefault(conversation, {}).update(queries)
    except (OSError, ValueError):
        pass


def _persist_ledger() -> None:
    try:
        _ledger_path().write_text(
            json.dumps(_EXECUTED_WRITE_QUERIES, ensure_ascii=False), encoding="utf-8"
        )
    except OSError:
        pass
_CURRENT_USER_MESSAGE: ContextVar[str] = ContextVar("week06_current_user_message", default="")
_CURRENT_OPERATION_ID: ContextVar[str] = ContextVar("week06_current_operation_id", default="")


def _write_succeeded(content: dict[str, Any]) -> bool:
    """쓰기 tool 결과가 실제 상태 변경 성공인지 증거 필드로 판정합니다."""

    if content.get("ok") is not True:
        return False
    for key in _WRITE_EVIDENCE_KEYS:
        value = content.get(key)
        if isinstance(value, int):
            if value > 0:
                return True
        elif value:
            return True
    return False


def _replayed_write_query(query: str) -> bool:
    """이미 성공한 쓰기 query의 중복 실행인지 operation_id로 판정합니다.

    - 같은 operation(같은 사용자 턴)에서 같은 쓰기 query가 다시 오면 무조건 중복이다.
      한 턴의 요청 하나는 같은 상태 변경을 한 번만 수행해야 한다.
    - 다른 operation(이전 턴)에서 성공한 query가 다시 오면 재전송(replay)으로 차단하되,
      현재 사용자 메시지가 그 query와 겹치면(같은 문장을 지금 다시 요청) 사용자의
      새로운 의사이므로 허용한다.
    """

    from fixed.session_scope import current_session_scope

    _load_ledger()
    executed_op = _EXECUTED_WRITE_QUERIES.get(current_session_scope(), {}).get(query.strip())
    if executed_op is None:
        return False
    if executed_op == _CURRENT_OPERATION_ID.get():
        return True
    current_message = _CURRENT_USER_MESSAGE.get().strip()
    if current_message and (query.strip() in current_message or current_message in query.strip()):
        return False
    # 바로 이전 턴에 tool 계층이 중복 확인 경고를 발행했다면, 이번 턴의 재위임은
    # 사용자 확인에 대한 후속일 수 있으므로 통과시킨다(층간 충돌 방지 — 앱 검증에서
    # 확인된 저장이 이 게이트에 막히는 실패가 재현됨). 최종 판정은 tool 계층의
    # op 멱등성·중복 확인이 한다.
    if has_fresh_duplicate_confirmation(current_session_scope(), WRITE_TURN_INDEX.get()):
        return False
    return True


def _remember_write_query(query: str, events: list[dict[str, Any]]) -> None:
    """상태 변경에 실제로 성공한 위임 query를 현재 operation_id와 함께 기억합니다."""

    from fixed.session_scope import current_session_scope

    for event in events:
        if event.get("event") != "tool_result" or event.get("tool_name") not in _WRITE_TOOL_NAMES:
            continue
        content = event.get("content")
        if isinstance(content, dict) and _write_succeeded(content):
            _load_ledger()
            _EXECUTED_WRITE_QUERIES.setdefault(current_session_scope(), {})[query.strip()] = _CURRENT_OPERATION_ID.get()
            _persist_ledger()
            return


def _replay_refusal_payload(agent_tool: str, query: str) -> str:
    """중복 재실행을 막고 supervisor가 읽고 교정할 수 있는 payload를 돌려줍니다."""

    return json.dumps(
        {
            "ok": False,
            "selected_agent": agent_tool,
            "error": (
                "이 query는 이 대화에서 이미 저장/삭제 등 상태 변경을 수행한 요청과 동일해 "
                "재실행하지 않았습니다. 이전 요청을 다시 보내지 말고, 이번에 필요한 작업만 "
                "새 query로 위임하세요."
            ),
        },
        ensure_ascii=False,
    )


def _run_subagent(agent: Any, query: str) -> tuple[str, list[dict[str, Any]]]:
    """하위 agent를 한 번 실행하고 (answer, trace events)를 반환합니다."""

    result = agent.invoke({"messages": [{"role": "user", "content": query}]})
    return extract_final_text(result), extract_agent_events(result)


def _subagent_error_payload(agent_tool: str, exc: Exception) -> str:
    """하위 agent 실행 실패를 supervisor가 읽고 대응할 수 있는 실패 payload로 바꿉니다.

    예외를 그대로 전파하면 supervisor 턴 전체가 죽지만, 실패 payload면 supervisor가
    실패 사실을 사용자에게 설명하거나 다른 방식으로 우회할 수 있다(Week 5 soft-fail과 같은 규칙).
    리뷰 반영: str(exc)에는 내부 경로·API 메시지가 섞일 수 있어 사용자용 문구(user_message)와
    디버그 정보(error_type/debug_reason)를 분리한다. 사용자 답변에는 user_message만 쓰게
    supervisor prompt가 지시하고, 상세 원인은 trace(tool 결과)에 보존된다.
    """

    agent_label = "Nana" if agent_tool == "nana_agent" else "Kana"
    return json.dumps(
        {
            "ok": False,
            "selected_agent": agent_tool,
            "user_message": f"{agent_label} 하위 에이전트 처리 중 문제가 발생했습니다. 잠시 후 다시 시도해 주세요.",
            "error_type": type(exc).__name__,
            "debug_reason": str(exc),
        },
        ensure_ascii=False,
    )


@tool(args_schema=AgentQueryInput)
def nana_agent(query: str) -> str:
    """개인 일정과 개인 RAG 작업을 프롬프트 기반 Nana 하위 에이전트에게 위임합니다."""

    global _NANA_SUBAGENT
    if _NANA_SUBAGENT is None:
        _NANA_SUBAGENT = create_agent(
            model=chat_model(), tools=week04_tools(), system_prompt=nana_system_prompt()
        )
    if _replayed_write_query(query):
        return _replay_refusal_payload("nana_agent", query)
    try:
        answer, events = _run_subagent(_NANA_SUBAGENT, query)
    except Exception as exc:
        return _subagent_error_payload("nana_agent", exc)
    _remember_write_query(query, events)
    return json.dumps(
        {
            "ok": True,
            "selected_agent": "nana_agent",
            "answer": answer,
            "trace": {"events": events},
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
            model=chat_model(), tools=kana_tools(), system_prompt=kana_system_prompt()
        )
    if _replayed_write_query(query):
        return _replay_refusal_payload("kana_agent", query)
    try:
        answer, events = _run_subagent(_KANA_SUBAGENT, query)
    except Exception as exc:
        return _subagent_error_payload("kana_agent", exc)
    _remember_write_query(query, events)

    # 하위 trace를 훑어 최종 시간 결정 payload를 top-level로 끌어올린다.
    # supervisor와 UI가 하위 event 전체를 뒤지지 않고도 최종 결정을 읽을 수 있게 하기 위해서다.
    final_slot_payload: dict[str, Any] | None = None
    final_decision_payload: dict[str, Any] | None = None
    for event in events:
        content = event.get("content")
        if not isinstance(content, dict):
            continue
        if event.get("event") == "tool_result" and "final_slot" in content:
            final_slot_payload = content
        if content.get("final_decision"):
            final_decision_payload = content["final_decision"]
    return json.dumps(
        {
            "ok": True,
            "selected_agent": "kana_agent",
            "answer": answer,
            "trace": {"events": events},
            "inner_tool_names": _tool_call_names(events),
            "final_slot_payload": final_slot_payload,
            "final_decision_payload": final_decision_payload,
        },
        ensure_ascii=False,
    )


class _SupervisorRunner:
    """supervisor 실행을 감싸 현재 사용자 메시지를 턴 범위로 기록합니다.

    멘토 리뷰의 operation_id 구조: 사용자 턴마다 runner(코드)가 operation_id를
    발급하고, 쓰기 게이트가 성공한 쓰기를 그 ID와 함께 저장해 중복을 판정한다.
    같은 operation 안의 같은 쓰기 재위임은 무조건 차단되고(한 요청 = 한 번 실행),
    이전 operation의 query 재등장은 현재 사용자 메시지와 겹칠 때만(의도적 반복) 허용된다.
    LLM이 아니라 invoke/stream 경계에서 ID를 만들므로 프롬프트 의존이 없다.
    """

    def __init__(self, agent: Any) -> None:
        self._agent = agent
        # 턴 순번은 대화별로 센다. 러너 전역이면 다른 대화가 한 턴 끼기만 해도
        # 확인 창(경고 바로 다음 턴)이 소모되는 오탐이 난다.
        self._turn_index_by_conversation: dict[str, int] = {}

    @staticmethod
    def _next_operation_id() -> str:
        # 재시작 후에도 과거 영속 기록과 충돌하지 않도록 전역 고유 ID를 쓴다.
        return f"op_{uuid.uuid4().hex[:12]}"

    @staticmethod
    def _last_user_content(payload: Any) -> str:
        messages = payload.get("messages") if isinstance(payload, dict) else None
        last = messages[-1] if messages else None
        if isinstance(last, dict):
            return str(last.get("content") or "")
        return str(getattr(last, "content", "") or "")

    def invoke(self, payload: Any, **kwargs: Any) -> Any:
        operation_id = self._next_operation_id()
        from fixed.session_scope import current_session_scope
        conversation = current_session_scope()
        turn = self._turn_index_by_conversation.get(conversation, 0) + 1
        self._turn_index_by_conversation[conversation] = turn
        message_token = _CURRENT_USER_MESSAGE.set(self._last_user_content(payload))
        op_token = _CURRENT_OPERATION_ID.set(operation_id)
        write_token = WRITE_OPERATION_ID.set(operation_id)
        turn_token = WRITE_TURN_INDEX.set(turn)
        try:
            return self._agent.invoke(payload, **kwargs)
        finally:
            WRITE_TURN_INDEX.reset(turn_token)
            WRITE_OPERATION_ID.reset(write_token)
            _CURRENT_OPERATION_ID.reset(op_token)
            _CURRENT_USER_MESSAGE.reset(message_token)

    def stream(self, payload: Any, **kwargs: Any) -> Any:
        operation_id = self._next_operation_id()
        from fixed.session_scope import current_session_scope
        conversation = current_session_scope()
        turn = self._turn_index_by_conversation.get(conversation, 0) + 1
        self._turn_index_by_conversation[conversation] = turn
        message_token = _CURRENT_USER_MESSAGE.set(self._last_user_content(payload))
        op_token = _CURRENT_OPERATION_ID.set(operation_id)
        write_token = WRITE_OPERATION_ID.set(operation_id)
        turn_token = WRITE_TURN_INDEX.set(turn)
        try:
            yield from self._agent.stream(payload, **kwargs)
        finally:
            WRITE_TURN_INDEX.reset(turn_token)
            WRITE_OPERATION_ID.reset(write_token)
            _CURRENT_OPERATION_ID.reset(op_token)
            _CURRENT_USER_MESSAGE.reset(message_token)


def build_langchain_supervisor_agent() -> object:
    """nana_agent와 kana_agent 위임 도구만 노출하는 LangChain v1 슈퍼바이저입니다."""

    global _SUPERVISOR_AGENT
    if _SUPERVISOR_AGENT is None:
        _SUPERVISOR_AGENT = _SupervisorRunner(
            create_agent(
                model=chat_model(),
                tools=supervisor_tools(),
                system_prompt=supervisor_system_prompt(),
            )
        )
    return _SUPERVISOR_AGENT


def build_week_agent() -> object:
    """active-week registry가 호출하는 표준 Week agent builder입니다."""

    return build_langchain_supervisor_agent()
