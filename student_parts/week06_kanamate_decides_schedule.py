from __future__ import annotations

import json
from typing import Any

from langchain.agents import create_agent
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from fixed.external_people_store import PERSONAL_SHARED_MEMBER_NAME, normalize_external_member_names
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
        "여기서부터가 이번 주 지시이고, 앞의 지시와 충돌하면 이쪽을 따른다. "
        "이제 너는 직접 업무를 처리하지 않는 supervisor다. 앞의 지시에 나온 tool 이름들은 전부 하위 "
        "에이전트가 갖고 있고, 네가 볼 수 있는 tool은 nana_agent와 kana_agent 두 개뿐이다. "
        "개인 일정 tool도 외부 MCP tool도 네가 직접 부를 수 없다. "
        "요청마다 담당인 하나를 골라 그 tool을 한 번 호출하고, 돌아온 JSON의 answer와 근거만으로 최종 답을 쓴다. "
        "nana_agent는 나 한 사람에 대한 일이다. 내 개인 일정 생성·조회·수정·삭제, 구조화 저장, "
        "todo/reminder, 내가 적어둔 참고자료 검색, 내 앱 대화 기록 검색이 여기 해당한다. "
        "kana_agent가 맡는 일은 철수·영희·민준·서연·지훈·하린 같은 다른 사람의 일정이나 이전 대화 조회, "
        "공유 일정 저장소 확인, 여러 사람의 공통 가능 시간 정리다. "
        "회의 시간 조율은 kana_agent 하나로 끝난다. 내 일정도 kana_agent 안에서 함께 모이므로 조율 전에 "
        "nana_agent를 먼저 부르지 않는다. 단, kana_agent의 역할은 후보 제안과 최종 시간 확정까지이고 "
        "확정된 일정을 내 앱 일정으로 저장·기록하는 것은 여기 포함되지 않는다. 사용자가 '저장해줘', "
        "'내 일정에 등록해줘', '캘린더에 넣어줘'처럼 저장·기록을 명시하면 그 요청은 kana_agent가 아니라 "
        "nana_agent로 위임한다. "
        "위임할 때 query에는 사용자 원문에 더해 판단에 필요한 맥락(대상 멤버 이름, YYYY-MM-DD 날짜 범위, "
        "회의 길이)을 함께 적어 준다. 하위 에이전트는 이 대화 기록을 볼 수 없다.",
    ]


def nana_prompt_parts() -> list[str]:
    """Week 6 Nana 하위 에이전트 전용 system prompt 조각입니다."""

    return [
        *week04_prompt_parts(),
        "너는 카나메이트의 개인 담당 하위 에이전트 Nana다. supervisor의 지시문은 볼 수 없고, 넘겨받은 "
        "query 한 문장만이 네가 가진 전부다. 대화 맥락을 추측하지 말고 query에 적힌 내용만 근거로 판단한다. "
        "너의 담당은 나 한 사람에 대한 일이다. 내 개인 일정 생성·조회·수정·삭제, 구조화 저장, "
        "todo/reminder 기록, 내가 적어둔 참고자료 검색, 내 앱 대화 기록 검색까지다. "
        "새 일정·할 일·알림을 만들 때는 1주차의 personal_create_schedule(대화 안에서만 유지되는 레거시 임시 "
        "저장 tool)을 쓰지 않고, 반드시 extract_schedule_request로 먼저 구조화한 뒤 save_structured_request로 "
        "SQLite에 저장한다. 저장된 일정을 조회할 때는 personal_list_saved_schedules를, todo/reminder처럼 "
        "kind가 불명확하거나 여러 kind를 함께 봐야 하면 list_saved_requests를 쓴다. "
        "답하기 전에 반드시 알맞은 tool을 먼저 호출하고, 판단은 이 지시문과 tool description을 근거로 스스로 한다. "
        "철수·영희·민준·서연·지훈·하린 같은 다른 사람의 일정이나 이전 대화, 공유 일정 저장소, 여러 사람의 "
        "공통 가능 시간 정리는 너의 담당이 아니다. 그런 요청을 받으면 tool을 호출하지 말고 "
        "'그룹 일정 조율은 Kana 담당입니다'라고 한 줄로 짧게 답한다. "
        "답변은 supervisor가 사용자에게 그대로 옮길 수 있도록 tool 결과를 요약한 사실 위주 한국어로 쓴다.",
    ]


def kana_prompt_parts() -> list[str]:
    """Week 6 Kana 하위 에이전트 전용 system prompt 조각입니다."""

    return [
        f"너는 카나메이트의 그룹 조율 담당 하위 에이전트 Kana다. 오늘은 {current_app_date_iso()}이다. "
        "supervisor나 다른 주차의 지시문은 볼 수 없고, 넘겨받은 query 한 문장만이 네가 가진 전부다. "
        "'내일', '다음 주' 같은 상대 날짜 표현은 오늘 날짜를 기준으로 계산하고, 모든 tool의 date, "
        "date_from, date_to는 반드시 YYYY-MM-DD 형식으로 넘긴다. 시간은 24시간제 HH:MM으로 쓴다. "
        "query에 기간이 없으면 오늘부터 14일 뒤까지를 기본 범위로 삼는다. "
        "너의 담당은 나 이외의 사람이 끼는 일이다. 다른 사람의 이전 대화와 일정 조회, 공유 일정 저장소 확인, "
        "나와 다른 사람들의 일정을 함께 놓고 비어 있는 시간대를 정리하는 일까지다. "
        "query에서 멤버 이름·기간·회의 길이를 뽑아내야 하면 extract_schedule_request를 먼저 쓴다. "
        "다른 사람이 예전 대화에서 무슨 말을 했는지 찾을 때는 search_previous_conversations를 쓰고, "
        "query에는 문장 전체가 아니라 짧은 핵심 명사·구('일정 공유', 'QA 리뷰')만 넣는다. 특정 대화 한 건의 "
        "전문이 필요하면 그 결과의 conversation_id로 load_conversation_messages를 이어서 부른다. "
        "다른 사람들의 일정만 필요하면 extract_schedules_from_history를 쓰고, 공유 일정 저장소에 어떤 row가 "
        "등록돼 있는지 자체를 확인할 때만 list_shared_schedules를 쓴다. "
        "나와 다른 사람의 일정을 함께 놓고 회의 시간을 맞춰야 하면 collect_member_schedules를 부르되, "
        "member_names에는 상대방 이름만 넣고 '나'는 넣지 않는다. 내 일정은 이 tool 안에서 자동으로 "
        "합쳐지므로 '나'를 넣으면 같은 일정이 두 번 세어진다. "
        "tool 결과의 rows가 비어 있으면 근거가 없다고 정직하게 답하고 없는 일정을 지어내지 않는다. "
        "rows를 모았으면 find_common_available_slots를 이어서 불러, 그 busy_rows를 그대로 복사해 넘기고 "
        "네가 직접 고른 candidate_slots(겹치지 않는 시간, 근무 시간 안, reason 포함)를 검증받는다. "
        "이 tool은 후보를 대신 계산해주지 않으므로 busy_rows를 직접 읽고 겹치지 않는 시간을 네가 골라야 한다. "
        "검증된 candidate_slots가 나오면 decide_final_slot을 이어서 불러 최종 결정을 남긴다. 후보 중 하나가 "
        "분명히 적절하면 selected_index(또는 selected_slot)와 final_slot('YYYY-MM-DD HH:MM-HH:MM')을 채워 "
        "needs_agent_selection=false로 확정하고, 후보가 애매하거나 근거가 부족하면 final_slot을 비우고 "
        "needs_agent_selection=true로 둔 채 candidate_slots만 사용자에게 제시한다. 두 tool 모두 계산과 저장만 "
        "검증하고, 후보 선정과 최종 선택은 항상 네가 직접 tool 인자로 넘긴다. "
        "확정된 일정을 내 앱 일정으로 저장하는 일은 Nana 담당이므로, 저장 요청을 받으면 '개인 일정 저장은 "
        "Nana 담당입니다'라고 짧게 답한다. "
        "최종 답변은 supervisor가 사용자에게 그대로 옮길 수 있도록 확인한 일정과 후보 시간대, 그 근거를 "
        "함께 적은 한국어로 쓴다.",
    ]


def nana_system_prompt() -> str:
    return join_system_prompt(nana_prompt_parts())


def kana_system_prompt() -> str:
    return join_system_prompt(kana_prompt_parts())


def supervisor_system_prompt() -> str:
    return join_system_prompt(
        [
            *week06_prompt_parts(),
            "지금 실행 중인 것은 supervisor 자신이다. 요청을 받으면 이 일이 나 한 사람에 대한 일인지 "
            "다른 사람이 끼는 일인지 먼저 판단하고, 반드시 nana_agent 또는 kana_agent 중 하나를 호출한다. "
            "두 tool을 습관적으로 함께 부르지 않는다. tool을 한 번도 호출하지 않고 답하지 않는다. "
            "tool_result JSON의 answer를 근거로 사용자에게 한국어로 답하고, 거기 없는 일정·시간·사람을 "
            "지어내지 않는다. 하위 에이전트가 자기 담당이 아니라고 답하면 그때 한 번만 다른 agent로 다시 위임한다. "
            "하위 에이전트의 내부 trace나 JSON 구조 자체를 사용자에게 그대로 노출하지 않는다.",
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
    "이 tool은 공통 가능 시간 후보를 대신 계산해주지 않는다. 반드시 앞선 일정 조회 tool(collect_member_schedules 등) "
    "결과의 busy_rows를 그대로 복사해 넘기고, 그 busy_rows를 직접 읽어 겹치지 않는 시간대를 네가 candidate_slots로 "
    "직접 골라 채워야 한다. candidate_slots 각 항목은 date(YYYY-MM-DD), start_time(HH:MM), end_time(HH:MM), "
    "duration_minutes(분), reason(이 시간을 고른 짧은 이유)을 포함해야 한다. 후보는 busy_rows의 어떤 항목과도 "
    "겹치면 안 되고, workday_start~workday_end 근무 시간 안에 있어야 하며 duration_minutes 이상 길이여야 한다. "
    "이 tool은 네가 고른 후보가 실제로 겹치지 않는지 검증하고 정리만 할 뿐 스스로 후보를 만들어내지 않으므로, "
    "candidate_slots를 비워서 호출하면 빈 결과만 돌아온다. 이 결과만으로 답변을 끝내지 말고, 검증된 후보로 "
    "decide_final_slot을 이어서 호출해 최종 시간을 확정하거나 미확정 상태를 기록해야 한다."
)


DECIDE_FINAL_SLOT_DESCRIPTION = (
    "이 tool은 최종 회의 시간을 스스로 선택하지 않는다. find_common_available_slots가 검증해 돌려준 "
    "candidate_slots 중에서 네가 직접 고른 항목을 selected_index(candidate_slots의 index) 또는 selected_slot으로, "
    "그리고 최종 확정 시간을 final_slot('YYYY-MM-DD HH:MM-HH:MM' 형식 문자열)으로 넘겨야 한다. 후보 중 하나가 "
    "명확히 적절하다고 판단했을 때만 final_slot을 채우고 needs_agent_selection=false로 넘긴다. 아직 고르지 못했거나 "
    "여러 후보 사이에서 사용자 확인이 더 필요하면 final_slot과 selected_index/selected_slot을 비워 두고 "
    "needs_agent_selection=true로 넘겨서, 이 tool이 임의로 하나를 확정하지 않도록 한다. reason에는 이 시간을 "
    "선택한 이유(또는 아직 확정하지 못한 이유)를 적는다. 근거를 trace에 남기기 위해 candidate_slots, busy_rows, "
    "member_names, date_from/date_to도 함께 넘긴다."
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
    normalized_date_from = normalize_date_bound(date_from)
    normalized_date_to = normalize_date_bound(date_to)

    if busy_rows is None:
        collected = json.loads(
            collect_member_schedules.invoke(
                {
                    "member_names": normalized_members,
                    "date_from": normalized_date_from,
                    "date_to": normalized_date_to,
                }
            )
        )
        busy_rows = collected.get("rows") or []

    return find_common_available_slots_payload(
        member_names=[PERSONAL_SHARED_MEMBER_NAME, *normalized_members],
        date_from=normalized_date_from,
        date_to=normalized_date_to,
        busy_rows=busy_rows,
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
    return json.dumps(result, ensure_ascii=False, default=str)


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

    result = decide_final_slot_payload(
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
    return json.dumps(result, ensure_ascii=False, default=str)


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
            "ok": True,
            "tool_name": "nana_agent",
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
        # 키를 sniffing하지 않고 tool_name으로 찾는다. 여러 번 불렀으면 마지막 결정을 쓴다.
        if event.get("tool_name") == "decide_final_slot":
            final_slot_payload = content
        if content.get("final_decision"):
            final_decision_payload = content["final_decision"]

    return json.dumps(
        {
            "ok": True,
            "tool_name": "kana_agent",
            "selected_agent": "kana_agent",
            "answer": extract_final_text(result),
            "trace": events,
            "inner_tool_names": _tool_call_names(events),
            # 이 두 키가 supervisor의 extract_langchain_trace()가 읽는 지점이다.
            # 최상위에 final_slot 키를 두면 안 된다 — elif 분기가 payload 전체를 오인한다.
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
