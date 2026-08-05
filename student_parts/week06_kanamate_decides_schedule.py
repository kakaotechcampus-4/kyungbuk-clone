from __future__ import annotations

import json
from typing import Any

from langchain.agents import create_agent
from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator

from fixed.external_people_store import PERSONAL_SHARED_MEMBER_NAME, normalize_external_member_names
from fixed.langchain_trace import extract_agent_events, extract_final_text
from fixed.llm import chat_model
from fixed.runtime_clock import current_app_date_iso
from fixed.schedule_decision import (
    CommonSlotCandidate,
    busy_rows_overlap,
    date_range,
    decide_final_slot_payload,
    find_common_available_slots_payload,
    format_time_minutes,
    normalize_date_bound,
    parse_time_minutes,
)
from student_parts.week01_wake_up_nana import join_system_prompt
from student_parts.week02_structure_natural_language_requests import extract_schedule_request
from student_parts.week04_retrieve_nanas_memory import week04_prompt_parts, week04_tools
from student_parts.week05_load_kanas_past_conversations import (
    collect_member_schedules,
    # alias 중복 제거 규칙은 Week 5에 이미 한 벌 있다. 같은 판정을 여기서 다시 쓰면 두 payload의
    # member 목록 계약이 갈라질 수 있어 helper를 그대로 가져와 재사용한다.
    dedupe_preserving_order,
    extract_schedules_from_history,
    json_payload,
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


# 시간 정보가 불완전한 row를 어떻게 다루는지 Kana agent와 사람이 같은 문장을 읽게 상수로 둔다.
# Week 5에서 row마다 time_status(complete/start_only/date_only)를 붙여 뒀고, Week 6은 그 계약을
# 그대로 이어받아 "시간 계산에 쓸 수 있는 row"와 "그날 일정이 있다는 근거로만 쓸 row"를 구분한다.
WEEK06_TIME_STATUS_PROMPT = (
    "collect_member_schedules가 돌려주는 busy row에는 time_status가 붙어 있고, 값에 따라 쓰는 방법이 다르다. "
    "complete는 시작·종료 시간이 모두 있어 겹침 계산에 그대로 쓴다. "
    "start_only는 시작 시간만 있어 종료를 알 수 없으므로 그 시각부터 회의 길이만큼은 막힌 것으로 보고 후보를 겹치지 않게 고른다. "
    "date_only는 시간을 전혀 몰라서 특정 시각을 막는 근거로 쓸 수 없다. 그날을 통째로 지우지 말고, "
    "후보는 정상적으로 제안하되 '그날 시간 미정 일정이 있어 확정 전에 확인이 필요하다'를 답변에 함께 밝힌다. "
    "시간을 모른다는 이유로 없는 시간을 만들어 내거나, 반대로 그날 전체를 불가능하다고 단정하지 않는다."
)

WEEK06_SUPERVISOR_PROMPT = (
    "Week 6부터 이 agent는 업무를 직접 처리하지 않는 supervisor다. "
    "가진 tool은 nana_agent와 kana_agent 두 개뿐이고, 일정 조회·저장·검색·조율에 필요한 실제 tool은 하위 agent만 갖고 있다. "
    "그래서 사용자 요청을 받으면 먼저 담당을 정하고, 반드시 nana_agent 또는 kana_agent를 호출한 뒤 그 결과만 근거로 답한다. "
    "하위 agent를 부르지 않고 내 지식이나 이전 대화 기억만으로 일정 내용을 답하지 않는다.\n"
    "Nana 담당(nana_agent): 나 혼자의 일 전부다. 내 개인 일정 조회·생성·수정·삭제, 할 일과 알림 저장, "
    "개인 참고자료 검색, 이 앱에서 나와 나눈 대화 검색(RAG), 조율이 끝난 회의를 내 일정으로 저장하는 일.\n"
    "Kana 담당(kana_agent): 나 이외의 사람이 끼는 일 전부다. 외부 멤버의 지난 대화 검색과 원문 확인, "
    "외부 멤버 일정 추출, 공유 일정 저장소 row 조회, 여러 사람의 공통 가능 시간 후보 검증과 최종 회의 시간 결정.\n"
    "판단 기준은 '요청에 나 말고 다른 사람이 등장하는가'다. 사람 이름이나 '같이', '다들', '회의', '조율'처럼 "
    "여러 사람을 전제하는 표현이 있으면 kana_agent, 내 일정과 내 기록만 다루면 nana_agent다.\n"
    "한 요청이 두 담당에 걸치면 순서대로 위임한다. 예를 들어 '민준이랑 시간 맞춰서 내 일정에 넣어줘'는 "
    "먼저 kana_agent로 회의 시간을 정하고, 그 결과의 final_slot을 kana_agent 답변에서 읽어 "
    "nana_agent에 넘겨 저장한다. 시간이 확정되지 않았으면(needs_agent_selection이 true) 저장을 위임하지 않고 "
    "사용자에게 확정할 시간을 먼저 확인한다.\n"
    "하위 agent 결과는 JSON이다. answer를 사람 말로 옮겨 답하고, ok가 false면 무엇이 실패했는지 error를 그대로 밝힌다. "
    "실패한 위임을 성공한 것처럼 요약하거나, 결과에 없는 일정을 추측해 채우지 않는다."
)

WEEK06_NANA_PROMPT = (
    "너는 supervisor에게 개인 업무를 위임받은 Nana 하위 agent다. supervisor의 prompt를 공유하지 않으므로 "
    "위임받은 query만 보고 스스로 판단해 tool을 고른다. "
    "담당은 나 혼자의 일이다. 내 개인 일정 조회·생성·수정·삭제, 할 일과 알림 저장, 개인 참고자료 검색, "
    "이 앱에서 나와 나눈 대화 검색이 전부 내 몫이다. "
    "일정을 물으면 기억으로 답하지 않고 반드시 조회 tool로 실제 저장값을 읽어 근거로 삼는다. "
    "외부 멤버의 일정이나 지난 대화를 가져오거나, 여러 사람의 공통 가능 시간을 계산하는 tool은 갖고 있지 않다. "
    "그런 요청을 받으면 추측해서 답하지 말고 '외부 멤버 일정과 그룹 시간 조율은 Kana 담당'이라고 짧게 알린다. "
    "다만 이미 시간이 정해진 회의를 내 일정으로 저장해 달라는 요청은 내 담당이므로 그대로 저장한다."
)

WEEK06_KANA_PROMPT = (
    "너는 supervisor에게 외부 멤버·그룹 조율 업무를 위임받은 Kana 하위 agent다. supervisor의 prompt를 공유하지 "
    "않으므로 위임받은 query만 보고 스스로 판단해 tool을 고른다. "
    "담당은 나 이외의 사람이 끼는 일이다. 외부 멤버의 지난 대화 검색과 원문 확인, 외부 멤버 일정 추출, "
    "공유 일정 저장소 row 조회, 여러 사람의 공통 가능 시간 후보 검증과 최종 회의 시간 결정이 내 몫이다. "
    "외부 멤버 데이터는 SQL로 직접 읽지 않고 반드시 주어진 MCP wrapper tool로만 접근한다.\n"
    "외부 멤버가 무슨 말을 했는지 찾을 때는 search_previous_conversations로 대화를 검색해 conversation_id를 얻고, "
    "원문을 그대로 봐야 할 때만 그 id를 load_conversation_messages에 넘긴다. conversation_id를 사용자에게 묻지 않는다.\n"
    "멤버 이름은 조사를 뗀 이름만 넘긴다. '민준이랑', '민준이가'는 '민준'으로, '하린과'는 '하린'으로 넘긴다. "
    "조사가 붙은 이름은 외부 저장소에서 찾지 못해 그 사람의 일정이 0건으로 돌아오고, "
    "그러면 실제로는 바쁜 사람이 한가한 것처럼 보여 이미 잡힌 시간에 회의를 잡게 된다.\n"
    "'언제 시간이 되는지', '회의 시간 잡자'처럼 여러 사람의 시간을 맞추는 요청은 collect_member_schedules 하나로 "
    "내 일정과 외부 멤버 일정을 같은 rows 구조로 모은다. extract_schedules_from_history와 개인 일정 조회를 따로 "
    "부르지 않는다. collect_member_schedules 결과의 external_status가 ok가 아니면 외부 멤버 일정을 못 가져온 "
    "상태이므로, 내 일정만 근거로 남았다고 먼저 밝히고 외부 멤버가 한가하다고 단정하지 않는다.\n"
    "조율 요청은 rows를 모으는 데서 멈추지 않는다. rows를 읽고 겹치지 않는 시간을 직접 골라 "
    "find_common_available_slots에 candidate_slots로 넘겨 검증하고, 이어서 decide_final_slot으로 최종 시간을 기록한다. "
    "이 두 tool은 후보나 최종 시간을 대신 계산해 주지 않으므로, 내가 고르지 않으면 결과가 비어 있게 된다.\n"
    "find_common_available_slots가 빈 후보를 돌려주면 결과의 rejected_candidates와 notes에 걸러진 이유가 들어 있다. "
    "그 이유만 근거로 다른 시간대를 골라 한 번 더 시도하고, 그래도 후보를 못 찾으면 "
    "decide_final_slot에 final_slot=null, needs_agent_selection=true와 이유를 넣어 반드시 기록한 뒤 답한다. "
    "find_common_available_slots만 여러 번 부르고 decide_final_slot 없이 답을 끝내지 않는다. "
    "후보가 걸러진 이유를 넘어서 '그날 업무 시간이 전부 예약됐다'처럼 rows에 없는 내용을 추측해 말하지 않는다. "
    "겹친다고 말할 때는 어떤 사람의 어떤 일정과 겹치는지 rows에 있는 값으로 밝힌다.\n"
    "확정된 회의를 내 일정으로 저장하는 tool은 갖고 있지 않다. 저장 요청을 받으면 '확정 일정 저장은 Nana 담당'이라고 "
    "짧게 알리고, 내가 정한 최종 시간은 답변에 분명히 남겨 supervisor가 그대로 넘길 수 있게 한다.\n"
    "답변 근거는 tool 결과 rows와 payload에 실제로 있는 값만 쓴다. 없는 일정이나 없는 후보를 만들어 내지 않는다.\n"
    + WEEK06_TIME_STATUS_PROMPT
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

    # 다른 주차 prompt를 누적하지 않는다. Week 1-4 조각에는 "개인 일정은 개인 일정 tool로 확인한다" 같은
    # Nana 규칙이 섞여 있는데, Kana는 그 tool을 갖고 있지 않아서 없는 tool을 부르려는 trace가 생긴다.
    # 그래서 Kana 역할과 오늘 날짜만 처음부터 직접 적는다.
    return [
        WEEK06_KANA_PROMPT,
        f"외부 멤버 일정과 회의 후보 날짜를 계산할 때 오늘 날짜는 {current_app_date_iso()}이며, "
        "date_from/date_to와 후보 date는 항상 이 날짜를 기준으로 계산한 YYYY-MM-DD 문자열로 넘긴다.",
    ]


def nana_system_prompt() -> str:
    return join_system_prompt(nana_prompt_parts())


def kana_system_prompt() -> str:
    return join_system_prompt(kana_prompt_parts())


def supervisor_system_prompt() -> str:
    return join_system_prompt(
        [
            *week06_prompt_parts(),
            # 누적 prompt에서는 뒤 조각이 우선한다. Week 1-5 조각은 "이 tool을 직접 불러라"를 전제로 쓰여 있어
            # supervisor가 갖고 있지 않은 tool을 부르려 하거나, 위임 없이 스스로 답하려는 trace가 생긴다.
            # 그래서 "직접 처리 금지 · 위임 후 결과만 근거로 답한다"를 마지막 조각에서 다시 못박는다.
            "위 주차별 지시에 등장하는 개인 일정·기록·외부 대화 tool은 모두 하위 agent가 갖고 있고 supervisor에게는 없다. "
            "그 지시들은 어떤 일이 어느 담당인지 판단하는 배경으로만 읽고, 실제 실행은 항상 nana_agent 또는 kana_agent에 위임한다. "
            "위임 없이 스스로 답을 만들지 않고, 위임 결과 JSON의 answer와 payload에 있는 값만 근거로 최종 답변을 쓴다.",
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
    # 한 요청이 두 담당에 걸치면 supervisor가 순서대로 두 번 위임한다("민준이랑 시간 맞춰서 저장해줘"는
    # kana_agent → nana_agent). selected_agent는 마지막 위임만 남아서 첫 위임이 trace에서 사라진다.
    # 기존 키는 UI 호환을 위해 그대로 두고, 위임 순서를 함께 기록해 두 단계가 모두 보이게 한다.
    selected_agents: list[str] = []

    for event in events:
        if event.get("event") == "tool_call" and event.get("tool_name") in {"nana_agent", "kana_agent"}:
            selected_agent = event["tool_name"]
            selected_agents.append(event["tool_name"])
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
        "supervisor_selected_agents": selected_agents,
        "inner_tool_names": inner_tool_names,
        "final_slot_payload": final_slot_payload,
        "final_decision_payload": final_decision_payload,
    }


def tool_name(tool_object: Any) -> str:
    return getattr(tool_object, "name", getattr(tool_object, "__name__", str(tool_object)))


FIND_COMMON_AVAILABLE_SLOTS_DESCRIPTION = (
    "여러 사람이 함께 비어 있는 회의 시간 후보를 검증하고 기록합니다. "
    "이 tool은 후보를 계산해 주지 않습니다. 빈 시간을 찾는 일은 네가 직접 해야 하며, "
    "candidate_slots를 비워서 부르면 후보 없음으로 기록되고 회의 시간을 정할 수 없습니다.\n"
    "부르기 전에 할 일: collect_member_schedules로 busy row를 모으고, 그 rows를 눈으로 읽어 "
    "아무 일정과도 겹치지 않는 시간대를 직접 골라 candidate_slots에 채웁니다.\n"
    "candidate_slots의 각 항목은 date(YYYY-MM-DD), start_time(HH:MM 24시간), end_time(HH:MM 24시간), "
    "duration_minutes(분), reason(이 시간을 고른 짧은 근거)을 모두 포함해야 합니다. "
    "후보는 date_from~date_to 범위 안에 있어야 하고, workday_start~workday_end 업무 시간 안에 들어야 하며, "
    "duration_minutes 이상 길어야 하고, 어떤 busy row와도 시간이 겹치면 안 됩니다. "
    "이 조건을 어긴 후보는 검증 단계에서 조용히 버려지므로, 겹치는 시간을 넣지 말고 후보를 2개 이상 여유 있게 제안합니다.\n"
    "busy_rows에는 collect_member_schedules 결과의 rows를 그대로 복사해 넘깁니다. busy_rows를 비우면 "
    "이 tool이 collect_member_schedules를 대신 호출해 다시 모으므로, 이미 모아 둔 rows가 있으면 복사해 넘기는 편이 정확합니다.\n"
    "busy row의 time_status에 따라 후보를 고르는 방법이 다릅니다. complete는 그 시간대를 피하고, "
    "start_only는 시작 시각부터 회의 길이만큼 피하고, date_only는 시간을 모르므로 그날을 통째로 버리지 말고 "
    "후보를 제안한 뒤 확인이 필요하다고 답변에 밝힙니다.\n"
    "이 tool 결과로 답을 끝내지 마세요. 검증된 candidate_slots를 받으면 그중 하나를 골라 "
    "decide_final_slot을 이어서 호출해 최종 시간을 확정해야 조율이 끝납니다."
)


DECIDE_FINAL_SLOT_DESCRIPTION = (
    "검증된 후보 중에서 네가 고른 최종 회의 시간을 확정 payload로 기록합니다. "
    "이 tool은 최종 시간을 대신 골라 주지 않습니다. 어떤 후보가 가장 좋은지는 네가 판단해서 인자로 넘겨야 합니다.\n"
    "확정할 때: selected_index에 find_common_available_slots 결과 candidate_slots의 index(0부터)를 넣거나 "
    "selected_slot에 고른 후보 객체를 그대로 넣고, final_slot에 'YYYY-MM-DD HH:MM-HH:MM' 형식 문자열을 채우고 "
    "needs_agent_selection은 false로 둡니다. reason에는 왜 이 시간을 골랐는지 사용자에게 그대로 보여줄 설명을 씁니다.\n"
    "아직 확정하지 못할 때: final_slot은 null, needs_agent_selection은 true로 두고 reason에 왜 확정하지 못했는지 "
    "(후보가 없음, 외부 일정 조회 실패, 사용자 확인 필요 등) 남깁니다. 후보가 없는데 시간을 임의로 만들어 확정하지 않습니다.\n"
    "근거를 trace에 남기기 위해 candidate_slots, busy_rows, member_names, date_from, date_to, duration_minutes도 "
    "앞선 tool 결과에서 복사해 함께 넘깁니다. 이 값이 빠지면 나중에 왜 이 시간이 확정됐는지 확인할 수 없습니다."
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

    query: str = Field(
        min_length=1,
        description="하위 agent가 그대로 처리할 사용자 요청 문장입니다. 빈 값은 허용하지 않습니다.",
    )

    @field_validator("query")
    @classmethod
    def _reject_blank_query(cls, value: str) -> str:
        """공백만 있는 query로 하위 agent를 실행하지 않도록 위임 입구에서 막습니다.

        min_length=1은 "   "를 통과시킨다. 빈 query를 그대로 넘기면 하위 agent가 맥락 없이
        아무 tool이나 고르거나 "무엇을 도와드릴까요"로 되묻는 turn이 trace에 남아서,
        "위임이 잘못됐다"와 "하위 agent가 실패했다"를 구분하기 어려워진다.
        """

        stripped = value.strip()
        if not stripped:
            raise ValueError(
                "query가 비어 있습니다. 하위 agent에 위임할 사용자 요청을 그대로 넘기세요."
            )
        return stripped


def _collected_busy_rows(
    *,
    member_names: list[str],
    date_from: str,
    date_to: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """busy_rows를 넘겨받지 못했을 때 collect_member_schedules로 직접 모읍니다.

    Week 5 wrapper는 실패해도 예외를 던지지 않고 ok/external_status로 상태를 알려 주므로,
    여기서도 같은 계약을 이어받아 "왜 근거가 비었는지"를 payload에 남긴다. 조회가 실패했는데
    rows가 빈 것을 "모두 한가함"으로 읽으면 이미 잡힌 시간에 회의를 잡게 되므로,
    collection_status를 후보 검증 payload까지 끌고 올라간다.
    """

    try:
        payload_text = collect_member_schedules.invoke(
            {
                "member_names": member_names,
                "date_from": date_from,
                "date_to": date_to,
            }
        )
    except Exception as exc:
        return [], {
            "collection_status": "failed",
            "collection_error": f"{type(exc).__name__}: {exc}",
        }

    try:
        payload = json.loads(payload_text)
    except (json.JSONDecodeError, TypeError) as exc:
        return [], {
            "collection_status": "failed",
            "collection_error": f"collect_member_schedules가 JSON을 반환하지 않았습니다: {exc}",
        }

    if not isinstance(payload, dict):
        return [], {"collection_status": "failed", "collection_error": "collect_member_schedules 응답이 dict가 아닙니다."}
    if payload.get("ok") is False:
        return [], {
            "collection_status": "failed",
            "collection_error": str(payload.get("error") or "collect_member_schedules가 ok=False를 반환했습니다."),
        }

    rows = payload.get("rows")
    if not isinstance(rows, list):
        return [], {
            "collection_status": "failed",
            "collection_error": f"rows가 list가 아닙니다: {type(rows).__name__}",
        }

    collected = [row for row in rows if isinstance(row, dict)]
    # 외부 MCP 조회만 실패한 부분 성공도 있다. 이때 rows에는 내 일정만 들어 있으므로
    # "외부 멤버 근거가 빠졌다"를 그대로 올려 보내 agent가 한가하다고 단정하지 않게 한다.
    external_status = payload.get("external_status")
    if external_status not in (None, "ok", "skipped"):
        return collected, {
            "collection_status": "partial",
            "collection_error": str(payload.get("external_error") or "외부 멤버 일정 조회에 실패했습니다."),
        }
    return collected, {"collection_status": "ok", "collection_error": None}


def _busy_rows_for_overlap_check(
    rows: list[dict[str, Any]],
    duration_minutes: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """겹침 검증에 쓸 row와 시간을 몰라 검증에 못 쓰는 row를 나눕니다.

    Week 5에서 붙인 time_status 계약을 Week 6 시간 계산에서 실제로 적용하는 자리다.
    fixed/schedule_decision.py의 busy_rows_overlap은 start_time이 비거나 "미정"이면 0분,
    end_time이 비거나 "미정"이면 24:00으로 보고 겹침을 판정한다. 그래서 그대로 넘기면
      - date_only row 하나가 그날 전체를 막아 후보가 전멸하고,
      - start_only row 하나가 시작 시각부터 자정까지를 막아 오후가 통째로 사라진다.
    시간을 모른다는 것이 "하루 종일 바쁘다"는 뜻은 아니므로 종류별로 다르게 다룬다.
      - complete   : 그대로 hard blocker로 쓴다.
      - start_only : 종료를 모르니 회의 길이만큼만 막힌 것으로 보정해 hard blocker로 쓴다.
                     시작 시각 근처는 실제로 바쁘므로 검증에서 빼면 겹치는 후보가 통과한다.
      - date_only  : 특정 시각을 막는 근거가 없어 검증에서 빼고 soft blocker로 따로 보고한다.
                     후보는 제안하되 확정 전에 확인이 필요하다는 사실을 payload에 남긴다.
    """

    hard_rows: list[dict[str, Any]] = []
    soft_rows: list[dict[str, Any]] = []
    assumed_minutes = max(30, int(duration_minutes or 60))

    for row in rows:
        time_status = str(row.get("time_status") or "").strip()
        start_minutes = parse_time_minutes(row.get("start_time"), -1)

        # time_status가 없는 row(외부 store가 직접 만든 row 등)는 값으로 직접 판정한다.
        if not time_status:
            end_minutes = parse_time_minutes(row.get("end_time"), -1)
            if start_minutes < 0:
                time_status = "date_only"
            elif end_minutes < 0:
                time_status = "start_only"
            else:
                time_status = "complete"

        if time_status == "complete":
            hard_rows.append(row)
            continue
        if time_status == "start_only" and start_minutes >= 0:
            # 원본 row를 바꾸지 않고 검증용 사본에만 종료 시간을 채운다. payload에 남는 busy_rows는
            # 원본이어야 "우리가 무엇을 보정했는지"와 "실제 저장값"을 함께 확인할 수 있다.
            hard_rows.append(
                {
                    **row,
                    "end_time": format_time_minutes(min(start_minutes + assumed_minutes, 24 * 60)),
                    "overlap_end_assumed": True,
                }
            )
            continue
        soft_rows.append(row)

    return hard_rows, soft_rows


def _candidate_rejection_reason(
    candidate: Any,
    *,
    date_from: str,
    date_to: str,
    busy_rows: list[dict[str, Any]],
    duration_minutes: int,
    workday_start: str,
    workday_end: str,
) -> str:
    """검증에서 걸러진 후보가 왜 걸러졌는지 한 줄로 설명합니다.

    후보를 걸러내는 권한은 fixed/schedule_decision.py의 normalize_llm_candidate_slots에 있고
    그 함수는 조건에 안 맞는 후보를 조용히 버립니다. 그래서 agent는 "후보가 0개"만 보고
    이유를 알 수 없어, 실제로는 비어 있는 시간을 "전부 예약됨"으로 잘못 설명하는 답변이 나왔다.

    판정 규칙을 여기서 다시 만들지 않고 fixed가 쓰는 같은 primitive(date_range,
    parse_time_minutes, busy_rows_overlap)로 이유만 되짚는다. 그래서 이 함수는 걸러낼 권한이
    없고 설명만 담당한다. 규칙이 바뀌면 fixed 한 곳만 바뀐다.
    """

    slot = candidate.model_dump() if hasattr(candidate, "model_dump") else candidate
    if not isinstance(slot, dict):
        return "후보 형식이 date/start_time/end_time을 가진 객체가 아닙니다."

    day = normalize_date_bound(str(slot.get("date") or ""))
    start_minutes = parse_time_minutes(slot.get("start_time"), -1)
    end_minutes = parse_time_minutes(slot.get("end_time"), -1)
    work_start = parse_time_minutes(workday_start, 9 * 60)
    work_end = parse_time_minutes(workday_end, 18 * 60)
    requested = max(30, int(duration_minutes or 60))

    if day not in set(date_range(date_from, date_to)):
        return f"{day}는 요청한 날짜 범위({date_from}~{date_to}) 밖입니다."
    if start_minutes < 0 or end_minutes < 0:
        return "start_time 또는 end_time이 HH:MM 형식이 아닙니다."
    if end_minutes <= start_minutes:
        return "종료 시간이 시작 시간보다 늦지 않습니다."
    if start_minutes < work_start or end_minutes > work_end:
        return f"업무 시간({workday_start}~{workday_end}) 밖입니다."
    if end_minutes - start_minutes < requested:
        return f"길이가 요청한 {requested}분보다 짧습니다."

    blockers = busy_rows_overlap(busy_rows, day, start_minutes, end_minutes)
    if blockers:
        names = ", ".join(
            f"{row.get('member_name') or '?'} {row.get('title') or ''}"
            f"({row.get('start_time')}-{row.get('end_time')})"
            for row in blockers[:3]
        )
        return f"이미 잡힌 일정과 겹칩니다: {names}"
    return "검증에서 제외됐지만 확인된 원인이 없습니다."


def _rejected_candidate_reports(
    submitted: list[Any] | None,
    accepted: list[dict[str, Any]],
    *,
    date_from: str,
    date_to: str,
    busy_rows: list[dict[str, Any]],
    duration_minutes: int,
    workday_start: str,
    workday_end: str,
) -> list[dict[str, Any]]:
    """agent가 낸 후보 중 검증을 통과하지 못한 것만 이유와 함께 모읍니다."""

    accepted_keys = {
        (str(slot.get("date")), str(slot.get("start_time")), str(slot.get("end_time")))
        for slot in accepted
    }
    reports: list[dict[str, Any]] = []
    for candidate in submitted or []:
        slot = candidate.model_dump() if hasattr(candidate, "model_dump") else candidate
        if not isinstance(slot, dict):
            reports.append({"candidate": str(candidate), "reason": "후보 형식이 올바르지 않습니다."})
            continue
        # 통과한 후보는 시간이 HH:MM으로 정규화돼 돌아오므로 같은 기준으로 맞춰 비교한다.
        start_minutes = parse_time_minutes(slot.get("start_time"), -1)
        end_minutes = parse_time_minutes(slot.get("end_time"), -1)
        key = (
            normalize_date_bound(str(slot.get("date") or "")),
            format_time_minutes(start_minutes) if start_minutes >= 0 else str(slot.get("start_time")),
            format_time_minutes(end_minutes) if end_minutes >= 0 else str(slot.get("end_time")),
        )
        if key in accepted_keys:
            continue
        reports.append(
            {
                "candidate": f"{slot.get('date')} {slot.get('start_time')}-{slot.get('end_time')}",
                "reason": _candidate_rejection_reason(
                    slot,
                    date_from=date_from,
                    date_to=date_to,
                    busy_rows=busy_rows,
                    duration_minutes=duration_minutes,
                    workday_start=workday_start,
                    workday_end=workday_end,
                ),
            }
        )
    return reports


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

    # 이름/날짜 정규화는 여기서 한 번만 하고, 같은 값을 일정 수집과 후보 검증에 함께 쓴다.
    # 두 단계가 서로 다른 범위를 보면 "범위 안에서 겹치지 않는다"는 검증 결과를 믿을 수 없다.
    # alias와 실제 이름이 함께 들어오면(["A", "철수"] → ["철수", "철수"]) 정규화 결과가 겹친다.
    # 그대로 두면 payload의 members에 같은 사람이 두 번 남아 "누구 일정을 봤는지"가 부정확해진다.
    normalized_members = dedupe_preserving_order(normalize_external_member_names(member_names))
    normalized_date_from = normalize_date_bound(date_from)
    normalized_date_to = normalize_date_bound(date_to)

    if busy_rows is None:
        collected_rows, collection_state = _collected_busy_rows(
            member_names=normalized_members,
            date_from=normalized_date_from,
            date_to=normalized_date_to,
        )
    else:
        collected_rows = [row for row in busy_rows if isinstance(row, dict)]
        # agent가 앞선 tool output에서 복사해 넘긴 rows다. 다시 모으지 않으므로 조회 상태를 알 수 없고,
        # 여기서 ok로 단정하면 실패한 조회 결과를 성공으로 덮어쓸 수 있어 provided로 구분해 둔다.
        collection_state = {"collection_status": "provided", "collection_error": None}

    hard_rows, soft_rows = _busy_rows_for_overlap_check(collected_rows, duration_minutes)

    # 내 일정도 겹침 판정의 근거라서 조회 대상에 "나"를 함께 남긴다. 이 값은 검증 payload의
    # members로 그대로 들어가 "누구 일정을 보고 고른 후보인지"를 설명한다. alias 정규화 결과에
    # 이미 "나"가 있을 수 있으므로 Week 5 collect_member_schedules와 같은 방식으로 중복을 막는다.
    evidence_members = [
        PERSONAL_SHARED_MEMBER_NAME,
        *[name for name in normalized_members if name != PERSONAL_SHARED_MEMBER_NAME],
    ]

    payload = find_common_available_slots_payload(
        member_names=evidence_members,
        date_from=normalized_date_from,
        date_to=normalized_date_to,
        # 겹침 검증에는 시간을 아는 row만 넘긴다. date_only row까지 넘기면 그날 후보가 전멸한다.
        busy_rows=hard_rows,
        duration_minutes=duration_minutes,
        workday_start=workday_start,
        workday_end=workday_end,
        limit=limit,
        candidate_slots=candidate_slots,
        llm_reason=llm_reason,
    )

    # payload의 busy_rows는 검증에 실제로 쓴 row다. 원본과 시간 미정 row를 함께 남겨
    # "무엇을 근거로 걸렀고 무엇을 못 걸렀는지"를 trace에서 확인할 수 있게 한다.
    payload["busy_rows_all"] = collected_rows
    payload["time_unknown_rows"] = soft_rows
    payload["date_from"] = normalized_date_from
    payload["date_to"] = normalized_date_to
    payload["duration_minutes"] = max(30, int(duration_minutes or 60))
    payload.update(collection_state)

    notes: list[str] = []
    if collection_state["collection_status"] == "failed":
        notes.append(
            f"일정 조회에 실패해 근거 row가 없습니다({collection_state['collection_error']}). "
            "빈 시간으로 단정하지 말고 조회 실패를 먼저 알리세요."
        )
    elif collection_state["collection_status"] == "partial":
        notes.append(
            f"외부 멤버 일정을 가져오지 못했습니다({collection_state['collection_error']}). "
            "아래 후보는 내 일정만 근거로 검증됐습니다."
        )
    if soft_rows:
        notes.append(
            f"시간이 미정인 일정 {len(soft_rows)}건은 겹침 검증에 쓸 수 없었습니다. "
            "해당 날짜 후보는 확정 전에 사용자 확인이 필요합니다."
        )

    # rows가 하나도 없는 멤버를 짚어 준다. 이름에 조사가 붙어 있으면("민준이") 외부 저장소에서
    # 못 찾아 0건이 돌아오는데, 그 상태는 "일정이 없어 한가함"과 payload 모양이 같다.
    # EXTERNAL_MEMBER_ALIAS가 비어 있어 정규화가 이름을 고쳐 주지 않으므로 여기서 드러내 준다.
    # 정말 일정이 없을 수도 있으니 오류로 막지 않고 확인만 요청한다.
    members_with_rows = {str(row.get("member_name") or "").strip() for row in collected_rows}
    members_without_rows = [name for name in evidence_members if name not in members_with_rows]
    payload["members_without_rows"] = members_without_rows
    if members_without_rows and collection_state["collection_status"] in ("ok", "provided"):
        notes.append(
            f"{', '.join(members_without_rows)}의 일정이 rows에 하나도 없습니다. "
            "이름에 조사가 붙어 있지 않은지(예: '민준이' → '민준') 확인하고, "
            "정말 일정이 없는 것이라면 그대로 진행하세요. 확인 없이 한가하다고 단정하지 마세요."
        )
    # 걸러진 후보와 그 이유를 남긴다. 이유 없이 "후보 0개"만 주면 agent가 원인을 추측해서
    # 실제로는 비어 있는 시간을 "전부 예약됨"으로 설명하는 답변이 나온다(실행에서 관찰).
    rejected = _rejected_candidate_reports(
        candidate_slots,
        payload.get("candidate_slots") or [],
        date_from=normalized_date_from,
        date_to=normalized_date_to,
        busy_rows=hard_rows,
        duration_minutes=duration_minutes,
        workday_start=workday_start,
        workday_end=workday_end,
    )
    payload["submitted_candidate_count"] = len(candidate_slots or [])
    payload["rejected_candidates"] = rejected

    if not payload.get("candidate_slots"):
        if not candidate_slots:
            notes.append(
                "candidate_slots를 넣지 않아 후보가 없습니다. busy_rows를 읽고 겹치지 않는 시간을 직접 골라 "
                "candidate_slots에 채워 다시 호출하세요."
            )
        else:
            detail = " / ".join(f"{item['candidate']} → {item['reason']}" for item in rejected[:5])
            notes.append(
                f"제안한 후보 {len(candidate_slots)}개가 모두 검증에서 제외됐습니다. {detail} "
                "이 이유만 근거로 다른 시간대를 다시 고르세요. 걸러진 이유를 넘어서 "
                "'하루 전체가 예약됐다'처럼 추측해 말하지 마세요."
            )
        notes.append(
            "다시 시도해도 후보를 찾지 못하면 decide_final_slot에 final_slot=null, "
            "needs_agent_selection=true와 이유를 넣어 반드시 기록한 뒤 답하세요."
        )
    payload["notes"] = notes
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

    # 검증 규칙은 helper 한 곳에만 두고 @tool은 JSON 직렬화만 맡는다. 같은 판정이 두 군데로
    # 갈라지면 tool 경로와 직접 호출 경로가 다른 결과를 내기 때문이다.
    return json_payload(
        find_common_available_slots_dict(
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
    )


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

    # 여기서 최종 시간을 대신 고르지 않는다. 후보가 있어도 agent가 selected_index/final_slot을 넘기지
    # 않았다면 needs_agent_selection=True 상태를 그대로 유지해야, "agent가 아직 안 골랐다"와
    # "코드가 임의로 골랐다"가 trace에서 섞이지 않는다. 그 판정은 decide_final_slot_payload가 갖고 있다.
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
    # course repo 계약은 top-level final_slot/reason/candidates다. ok/tool_name을 함께 실어
    # 다른 wrapper tool과 같은 모양으로 읽히게 하되, 계약 키를 감싸지 않고 그대로 남긴다.
    return json_payload({"ok": True, "tool_name": "decide_final_slot", **payload})


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


def _build_nana_subagent() -> Any:
    """Week 4 tool을 가진 Nana 하위 agent를 한 번만 만들고 재사용합니다."""

    global _NANA_SUBAGENT
    if _NANA_SUBAGENT is None:
        _NANA_SUBAGENT = create_agent(
            model=chat_model(),
            tools=week04_tools(),
            system_prompt=nana_system_prompt(),
        )
    return _NANA_SUBAGENT


def _build_kana_subagent() -> Any:
    """Week 5 wrapper와 조율 tool을 가진 Kana 하위 agent를 한 번만 만들고 재사용합니다."""

    global _KANA_SUBAGENT
    if _KANA_SUBAGENT is None:
        _KANA_SUBAGENT = create_agent(
            model=chat_model(),
            tools=kana_tools(),
            system_prompt=kana_system_prompt(),
        )
    return _KANA_SUBAGENT


def _run_subagent(selected_agent: str, query: str, build: Any) -> dict[str, Any]:
    """하위 agent를 실행해 supervisor가 읽을 위임 결과 payload로 정리합니다.

    하위 agent 실행이 깨져도 예외를 supervisor 쪽으로 올리지 않는다. tool이 예외로 끝나면
    supervisor는 "위임이 실패했다"는 사실만 알고 이유를 알 수 없어서, 근거 없이 스스로 답을
    지어내거나 같은 위임을 반복하는 turn이 생긴다. 그래서 Week 5 wrapper와 같은 방식으로
    ok=False와 error를 payload에 남겨, 실패 이유가 trace와 최종 답변에 함께 드러나게 한다.
    """

    try:
        result = build().invoke({"messages": [{"role": "user", "content": query}]})
    except Exception as exc:
        return {
            "ok": False,
            "tool_name": selected_agent,
            "selected_agent": selected_agent,
            "query": query,
            "answer": "",
            "error": f"{type(exc).__name__}: {exc}",
            "trace": {"events": []},
            "inner_tool_names": [],
        }

    events = extract_agent_events(result)
    return {
        "ok": True,
        "tool_name": selected_agent,
        "selected_agent": selected_agent,
        "query": query,
        "answer": extract_final_text(result),
        "trace": {"events": events},
        "inner_tool_names": _tool_call_names(events),
    }


def _lift_slot_payloads(events: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Kana 하위 trace에서 최종 시간 결정 payload를 끌어올립니다.

    supervisor는 하위 agent의 trace를 직접 보지 않고 위임 결과 JSON만 읽는다. 그래서 최종 시간이
    하위 trace 안에만 남아 있으면 supervisor가 확정 시간을 근거로 답하거나 Nana에게 저장을 넘길 수
    없다. decide_final_slot 결과를 payload 최상위로 올려 위임 경계를 넘어가게 만든다.

    같은 조율에서 decide_final_slot이 여러 번 호출될 수 있으므로(후보 보류 후 재확정) 마지막 결과를
    남긴다. 마지막 호출이 agent의 최종 판단이기 때문이다.
    """

    final_slot_payload: dict[str, Any] | None = None
    final_decision_payload: dict[str, Any] | None = None

    for event in events:
        if event.get("event") != "tool_result":
            continue
        content = event.get("content")
        if not isinstance(content, dict):
            continue
        if event.get("tool_name") == "decide_final_slot" or "final_slot" in content:
            final_slot_payload = content
        # propose_group_schedule 호환 경로는 final_decision에 최종 결정을 담는다.
        if isinstance(content.get("final_decision"), dict):
            final_decision_payload = content["final_decision"]

    return final_slot_payload, final_decision_payload


@tool(args_schema=AgentQueryInput)
def nana_agent(query: str) -> str:
    """개인 일정과 개인 RAG 작업을 프롬프트 기반 Nana 하위 에이전트에게 위임합니다."""

    return json_payload(_run_subagent("nana_agent", query, _build_nana_subagent))


@tool(args_schema=AgentQueryInput)
def kana_agent(query: str) -> str:
    """그룹 일정 종합 작업을 프롬프트 기반 Kana 하위 에이전트에게 위임합니다."""

    payload = _run_subagent("kana_agent", query, _build_kana_subagent)
    final_slot_payload, final_decision_payload = _lift_slot_payloads(payload["trace"]["events"])
    payload["final_slot_payload"] = final_slot_payload
    payload["final_decision_payload"] = final_decision_payload
    # 확정 시간을 supervisor가 답변과 저장 위임에 바로 쓸 수 있도록 문자열로도 함께 올린다.
    #
    # 결정이 없을 때 final_slot 키를 None으로 넣지 않는다. extract_langchain_trace는
    # content에 final_slot 키가 있으면 그 dict 전체를 final_slot_payload로 본다("final_slot" in content).
    # 키를 항상 넣으면 결정이 없는 위임에서도 위임 payload 전체가 최종 시간 payload로 잡혀
    # trace와 UI에 엉뚱한 값이 남는다. 그래서 실제로 결정이 있을 때만 올린다.
    if final_slot_payload:
        payload["final_slot"] = final_slot_payload.get("final_slot")
        payload["needs_agent_selection"] = final_slot_payload.get("needs_agent_selection")
    return json_payload(payload)


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
