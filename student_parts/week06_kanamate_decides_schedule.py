from __future__ import annotations

import json
from typing import Any

from langchain.agents import create_agent
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from fixed.external_mcp import PERSONAL_SHARED_MEMBER_NAME
from fixed.external_people_store import normalize_external_member_names
from fixed.langchain_trace import extract_agent_events, extract_final_text
from fixed.llm import chat_model
from fixed.runtime_clock import current_app_date_iso
from fixed.schedule_decision import (
    CommonSlotCandidate,
    date_range,
    decide_final_slot_payload,
    find_common_available_slots_payload,
    normalize_date_bound,
    normalize_llm_candidate_slots,
    parse_time_minutes,
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
#   - [메인] SUPERVISOR_DELEGATION_PROMPT / SUPERVISOR_TOOL_SCOPE_PROMPT / SUPERVISOR_EXECUTION_PROMPT
#     supervisor prompt 조각입니다. 누적된 Week 1-5 tool 규칙은 하위 agent용이므로
#     TOOL_SCOPE 조각에서 supervisor의 tool 범위를 nana_agent/kana_agent 둘로 다시 못 박습니다.
#
#   - [메인] NANA_ROLE_PROMPT / KANA_ROLE_PROMPT / KANA_TOOL_CALL_PROMPT
#     하위 agent 역할과 담당 경계, Kana의 조율 3단(수집 → 후보 검증 → 최종 확정) 호출 규칙입니다.
#     Kana는 다른 주차 prompt를 누적하지 않으므로 오늘 날짜도 여기서 직접 알려 줍니다.
#
#   - [추가] _free_window_exists(...) / _find_common_available_slots_note(...)
#     후보 검증 결과에 다음 행동을 실어 보냅니다. tool description은 호출 전 근거이고
#     이 note는 결과를 받은 뒤의 판단 지점이라, 같은 계약을 두 지점에 함께 둡니다.
#     범위 전체가 이미 차 있으면 "다른 시간을 골라 다시 호출하라"가 만족될 수 없어 재호출이 반복되므로,
#     _free_window_exists()로 그 상황을 먼저 판정해 note를 재시도 대신 종료 안내로 바꿉니다.
#
#   - [메인] _run_subagent(agent, query) / _final_payloads_from_events(events) / _kana_result_note(...)
#     하위 agent 실행과 trace 정리, Kana trace에서 최종 시간 payload 끌어올리기,
#     조율이 확정까지 가지 못한 경우를 supervisor에게 알리는 note를 만듭니다.
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


# 누적 prompt에는 Week 1-5의 tool 호출 규칙이 그대로 남아 있는데, supervisor가 실제로 가진 tool은
# nana_agent/kana_agent 둘뿐입니다. join_system_prompt는 "뒤에 있는 지시를 우선한다"고 선언하므로
# 누적 조각 뒤에서 tool 범위를 명시적으로 무효화해 supervisor가 없는 tool을 부르지 않게 합니다.
SUPERVISOR_DELEGATION_PROMPT = """Week 6부터 당신은 직접 일하지 않는 supervisor입니다. 실제 작업은 두 하위 agent가 합니다.
Week 5의 "여러 사람의 최종 회의 시간을 직접 확정하지는 않는다"는 지시는 Week 6에서는 적용하지 않습니다.
최종 회의 시간 확정은 Week 6의 주력 기능이고 kana_agent가 담당합니다.
- nana_agent: 내 개인 일정 생성·조회·수정·삭제, 할 일/알림 저장, 개인 참고자료와 앱 대화 검색(RAG).
- kana_agent: 다른 사람의 이전 대화와 바쁜 시간 조회, 공유 일정 저장소 조회,
  여러 사람의 공통 가능 시간 후보 검증과 최종 회의 시간 결정.

위임 판단 기준은 "누구의 기록이 필요한가"입니다.
- 내 앱 DB만 보면 되는 요청 → nana_agent
- 다른 사람의 기록이 필요하거나 여러 사람의 시간을 맞춰야 하는 요청 → kana_agent
- 날짜와 시각(HH:MM)이 둘 다 이미 정해진 등록·저장 요청은 조율이 아니라 저장이므로 nana_agent입니다.
- 시각이 정해지지 않은 요청은 "등록해줘"라는 말이 있어도 아직 조율입니다.
  날짜만 있고 시각이 없으면 먼저 kana_agent로 시간을 확정합니다."""

SUPERVISOR_TOOL_SCOPE_PROMPT = """위에 누적된 Week 1-5의 tool 호출 규칙은 전부 하위 agent가 쓰는 지식입니다.
supervisor인 당신이 실제로 가진 tool은 nana_agent와 kana_agent 두 개뿐이고,
personal_로 시작하는 tool, save_structured_request, search_로 시작하는 tool,
collect_member_schedules, find_common_available_slots, decide_final_slot은 당신에게 없습니다.
그 tool들을 직접 부르려 하지 말고, 해야 할 일을 문장으로 정리해 담당 하위 agent에 넘기세요."""

SUPERVISOR_EXECUTION_PROMPT = """supervisor 실행 규칙:
- 사용자 요청에 답하기 전에 반드시 nana_agent 또는 kana_agent를 한 번은 호출합니다.
- query에는 사용자 원문의 의도, 대상 인물, 날짜 범위를 그대로 담아 넘깁니다. 요약하다 날짜를 빠뜨리지 않습니다.
- 하위 agent가 돌려준 answer와 payload만 근거로 답합니다. 하위 agent가 조회하지 않은 내용을 지어내지 않습니다.
- 하위 agent 결과에 note가 있으면 그 지시를 먼저 따릅니다.
  특히 최종 시간이 미확정이면 확정된 것처럼 답하지 말고 후보와 미확정 사실을 그대로 전합니다.
- 한 번의 요청에는 하나의 하위 agent만 호출하는 것이 기본입니다.
  "시간을 정해줘", "언제가 좋을지 알아봐"는 조율까지가 요청이므로 kana_agent 하나로 끝냅니다.
  사용자가 "정해서 등록까지 해줘"처럼 저장을 명시적으로 요청했을 때만
  kana_agent로 시간을 확정한 뒤 nana_agent에 저장을 이어서 위임합니다.
  이때 nana_agent에 넘기는 query에는 kana_agent가 확정한 날짜·시각을 그대로 적어 줍니다.
  시각이 아직 없는 상태로 nana_agent에 저장을 넘기지 않습니다.
- 하위 agent가 "담당이 아니다"라고 답하면 같은 agent에 다시 묻지 말고 다른 agent로 넘깁니다.
- 최종 답변은 사용자에게 하는 한국어 문장으로 씁니다. 하위 agent JSON을 그대로 붙여넣지 않습니다."""

NANA_ROLE_PROMPT = """당신은 Week 6 supervisor 아래에서 개인 업무를 맡은 Nana 하위 agent입니다.
supervisor의 프롬프트를 공유하지 않으므로 넘겨받은 query만으로 판단합니다.

담당: 내 개인 일정 생성·조회·수정·삭제, 할 일/알림 저장, 개인 참고자료와 앱 대화 RAG.
담당이 아닌 것: 다른 사람의 이전 대화 조회, 여러 사람의 공통 가능 시간 찾기, 최종 회의 시간 결정.
담당이 아닌 요청을 받으면 추측으로 답하지 말고 "그룹 일정 조율은 Kana 담당입니다"라고 한 줄로 알립니다.

단, 날짜와 시간이 이미 정해진 그룹 일정을 저장하는 것은 조율이 아니라 저장이므로 내 담당입니다.
확정된 시간을 받아 저장하라는 요청은 Kana에게 넘기지 말고 Week 3 규칙대로
extract_schedule_request → save_structured_request로 직접 저장한 뒤 결과를 알립니다.
tool 결과 JSON을 그대로 붙여넣지 말고 한국어 문장으로 정리해 답합니다."""

KANA_ROLE_PROMPT = f"""당신은 Kana, Week 6 supervisor 아래에서 여러 사람의 일정 조율을 맡은 하위 agent입니다.
오늘 날짜는 {current_app_date_iso()}입니다.
supervisor의 프롬프트를 공유하지 않으므로 넘겨받은 query만으로 판단합니다.

담당: 다른 사람의 이전 대화 검색, 외부 멤버의 바쁜 시간 조회, 공유 일정 저장소 row 조회,
여러 사람의 공통 가능 시간 후보 검증과 최종 회의 시간 결정.
담당이 아닌 것: 확정된 일정을 앱에 저장하는 것, 내 개인 일정/할 일/알림 저장, 개인 참고자료 RAG.
저장이 필요하면 직접 하지 말고 "확정된 일정 저장은 Nana 담당입니다"라고 한 줄로 알립니다.

- date_from/date_to는 항상 YYYY-MM-DD로 넘깁니다. "다음 주" 같은 상대 표현은 오늘 날짜 기준으로 계산합니다.
- 조회 결과가 0건이어도 "일정이 없다"고 단정하지 않고 tool 결과의 note를 먼저 읽습니다.
- tool 결과 JSON을 그대로 붙여넣지 말고 한국어 문장으로 정리해 답합니다."""

KANA_TOOL_CALL_PROMPT = """Kana tool 호출 규칙:
- 남의 과거 대화 내용이 필요하면 search_previous_conversations로 찾고, 전문이 필요하면
  그 conversation_id로 load_conversation_messages를 잇습니다.
  query에는 문장이나 조사를 넣지 말고 "일정", "회의"처럼 짧은 핵심 명사만 넣습니다.
- 특정 멤버의 바쁜 시간만 필요하면 extract_schedules_from_history를 씁니다.
- 회의 시간을 찾아야 하는 조율 요청은 다음 세 tool을 이 순서로 이어서 호출합니다.
  1) collect_member_schedules — 내 일정과 대상 멤버의 바쁜 시간을 한 번에 모읍니다.
  2) find_common_available_slots — 1)의 rows를 busy_rows에 그대로 복사해 넘기고,
     그 rows와 겹치지 않는 시간을 직접 골라 candidate_slots에 채웁니다. 이 tool은 후보를 대신 계산하지 않습니다.
     첫 호출부터 candidate_slots를 채웁니다. 빈 목록으로 호출하면 후보 0건이 돌아올 뿐입니다.
     후보를 채우지 못한 상태로 사용자에게 답하지 말고, tool이 어떻게 동작하는지도 사용자에게 설명하지 않습니다.
  3) decide_final_slot — 2)의 후보 중 가장 이른 시간을 기본으로 하나를 골라
     selected_index와 final_slot('YYYY-MM-DD HH:MM-HH:MM')을 채우고 needs_agent_selection=false로 확정합니다.
- 조율 요청의 기본 동작은 확정입니다. 후보를 나열하고 사용자에게 고르라고 되묻지 마세요.
  "후보만 알려줘", "어떤 시간대가 비어 있어?"처럼 나열만 명시적으로 요청한 경우에만
  final_slot을 null, needs_agent_selection을 true로 두고 reason에 이유를 적습니다.
- 요청 범위에 회의 길이만큼 비는 구간이 아예 없다는 결과(range_fully_busy)를 받으면 후보를 다시 만들지 않습니다.
  decide_final_slot을 needs_agent_selection=true로 호출해 마무리하고,
  사용자에게 가능한 시간이 없다는 사실과 날짜 범위를 넓힐지를 함께 물어봅니다.
- "후보만 알려줘"도 밟는 절차는 똑같습니다. candidate_slots를 직접 채워 find_common_available_slots를 부르고,
  decide_final_slot을 needs_agent_selection=true로 부른 뒤 후보를 안내합니다.
  후보를 채우지도 않은 채 "가능한 시간이 없다"거나 "후보를 직접 골라 달라"고 답하지 않습니다.
- 세 tool 중 어디서도 중간에 멈추지 않습니다. 멈추면 supervisor가 결과를 근거로 답할 수 없습니다.
- rows의 notes에 "그룹 일정 참석자"가 있으면 이미 잡혀 있는 회의이므로 그 시간은 후보에서 뺍니다.
- 날짜와 시간이 이미 정해진 저장 요청은 조율이 아니므로 위 세 tool을 부르지 않습니다."""


def week06_system_prompt() -> str:
    """6주차 supervisor agent가 따르는 시스템 프롬프트입니다."""

    return supervisor_system_prompt()


def week06_prompt_parts() -> list[str]:
    """1~6주차 supervisor system prompt 조각을 누적합니다."""

    return [
        *week05_prompt_parts(),
        SUPERVISOR_DELEGATION_PROMPT,
        SUPERVISOR_TOOL_SCOPE_PROMPT,
    ]


def nana_prompt_parts() -> list[str]:
    """Week 6 Nana 하위 에이전트 전용 system prompt 조각입니다."""

    return [
        *week04_prompt_parts(),
        NANA_ROLE_PROMPT,
    ]


def kana_prompt_parts() -> list[str]:
    """Week 6 Kana 하위 에이전트 전용 system prompt 조각입니다."""

    return [
        KANA_ROLE_PROMPT,
        KANA_TOOL_CALL_PROMPT,
    ]


def nana_system_prompt() -> str:
    return join_system_prompt(nana_prompt_parts())


def kana_system_prompt() -> str:
    return join_system_prompt(kana_prompt_parts())


def supervisor_system_prompt() -> str:
    return join_system_prompt(
        [
            *week06_prompt_parts(),
            SUPERVISOR_EXECUTION_PROMPT,
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
    "여러 사람의 공통 가능 시간 후보를 검증하고 기록합니다. "
    "이 tool은 후보를 대신 계산해 주지 않습니다. busy_rows를 직접 읽고 겹치지 않는 시간을 "
    "당신이 골라 candidate_slots에 채워 넘겨야 하며, 비워서 호출하면 후보는 0건으로 돌아옵니다.\n"
    "- candidate_slots의 각 항목: date('YYYY-MM-DD'), start_time('HH:MM'), end_time('HH:MM'), "
    "duration_minutes(정수 분), reason(이 시간을 고른 짧은 근거).\n"
    "- 후보는 어떤 busy row와도 겹치면 안 되고, workday_start~workday_end 안에 들어가야 하며, "
    "duration_minutes만큼 길어야 합니다. 이 조건을 어긴 후보는 결과에서 조용히 제외됩니다.\n"
    "- busy_rows에는 앞서 호출한 collect_member_schedules 결과의 rows를 그대로 복사해 넘깁니다. "
    "생략하면 이 tool이 member_names와 날짜 범위로 직접 모읍니다.\n"
    "- 이 결과로 답변을 끝내지 말고, 돌아온 candidate_slots로 decide_final_slot을 이어서 호출하세요."
)


DECIDE_FINAL_SLOT_DESCRIPTION = (
    "회의 최종 시간 결정을 기록합니다. "
    "이 tool은 최종 시간을 대신 고르지 않습니다. 후보 중 무엇을 확정할지는 당신이 판단해 "
    "selected_index(또는 selected_slot)와 final_slot을 채워 넘겨야 합니다.\n"
    "- final_slot 형식: 'YYYY-MM-DD HH:MM-HH:MM'. 확정했으면 needs_agent_selection=false로 둡니다.\n"
    "- 조율 요청의 기본은 확정입니다. 후보가 하나라도 있으면 그중 하나를 골라 확정하세요.\n"
    "- 사용자가 후보 나열만 명시적으로 요청했거나 후보가 0건일 때만 final_slot은 null, "
    "needs_agent_selection은 true로 두고 reason에 확정하지 않은 이유를 적습니다.\n"
    "- reason은 사용자에게 그대로 보여줄 한국어 설명으로 씁니다.\n"
    "- 근거를 남겨야 하므로 candidate_slots, busy_rows, member_names, date_from, date_to도 "
    "앞선 tool output에서 복사해 함께 넘깁니다."
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


def _free_window_exists(
    *,
    busy_rows: list[dict[str, Any]],
    date_from: str,
    date_to: str,
    duration_minutes: int,
    workday_start: str,
    workday_end: str,
) -> bool:
    """요청 범위 안에 duration_minutes만큼 연속으로 비는 구간이 하나라도 있는지 봅니다.

    후보를 대신 고르는 것이 아니라 "더 고를 수 있는가"만 판정합니다.
    범위 전체가 이미 차 있으면 "다른 시간을 골라 다시 호출하라"는 안내가 영원히 만족될 수 없어
    같은 호출이 반복되므로, 그때는 note를 재시도가 아니라 종료 쪽으로 바꾸기 위해 필요합니다.
    """

    work_start = parse_time_minutes(workday_start, 9 * 60)
    work_end = parse_time_minutes(workday_end, 18 * 60)
    needed = max(30, int(duration_minutes or 60))
    for day in date_range(date_from, date_to):
        blocked: list[tuple[int, int]] = []
        for row in busy_rows:
            if str(row.get("date") or "") != day:
                continue
            start = max(work_start, parse_time_minutes(row.get("start_time"), 0))
            end = min(work_end, parse_time_minutes(row.get("end_time"), 24 * 60))
            if end > start:
                blocked.append((start, end))
        cursor = work_start
        for start, end in sorted(blocked):
            if start - cursor >= needed:
                return True
            cursor = max(cursor, end)
        if work_end - cursor >= needed:
            return True
    return False


def _find_common_available_slots_note(
    *,
    requested_candidate_count: int,
    accepted_candidate_count: int,
    rejected_candidate_count: int,
    truncated_candidate_count: int,
    busy_row_count: int,
    free_window_exists: bool,
) -> str:
    """후보 검증 결과에 다음 행동을 함께 실어 보냅니다.

    이 tool은 후보를 대신 계산하지 않으므로 candidate_slots가 비면 결과도 조용히 0건입니다.
    description(호출 전 근거)만으로는 확률이라, tool 결과(호출 후 판단 지점)에도 같은 계약을 남깁니다.
    재시도를 유도하는 note는 "다시 하면 되는 상황"에서만 안전하므로, 빈 구간이 아예 없을 때는
    같은 안내가 무한 재호출이 되기 전에 종료 조건을 대신 알려 줍니다.
    """

    if not free_window_exists:
        return (
            f"요청한 날짜 범위와 업무 시간 안에는 {busy_row_count}건의 일정 때문에 "
            "회의 길이만큼 연속으로 비는 구간이 없습니다. 후보를 다시 골라 이 tool을 재호출하지 마세요. "
            "decide_final_slot을 final_slot=null, needs_agent_selection=true로 호출해 마무리하고, "
            "사용자에게는 가능한 시간이 없다는 사실과 함께 날짜 범위를 넓힐지 물어보세요."
        )
    if not requested_candidate_count:
        return (
            f"이 tool은 후보를 계산하지 않습니다. 후보 0건은 '가능한 시간이 없다'는 뜻이 아니라 "
            f"candidate_slots를 안 채웠다는 뜻입니다. busy_rows {busy_row_count}건을 직접 읽고 "
            "겹치지 않는 시간을 골라 candidate_slots에 채워 다시 호출하세요. "
            "후보를 채우지 않은 채로 사용자에게 답하지 말고, 이 안내를 사용자에게 그대로 전달하지도 마세요."
        )
    if not accepted_candidate_count:
        return (
            f"넘긴 후보 {requested_candidate_count}건이 모두 제외됐습니다. "
            "busy_rows와 겹치거나, 업무 시간 밖이거나, duration_minutes보다 짧은 후보입니다. "
            "비어 있는 구간은 남아 있으니 busy_rows를 다시 읽고 다른 시간을 골라 한 번 더 호출하세요. "
            "그래도 못 찾으면 decide_final_slot을 needs_agent_selection=true로 호출해 마무리하세요."
        )
    detail: list[str] = []
    if rejected_candidate_count:
        detail.append(f"겹침·업무시간·길이 조건으로 {rejected_candidate_count}건 제외")
    if truncated_candidate_count:
        detail.append(f"limit 초과로 {truncated_candidate_count}건 생략")
    detail_text = f" ({', '.join(detail)})" if detail else ""
    return (
        f"후보 {accepted_candidate_count}건이 검증됐습니다{detail_text}. "
        "여기서 답변을 끝내지 말고 이 candidate_slots로 decide_final_slot을 호출해 "
        "그중 하나를 최종 시간으로 확정하세요. 사용자가 나열만 요청한 경우에만 미확정으로 둡니다."
    )


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
    # 내 일정도 겹침 판단의 근거이므로 조회 대상에 "나"를 함께 넣습니다.
    members_with_me = [
        PERSONAL_SHARED_MEMBER_NAME,
        *[name for name in normalized_members if name != PERSONAL_SHARED_MEMBER_NAME],
    ]
    normalized_date_from = normalize_date_bound(date_from)
    normalized_date_to = normalize_date_bound(date_to)

    rows = busy_rows
    if rows is None:
        collected = json.loads(
            collect_member_schedules.invoke(
                {
                    "member_names": members_with_me,
                    "date_from": normalized_date_from,
                    "date_to": normalized_date_to,
                }
            )
        )
        rows = collected.get("rows", [])

    requested_candidate_count = len(candidate_slots or [])
    payload = find_common_available_slots_payload(
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
    # "검증에서 탈락한 후보"와 "limit 때문에 결과에서 잘린 후보"는 원인이 다르므로 따로 셉니다.
    # limit을 후보 수만큼 풀어 한 번 더 정규화하면 검증만 통과한 개수를 알 수 있습니다.
    validated_candidate_count = len(
        normalize_llm_candidate_slots(
            candidate_slots=candidate_slots,
            llm_reason=llm_reason,
            date_from=normalized_date_from,
            date_to=normalized_date_to,
            busy_rows=rows,
            duration_minutes=duration_minutes,
            workday_start=workday_start,
            workday_end=workday_end,
            limit=max(requested_candidate_count, 1),
        )
    )
    accepted_candidate_count = len(payload["candidate_slots"])
    free_window_exists = _free_window_exists(
        busy_rows=rows,
        date_from=normalized_date_from,
        date_to=normalized_date_to,
        duration_minutes=duration_minutes,
        workday_start=workday_start,
        workday_end=workday_end,
    )

    # decide_final_slot에 그대로 넘길 값이라 결과에 함께 남깁니다.
    payload["date_from"] = normalized_date_from
    payload["date_to"] = normalized_date_to
    payload["duration_minutes"] = duration_minutes
    payload["rejected_candidate_count"] = requested_candidate_count - validated_candidate_count
    payload["truncated_candidate_count"] = validated_candidate_count - accepted_candidate_count
    payload["range_fully_busy"] = not free_window_exists
    payload["note"] = _find_common_available_slots_note(
        requested_candidate_count=requested_candidate_count,
        accepted_candidate_count=accepted_candidate_count,
        rejected_candidate_count=payload["rejected_candidate_count"],
        truncated_candidate_count=payload["truncated_candidate_count"],
        busy_row_count=len(rows),
        free_window_exists=free_window_exists,
    )
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
    # top-level final_slot / reason / candidates는 course repo 계약이라 그대로 두고 note만 덧붙입니다.
    if payload.get("needs_agent_selection") and payload.get("candidates"):
        payload["note"] = (
            "최종 시간이 아직 확정되지 않았습니다. 확정할 후보를 골랐다면 selected_index와 final_slot을 "
            "채워 다시 호출하고, 사용자에게는 확정된 것처럼 답하지 말고 후보와 미확정 사실을 그대로 전하세요."
        )
    return json.dumps(payload, ensure_ascii=False)


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


def _run_subagent(agent: Any, query: str) -> tuple[str, list[dict[str, Any]], list[str]]:
    """하위 agent를 실행하고 최종 답변, trace event, 내부 tool 호출 이름을 함께 뽑습니다."""

    result = agent.invoke({"messages": [{"role": "user", "content": query}]})
    events = extract_agent_events(result)
    return extract_final_text(result), events, _tool_call_names(events)


def _final_payloads_from_events(
    events: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Kana 하위 trace에서 최종 시간 결정 payload를 끌어올립니다.

    supervisor는 하위 agent의 event를 직접 보지 못하므로, decide_final_slot 결과를
    kana_agent 반환 JSON까지 올려야 extract_langchain_trace()가 UI trace에 실을 수 있습니다.
    """

    final_slot_payload: dict[str, Any] | None = None
    final_decision_payload: dict[str, Any] | None = None
    for event in events:
        if event.get("event") != "tool_result":
            continue
        content = event.get("content")
        if not isinstance(content, dict):
            continue
        if "final_slot" in content:
            final_slot_payload = content
        if content.get("final_decision"):
            final_decision_payload = content["final_decision"]
    return final_slot_payload, final_decision_payload


def _kana_result_note(
    final_slot_payload: dict[str, Any] | None,
    inner_tool_names: list[str],
) -> str | None:
    """조율이 최종 확정까지 가지 못한 경우를 supervisor의 판단 지점에 남깁니다."""

    if final_slot_payload is None:
        if "find_common_available_slots" in inner_tool_names:
            return (
                "Kana가 후보까지만 만들고 decide_final_slot을 호출하지 않았습니다. "
                "최종 시간이 확정된 것처럼 답하지 말고 후보와 미확정 사실을 그대로 전하세요."
            )
        return None
    if final_slot_payload.get("needs_agent_selection") or not final_slot_payload.get("final_slot"):
        return (
            "Kana가 최종 시간을 확정하지 않았습니다(needs_agent_selection). "
            "candidates를 후보로만 안내하고 확정 표현을 쓰지 마세요."
        )
    return None


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
    answer, events, inner_tool_names = _run_subagent(_NANA_SUBAGENT, query)
    return json.dumps(
        {
            "ok": True,
            "tool_name": "nana_agent",
            "selected_agent": "nana_agent",
            "answer": answer,
            "trace": events,
            "inner_tool_names": inner_tool_names,
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
    answer, events, inner_tool_names = _run_subagent(_KANA_SUBAGENT, query)
    final_slot_payload, final_decision_payload = _final_payloads_from_events(events)
    payload: dict[str, Any] = {
        "ok": True,
        "tool_name": "kana_agent",
        "selected_agent": "kana_agent",
        "answer": answer,
        "trace": events,
        "inner_tool_names": inner_tool_names,
        "final_slot_payload": final_slot_payload,
        "final_decision_payload": final_decision_payload,
    }
    note = _kana_result_note(final_slot_payload, inner_tool_names)
    if note:
        payload["note"] = note
    return json.dumps(payload, ensure_ascii=False)


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
