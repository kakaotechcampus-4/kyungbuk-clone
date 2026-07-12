from __future__ import annotations

import json
from typing import Any, Literal

from langchain.agents import create_agent
from langchain.tools import tool
from pydantic import BaseModel, Field

from fixed.config import CONFIG
from fixed.llm import chat_model
from fixed.runtime_clock import current_app_date_iso
from student_parts.week01_wake_up_nana import join_system_prompt, week01_prompt_parts, week01_tools


RequestKind = Literal["personal_schedule", "group_schedule", "todo", "reminder", "unknown"]
_WEEK02_AGENT: Any | None = None


# [2주차 1회차 수강생 구현 가이드]
#
# 목표
#   Week 1 tool이 만든 JSON payload나 사용자의 한국어 자연어 요청을 일정 앱이 읽을 수 있는
#   StructuredRequest/StructuredRequestBatch로 바꿉니다. Week 1은 이미 정해진 인자를 받아
#   임시 일정을 만들었다면, Week 2는 그 tool 결과 JSON과 "내일 오후 3시" 같은 자연어를
#   날짜/시간/종류/멤버 필드로 구조화하는 단계입니다. 구조화 결과는 아직 저장하지 않습니다.
#
# 구현 위치와 사용할 코드
#   - 이 파일(student_parts/week02_structure_natural_language_requests.py)의 StructuredRequest 스키마와
#     StructuredRequestBatch, week02_tools(), week02_prompt_parts(), week02_system_prompt(),
#     build_week02_agent()를 확인합니다.
#   - build_week02_agent()는 langchain.agents.create_agent, fixed/llm.py의 chat_model(),
#     week02_system_prompt(), response_format=StructuredRequestBatch를 사용해 Week 2 agent를 만듭니다.
#   - week02_tools()는 Week 1 도구 목록을 그대로 가져옵니다. Week 2 agent는 개인 일정 생성 요청에서
#     personal_create_schedule이 반환한 created_schedule JSON payload를 읽고
#     response_format=StructuredRequestBatch로 최종 구조화 결과를 확인합니다.
#   - week02_prompt_parts()는 student_parts/week01_wake_up_nana.py의 week01_prompt_parts() 위에
#     Week 2 구조화 지시를 추가합니다.
#
# 구현 대상
#   1. StructuredRequest 스키마
#      - kind/title/date/start_time/end_time/members/priority/reason/original_text 필드가
#        이후 Week 3 저장 payload의 기준이 됩니다.
#      - kind는 RequestKind Literal에 들어 있는 값만 허용합니다.
#      - 각 필드에는 LLM structured output이 이해할 수 있도록 한국어 description을 붙입니다.
#
#   2. StructuredRequestBatch 스키마
#      - requests에는 StructuredRequest 목록을 담고, 요청이 하나뿐이어도 list 형태를 유지합니다.
#      - base_date에는 상대 날짜 해석 기준일(current_app_date_iso)을 담습니다.
#
#   3. Week 2 agent 세로 슬라이스
#      - week02_tools()는 Week 1 tool 목록을 그대로 반환합니다.
#      - week02_prompt_parts()와 week02_system_prompt()에는 자연어/Week 1 tool JSON을
#        StructuredRequestBatch로 구조화하라는 지시를 넣습니다.
#      - build_week02_agent()에 response_format=StructuredRequestBatch를 연결해
#        ./run.sh --week2가 동작하게 합니다.
#      - 개인 일정 생성 요청에서는 Week 1 personal_create_schedule tool 결과의 created_schedule JSON을
#        LLM이 읽어 StructuredRequestBatch로 최종 변환하는 흐름을 확인합니다.
#
# StructuredRequest 읽는 법
#   - kind: personal_schedule, group_schedule, todo, reminder, unknown 중 하나입니다.
#   - title/date/start_time/end_time: 일정 앱이 실제 저장이나 생성에 사용할 핵심 필드입니다.
#   - members: 참석자/관련 멤버 list입니다. 모르면 빈 list로 둡니다.
#   - priority/reason/original_text: 할 일 우선순위, 판단 근거, 원문 보존용 필드입니다.
#   - 모르는 값을 억지로 만들지 않는 것이 중요합니다. 확실하지 않으면 None 또는 빈 list가 안전합니다.
#   - date/start_time/end_time은 확실할 때만 YYYY-MM-DD, HH:MM 형식으로 채웁니다.
#
# 참고 코드
#   - week01_prompt_parts()
#      Week 1 system prompt를 이어받아 Week 2 구조화 지시를 누적할 때 사용합니다.
#   - week01_tools()
#      Week 1 개인 일정 tool 목록입니다. Week 2 agent는 이 tool 결과 JSON을 구조화 근거로 씁니다.
#
# 검증 방법
#   ./run.sh --week2로 실행한 뒤 "다음 주 화요일 오후 3시에 철수랑 회의 잡아줘" 같은 문장을 입력합니다.
#   최종 답변이 StructuredRequestBatch class 형식의 structured_response로 나오는지 확인합니다.
#
# 함수별 동작 설명
#   - StructuredRequest
#     Week 2 structured output의 중심 스키마입니다. LLM이 자연어에서 뽑은 요청 종류, 제목, 날짜, 시간,
#     멤버, 우선순위, 근거, 원문을 이 class 필드에 맞춰 반환합니다.
#
#   - StructuredRequestBatch
#     StructuredRequest 여러 개와 base_date를 함께 담는 최종 structured_response 스키마입니다.
#     요청이 하나뿐이어도 requests list 안에 StructuredRequest 하나를 담습니다.
#
#   - week02_tools()
#     Week 1 개인 일정 tool을 그대로 노출합니다. Week 2 agent는 개인 일정 생성 요청에서
#     created_schedule JSON을 structured_response의 근거로 사용할 수 있습니다.
#
#   - week02_system_prompt() / week02_prompt_parts()
#     Week 1 prompt 위에 "자연어를 StructuredRequestBatch로 출력한다"는 Week 2 지시를 누적합니다.
#
#   - build_week02_agent() / build_week_agent()
#     response_format=StructuredRequestBatch가 설정된 agent를 만들고 재사용합니다.
#     build_week_agent()는 실행기가 찾는 표준 entry point입니다.


class StructuredRequest(BaseModel):
    """LLM structured output으로 추출되는 2주차 요청 스키마입니다."""

    kind: RequestKind = Field(description="요청 종류: personal_schedule(개인 일정), group_schedule(그룹 일정), todo(할 일), reminder(리마인더), unknown(알 수 없음) 중 하나")
    title: str | None = Field(default=None, description="일정 또는 할 일의 제목. 확실하지 않으면 None")
    date: str | None = Field(default=None, description="일정 날짜. 확실한 경우에만 YYYY-MM-DD 형식으로 채움. 모르면 None")
    start_time: str | None = Field(default=None, description="시작 시각. 확실한 경우에만 HH:MM 형식으로 채움. 모르면 None")
    end_time: str | None = Field(default=None, description="종료 시각. 확실한 경우에만 HH:MM 형식으로 채움. 모르면 None")
    members: list[str] = Field(default_factory=list, description="참석자 또는 관련 멤버 이름 목록. 정보가 없으면 빈 리스트")
    priority: str | None = Field(default=None, description="할 일 우선순위(예: high/medium/low). 해당 없으면 None")
    reason: str | None = Field(default=None, description="요청 종류나 필드 값을 선택한 판단 근거. 없으면 None")
    original_text: str = Field(default="", description="사용자가 입력한 원문 텍스트. 원본 보존용")


class StructuredRequestBatch(BaseModel):
    """여러 자연어 의도를 StructuredRequest 목록으로 나누는 2차 과제 스키마입니다."""

    requests: list[StructuredRequest] = Field(default_factory=list, description="구조화된 요청 목록. 요청이 하나뿐이어도 반드시 리스트 형태를 유지")
    base_date: str = Field(default_factory=current_app_date_iso, description="상대 날짜(내일, 다음 주 등) 해석의 기준이 되는 오늘 날짜 (YYYY-MM-DD)")


def _coerce_structured_request(value: Any) -> StructuredRequest:
    """이후 회차에서 사용할 StructuredRequest 정규화 예약 함수입니다."""

    ...


def extract_structured_request(text: str) -> StructuredRequest:
    """이후 회차에서 사용할 단건 구조화 예약 함수입니다."""

    ...


@tool
def extract_schedule_request(query: str) -> str:
    """이후 회차에서 저장 흐름과 연결할 예약 tool입니다."""

    ...


def week02_tools() -> list[Any]:
    """Week 2 agent에 Week 1 도구를 노출해 tool JSON을 structured_response 근거로 씁니다."""

    return week01_tools()


def week02_system_prompt() -> str:
    """2주차 agent가 따르는 시스템 프롬프트입니다."""

    return join_system_prompt([
        *week02_prompt_parts(),
        f"""
        개인 일정 생성 요청에서는 반드시 personal_create_schedule tool을 먼저 호출하고,
        그 결과 JSON의 created_schedule 필드를 읽어 StructuredRequest 필드를 채운다.
        tool을 호출하기 전에 structured_response를 먼저 반환하지 않는다.

        최종 답변은 반드시 structured_response(StructuredRequestBatch) 형식으로만 출력한다.
        요청이 하나뿐이어도 requests 목록에 StructuredRequest 하나를 담아 반환한다.
        personal_create_schedule tool을 호출한 경우, 그 결과 JSON의 created_schedule 필드를 읽어
        StructuredRequest의 title/date/start_time/end_time/members 필드를 채운다.
        """,
    ])


def week02_prompt_parts() -> list[str]:
    """2주차 structured output agent가 따르는 system prompt 조각입니다."""

    return [
        *week01_prompt_parts(),
        f"""
        너는 Week 2 구조화 단계를 담당한다. 오늘 날짜는 {current_app_date_iso()}를 기준으로 한다.
        사용자의 자연어 요청 또는 Week 1 tool이 반환한 JSON을 읽어
        StructuredRequestBatch(requests, base_date)로 구조화해 반환한다.

        각 요청은 StructuredRequest의 kind/title/date/start_time/end_time/members/priority/reason/original_text 필드로 분류한다.
        - kind는 personal_schedule, group_schedule, todo, reminder, unknown 중 하나다.
        - members(참석자)가 명시된 일정 요청은 group_schedule, 참석자 언급 없이 본인 일정만 다루면 personal_schedule로 분류한다.
        - 완료해야 할 작업을 나열한 요청은 todo, 특정 시점에 알려/리마인드해 달라는 요청은 reminder로 분류, 위 기준으로도 판단이 어려우면 unknown으로 분류한다.
        - date는 YYYY-MM-DD, start_time/end_time은 HH:MM 형식으로 확실할 때만 채운다.
        - 모르는 값은 억지로 만들지 않고 None 또는 빈 리스트로 둔다.

        Week 1 tool 결과 JSON이 이미 있으면 tool을 다시 호출하지 않고 해당 payload를 읽어 structured_response를 만든다.
        Week 2에서는 SQLite 저장, RAG 검색, 외부 멤버 일정 조율은 하지 않는다.
        """,
    ]


def build_week02_agent() -> object:
    """Week 2 대화에서 structured_response를 직접 반환하는 단일 LangChain agent를 만듭니다."""

    if not CONFIG.has_openai_key:
        raise RuntimeError("PROXY_TOKEN이 .env에 필요합니다.")
    global _WEEK02_AGENT
    if _WEEK02_AGENT is None:
        _WEEK02_AGENT = create_agent(
            model=chat_model(),
            tools=week02_tools(),
            response_format=StructuredRequestBatch,
            system_prompt=week02_system_prompt(),
        )
    return _WEEK02_AGENT


def build_week_agent() -> object:
    """active-week registry가 호출하는 표준 Week agent builder입니다."""

    return build_week02_agent()
