from __future__ import annotations

import json
import re
from typing import Any, Literal

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import tool
from pydantic import BaseModel, Field, field_validator

from fixed.config import CONFIG
from fixed.llm import chat_model
from fixed.runtime_clock import current_app_date_iso
from student_parts.week01_wake_up_nana import join_system_prompt, week01_prompt_parts, week01_tools


RequestKind = Literal["personal_schedule", "group_schedule", "todo", "reminder", "unknown"]
_WEEK02_AGENT: Any | None = None
DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
TIME_PATTERN = re.compile(r"^\d{2}:\d{2}$")
UNKNOWN_TEXT_VALUES = {"", "미정", "모름", "알 수 없음", "unknown", "none", "null"}


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

    kind: RequestKind = Field(
        default="unknown",
        description="요청의 종류입니다. 개인 일정은 personal_schedule, 단체 일정은 group_schedule, 할 일은 todo, 알림은 reminder, 판단이 어려우면 unknown입니다.",
    )
    title: str | None = Field(default=None, description="일정, 할 일, 알림의 제목 또는 사용자가 요청한 핵심 내용입니다.")
    date: str | None = Field(default=None, description="요청 날짜입니다. 확실할 때만 YYYY-MM-DD 형식으로 채웁니다.")
    start_time: str | None = Field(default=None, description="시작 시간입니다. 확실할 때만 HH:MM 형식으로 채웁니다.")
    end_time: str | None = Field(default=None, description="종료 시간입니다. 확실할 때만 HH:MM 형식으로 채우고 모르면 None입니다.")
    members: list[str] = Field(default_factory=list, description="참석자나 관련 멤버 이름 목록입니다. 모르면 빈 목록입니다.")
    priority: str | None = Field(default=None, description="할 일이나 알림의 우선순위입니다. 사용자가 말하지 않았거나 판단하기 어려우면 None입니다.")
    reason: str | None = Field(default=None, description="필드를 이렇게 구조화한 짧은 판단 근거입니다.")
    original_text: str = Field(default="", description="구조화의 근거가 된 사용자 원문 또는 tool JSON 원문입니다.")

    @field_validator("date")
    @classmethod
    def validate_date_format(cls, value: str | None) -> str | None:
        """날짜는 알 수 없으면 None, 값이 있으면 YYYY-MM-DD만 허용합니다."""

        if value is None:
            return None
        value = value.strip()
        if value.lower() in UNKNOWN_TEXT_VALUES:
            return None
        if not DATE_PATTERN.fullmatch(value):
            raise ValueError("date는 YYYY-MM-DD 형식이거나 None이어야 합니다.")
        return value

    @field_validator("start_time", "end_time")
    @classmethod
    def validate_time_format(cls, value: str | None) -> str | None:
        """시간은 알 수 없으면 None, 값이 있으면 HH:MM만 허용합니다."""

        if value is None:
            return None
        value = value.strip()
        if value.lower() in UNKNOWN_TEXT_VALUES:
            return None
        if not TIME_PATTERN.fullmatch(value):
            raise ValueError("시간 필드는 HH:MM 형식이거나 None이어야 합니다.")
        return value


class StructuredRequestBatch(BaseModel):
    """여러 자연어 의도를 StructuredRequest 목록으로 나누는 2차 과제 스키마입니다."""

    requests: list[StructuredRequest] = Field(
        default_factory=list,
        description="사용자 요청을 의도별로 나눈 구조화 결과 목록입니다. 요청이 하나뿐이어도 목록에 하나를 담습니다.",
    )
    base_date: str = Field(
        default_factory=current_app_date_iso,
        description="오늘, 내일, 다음 주 같은 상대 날짜 표현을 해석할 때 기준이 되는 앱 실행 날짜입니다.",
    )


def _coerce_structured_request(value: Any) -> StructuredRequest:
    """LangChain structured output 결과를 StructuredRequest로 정규화합니다."""

    if isinstance(value, StructuredRequest):
        return value
    if isinstance(value, dict):
        return StructuredRequest.model_validate(value)
    raise RuntimeError(
        "StructuredRequest 또는 dict 형태의 structured output이 필요합니다. "
        f"현재 타입: {type(value).__name__}"
    )


def extract_structured_request(text: str) -> StructuredRequest:
    """Week 3 이상에서 agent를 새로 띄우지 않고 자연어를 StructuredRequest로 바꿉니다."""

    structured_model = chat_model().with_structured_output(
        StructuredRequest,
        method="function_calling",
    )
    result = structured_model.invoke(
        [
            {"role": "system", "content": join_system_prompt(week02_prompt_parts())},
            {"role": "user", "content": text},
        ]
    )
    return _coerce_structured_request(result)


@tool
def extract_schedule_request(query: str) -> str:
    """Week 3 이상 agent가 저장/조율 전에 호출하는 구조화 bridge tool입니다."""

    structured = extract_structured_request(query)
    return json.dumps(
        {
            "ok": True,
            "tool_name": "extract_schedule_request",
            "base_date": current_app_date_iso(),
            "structured_request": structured.model_dump(),
        },
        ensure_ascii=False,
    )


def week02_tools() -> list[Any]:
    """Week 2 agent에 Week 1 도구를 노출해 tool JSON을 structured_response 근거로 씁니다."""

    return week01_tools()


def week02_system_prompt() -> str:
    """2주차 agent가 따르는 시스템 프롬프트입니다."""

    return join_system_prompt(
        [
            *week02_prompt_parts(),
            """
            Week 2 최종 답변 규칙:
            - 최종 출력은 반드시 StructuredRequestBatch structured_response로 만든다.
            - 요청이 하나뿐이어도 requests 목록 안에 StructuredRequest 하나를 담는다.
            - base_date에는 상대 날짜 해석 기준일을 유지한다.
            - personal_create_schedule tool을 호출한 경우 tool 결과 JSON의 created_schedule을 읽어
              title/date/start_time/end_time/members 필드를 채운다.
            - tool 이름이 personal_create_schedule이어도, 원문에 참석자/상대방/멤버가 있으면
              kind는 personal_schedule이 아니라 group_schedule로 분류한다.
            - 사용자에게 별도 저장 완료, DB 반영, 외부 캘린더 연동을 약속하지 않는다.
            """,
        ]
    )


def week02_prompt_parts() -> list[str]:
    """2주차 structured output agent가 따르는 system prompt 조각입니다."""

    return [
        *week01_prompt_parts(),
        f"""
        너는 Week 2 요청 구조화 agent다.
        오늘 날짜이자 상대 날짜 해석 기준일은 {current_app_date_iso()}다.

        역할:
        - 사용자의 한국어 자연어 요청을 일정 앱이 읽을 수 있는 StructuredRequestBatch로 구조화한다.
        - 각 요청은 kind/title/date/start_time/end_time/members/priority/reason/original_text 필드로 나눈다.
        - personal_schedule, group_schedule, todo, reminder 중 맞는 kind를 고르고 확실하지 않으면 unknown으로 둔다.
        - 모르는 값을 억지로 만들지 말고 None 또는 빈 목록으로 둔다.
        - date는 YYYY-MM-DD, 시간은 HH:MM 형식을 사용한다.

        Week 1 tool JSON 처리:
        - 개인 일정 생성처럼 Week 1 tool 호출이 필요한 요청은 personal_create_schedule을 사용할 수 있다.
        - personal_create_schedule은 Week 1에서 만든 임시 tool 이름일 뿐이므로 kind 판단 근거로 삼지 않는다.
        - tool 결과 JSON을 받으면 다시 같은 tool을 호출하지 말고 created_schedule payload를 읽어 structured_response로 옮긴다.
        - attendees는 StructuredRequest.members로 변환한다.
        - attendees/members가 비어 있지 않거나 원문에 함께할 사람이 있으면 kind는 group_schedule이다.

        Week 2 범위 제한:
        - 이번 주차는 자연어와 Week 1 tool JSON을 구조화하는 단계까지만 수행한다.
        - SQLite 저장, RAG 검색, 외부 멤버 일정 조율, 실제 캘린더 연동은 하지 않는다.
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
            response_format=ToolStrategy(StructuredRequestBatch),
            system_prompt=week02_system_prompt(),
        )
    return _WEEK02_AGENT


def build_week_agent() -> object:
    """active-week registry가 호출하는 표준 Week agent builder입니다."""

    return build_week02_agent()
