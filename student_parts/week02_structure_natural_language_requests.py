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


class StructuredRequest(BaseModel):
    """LLM structured output으로 추출되는 2주차 요청 스키마입니다."""

    kind: RequestKind = Field(
        description="요청 종류. personal_schedule/group_schedule/todo/reminder/unknown 중 하나."
    )
    title: str | None = Field(default=None, description="일정 또는 할 일의 제목. 확실하지 않으면 None.")
    date: str | None = Field(default=None, description="YYYY-MM-DD 형식의 날짜. 확실하지 않으면 None.")
    start_time: str | None = Field(default=None, description="HH:MM 형식의 시작 시각. 확실하지 않으면 None.")
    end_time: str | None = Field(default=None, description="HH:MM 형식의 종료 시각. 확실하지 않으면 None.")
    members: list[str] = Field(default_factory=list, description="참석자 또는 관련 멤버 목록. 모르면 빈 list.")
    priority: str | None = Field(default=None, description="할 일의 우선순위. 확실하지 않으면 None.")
    reason: str | None = Field(default=None, description="이 요청을 이렇게 구조화한 판단 근거.")
    original_text: str = Field(default="", description="구조화의 근거가 된 사용자 원문 텍스트.")


class StructuredRequestBatch(BaseModel):
    """여러 자연어 의도를 StructuredRequest 목록으로 나누는 2차 과제 스키마입니다."""

    requests: list[StructuredRequest] = Field(
        default_factory=list,
        description="자연어에서 구조화한 StructuredRequest 목록. 요청이 하나여도 list.",
    )
    base_date: str = Field(
        default_factory=current_app_date_iso,
        description="상대 날짜(예: '내일', '다음 주 화요일') 해석 기준일.",
    )


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

    final_answer_rules = (
        "최종 답변은 반드시 StructuredRequestBatch 형식의 structured_response로 반환한다. "
        "요청이 하나뿐이어도 requests 목록 안에 StructuredRequest 하나를 담는다. "
        "자유 텍스트로 답하지 않는다."
    )
    return join_system_prompt([*week02_prompt_parts(), final_answer_rules])


def week02_prompt_parts() -> list[str]:
    """2주차 structured output agent가 따르는 system prompt 조각입니다."""

    return [
        *week01_prompt_parts(),
        (
            f"너는 이제 2주차 요청 구조화 agent다. 오늘 기준일은 {current_app_date_iso()}이다. "
            "'내일', '다음 주 화요일' 같은 상대 날짜는 이 기준일로 계산한다."
        ),
        (
            "사용자의 자연어 요청을 StructuredRequest 필드(kind/title/date/start_time/end_time/"
            "members/priority/reason/original_text)로 구조화한다. "
            "확실하지 않은 값은 억지로 채우지 말고 None 또는 빈 list로 둔다."
        ),
        (
            "personal_create_schedule 같은 Week 1 tool을 호출해 JSON 결과를 받은 경우, "
            "그 tool을 다시 호출하지 말고 반환된 created_schedule payload를 읽어 "
            "structured_response의 필드를 채우는 데 사용한다."
        ),
        (
            "Week 2에서는 SQLite 저장, RAG 검색, 외부 멤버 일정 조율을 하지 않는다. "
            "구조화 결과를 만드는 것까지만 한다."
        ),
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