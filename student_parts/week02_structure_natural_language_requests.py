from __future__ import annotations

import json
from typing import Any, Literal

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from fixed.config import CONFIG
from fixed.llm import chat_model
from fixed.runtime_clock import current_app_date_iso
from student_parts.week01_wake_up_nana import join_system_prompt, week01_prompt_parts, week01_tools


RequestKind = Literal["personal_schedule", "group_schedule", "todo", "reminder", "unknown"]
_WEEK02_AGENT: Any | None = None


class StructuredRequest(BaseModel):
    """LLM structured output으로 추출되는 2주차 요청 스키마입니다."""

    kind: RequestKind = Field(description="요청 종류: personal_schedule, group_schedule, todo, reminder, unknown 중 하나")
    title: str | None = Field(None, description="일정/할 일 제목")
    date: str | None = Field(None, description="일정 날짜 (YYYY-MM-DD 형식)")
    start_time: str | None = Field(None, description="일정 시작 시간 (HH:MM 형식)")
    end_time: str | None = Field(None, description="일정 종료 시간 (HH:MM 형식)")
    members: list[str] = Field(default_factory=list, description="참석자/관련 멤버 목록")
    priority: str | None = Field(None, description="할 일 우선순위")
    reason: str | None = Field(None, description="kind 분류 판단 근거")
    original_text: str = Field("", description="사용자가 입력한 원문 텍스트")


class StructuredRequestBatch(BaseModel):
    """여러 자연어 의도를 StructuredRequest 목록으로 나누는 2차 과제 스키마입니다."""

    requests: list[StructuredRequest] = Field(default_factory=list, description="구조화된 요청 목록. 요청이 하나뿐이어도 리스트로 담는다")
    base_date: str = Field(default_factory=current_app_date_iso, description="상대 날짜 해석 기준일 (YYYY-MM-DD 형식)")


def _coerce_structured_request(value: Any) -> StructuredRequest:
    """LangChain structured output 결과를 StructuredRequest로 정규화합니다."""

    if isinstance(value, StructuredRequest):
        return value
    if isinstance(value, dict):
        return StructuredRequest.model_validate(value)
    raise RuntimeError(f"예상하지 못한 structured output 형태입니다: {type(value)!r}")


def extract_structured_request(text: str) -> StructuredRequest:
    """Week 3 이상에서 agent를 새로 띄우지 않고 자연어를 StructuredRequest로 바꿉니다."""

    structured_llm = chat_model().with_structured_output(StructuredRequest, method="function_calling")
    result = structured_llm.invoke(
        [
            SystemMessage(content=join_system_prompt(week02_prompt_parts())),
            HumanMessage(content=text),
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


def week02_prompt_parts() -> list[str]:
    """2주차 structured output agent가 따르는 system prompt 조각입니다."""

    return [
        *week01_prompt_parts(),
        f"""오늘은 {current_app_date_iso()}이다.
너는 사용자의 자연어 요청을 StructuredRequestBatch로 구조화하는 역할이다.

요청 종류(kind)는 반드시 아래 중 하나로 분류한다:
- personal_schedule: 혼자 하는 개인 일정
- group_schedule: 다른 사람과 함께하는 일정
- todo: 해야 할 일
- reminder: 알림
- unknown: 위 중 어느 것도 아닌 경우

Week 1 tool JSON을 이미 받은 경우 다시 tool을 호출하지 않고
created_schedule payload를 읽어서 structured_response로 만든다.
Week 2에서는 SQLite 저장, RAG, 외부 멤버 일정 조율을 하지 않는다.""",
    ]


def week02_system_prompt() -> str:
    """2주차 agent가 따르는 시스템 프롬프트입니다."""

    return join_system_prompt(
        week02_prompt_parts() + [
            """최종 답변은 반드시 StructuredRequestBatch 형식으로만 반환한다.
요청이 하나뿐이어도 requests 목록에 StructuredRequest 하나를 담는다.
personal_create_schedule tool 결과의 created_schedule JSON을 읽어 필드를 채운다.
tool을 이미 호출한 경우 다시 호출하지 않고 결과를 읽어 structured_response로 만든다.""",
        ]
    )


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