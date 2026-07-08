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

    kind: RequestKind = Field(description="요청 종류를 나타내는 필드입니다.")
    title: str | None = Field(None, description="일정 제목을 나타내는 필드입니다.")
    date: str | None = Field(None, description="일정 날짜를 YYYY-MM-DD 형식으로 나타내는 필드입니다.")
    start_time: str | None = Field(None, description="일정 시작 시간을 HH:MM 형식으로 나타내는 필드입니다.")
    end_time: str | None = Field(None, description="일정 종료 시간을 HH:MM 형식으로 나타내는 필드입니다.")
    members: list[str] = Field(default_factory=list, description="참석자 list 필드입니다.")
    priority: str | None = Field(None, description="우선순위를 나타내는 필드입니다.")
    reason: str | None = Field(None, description="판단 이유를 나타내는 필드입니다.")
    original_text: str = Field("", description="원문을 나타내는 필드입니다.")
    

class StructuredRequestBatch(BaseModel):
    """여러 자연어 의도를 StructuredRequest 목록으로 나누는 2차 과제 스키마입니다."""

    requests: list[StructuredRequest] = Field(default_factory=list, description="구조화된 요청 목록 필드입니다.")
    base_date: str = Field(default_factory=current_app_date_iso, description="상대 날짜 기준일 필드입니다.")



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

    return join_system_prompt([*week02_prompt_parts(),
    "요청이 하나만 있어도 요청 목록에 구조화된 요청 하나를 담으시오.",
    "personal_create_schedule tool 결과 JSON의 created_schedule을 읽어 필드를 채우시오."])


def week02_prompt_parts() -> list[str]:
    """2주차 structured output agent가 따르는 system prompt 조각입니다."""

    return [
        *week01_prompt_parts(),
        "너는 지금부터 사용자가 입력한 요청을 구조화하는 agent야. 오늘 날짜는 current_app_date_iso() 기준으로 판단해.",
        "명령어를 받으면 StructuredRequest 필드로 구조화하세요.",
        "week01 tool JSON 받은 경우에는 다시 tool을 호출하지 말고 payload를 읽어 structured_response로 만드시오.",
        "week02에서는 SQLite 저장, RAG, 외부 멤버 일정 조율을 하지 않습니다."
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
