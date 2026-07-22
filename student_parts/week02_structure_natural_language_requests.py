from __future__ import annotations

import json
import re
from typing import Any, Literal

from langchain.agents import create_agent
from langchain.tools import tool
from pydantic import BaseModel, Field, field_validator

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

    kind: RequestKind = Field(
        description="요청 종류. personal_schedule/group_schedule/todo/reminder/unknown 중 하나."
    )
    title: str | None = Field(default=None, description="일정/할 일 제목. 확실하지 않으면 None.")
    date: str | None = Field(default=None, description="YYYY-MM-DD 형식 날짜. 확실하지 않으면 채우지 않고 None으로 둔다.")
    start_time: str | None = Field(default=None, description="HH:MM 형식 시작 시각. 확실하지 않으면 채우지 않고 None으로 둔다.")
    end_time: str | None = Field(default=None, description="HH:MM 형식 종료 시각. 확실하지 않으면 채우지 않고 None으로 둔다.")
    members: list[str] = Field(default_factory=list, description="참석자/관련 멤버 이름 목록. 모르면 빈 목록.")
    priority: str | None = Field(default=None, description="할 일 우선순위. 확실하지 않으면 None.")
    reason: str | None = Field(default=None, description="이 kind로 분류한 판단 근거.")
    original_text: str = Field(default="", description="이 요청에 해당하는 사용자 원문 문장 보존용 필드.")

    @field_validator("date")
    @classmethod
    def _validate_date_format(cls, value: str | None) -> str | None:
        """YYYY-MM-DD 형식이 아니면 확실하지 않은 값으로 보고 None으로 정규화합니다."""

        if value is not None and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            return None
        return value

    @field_validator("start_time", "end_time")
    @classmethod
    def _validate_time_format(cls, value: str | None) -> str | None:
        """HH:MM 형식이 아니면 확실하지 않은 값으로 보고 None으로 정규화합니다."""

        if value is not None and not re.fullmatch(r"\d{2}:\d{2}", value):
            return None
        return value


class StructuredRequestBatch(BaseModel):
    """여러 자연어 의도를 StructuredRequest 목록으로 나누는 2차 과제 스키마입니다."""

    requests: list[StructuredRequest] = Field(
        default_factory=list,
        description="구조화된 개별 요청 목록. 요청이 하나뿐이어도 리스트 안에 하나만 담는다.",
    )
    base_date: str = Field(
        default_factory=current_app_date_iso,
        description="상대 날짜 표현(내일, 다음 주 등)을 해석하는 기준일(YYYY-MM-DD).",
    )


def _coerce_structured_request(value: Any) -> StructuredRequest:
    """이후 회차에서 사용할 StructuredRequest 정규화 예약 함수입니다."""

    if isinstance(value, StructuredRequest):
        return value
    if isinstance(value, dict):
        return StructuredRequest.model_validate(value)
    raise RuntimeError(
        "StructuredRequest 또는 dict 형태의 structured output이 필요합니다. "
        f"현재 타입: {type(value).__name__}"
    )


def extract_structured_request(text: str) -> StructuredRequest:
    """이후 회차에서 사용할 단건 구조화 예약 함수입니다."""

    structured_model = chat_model().with_structured_output(StructuredRequest, method="function_calling")
    result = structured_model.invoke(
        [
            {"role": "system", "content": join_system_prompt(week02_prompt_parts())},
            {"role": "user", "content": text},
        ]
    )
    return _coerce_structured_request(result)


@tool
def extract_schedule_request(query: str) -> str:
    """이후 회차에서 저장 흐름과 연결할 예약 tool입니다."""

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

    return join_system_prompt([
        "너는 이제 자연어 요청을 구조화하는 2주차 담당자다.",
        *week02_prompt_parts(),
        "최종 답변은 항상 StructuredRequestBatch 형식이며, 요청이 하나뿐이어도 requests 리스트 안에 "
        "StructuredRequest 하나를 담아 반환한다.",
        "kind가 personal_schedule로 확정된 요청에 한해서만, personal_create_schedule tool 결과 JSON의 "
        "created_schedule 필드(title/date/start_time/end_time/attendees)를 필드 채우기 근거로 쓴다. "
        "tool을 호출했다는 사실 자체가 kind를 personal_schedule로 바꾸는 근거는 아니다.",
        "created_schedule의 start_time/end_time이 '미정'이거나 원문에 명시적 근거가 없는 값이면, 그 필드는 "
        "채우지 않고 None으로 남긴다.",
    ])


def week02_prompt_parts() -> list[str]:
    """2주차부터 누적되는, 이후 주차에서도 계속 유지돼야 하는 행동 규칙입니다."""

    return [
        *week01_prompt_parts(),
        "kind는 반드시 사용자 원문 문장의 의미만으로 먼저 판단한다. tool을 호출했는지 여부는 kind 판단의 "
        "근거가 될 수 없다.",
        "personal_schedule과 group_schedule은 members에 이름이 특정된 상대가 있는지로 구분한다. "
        "문장에서 이름이 특정된 상대가 1명 이상 확인되면 그 이름들을 members에 채우고 kind는 "
        "group_schedule이다. 이름이 특정된 상대가 없으면(병원 예약, 헬스장처럼 기관/장소만 있는 경우 "
        "포함) kind는 personal_schedule이고 members는 빈 리스트로 둔다.",
        "예: '철수 만나기로 했어' → group_schedule, members=['철수']. '지수랑 저녁 먹기로 했어' → "
        "group_schedule, members=['지수']. '병원 예약 했어' → personal_schedule, members=[].",
        "personal_create_schedule은 회의, 약속, 병원 예약처럼 특정 시각에 실제로 만나거나 참석해야 "
        "하는 이벤트이고 날짜·시간이 문장에서 확인 가능할 때 호출한다(kind가 personal_schedule이든 "
        "group_schedule이든 상관없다). '장보기', '청소하기', '빨래하기'처럼 시간 표현이 있어도 "
        "완료해야 할 작업(chore) 성격이면 kind는 todo이고, tool을 호출하지 않는다. 날짜·시간이 "
        "불명확해도 호출하지 않는다.",
        "'3시에 약 먹으라고 알려줘', '내일 9시에 깨워줘'처럼 특정 시각에 스스로에게 알림만 받으면 되는 "
        "요청은 kind를 reminder로 분류한다. 완료 여부가 중요한 작업(장보기, 청소 등)은 todo, 만나거나 "
        "참석해야 하는 이벤트는 personal_schedule/group_schedule, 시각을 못 놓치도록 알려주기만 하면 "
        "되는 요청은 reminder로 구분한다.",
        "사용자의 자연어 요청이나 Week 1 tool이 반환한 JSON을 kind/title/date/start_time/end_time/"
        "members/priority/reason/original_text 필드로 구조화한다.",
        "Week 1 tool(personal_create_schedule 등) 결과 JSON을 이미 받았다면 같은 tool을 다시 호출하지 않고, "
        "그 JSON payload를 읽어 구조화 결과를 채운다.",
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
