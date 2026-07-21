from __future__ import annotations

import json
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
    """
        사용자의 요청을 구조화된 결과를 담는 스키마입니다.

        요청의 유형, 일정의 제목, 날짜, 일정의 시작 날짜, 종료 날짜, 일정에 포함되는 사람들의 이름, 일정의 우선순위,
        일정의 우선순위 지정 이유, 요청받은 원본 텍스트를 필드 형태로 표현합니다.

        kind 같은 경우 RequestKind Literal에 정의된 값만 허용하며 다음 규칙을 따릅니다.

        kind 결정 우선순위:
        1. original_text에 다른 사람 이름이나 참석자가 있으면 group_schedule
        2. 혼자 하는 일정이면 personal_schedule
        3. 할 일이면 todo
        4. 알림이면 reminder
        5. 판단 불가하면 unknown

        단, tool 호출 결과나 tool 이름은 kind 결정에 절대 사용하지 않습니다.
        
        priority 같은 경우 일정의 중요도를 평가하며, "중요하다"와 유사한 표현이 들어간다면 high, "보통이다"와 유사한 표현이 들어간다면 medium, "중요하지 않다"와 유사한 표현이 들어간다면 low, 
        그 외에는 None으로 채웁니다. priority 값은 "high","medium","low"중 하나만 허용하며, validate_priority validator를 통해 값을 필수적으로 검증합니다.
        
        확실하지 않을경우 억지로 채우려 하지말고, default값을 담습니다.
    """

    kind : RequestKind = Field(description="요청의 유형 지정")
    title : str | None = Field(description="일정 제목", default = None)
    date : str | None = Field(description="YYYY-MM-DD", default = None)
    start_time : str | None = Field(description="HH:MM", default = None)
    end_time : str | None = Field(description="HH:MM", default = None)
    members : list[str] = Field(description="멤버들의 이름을 list로 저장", default_factory=list)
    priority : str | None = Field(description="우선순위 지정", default=None)
    reason : str | None = Field(description="우선순위 지정 이유", default=None)
    original_text : str = Field(description="원본 프롬프트 저장", default = "")

    @field_validator("priority")
    @classmethod
    def validate_priority(cls,value: str | None) -> str | None:
        """priority 필드에 들어오는 값이 high, medium, low 중 하나인지 검증하는 validator입니다."""
        if value is None:
            return None
        
        allowed = {"high","medium","low"}

        if value not in allowed :
            raise ValueError(f"priority는 {allowed} 중 하나여야 합니다. 현재 값: {value}")
            
        return value


class StructuredRequestBatch(BaseModel):
    """
    여러 자연어 요청을 StructuredRequest 목록으로 분리한 결과를 담는 스키마입니다.

    requests에는 분리한 요청들을 list type으로 저장하고, base_date에는 상대 날짜 표현 해석할 때 기준이 되는 날짜를 저장합니다.
    """

    requests : list[StructuredRequest] = Field(description="요청받은 StructuredRequest들을 저장  ",default_factory=list)
    base_date : str = Field(description="날짜 표현 해석할 때 기준이 되는 오늘 날짜",default_factory=current_app_date_iso)



def _coerce_structured_request(value: Any) -> StructuredRequest:
    """LLM structured output 결과를 StructuredRequest 하나로 정규화합니다.

    with_structured_output이 리턴하는 값은 실행 환경에 따라 이미 StructuredRequest
    인스턴스일 수도, 그 필드를 담은 dict일 수도 있습니다. 둘 다 여기서 StructuredRequest로
    통일해서 이후 코드가 항상 같은 타입을 다루게 합니다. 그 외 타입은 정규화할 방법이
    없으므로 TypeError를 던집니다.
    """

    if isinstance(value,StructuredRequest) :
        return value
    if isinstance(value,dict):
        return StructuredRequest.model_validate(value)
    raise TypeError("dict/StructuredRequest type만이 사용되야합니다.")


def extract_structured_request(text: str) -> StructuredRequest:
    """자연어 문장 하나를 별도 LLM 호출로 구조화해 StructuredRequest 하나를 반환합니다.

    Week 2 agent와 달리 이 함수는 batch(StructuredRequestBatch)가 아니라 단건만 다루는
    "bridge" 함수입니다. Week 3 이상에서 agent를 새로 띄우지 않고, 이미 만든 chat_model에
    with_structured_output(StructuredRequest)을 붙여 즉석에서 자연어를 구조화할 때 씁니다.
    system 메시지는 Week 2 system prompt를 그대로 재사용해 구조화 기준(kind 판단 규칙,
    priority 규칙 등)을 동일하게 따르게 합니다.
    """

    structured_model = chat_model().with_structured_output(
        StructuredRequest,
        method = "function_calling"
    )

    result = structured_model.invoke([
        {"role": "system", "content": week02_system_prompt()},
        {"role": "user", "content" : text}
    ])

    return _coerce_structured_request(result)


@tool
def extract_schedule_request(query: str) -> str:
    """Week 3 이상 agent가 저장/조회 전에 호출하는 구조화 bridge tool입니다.

    사용자의 자연어 요청(query)을 extract_structured_request로 구조화한 뒤,
    그 StructuredRequest를 그대로 저장 tool(save_structured_request)에 넘길 수 있도록
    ok/tool_name/base_date와 함께 JSON 문자열로 감싸 반환합니다. agent는 이 tool을 먼저
    호출해 자연어를 구조화하고, 반환된 structured_request 필드를 다음 저장 tool 호출의
    인자로 그대로 사용합니다.
    """

    structured = extract_structured_request(query)

    return json.dumps({
        "ok" : True,
        "tool_name" : "extract_schedule_request",
        "base_date" : current_app_date_iso(),
        "structured_request" : structured.model_dump()
    }, ensure_ascii=False
    )


def week02_tools() -> list[Any]:
    """Week 2 agent에 Week 1 도구를 노출해 tool JSON을 structured_response 근거로 씁니다."""

    return week01_tools()


def week02_system_prompt() -> str:
    """2주차 agent가 따르는 시스템 프롬프트입니다."""

    return join_system_prompt([
        *week02_prompt_parts(),
        """
        최종 응답형태인 structured_response는 반드시 StructuredRequestBatch 형태로 작성해야합니다.
        요청이 하나뿐이어도 requests 목록에 StructuredRequest 하나를 담아야합니다.

        일정 생성 tool을 사용한 결과, JSON의 created_schedule 값을 읽어 StructuredRequest의 필드를 채우세요.
        """
    ])


def week02_prompt_parts() -> list[str]:
    """2주차 structured output agent가 따르는 system prompt 조각입니다."""

    FEW_SHOT_EXAMPLES = """
    예시 1.
        "다음 주 화요일 오후 3시에 철수랑 회의 잡아줘"
        
        오늘 날짜와 요일이 2026-07-09 목요일 이라면, 다음주 화요일은 5일 뒤인 2026-07-14 화요일입니다.
        
        -> structured_response
        {
            "requests": [
                {
                    "kind": "group_schedule",
                    "title": "회의",
                    "date": "2026-07-14",
                    "start_time": "15:00",
                    "end_time": null,
                    "members": ["철수"],
                    "priority": null,
                    "reason": null,
                    "original_text": "다음 주 화요일 오후 3시에 철수랑 회의 잡아줘"
                }
            ],
            "base_date": "2026-07-09"
        },

    예시 2.
        "다음주 수요일 오전 10시에 IT1호관에서 철수랑 영희랑 동아리 스터디 일정 잡아줘. 중요한거니깐 잊으면 안돼"
        오늘 날짜와 요일이 2026-07-10 금요일 이라면, 다음주 수요일은 5일 뒤인 2026-07-15 수요일입니다.
        -> structured_response
        {
            "requests": [
                {
                    "kind": "group_schedule",
                    "title": "동아리 스터디",
                    "date": "2026-07-15",
                    "start_time": "10:00",
                    "end_time": null,
                    "members": ["철수","영희"],
                    "priority": "high",
                    "reason": "\"중요한\" 표현이 포함되어, 중요한 일정으로 판단됨",
                    "original_text": "다음주 수요일 오전 10시에 IT1호관에서 철수랑 영희랑 동아리 스터디 일정 잡아줘. 중요한거니깐 잊으면 안돼"
                }
            ],
            "base_date": "2026-07-10"
        }

    """
    return [
        *week01_prompt_parts(),
        """사용자의 자연어 요청을 StructuredRequest 스키마에 맞게 구조화하세요.""",
        """각 필드의 작성기준은 StructuredRequest의 docstring과 Field description을 따르세요.""",
        """ Week 1 tool 실행 결과로 JSON이 이미 주어진 경우에는 같은 도구를 다시 호출하지 않으며, payload를 읽어 structured_response 스키마에 맞춰만듭니다.""",
        """ Week 2에서 다음 작업은 수행하지 않습니다.
            - SQLite 저장
            - RAG 검색
            - 외부 멤버 일정 조율,조회
        """,
        FEW_SHOT_EXAMPLES
    ]


def build_week02_agent() -> object:
    """Week 2 대화에서 structured_response를 직접 반환하는 단일 LangChain agent를 만듭니다."""

    global _WEEK02_AGENT
    if not CONFIG.has_openai_key:
        raise RuntimeError("PROXY_TOKEN이 .env에 필요합니다.")
    if not _WEEK02_AGENT : 
        _WEEK02_AGENT = create_agent(
            model = chat_model(),
            tools = week02_tools(),
            response_format=StructuredRequestBatch,
            system_prompt=week02_system_prompt(),
        )
    return _WEEK02_AGENT


def build_week_agent() -> object:
    """active-week registry가 호출하는 표준 Week agent builder입니다."""

    return build_week02_agent()
