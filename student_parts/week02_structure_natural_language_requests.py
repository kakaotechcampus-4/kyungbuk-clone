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

from langchain.agents.structured_output import ToolStrategy

RequestKind = Literal["personal_schedule", "group_schedule", "todo", "reminder", "unknown"]
_WEEK02_AGENT: Any | None = None


# [2주차 수강생 구현 가이드]
#
# 목표
#   Week 2의 핵심은 사용자의 한국어 자연어 요청이나 Week 1 tool이 만든 JSON payload를
#   일정 앱이 읽을 수 있는 StructuredRequest/StructuredRequestBatch로 바꾸는 것입니다.
#   Week 1이 이미 정해진 인자를 받아 임시 일정을 만들었다면, Week 2는 "내일 오후 3시" 같은
#   자연어와 created_schedule JSON을 날짜/시간/종류/멤버 필드로 구조화합니다.
#   구조화 결과는 아직 SQLite, RAG, 외부 멤버 일정 조율 흐름에 저장하지 않습니다.
#
# 과제 구성
#   - 메인과제: Week 2 agent가 자연어 또는 Week 1 tool JSON을 StructuredRequestBatch로
#     최종 반환하는 세로 슬라이스를 완성합니다.
#   - 추가 과제: 메인과제에서 만든 StructuredRequest 스키마를 Week 3 이상 저장/조율 흐름에서
#     재사용할 수 있도록 bridge 함수를 완성합니다.
#
# 구현 위치와 사용할 코드
#   - 이 파일(student_parts/week02_structure_natural_language_requests.py)의
#     StructuredRequest, StructuredRequestBatch, week02_tools(), week02_prompt_parts(),
#     week02_system_prompt(), build_week02_agent()를 확인합니다.
#   - build_week02_agent()는 langchain.agents.create_agent, fixed/llm.py의 chat_model(),
#     week02_system_prompt(), response_format=StructuredRequestBatch를 사용해 Week 2 agent를 만듭니다.
#   - week02_tools()는 Week 1 도구 목록을 그대로 가져옵니다. Week 2 agent는 개인 일정 생성 요청에서
#     personal_create_schedule이 반환한 created_schedule JSON payload를 읽고
#     response_format=StructuredRequestBatch로 최종 구조화 결과를 확인합니다.
#   - week02_prompt_parts()는 student_parts/week01_wake_up_nana.py의 week01_prompt_parts() 위에
#     Week 2 구조화 지시를 추가합니다.
#   - _coerce_structured_request(), extract_structured_request(), extract_schedule_request()는
#     Week 3 이상에서 재사용되는 구조화 bridge입니다. Week 2 파일에 있지만 Week 2 agent에
#     공개되는 tool은 아닙니다.
#
# 메인과제 구현 대상
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
# 추가 과제 구현 대상
#   1. _coerce_structured_request
#      - LangChain structured output 결과가 이미 StructuredRequest이면 그대로 반환합니다.
#      - dict이면 StructuredRequest.model_validate(...)로 검증해 반환합니다.
#      - 예상한 형태가 아니면 RuntimeError를 발생시켜 잘못된 LLM 응답을 조용히 통과시키지 않습니다.
#
#   2. extract_structured_request
#      - chat_model().with_structured_output(StructuredRequest, method="function_calling")를 사용합니다.
#      - system 메시지에는 join_system_prompt(week02_prompt_parts())를 넣고,
#        user 메시지에는 text를 넣어 structured LLM을 호출합니다.
#      - 자연어 또는 JSON 문자열을 StructuredRequest 하나로 검증/구조화합니다.
#
#   3. extract_schedule_request
#      - extract_structured_request(query) 결과에 ok/tool_name/base_date를 붙입니다.
#      - structured_request에는 model_dump() 결과를 넣고, json.dumps(..., ensure_ascii=False)로 반환합니다.
#      - Week 3 이상 저장 tool이 structured_request 필드를 그대로 받을 수 있게 만듭니다.
#
# StructuredRequest 읽는 법
#   - kind: personal_schedule, group_schedule, todo, reminder, unknown 중 하나입니다.
#   - title/date/start_time/end_time: 일정 앱이 실제 저장이나 생성에 사용할 핵심 필드입니다.
#   - members: 참석자/관련 멤버 list입니다. 모르면 빈 list로 둡니다.
#   - priority/reason/original_text: 할 일 우선순위, 판단 근거, 원문 보존용 필드입니다.
#   - 모르는 값을 억지로 만들지 않는 것이 중요합니다. 확실하지 않으면 None 또는 빈 list가 안전합니다.
#   - date/start_time/end_time은 확실할 때만 YYYY-MM-DD, HH:MM 형식으로 채웁니다.
#
# bridge 동작 기준
#   - 요청이 하나뿐이어도 Week 2 agent의 structured_response에는 StructuredRequest 하나를 담습니다.
#   - 여러 일정/할 일/알림 의도가 한 문장에 섞이면 Week 2 agent에서는 여러 StructuredRequest로 나눕니다.
#   - extract_structured_request()는 bridge 용도라 StructuredRequest 하나만 반환합니다.
#   - Week 1 personal_create_schedule은 이미 분해된 인자로 임시 일정을 생성하고,
#     Week 2 agent와 bridge는 그 JSON payload를 읽어 저장 가능한 구조로 최종 변환한다는 차이를 비교합니다.
#
# 참고 코드
#   - week01_prompt_parts()
#      Week 1 system prompt를 이어받아 Week 2 구조화 지시를 누적할 때 사용합니다.
#   - week01_tools()
#      Week 1 개인 일정 tool 목록입니다. Week 2 agent는 이 tool 결과 JSON을 구조화 근거로 씁니다.
#   - extract_structured_request / extract_schedule_request
#      Week 3 이상에서 DB 저장/조율 tool chain에 쓰는 bridge 코드입니다.
#      query 문자열이 자연어든 Week 1 tool JSON이든, Python rule/parser로 매핑하지 않고
#      structured LLM 호출로 구조화한 뒤 JSON tool payload로 감쌉니다.
#
# 검증 방법
#   - 메인과제: ./run.sh --week2로 실행한 뒤 "다음 주 화요일 오후 3시에 철수랑 회의 잡아줘" 같은
#     문장을 입력합니다. 최종 답변이 StructuredRequestBatch class 형식의 structured_response로
#     나오는지 확인합니다.
#   - 추가 과제: Week 3을 실행한 뒤 trace에서 extract_schedule_request 이후
#     save_structured_request가 호출되는지 봅니다. extract_schedule_request의 반환 JSON에
#     ok/tool_name/base_date/structured_request가 들어 있는지 확인합니다.
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
#
#   - _coerce_structured_request(value)
#     LangChain structured output 결과가 이미 StructuredRequest이면 그대로 쓰고, dict이면 Pydantic 검증을 거쳐
#     StructuredRequest로 바꿉니다. 예상한 형태가 아니면 오류를 내서 잘못된 LLM 응답을 조용히 통과시키지 않습니다.
#
#   - extract_structured_request(text)
#     agent loop를 새로 만들지 않고 chat_model().with_structured_output(...)만 사용해 자연어 또는 JSON 문자열을
#     StructuredRequest로 검증/구조화합니다. Week 3 이상에서 저장/조율 직전 입력을 구조화해야 할 때 재사용하는 bridge 함수입니다.
#
#   - extract_schedule_request(query)
#     Week 3 이상 agent가 저장/조율 전에 호출하는 LangChain bridge tool입니다.
#     extract_structured_request(...) 결과에 ok/tool_name/base_date를 붙여 JSON 문자열로 반환하므로,
#     이후 저장 tool이 structured_request 필드를 그대로 받을 수 있습니다.


class StructuredRequest(BaseModel):
    """LLM structured output으로 추출되는 2주차 요청 스키마입니다."""

    # TODO: kind 필드를 RequestKind 타입으로 선언하고 Field(description=...)를 붙이세요.
    # TODO: title/date/start_time/end_time 필드를 str | None 타입으로 선언하고 기본값은 None으로 두세요.
    # TODO: members 필드를 list[str] 타입으로 선언하고 default_factory=list를 사용하세요.
    # TODO: priority/reason 필드를 str | None 타입으로 선언하고 기본값은 None으로 두세요.
    # TODO: original_text 필드를 str 타입으로 선언하고 기본값은 ""로 두세요.
    # TODO: 각 필드에는 LLM structured output이 이해할 수 있도록 한국어 description을 달아주세요.
    kind: RequestKind = Field(
        description=(
            "사용자 요청의 종류를 분류한 값입니다. 어떤 Tool을 호출했는지가 아니라 요청 내용으로 판단하세요. " 
            "5가지 값 중 하나를 고릅니다.\n"
            "- personal_schedule: members에 본인 한 명뿐인 개인 일정\n"
            "- group_schedule: members에 본인 외 다른 사람이 1명 이상 포함된 일정\n"
            "- todo: 특정 시각이 정해지지 않은 완료 여부만 중요한 할 일\n"
            "- reminder: 특정 시점에 사용자에게 알려주기를 원하는 알림성 요청\n"
            "- unknown: 위 4가지 중 어디에도 명확히 속하지 않거나 종류를 판단할 정보가 부족한 경우\n"
        )
    )
    title: str | None = Field(default=None, description="요청과 관련된 일정/할 일의 제목입니다. 언급되지 않았다면 None으로 둡니다.")
    date: str | None = Field(default=None, description="요청과 관련된 날짜입니다. 언급되지 않았다면 None으로 둡니다.")
    start_time: str | None = Field(default=None, description="일정의 시작 시각입니다. 시각 형식은 (YYYY-MM-DD/HH:MM)입니다. 언급되지 않았다면 None으로 둡니다.")
    end_time: str | None = Field(default=None, description="일정의 종료 시각입니다. 시각 형식은 (YYYY-MM-DD/HH:MM)입니다. 언급되지 않았다면 None으로 둡니다.")
    members: list[str] = Field(default_factory=list, description="일정에 참여하는 사람들의 이름 목록입니다. 언급되지 않았다면 빈 리스트로 둡니다.")
    priority: str | None = Field(default=None, description="요청의 우선순위입니다. 언급되지 않았다면 None으로 둡니다.")
    reason: str | None = Field(default=None, description="요청이나 일정의 목적 또는 이유입니다. 언급되지 않았다면 None으로 둡니다.")
    original_text: str = Field(default="", description="사용자가 입력한 원본 문장 전체입니다.")


class StructuredRequestBatch(BaseModel):
    """여러 자연어 의도를 StructuredRequest 목록으로 나누는 메인과제 스키마입니다."""

    # TODO: requests 필드를 list[StructuredRequest] 타입으로 선언하고 default_factory=list를 사용하세요.
    # TODO: base_date 필드를 str 타입으로 선언하고 default_factory=current_app_date_iso를 사용하세요.
    # TODO: 각 필드에는 Week 2 구조화 결과와 상대 날짜 기준일을 설명하는 한국어 description을 달아주세요.
    requests: list[StructuredRequest] = Field(default_factory=list, description="StructuredRequest 목록입니다. 요청이 하나여도 리스트 형태로 담습니다.")
    base_date: str = Field(default_factory=current_app_date_iso, description="상대 날짜 표현을 해석할 때 기준이 되는 오늘 날짜입니다.")


def _coerce_structured_request(value: Any) -> StructuredRequest:
    """LangChain structured output 결과를 StructuredRequest로 정규화합니다."""

    # TODO: value가 이미 StructuredRequest이면 그대로 반환하세요.
    # TODO: value가 dict이면 StructuredRequest.model_validate(...)로 검증해 반환하세요.
    # TODO: 예상한 형태가 아니면 RuntimeError를 발생시켜 잘못된 LLM 응답을 조용히 통과시키지 마세요.
    if isinstance(value, StructuredRequest): return value
    elif isinstance(value, dict): return StructuredRequest.model_validate(value)
    else: raise RuntimeError(f"예상치 못한 Structed Output 형태입니다: {type(value)}")


def extract_structured_request(text: str) -> StructuredRequest:
    """Week 3 이상에서 agent를 새로 띄우지 않고 자연어를 StructuredRequest로 바꿉니다."""

    # TODO: chat_model().with_structured_output(StructuredRequest, method="function_calling")로 structured LLM을 만드세요.
    # TODO: system 메시지에는 join_system_prompt(week02_prompt_parts())를 넣고, user 메시지에는 text를 넣어 invoke하세요.
    # TODO: LLM 결과를 _coerce_structured_request(...)로 정규화해 StructuredRequest 하나로 반환하세요.
    structured_llm = chat_model().with_structured_output(
        StructuredRequest, method="function_calling"
    )
    result = structured_llm.invoke(
        [
            {"role": "system", "content": join_system_prompt(week02_prompt_parts())},
            {"role": "user", "content": text},
        ]
    )
    return _coerce_structured_request(result)


@tool
def extract_schedule_request(query: str) -> str:
    """Week 3 이상 agent가 저장/조율 전에 호출하는 구조화 bridge tool입니다."""

    # TODO: extract_structured_request(query)를 호출해 자연어 또는 Week 1 JSON payload를 구조화하세요.
    # TODO: ok/tool_name/base_date/structured_request 키를 가진 dict를 만들고 structured_request에는 model_dump() 결과를 넣으세요.
    # TODO: json.dumps(..., ensure_ascii=False)로 JSON 문자열을 반환하세요.
    structured_request = extract_structured_request(query)
    payload = {
        "ok": True,
        "tool_name": "extract_schedule_request",
        "base_date": current_app_date_iso(),
        "structured_request": structured_request.model_dump(),
    }
    return json.dumps(payload, ensure_ascii=False)


def week02_tools() -> list[Any]:
    """Week 2 agent에 Week 1 도구를 노출해 tool JSON을 structured_response 근거로 씁니다."""

    # TODO: Week 1에서 구현한 tool 목록을 그대로 반환하세요.
    return week01_tools()


def week02_system_prompt() -> str:
    """2주차 agent가 따르는 시스템 프롬프트입니다."""

    # TODO: join_system_prompt(...)로 week02_prompt_parts()와 Week 2 structured_response 최종 답변 규칙을 합치세요.
    # TODO: StructuredRequestBatch에는 요청이 하나뿐이어도 requests 목록에 StructuredRequest 하나를 담도록 지시하세요.
    # TODO: personal_create_schedule tool 결과 JSON의 created_schedule을 읽어 필드를 채우도록 지시하세요.
    return join_system_prompt(
        [
            *week02_prompt_parts(),
            "## 최종 답변 형식\n"
            "- 최종 답변은 반드시 StructuredRequestBatch 형식의 structured_response로 반환해.\n"
            "- 요청이 하나뿐이더라도 requests 목록 안에 StructuredRequest를 하나 담아.\n"
            "- 한 문장에 여러 일정/할 일/알림 의도가 섞여 있으면 별도의 StructuredRequest로 나눠."
            "- Tool 결과 JSON(created_schedule 등)을 그대로 복사해서 출력하지 마. "
            "필드 값을 채우는 데만 참고하고, 원본 payload 자체는 답변에 넣지 마.\n"
            "- 최종 출력에는 StructuredRequestBatch JSON 객체 '단 하나만' 있어야 해. "
            "그 앞이나 뒤에 다른 JSON, 자연어 설명, 인사말, 코드블록(```)을 절대 추가하지 마."
        ]
    )


def week02_prompt_parts() -> list[str]:
    """2주차 structured output agent가 따르는 system prompt 조각입니다."""

    return [
        # TODO: Week 2 요청 구조화 agent 역할과 현재 날짜(current_app_date_iso()) 기준을 추가하세요.
        # TODO: 자연어를 StructuredRequest 필드(kind/title/date/start_time/end_time/members 등)로 구조화하도록 지시하세요.
        # TODO: Week 1 tool JSON을 받은 경우 다시 tool을 호출하지 않고 payload를 읽어 structured_response로 만들도록 지시하세요.
        # TODO: Week 2에서는 SQLite 저장, RAG, 외부 멤버 일정 조율을 하지 않는다고 명시하세요.
        *week01_prompt_parts(),
        "## Week1 지시 재정의\n"
        "위의 'Tool 사용 규칙' 3번의 '처리 결과를 사용자에게 간결하게 답변해'는 이번 주차부터 적용하지 않아."
        "이번 주차의 최종 답변은 자연어 요약이 아니라 StructuredRequestBatch 하나여야 하고 이 지시가 앞선 주차의 답변 형식 지시보다 우선이야.\n"
        "\n"
        "- 너는 이제 사용자의 자연어 요청 또는 personal_* Tool이 반환한 JSON 결과를 StructuredRequest/StructuredRequestBatch 형태로 구조화하는 역할도 맡아.\n"
        f"- 위에서 안내한 현재 날짜({current_app_date_iso()})를 상대 날짜 해석 기준으로 계속 사용해.\n"
        "\n"
        "## 구조화 규칙\n"
        "1. 입력을 읽고 kind/title/date/start_time/end_time/members/priority/reason/original_text 필드를 채워.\n"
        "2. 확실하지 않은 값은 추측해서 채우지 말고 description을 확인하고 None 또는 빈 리스트로 남겨.\n"
        "3. 이미 personal_* Tool 결과(JSON)를 받았다면 같은 Tool을 다시 호출하지 말고, 그 payload는 내부적으로 참고만 해서 필드를 채우는 데 쓰고 절대 그대로 출력하지 마.\n"
        "4. 아직 SQLite 저장, RAG 검색, 외부 멤버 일정 조율은 하지 않아. 오직 구조화 결과만 반환해."
    ]


def build_week02_agent() -> object:
    """Week 2 대화에서 structured_response를 직접 반환하는 단일 LangChain agent를 만듭니다."""

    # TODO: CONFIG.has_openai_key가 없으면 RuntimeError("PROXY_TOKEN이 .env에 필요합니다.")를 발생시키세요.
    # TODO: 전역 _WEEK02_AGENT를 재사용하고, 아직 없을 때만 create_agent(...)로 새 agent를 만드세요.
    # TODO: create_agent에는 model=chat_model(), tools=week02_tools(), response_format=StructuredRequestBatch,
    #       system_prompt=week02_system_prompt()를 연결하세요.
    # TODO: 생성 또는 재사용한 _WEEK02_AGENT를 반환하세요.
    global _WEEK02_AGENT

    if not CONFIG.has_openai_key:
        raise RuntimeError("PROXY_TOKEN이 .env에 필요합니다.")
    
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
