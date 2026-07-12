from __future__ import annotations

import json
from typing import Any, Literal

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
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

    kind: RequestKind = Field(
        description=(
            "요청 종류입니다. personal_schedule(개인 일정), group_schedule(그룹/회의 일정), "
            "todo(할 일), reminder(알림), unknown(분류 불가) 중 하나만 사용하세요. "
            "확실하지 않으면 unknown으로 두세요."
        )
    )
    title: str | None = Field(
        default=None,
        description="일정이나 할 일의 제목입니다. 모르면 None으로 두세요.",
    )
    date: str | None = Field(
        default=None,
        description="날짜입니다. 확실할 때만 YYYY-MM-DD 형식으로 채우고, 모르면 None으로 두세요.",
    )
    start_time: str | None = Field(
        default=None,
        description="시작 시간입니다. 확실할 때만 HH:MM 형식으로 채우고, 모르면 None으로 두세요.",
    )
    end_time: str | None = Field(
        default=None,
        description="종료 시간입니다. 확실할 때만 HH:MM 형식으로 채우고, 모르면 None으로 두세요.",
    )
    members: list[str] = Field(
        default_factory=list,
        description="참석자나 관련 멤버 목록입니다. 모르면 빈 list로 두세요.",
    )
    priority: str | None = Field(
        default=None,
        description="할 일의 우선순위입니다. 정보가 없으면 None으로 두세요.",
    )
    reason: str | None = Field(
        default=None,
        description="이 요청 종류로 분류한 판단 근거입니다. 없으면 None으로 두세요.",
    )
    original_text: str = Field(
        default="",
        description="구조화 이전의 사용자 원문을 그대로 보존합니다.",
    )


class StructuredRequestBatch(BaseModel):
    """여러 자연어 의도를 StructuredRequest 목록으로 나누는 2차 과제 스키마입니다."""

    requests: list[StructuredRequest] = Field(
        default_factory=list,
        description=(
            "자연어에서 추출한 StructuredRequest 목록입니다. "
            "요청이 하나뿐이어도 반드시 list 안에 StructuredRequest 하나를 담습니다."
        ),
    )
    base_date: str = Field(
        default_factory=current_app_date_iso,
        description="'내일', '다음 주' 같은 상대 날짜를 해석할 때 기준이 되는 오늘 날짜(YYYY-MM-DD)입니다.",
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
    return join_system_prompt(
        [
            *week02_prompt_parts(),
            """
                최종 답변은 반드시 StructuredRequestBatch 형식의 structured_response로 반환합니다.
                요청이 하나뿐이어도 requests 목록에 StructuredRequest 하나를 담습니다.
                개인 일정 생성 요청에서 personal_create_schedule tool을 사용했다면, 이는 Week 1 tool이므로
                아래 "Week 1 tool 결과 payload" 규칙을 그대로 적용하세요: tool을 다시 호출하지 말고,
                그 결과 JSON의 created_schedule 값을 읽어 kind/title/date/start_time/end_time/members 필드로
                변환한 뒤 requests 목록에 반드시 포함시키세요.
            """,
        ]
    )


def week02_prompt_parts() -> list[str]:
    """2주차 structured output agent가 따르는 system prompt 조각입니다."""
    return [
        *week01_prompt_parts(),
        f"""
            당신은 Week 2 요청 구조화 assistant입니다.
            사용자의 한국어 문장을 StructuredRequest로 변환하세요.
            오늘 날짜는 {current_app_date_iso()}이며, '내일', '다음 주 화요일' 같은 상대 날짜는 이 날짜를 기준으로 해석합니다.
            이번 사용자 입력만 근거로 필드를 작성하며, 이전 대화 턴의 날짜/시간/멤버 정보를 재사용하지 마세요.

            요청 분리 규칙:
            - "하고", "그리고", "또", "도" 등으로 연결된 서로 다른 행동은 각각 독립적인 StructuredRequest로 분리하여 requests에 담으세요. 요청이 하나뿐이어도 반드시 list 형태를 유지합니다.
            - 원문의 모든 행동은 빠짐없이 포함되어야 합니다. 특히 문장 마지막 행동과, 이미 Week 1 tool로 처리된 요청도 절대 생략하지 마세요.
            - "A하고 B"라면 첫 요청의 original_text는 "A", 두 번째는 "B"만 담습니다. 서로 다른 요청의 original_text가 같거나 전체 원문과 동일하면 잘못된 분리입니다.
            - 각 요청의 kind/date/start_time/end_time/members는 오직 그 요청의 original_text에만 근거해서 작성합니다. 다른 요청의 정보를 절대 복사하지 마세요.
            - (검증) 응답 확정 전에, 원문의 독립 행동 개수와 requests 개수가 일치하는지, 그리고 서로 다른 요청 간에 date/start_time/members가 우연히 같아진 값이 실제로 그 요청의 original_text에 근거를 두고 있는지 확인하세요. 근거가 없으면 즉시 None 또는 빈 list로 고치세요.

            필드 작성 규칙:
            - date, start_time, members, kind는 서로 독립적입니다. 한 필드가 채워졌다고 다른 필드를 함께 만들거나 바꾸지 마세요.
            - kind: personal_schedule / group_schedule / todo / reminder / unknown 중 하나.
            - title: 사용자 표현의 의미를 유지하며 간결하게.
            - date: 해당 요청 원문에 날짜 표현이 있을 때만 YYYY-MM-DD로 작성. 없으면 반드시 None. 시간이 있다고 오늘 날짜를 자동 채우지 마세요.
            - start_time: 해당 요청 원문에 시간 표현이 있을 때만 HH:MM으로 작성. 없으면 반드시 None. 날짜와 시간은 독립적으로 판단합니다.
            - end_time: 종료 시간 표현이 있을 때만 작성. 없으면 None.
            - members: 구체적인 사람 이름이 명확할 때만 추가. "팀", "다들", "모두", "사람들", "팀원들", "다 같이" 같은 일반 명사는 넣지 말고 빈 list([]).
            - priority, reason: 사용자가 직접 제공한 경우에만 작성, 아니면 None. reason은 모델 추론을 적는 필드가 아닙니다.
            - original_text: 해당 요청으로 분리된 원문 "부분"만. 전체 원문 복사 금지, 빈 문자열 금지.

            None 규칙:
            - "미정", "없음", "" 등을 쓰지 말고 반드시 None을 사용합니다.
            - 언급되지 않은 날짜/시간/멤버/우선순위를 절대 생성하지 마세요. base_date는 상대 날짜 해석용일 뿐 자동 적용 금지.
            - 00:00은 절대 기본값으로 사용하지 않습니다. 사용자가 "자정", "밤 12시"라고 직접 말한 경우에만 허용됩니다.

            자체 검증 (필수):
            - date를 채우기 전에 확인: 이 요청의 original_text에 날짜 단어(오늘/내일/모레/특정 요일/특정 날짜)가 있는가? 있으면 반드시 date를 채우고, 없으면 반드시 None으로 둡니다. 둘 다 지켜야 하며 어느 한쪽으로 치우치지 마세요.
            - start_time도 마찬가지: 시간 단어(오전/오후/몇 시/자정 등)가 있으면 반드시 채우고, 없으면 반드시 None.
            - base_date나 옆 요청의 값을 추측이나 기본값으로 채우는 것은 금지이며, 반대로 원문에 근거가 있는데도 None으로 두는 것 역시 금지입니다.

            kind 분류 규칙:
            - personal_schedule: 다른 사람이 전혀 관련되지 않은 혼자만의 일정이며, "잡아줘/추가해줘/예약해줘" 같은 등록 의도 표현이 있는 경우에만 사용. 사람 이름이나 "~랑/~와 함께" 같은 동반 표현이 하나라도 있으면 personal_schedule이 아니라 group_schedule입니다. 행동을 나타내는 표현("작성하기", "준비하기" 등)은 날짜가 붙어도 todo입니다.
            - group_schedule: 둘 이상이 함께하는 일정 + 등록 의도 표현("잡아줘/추가해줘/일정 만들어줘/예약해줘"). "다들", "모두"처럼 일반 표현만 있어도 등록 의도가 함께 있으면 group_schedule이며, 이때 members는 빈 list여도 됩니다.
            - todo: 사용자가 직접 완료해야 하는 작업/행동, 또는 "해야 해/필요해"류 수행 의무 표현. 등록 의도 표현이 없다면 날짜/멤버가 붙어 있어도 kind는 todo이며 다른 kind로 바뀌지 않습니다.
            - reminder: "알려줘/알림/기억해줘/리마인드" 등 알림 요청 표현이 있으면 reminder(완료 행동이 포함되어 있어도 목적이 알림이면 우선).
            - unknown: 위 기준으로 분류 불가할 때만.

            추가 규칙:
            - 같은 입력에는 항상 동일한 kind. 확실하지 않은 정보는 None 또는 빈 list.
            - "회의해야 해", "미팅 잡아줘", "철수랑 만나야 해"처럼 종류와 사람만 있는 경우 날짜와 시간을 추측하지 않습니다.

            Week 1 tool 결과 JSON(payload)을 이미 받은 경우에는 tool을 다시 호출하지 말고,
            해당 payload를 그대로 읽어 StructuredRequest 필드로 변환한 뒤 structured_response에 반드시 포함시키세요.

            Week 2에서는 SQLite 저장, RAG, 외부 멤버 일정 조율을 하지 않습니다.
        """,
    ]


def build_week02_agent() -> object:
    """Week 2 대화에서 structured_response를 직접 반환하는 단일 LangChain agent를 만듭니다."""

    if not CONFIG.has_openai_key:
        raise RuntimeError("PROXY_TOKEN이 .env에 필요합니다.")
    global _WEEK02_AGENT
    if _WEEK02_AGENT is None:
        # response_format을 그냥 StructuredRequestBatch로 주면 LangChain이 provider
        # native 구조화 출력을 쓰는데, 프록시 모델(gpt-4.1-mini)이 JSON 뒤에 여분의
        # 텍스트/객체를 덧붙이는 경우가 있어 json.loads가 "Extra data"로 실패한다.
        # ToolStrategy로 감싸면 구조화 결과를 tool call 인자로 받으므로 뒤따르는
        # 텍스트에 오염되지 않고, handle_errors 기본값이 파싱 실패 시 재시도까지 해준다.
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
