from __future__ import annotations

import json
from typing import Any, Literal

from langchain.agents import create_agent
from langchain_core.tools import tool
from pydantic import BaseModel, Field, model_validator

from fixed.config import CONFIG
from fixed.llm import chat_model
from fixed.runtime_clock import current_app_date_iso
from fixed.app_store import AppSQLiteStore
from student_parts.week01_wake_up_nana import (
    join_system_prompt,
    personal_create_schedule as week01_personal_create_schedule,
    week01_tools,
)
from student_parts.week02_structure_natural_language_requests import (
    RequestKind,
    StructuredRequest,
    extract_schedule_request,
    extract_structured_request,
    week02_prompt_parts,
)


_WEEK03_AGENT: Any | None = None

# schedules 테이블에는 일정 kind만 들어간다(todo는 todos, reminder는 reminders 테이블).
# 그래서 일정 조회 tool의 kind는 RequestKind 전체가 아니라 아래 두 값만 받는다.
ScheduleKind = Literal["personal_schedule", "group_schedule"]

SQLITE_MEMORY_PROMPT = (
    "Week 3부터 일정/할 일/알림은 앱 SQLite DB에 영속 저장된다. "
    "Week 1 임시 메모리와 달리 새 대화를 시작해도 저장된 기록은 사라지지 않는다. "
    "따라서 저장된 일정/할 일/알림에 대한 질문은 기억이나 이전 대화 내용에 의존하지 말고 "
    "반드시 DB 조회 tool 결과를 기준으로 답한다. "
    "이때 개인/그룹 일정은 personal_list_saved_schedules로, 할 일(todo)과 알림(reminder)은 "
    "list_saved_requests(kind=...)로 조회한다. "
    "Week 1 임시 조회 tool(personal_list_schedules)은 SQLite에 저장된 기록을 보지 못하므로 "
    "저장된 기록 조회에 사용하지 않는다."
)

WEEK03_TOOL_CALL_PROMPT = """Week 3 tool 호출 순서 규칙:
- 일정/할 일/알림을 저장해야 하는 자연어 요청은 먼저 extract_schedule_request(query=사용자 원문)를 호출해 구조화합니다.
- 이어서 반환 JSON의 structured_request 필드(kind/title/date/start_time/end_time/members/priority/reason/original_text)를
  save_structured_request 인자로 그대로 전달해 SQLite에 저장합니다. 값을 새로 만들거나 바꾸지 않습니다.
- ok/tool_name/base_date 같은 wrapper 키 자체를 save_structured_request 인자로 넘기지 않습니다.
- 일정 생성/저장 요청은 personal_create_schedule이 아니라 위의
  extract_schedule_request → save_structured_request 순서로 처리합니다.
- 조회 tool은 대상에 따라 다음 기준으로 고릅니다.
  · 개인/그룹 "일정" 조회: personal_list_saved_schedules (kind는 personal_schedule/group_schedule만 받습니다)
  · "할 일"(todo), "알림"(reminder) 등 구조화 요청 조회: list_saved_requests(kind="todo" 또는 "reminder")
  · request_id를 이미 아는 단건 조회: get_saved_request
- "저장한 할 일 보여줘", "알림 뭐 있어?" 같은 요청에 personal_list_saved_schedules를 쓰지 않습니다.
  할 일/알림은 schedules 테이블에 없어 항상 빈 목록이 나옵니다. 반드시 list_saved_requests로 조회합니다.
- personal_list_saved_schedules는 kind를 지정하지 않으면 개인 일정(personal_schedule)만 반환합니다.
  그룹 일정은 kind=group_schedule로 조회하고, "모든 일정" 요청이면 두 kind를 각각 조회해 합쳐서 답합니다.
- 일정·할 일·알림을 가리지 않는 "저장한 거 전부 보여줘" 요청은 kind 없이 list_saved_requests를 호출합니다.
- 저장 일정 수정/삭제 전에는 personal_list_saved_schedules로 후보 schedule_id를 먼저 확인합니다.
  이때 사용자가 날짜를 말하지 않았다면 date_from/date_to를 넣지 말고 전체에서 제목으로 찾습니다.
  수정은 personal_update_saved_schedule에 바꿀 필드만 전달하고(전달하지 않은 필드는 유지됨),
  삭제는 personal_delete_saved_schedules에 확인한 schedule_ids나 명시 필터를 전달합니다.
- 사용자가 전체 삭제를 명확히 요청한 경우에만 delete_all=True를 사용합니다."""


# [3주차 수강생 구현 가이드]
#
# 목표
#   Week 2에서 만든 StructuredRequest를 Pydantic 입력 스키마로 검증한 뒤 SQLite에 저장하고,
#   저장된 요청/일정을 다시 조회/수정/삭제합니다. 여기서부터 Nana는 Week 1의 임시 메모리 대신
#   앱 DB에 남는 "기록장"을 갖게 됩니다.
#
# 과제 구성
#   - 메인과제: 구조화 결과를 SQLite에 저장하고 다시 조회하는 세로 슬라이스를 완성해
#     "저장 → 조회 → 새 대화에서도 유지"가 동작하는 최소 기록장을 만듭니다.
#   - 추가 과제: 저장된 일정을 수정/삭제하고 외부 공유 저장소와 동기화하며,
#     Week 1 호환 생성과 레거시 payload 정규화까지 다루는 확장 기능을 완성합니다.
#
# 핵심 흐름
#   1. LLM은 extract_schedule_request(query=사용자 요청)를 호출해 자연어를 Week 2 StructuredRequest로 바꿉니다.
#   2. LLM은 structured_request의 kind/title/date/start_time/end_time/members/priority/reason/original_text를
#      save_structured_request 인자로 그대로 전달합니다.
#   3. 각 tool에 붙은 @tool(args_schema=...)가 Pydantic class로 입력을 검증합니다.
#   4. Python tool 본문은 이미 검증된 인자를 AppSQLiteStore에 넘기고, 결과를 JSON 문자열로 반환합니다.
#
# 구현 위치와 사용할 코드
#   - StructuredRequest와 RequestKind는 week02_structure_natural_language_requests.py에서 재사용합니다.
#   - SaveStructuredRequestInput은 Week 2 StructuredRequest를 상속하고, Week 1 호환용 source_schedule_id만 추가합니다.
#   - SavedRequestListInput, SavedRequestGetInput, SavedScheduleListInput,
#     SavedScheduleUpdateInput, SavedScheduleDeleteInput은 조회/수정/삭제 tool 인자 스키마입니다.
#   - 실제 DB 접근은 fixed/app_store.py의 AppSQLiteStore를 사용하고, _store()가 CONFIG.app_db_path 기준
#     store 객체를 만들어 줍니다.
#   - save_structured_request_payload()와 delete_saved_schedules_dict()는 테스트/직접 호출/이전 trace 호환용 helper입니다.
#     agent가 일반적으로 호출하는 경로는 @tool(args_schema=...)가 붙은 tool 함수입니다.
#
# 메인과제 구현 대상
#   1. save_structured_request
#      - @tool(args_schema=SaveStructuredRequestInput)으로 Week 2 구조화 결과를 검증합니다.
#      - tool 본문에서는 Pydantic class를 다시 만들지 말고, 함수 인자로 들어온 값을 바로 저장 dict로 정리합니다.
#      - 자연어 문자열이나 ok/tool_name/base_date wrapper를 직접 저장하지 않습니다.
#
#   2. list_saved_requests / get_saved_request
#      - list는 kind/date_from/date_to 필터를 AppSQLiteStore.list_saved_requests(...)에 그대로 넘깁니다.
#      - 할 일(todo)/알림(reminder) 조회는 schedules 테이블이 아니라 이 tool로 합니다.
#      - get은 request_id 하나로 단건 조회합니다.
#      - 조회 결과가 없어도 예외를 던지지 말고 rows=[] 또는 row=None 형태를 유지합니다.
#
#   3. personal_list_saved_schedules
#      - 저장된 "일정" 목록을 반환해 "내 일정 보여줘" 같은 조회 질문과 이후 수정/삭제 후보 확인에 씁니다.
#      - kind는 ScheduleKind(personal_schedule/group_schedule)만 받습니다. todo/reminder는 schedules 테이블에
#        없으므로 스키마 단계에서 막고, 할 일/알림 조회는 list_saved_requests로 보냅니다.
#      - 날짜가 명확한 조회는 date_from/date_to로 범위를 좁히고, 너무 많은 row가 들어가지 않게 limit을 사용합니다.
#
# 추가 과제 구현 대상
#   1. personal_update_saved_schedule
#      - AppSQLiteStore.update_schedule(...) 결과를 JSON 응답으로 완성하고, 공유 일정 복사본 동기화 결과(shared_sync)도 함께 반환합니다.
#      - None으로 들어온 필드는 "수정하지 않음"이라는 뜻입니다. ID를 못 찾으면 ok=False로 답합니다.
#
#   2. personal_delete_saved_schedules
#      - schedule_ids, date, title, start_time, time_unspecified, delete_all 조건을 받습니다.
#      - 조건 없이 삭제하지 않도록 _delete_saved_schedules(...)에서 안전 규칙을 확인합니다.
#      - deleted_count, filters, deleted를 유지해야 trace에서 무엇이 지워졌는지 확인할 수 있습니다.
#
#   3. personal_create_schedule (Week 1 호환)
#      - Week 1과 같은 이름을 유지하면서 임시 일정 생성 결과를 SQLite에도 저장하는 이중 기록 tool입니다.
#      - week01_personal_create_schedule 결과를 structured_request_from_week01_schedule()로 변환해 저장합니다.
#
#   4. 레거시 payload 정규화
#      - SaveStructuredRequestInput.unwrap_legacy_payload는 예전 trace/테스트의 payload/structured_request wrapper를 저장 스키마로 풉니다.
#      - _save_input_from / save_structured_request_payload는 tool 없이 dict/JSON/자연어를 직접 저장할 때 쓰는 helper입니다.
#
# 반환 규칙
#   모든 @tool은 JSON 문자열을 반환합니다.
#   ok와 tool_name은 기본으로 넣고, 조회는 rows/row, 삭제는 deleted_count/filters/deleted를 유지하세요.
#
# 참고 코드
#   week03_tools()는 Week 1-2 도구에 SQLite 도구를 누적해 공개합니다.
#   Week 1 호환 personal_create_schedule은 week01_personal_create_schedule 결과를
#   structured_request_from_week01_schedule()로 SaveStructuredRequestInput에 맞춘 뒤 SQLite에 저장합니다.
#   삭제 요청은 먼저 personal_list_saved_schedules로 후보를 확인한 뒤
#   personal_delete_saved_schedules에 schedule_ids 또는 명시 필터를 넘기는 흐름으로 처리합니다.
#
# 검증 방법
#   - 메인과제: ./run.sh --week3에서 "내일 10시 개인 코칭 저장해줘"처럼 입력합니다.
#     trace에서 extract_schedule_request 다음에 save_structured_request가 호출되는지 보고,
#     이어서 "내 일정 보여줘"가 personal_list_saved_schedules로 조회되며, 앱을 다시 시작하거나
#     새 대화를 열어도 저장된 일정이 그대로 보이면 메인과제가 동작하는 것입니다.
#   - 추가 과제: 저장된 일정을 personal_list_saved_schedules로 확인한 뒤 personal_update_saved_schedule로 시간을 바꾸고,
#     personal_delete_saved_schedules에 schedule_ids 또는 명시 필터를 넘겨 삭제한 일정이 목록에서 사라지는지 봅니다.
#
# 함수별 동작 설명 ([메인]/[추가]/[공통]은 각 함수가 속한 과제 티어입니다)
#   - [공통] _store()
#     현재 CONFIG.app_db_path를 기준으로 AppSQLiteStore를 생성합니다. SQL은 store.py가 담당하고,
#     이 파일의 tool들은 store 메서드를 호출하는 얇은 입구 역할만 합니다.
#
#   - [공통] _tool_name(item)
#     LangChain tool 객체와 일반 함수 객체 모두에서 이름을 안전하게 꺼냅니다. week03_tools()에서 Week 1 tool을 교체할 때 사용합니다.
#
#   - [공통] json_payload(payload)
#     tool 결과 dict를 한글이 깨지지 않는 JSON 문자열로 바꿉니다.
#
#   - [공통] tool_result(tool_name, ok, **payload)
#     여러 tool이 공통으로 쓰는 응답 껍데기를 만듭니다. 필수 구조는 아니지만 ok/tool_name 반복을 줄이는 작은 helper입니다.
#
#   - [메인] SaveStructuredRequestInput
#     Week 2 StructuredRequest를 상속한 저장 입력 스키마입니다. LangChain의 @tool(args_schema=...)가 이 class를 보고
#     save_structured_request 인자를 검증합니다.
#
#   - [추가] SaveStructuredRequestInput.unwrap_legacy_payload(value)
#     예전 trace나 테스트에서 들어올 수 있는 payload/structured_request wrapper를 저장 스키마 형태로 풀어 줍니다.
#     일반적인 agent 경로에서는 LLM이 필드를 직접 넘기므로 이 함수가 크게 개입하지 않습니다.
#
#   - [추가] _save_input_from(value)
#     테스트나 직접 호출 helper에서 dict, JSON 문자열, StructuredRequest를 SaveStructuredRequestInput 하나로 맞춥니다.
#     자연어 문자열이 들어오면 Week 2 extract_structured_request(...)로 먼저 구조화합니다.
#
#   - [추가] save_structured_request_payload(...)
#     tool wrapper 없이 직접 저장을 테스트해야 할 때 쓰는 helper입니다. 입력을 검증한 뒤 AppSQLiteStore.save_structured_request(...)에 넘깁니다.
#
#   - [메인/추가] SavedRequestListInput / SavedRequestGetInput / SavedScheduleListInput / SavedScheduleUpdateInput / SavedScheduleDeleteInput
#     조회, 단건 조회, 일정 목록, 일정 수정, 일정 삭제 tool의 입력 스키마입니다. Pydantic이 기본값과 범위를 검증합니다.
#     앞의 셋(list/get/schedule list)은 메인과제, 수정/삭제 스키마는 추가 과제에서 씁니다.
#
#   - [추가] _delete_saved_schedules(...)
#     삭제 조건이 비어 있는지 먼저 확인하고, delete_all인지 필터 삭제인지에 따라 store 삭제 메서드를 호출합니다.
#     실제 SQL 삭제는 AppSQLiteStore가 수행하고, 이 함수는 안전 규칙과 응답 모양을 정리합니다.
#
#   - [추가] structured_request_from_week01_schedule(schedule)
#     Week 1의 임시 schedule dict를 Week 3 저장 입력으로 변환합니다. personal_create_schedule 호환 wrapper에서 사용합니다.
#
#   - [추가] personal_create_schedule(...)
#     Week 1과 같은 이름을 유지하는 호환 tool입니다. 먼저 Week 1 임시 일정을 만들고, 같은 내용을 SQLite에도 저장합니다.
#
#   - [메인] save_structured_request(...)
#     Week 2 structured_request 필드를 직접 받아 SQLite에 저장하는 Week 3 핵심 tool입니다.
#     args_schema가 입력 검증을 끝낸 뒤 들어오므로, 본문은 저장 dict를 만들어 store에 넘기는 일만 합니다.
#
#   - [메인] list_saved_requests(...) / get_saved_request(...)
#     SQLite에 저장된 structured_requests 원본 기록을 목록 또는 단건으로 조회합니다.
#     할 일(todo)/알림(reminder) 조회 경로도 이쪽입니다.
#
#   - [메인] personal_list_saved_schedules(...)
#     저장된 "일정" row만 조회합니다. 수정/삭제 전 후보 schedule_id를 확인하거나 일정 조회 질문에 답할 때 사용합니다.
#     kind는 ScheduleKind로 제한돼 todo/reminder를 넘기면 Pydantic 검증에서 막힙니다.
#
#   - [추가] delete_saved_schedules_dict(...)
#     테스트나 내부 코드에서 tool invoke 없이 삭제 로직을 호출할 수 있게 만든 dict 반환 helper입니다.
#
#   - [추가] personal_update_saved_schedule(...)
#     schedule_id로 저장 일정을 찾아 제목/날짜/시간/참석자를 수정합니다. 공유 일정 동기화 결과도 함께 반환합니다.
#
#   - [추가] personal_delete_saved_schedules(...)
#     schedule_ids나 날짜/제목/시간 필터로 저장 일정을 삭제하는 tool입니다. 조건 없는 삭제는 실패 응답으로 막습니다.
#
#   - [공통] week03_tools()
#     Week 1 tool 목록에 Week 2 구조화 tool과 Week 3 SQLite tool을 누적합니다. Week 1 personal_create_schedule은
#     SQLite 저장까지 수행하는 이 파일의 호환 tool로 교체합니다.
#
#   - [공통] week03_system_prompt() / week03_prompt_parts()
#     Week 3 agent가 "구조화 후 저장" 흐름을 따르도록 system prompt를 조립합니다.
#
#   - [공통] build_week03_agent() / build_week_agent()
#     Week 1~3 tool을 가진 agent를 한 번만 만들고 재사용합니다. build_week_agent()는 실행기가 호출하는 표준 entry point입니다.


def _store() -> AppSQLiteStore:
    return AppSQLiteStore(CONFIG.app_db_path)


def _tool_name(item: Any) -> str:
    return getattr(item, "name", getattr(item, "__name__", str(item)))


def json_payload(payload: dict[str, Any]) -> str:
    """도구 반환용 dict를 한글이 깨지지 않는 JSON 문자열로 변환합니다."""

    return json.dumps(payload, ensure_ascii=False)


def tool_result(tool_name: str, *, ok: bool = True, **payload: Any) -> dict[str, Any]:
    """Week 3 tool들이 공통으로 쓰는 JSON payload 껍데기를 만듭니다."""

    return {"ok": ok, "tool_name": tool_name, **payload}


class SaveStructuredRequestInput(StructuredRequest):
    """SQLite 저장 직전에 검증하는 Week 3 입력 스키마입니다."""

    kind: RequestKind = Field(default="unknown", description="분류된 요청 종류")
    source_schedule_id: str | None = Field(default=None, description="Week 1 임시 일정에서 넘어온 원본 일정 ID")

    @model_validator(mode="before")
    @classmethod
    def unwrap_legacy_payload(cls, value: Any) -> Any:
        """예전 trace의 payload wrapper만 짧게 풀고 실제 검증은 필드 스키마에 맡깁니다."""

        if isinstance(value, StructuredRequest):
            return value.model_dump()
        # {"ok":..., "structured_request": {...}} / {"payload": {...}} wrapper를 실제 필드 dict로 푼다.
        # wrapper가 조용히 extra-무시되면 kind=unknown 빈 row가 저장되므로 여기서 반드시 벗긴다.
        while isinstance(value, dict):
            inner = next(
                (value[key] for key in ("payload", "structured_request") if isinstance(value.get(key), dict)),
                None,
            )
            if inner is None:
                break
            # wrapper 바깥에만 있는 source_schedule_id는 보존한다 (Week 1 호환 경로)
            if "source_schedule_id" in value and "source_schedule_id" not in inner:
                inner = {**inner, "source_schedule_id": value["source_schedule_id"]}
            value = inner
        return value


def _save_input_from(value: SaveStructuredRequestInput | StructuredRequest | dict[str, Any] | str) -> SaveStructuredRequestInput:
    """저장 입력을 SaveStructuredRequestInput 하나로 모읍니다."""

    if isinstance(value, SaveStructuredRequestInput):
        return value
    if isinstance(value, (StructuredRequest, dict)):
        # wrapper 해제와 필드 검증은 unwrap_legacy_payload + 필드 스키마가 담당한다.
        return SaveStructuredRequestInput.model_validate(value)
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("{"):
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                data = None
            if isinstance(data, dict):
                return SaveStructuredRequestInput.model_validate(data)
        # JSON이 아닌 자연어는 Week 2 bridge로 먼저 구조화한다.
        return SaveStructuredRequestInput.model_validate(extract_structured_request(value).model_dump())
    raise TypeError(f"SaveStructuredRequestInput으로 변환할 수 없는 입력입니다: {type(value).__name__}")


def save_structured_request_payload(
    request: SaveStructuredRequestInput | StructuredRequest | dict[str, Any] | str,
    *,
    store: AppSQLiteStore | None = None,
) -> dict[str, Any]:
    """검증된 structured request를 앱 DB에 저장합니다."""

    save_input = _save_input_from(request)
    payload = {key: value for key, value in save_input.model_dump().items() if value is not None}
    saved = (store or _store()).save_structured_request(payload)
    return tool_result("save_structured_request", **saved)


class SavedRequestListInput(BaseModel):
    """저장 요청 목록 조회 입력입니다."""

    kind: RequestKind | None = None
    date_from: str | None = None
    date_to: str | None = None


class SavedRequestGetInput(BaseModel):
    """저장 요청 단건 조회 입력입니다."""

    request_id: str


class SavedScheduleListInput(BaseModel):
    """저장 일정 목록 조회 입력입니다."""

    limit: int = Field(default=50, ge=1, le=200)
    # 일정 종류만 허용한다. todo/reminder는 schedules 테이블에 없어 조회해도 항상 0건이므로,
    # 프롬프트 지시가 아니라 스키마 검증이 잘못된 tool 호출을 막는다.
    kind: ScheduleKind | None = Field(
        default=None,
        description=(
            "일정 종류만 지정합니다(personal_schedule/group_schedule). 미지정 시 personal_schedule. "
            "할 일(todo)·알림(reminder)은 이 tool이 아니라 list_saved_requests로 조회하세요."
        ),
    )
    date_from: str | None = None
    date_to: str | None = None


class SavedScheduleUpdateInput(BaseModel):
    """저장 일정 수정 입력입니다."""

    schedule_id: str
    title: str | None = None
    date: str | None = None
    start_time: str | None = None
    end_time: str | None = None
    attendees: list[str] | None = None


class SavedScheduleDeleteInput(BaseModel):
    """저장 일정 삭제 입력입니다."""

    schedule_ids: list[str] | None = None
    date: str | None = None
    title: str | None = None
    start_time: str | None = None
    time_unspecified: bool = False
    delete_all: bool = False


def _delete_saved_schedules(
    *,
    store: AppSQLiteStore,
    schedule_ids: list[str] | None = None,
    date: str | None = None,
    title: str | None = None,
    start_time: str | None = None,
    time_unspecified: bool = False,
    delete_all: bool = False,
) -> dict[str, Any]:
    """삭제 guard와 DB 호출을 한 곳에 둡니다."""

    filters = {
        "schedule_ids": schedule_ids,
        "date": date,
        "title": title,
        "start_time": start_time,
        "time_unspecified": time_unspecified,
        "delete_all": delete_all,
    }
    # 프롬프트 지시가 아니라 코드가 보장하는 안전 규칙: 조건 없는 삭제는 실행 자체를 거부한다.
    if not delete_all and not any([schedule_ids, date, title, start_time, time_unspecified]):
        return tool_result(
            "personal_delete_saved_schedules",
            ok=False,
            error="삭제 조건이 없습니다. schedule_ids/date/title/start_time/time_unspecified 중 하나를 지정하거나, 전체 삭제 의도라면 delete_all=True를 명시하세요.",
            deleted_count=0,
            filters=filters,
            deleted=[],
        )
    if delete_all:
        deleted = store.delete_all_schedules()
    else:
        deleted = store.delete_schedules_by_filter(
            schedule_ids=schedule_ids,
            date=date,
            title=title,
            start_time=start_time,
            time_unspecified=time_unspecified,
        )
    return tool_result(
        "personal_delete_saved_schedules",
        deleted_count=len(deleted),
        filters=filters,
        deleted=deleted,
    )


def _clean_dt(value: Any) -> str | None:
    """'미정'이나 빈 문자열처럼 형식이 아닌 값은 None(모름)으로 통일합니다."""

    text = str(value).strip() if value is not None else ""
    return text if text and text != "미정" else None


def structured_request_from_week01_schedule(schedule: dict[str, Any]) -> SaveStructuredRequestInput:
    """Week 1 임시 일정 dict를 Week 3 저장 입력으로 변환합니다."""

    return SaveStructuredRequestInput(
        kind="personal_schedule",
        title=schedule.get("title"),
        date=_clean_dt(schedule.get("date")),
        start_time=_clean_dt(schedule.get("start_time")),
        end_time=_clean_dt(schedule.get("end_time")),
        members=list(schedule.get("attendees") or []),
        reason="Week 1 personal_create_schedule 호환 경로로 생성된 개인 일정",
        original_text=str(schedule.get("title") or ""),
        source_schedule_id=schedule.get("id"),
    )


@tool("personal_create_schedule")
def personal_create_schedule(
    title: str,
    date: str,
    start_time: str,
    end_time: str = "미정",
    attendees: list[str] | None = None,
) -> str:
    """Nana의 개인 일정을 생성하고 Week 3+ 앱 SQLite DB에도 저장합니다."""

    # 1) Week 1 임시 일정을 그대로 생성해 이름/반환 계약을 유지한다 (이중 기록의 앞면).
    created = json.loads(
        week01_personal_create_schedule.invoke(
            {
                "title": title,
                "date": date,
                "start_time": start_time,
                "end_time": end_time,
                "attendees": attendees,
            }
        )
    )
    # 2) 같은 내용을 SQLite에도 저장한다 (뒷면). source_schedule_id 덕분에 재호출돼도 중복 저장되지 않는다.
    save_input = structured_request_from_week01_schedule(created.get("created_schedule") or {})
    sqlite_save = save_structured_request_payload(save_input)
    created["tool_name"] = "personal_create_schedule"
    created["structured_request"] = save_input.model_dump()
    created["sqlite_save"] = sqlite_save
    return json_payload(created)


@tool(args_schema=SaveStructuredRequestInput)
def save_structured_request(
    kind: RequestKind = "unknown",
    title: str | None = None,
    date: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
    members: list[str] | None = None,
    priority: str | None = None,
    reason: str | None = None,
    original_text: str = "",
    source_schedule_id: str | None = None,
) -> str:
    """Week 2 structured_request 필드를 검증한 뒤 SQLite에 저장합니다."""

    # args_schema가 검증을 끝냈으므로 본문은 저장 dict를 정리해 store에 넘기기만 한다.
    # None은 "모름"이라는 정보이므로 지어내지 않고 raw_json에서도 키를 제외해 모름을 명시적으로 남긴다.
    payload = {
        "kind": kind,
        "title": title,
        "date": date,
        "start_time": start_time,
        "end_time": end_time,
        "members": members if members is not None else [],
        "priority": priority,
        "reason": reason,
        "original_text": original_text,
        "source_schedule_id": source_schedule_id,
    }
    payload = {key: value for key, value in payload.items() if value is not None}
    saved = _store().save_structured_request(payload)
    result = tool_result("save_structured_request", **saved)
    # kind 누락/unknown은 감사 로그(structured_requests)에만 남고 일정/할 일 조회에 안 보인다.
    # 조용히 지나가지 않도록 결과에 경고를 담아 LLM이 kind를 채워 재시도하게 한다.
    if saved.get("kind") == "unknown":
        result["warning"] = (
            "kind가 unknown이라 structured_requests에만 저장되었고 일정/할 일/알림 조회에는 나타나지 않습니다. "
            "요청 종류를 알 수 있다면 kind를 지정해 다시 저장하세요."
        )
    return json_payload(result)


@tool(args_schema=SavedRequestListInput)
def list_saved_requests(
    kind: RequestKind | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> str:
    """SQLite에 저장된 구조화 요청을 조회합니다. 할 일(kind='todo')·알림(kind='reminder') 조회는 이 tool을 쓰고, kind를 비우면 전체를 반환합니다."""

    rows = _store().list_saved_requests(kind=kind, date_from=date_from, date_to=date_to)
    return json_payload(tool_result("list_saved_requests", rows=rows))


@tool(args_schema=SavedRequestGetInput)
def get_saved_request(request_id: str) -> str:
    """request_id로 구조화 요청 행 하나를 조회합니다."""

    # 결과가 없어도 예외를 던지지 않고 row=None을 유지해 agent가 "없음"을 그대로 읽게 한다.
    row = _store().get_saved_request(request_id)
    return json_payload(tool_result("get_saved_request", row=row))


@tool(args_schema=SavedScheduleListInput)
def personal_list_saved_schedules(
    limit: int = 50,
    kind: ScheduleKind | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> str:
    """앱 DB에 저장된 개인/그룹 '일정'만 반환합니다. 일정 조회와 수정/삭제 후보 확인용이며, 할 일·알림은 list_saved_requests로 조회하세요."""

    store = _store()
    resolved_kind = kind or "personal_schedule"
    schedules = store.list_schedules(limit=limit, kind=resolved_kind, date_from=date_from, date_to=date_to)
    filters = {"kind": resolved_kind, "date_from": date_from, "date_to": date_to, "limit": limit}
    result = tool_result("personal_list_saved_schedules", filters=filters, schedules=schedules)
    # 시스템 프롬프트 지시만으로는 LLM이 아래 두 실수를 자주 한다(확률적).
    # 결과 JSON 자체에 힌트를 담아 의사결정 지점에서 바로잡는다.
    notes: list[str] = []
    # 1) 날짜 언급 없는 수정/삭제 후보 확인에 오늘 날짜 필터를 넣어 0건으로 끝내는 실수
    if not schedules and (date_from or date_to):
        notes.append(
            "date 필터 범위에서 0건입니다. 사용자가 날짜를 명시하지 않았다면 "
            "date_from/date_to 없이 다시 조회해 제목으로 찾으세요."
        )
    # 2) kind 미지정 조회에서 그룹 일정 재조회를 건너뛰는 실수
    if kind is None:
        group_count = len(store.list_schedules(limit=limit, kind="group_schedule", date_from=date_from, date_to=date_to))
        if group_count > 0:
            notes.append(
                f"kind 미지정이라 personal_schedule만 반환했습니다. "
                f"같은 조건의 group_schedule 일정이 {group_count}건 따로 있습니다. "
                f"사용자가 전체/모든 일정을 물었다면 kind=group_schedule로 한 번 더 조회해 합쳐서 답하세요."
            )
    if notes:
        result["note"] = " ".join(notes)
    return json_payload(result)


def delete_saved_schedules_dict(
    schedule_ids: list[str] | None = None,
    date: str | None = None,
    title: str | None = None,
    start_time: str | None = None,
    time_unspecified: bool = False,
    delete_all: bool = False,
    app_store: AppSQLiteStore | None = None,
) -> dict[str, Any]:
    """tool invoke 없이 저장 일정 삭제 로직을 직접 호출합니다."""

    return _delete_saved_schedules(
        store=app_store or _store(),
        schedule_ids=schedule_ids,
        date=date,
        title=title,
        start_time=start_time,
        time_unspecified=time_unspecified,
        delete_all=delete_all,
    )


@tool(args_schema=SavedScheduleUpdateInput)
def personal_update_saved_schedule(
    schedule_id: str,
    title: str | None = None,
    date: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
    attendees: list[str] | None = None,
) -> str:
    """앱 DB에 저장된 내 일정 원본을 수정하고 공유 일정 복사본을 같은 값으로 갱신합니다."""

    # None은 "수정하지 않음"이라는 뜻 — store.update_schedule이 None 필드는 기존 값을 유지한다.
    updated = _store().update_schedule(
        schedule_id,
        title=title,
        date=date,
        start_time=start_time,
        end_time=end_time,
        attendees=attendees,
    )
    if updated is None:
        return json_payload(
            tool_result(
                "personal_update_saved_schedule",
                ok=False,
                error=f"schedule_id '{schedule_id}'에 해당하는 저장 일정을 찾지 못했습니다. personal_list_saved_schedules로 ID를 다시 확인하세요.",
                schedule_id=schedule_id,
            )
        )
    return json_payload(
        tool_result(
            "personal_update_saved_schedule",
            updated_schedule=updated["schedule"],
            shared_sync=updated["shared_sync"],
        )
    )


@tool(args_schema=SavedScheduleDeleteInput)
def personal_delete_saved_schedules(
    schedule_ids: list[str] | None = None,
    date: str | None = None,
    title: str | None = None,
    start_time: str | None = None,
    time_unspecified: bool = False,
    delete_all: bool = False,
) -> str:
    """Nana가 고른 일정 ID나 날짜/제목/시간 필터로 저장 일정을 삭제합니다."""

    return json_payload(
        _delete_saved_schedules(
            store=_store(),
            schedule_ids=schedule_ids,
            date=date,
            title=title,
            start_time=start_time,
            time_unspecified=time_unspecified,
            delete_all=delete_all,
        )
    )


def week03_tools() -> list[Any]:
    """Week 1 도구, Week 2 구조화 helper, SQLite 저장/조회/삭제 도구를 조립합니다."""

    base_tools = [
        personal_create_schedule if _tool_name(item) == "personal_create_schedule" else item for item in week01_tools()
    ]
    return [
        *base_tools,
        extract_schedule_request,
        save_structured_request,
        list_saved_requests,
        get_saved_request,
        personal_list_saved_schedules,
        personal_update_saved_schedule,
        personal_delete_saved_schedules,
    ]


def week03_system_prompt() -> str:
    """3주차 단일 agent가 따르는 시스템 프롬프트입니다."""

    return join_system_prompt(week03_prompt_parts())


def week03_prompt_parts() -> list[str]:
    """1~3주차 system prompt 조각을 누적합니다."""

    return [
        *week02_prompt_parts(),
        """당신은 Week 3 기록장 agent입니다. Week 2의 구조화 결과를 대화로 끝내지 않고
SQLite에 저장해 새 대화에서도 유지되는 기록으로 만듭니다.
Week 2의 'SQLite 저장을 하지 않는다'는 지시는 Week 3에서는 적용하지 않습니다.
최종 답변은 structured_response가 아니라 tool 결과 JSON을 근거로 한 자연어로 하세요.""",
        SQLITE_MEMORY_PROMPT,
        WEEK03_TOOL_CALL_PROMPT,
        """상대 날짜(내일, 다음 주 화요일 등)는 위에 안내된 오늘 날짜를 기준으로 해석합니다.
Week 3 tool 선택 기준: 일정/할 일/알림의 저장과 저장된 기록의 조회/수정/삭제는
SQLite tool(save_structured_request, personal_list_saved_schedules, list_saved_requests,
get_saved_request, personal_update_saved_schedule, personal_delete_saved_schedules)을
우선 사용하고, Week 1 임시 tool은 사용하지 않습니다.
조회 tool은 대상으로 구분합니다. 일정은 personal_list_saved_schedules,
할 일·알림은 list_saved_requests, ID를 아는 단건은 get_saved_request입니다.
Week 3에서는 RAG와 외부 멤버 일정 조율을 하지 않습니다.""",
    ]


def build_week03_agent() -> object:
    """Week 1-3 누적 tool 목록을 노출하는 단일 LangChain agent를 만듭니다."""

    if not CONFIG.has_openai_key:
        raise RuntimeError("PROXY_TOKEN이 .env에 필요합니다.")
    global _WEEK03_AGENT
    if _WEEK03_AGENT is None:
        # Week 2와 달리 최종 답변이 자연어이므로 response_format 없이 tool 루프만 연결한다.
        _WEEK03_AGENT = create_agent(
            model=chat_model(),
            tools=week03_tools(),
            system_prompt=week03_system_prompt(),
        )
    return _WEEK03_AGENT


def build_week_agent() -> object:
    """active-week registry가 호출하는 표준 Week agent builder입니다."""

    return build_week03_agent()
