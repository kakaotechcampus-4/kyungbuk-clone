from __future__ import annotations

import json
from typing import Any, Callable

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

SQLITE_MEMORY_PROMPT = """
Week 3부터 개인 일정/할 일/알림은 앱 SQLite DB에 저장된다.
Week 1의 personal_list_schedules/personal_delete_schedule은 이번 대화창에서만 보이는 임시 메모리이므로,
"저장된 내 일정"을 묻는 질문에는 반드시 personal_list_saved_schedules 같은 Week 3 SQLite 조회 tool을 사용한다.
SQLite에 저장된 내용은 새 대화를 열거나 앱을 다시 시작해도 그대로 남아 있다는 것을 전제로 답한다.
"""

WEEK03_TOOL_CALL_PROMPT = """
일정/할 일/알림을 저장할 때는 다음 순서를 따른다.
1. extract_schedule_request로 사용자의 자연어 요청을 구조화한다.
2. 구조화된 kind/title/date/start_time/end_time/members 등을 save_structured_request 인자로 그대로 넘겨 저장한다.
3. 저장된 내용을 조회할 때는 list_saved_requests, get_saved_request, personal_list_saved_schedules를 사용한다.
4. 일정을 수정하거나 삭제하기 전에는 먼저 personal_list_saved_schedules로 후보를 확인해 어떤 일정인지 특정한다.
5. schedule_id가 여러 개거나 특정할 수 없으면 삭제/수정하지 말고 사용자에게 어떤 일정인지 되묻는다.
6. delete_all은 사용자가 "전부 지워줘"처럼 명확히 전체 삭제를 요청했을 때만 사용한다.
7. tool 결과의 ok, error, error_type 같은 원본 JSON 필드를 사용자에게 그대로 보여주지 않는다.
   저장/조회/삭제가 실패했으면 그 이유를 한두 문장의 자연스러운 한국어로 풀어서 말한다.
"""


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
#      - get은 request_id 하나로 단건 조회합니다.
#      - 조회 결과가 없어도 예외를 던지지 말고 rows=[] 또는 row=None 형태를 유지합니다.
#
#   3. personal_list_saved_schedules
#      - 저장된 일정 목록을 반환해 "내 일정 보여줘" 같은 조회 질문과 이후 수정/삭제 후보 확인에 씁니다.
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
#
#   - [메인] personal_list_saved_schedules(...)
#     저장된 일정 row를 조회합니다. 수정/삭제 전 후보 schedule_id를 확인하거나 사용자의 일정 조회 질문에 답할 때 사용합니다.
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


def _store_call(tool_name: str, operation: Callable[[], dict[str, Any]]) -> str:
    """DB에 접근하는 모든 tool이 거쳐 가는 단일 관문입니다.

    tool 함수는 "무엇을 할지"만 operation으로 넘기고, 실패를 어떻게 표현할지는
    이 함수 한 곳에서만 결정한다. tool마다 try/except를 따로 복붙해두면 시간이 지나며
    tool별로 에러 응답 모양이 조금씩 어긋나기 쉽다.

    operation이 정상적으로 dict를 반환하면 그 dict를 그대로 결과에 실어준다.
    operation 스스로 "찾지 못함" 같은 정상적인 실패를 표현하고 싶으면
    반환 dict 안에 ok=False를 직접 넣어도 된다(예: personal_update_saved_schedule).
    operation 실행 중 예외가 나면(DB 연결 오류 등 예상 못 한 실패) 여기서 잡아서
    모든 tool이 동일한 모양(ok, error_type, error)으로 답하게 만든다.
    """

    try:
        return json_payload(tool_result(tool_name, **operation()))
    except Exception as exc:
        return json_payload(
            tool_result(tool_name, ok=False, error_type=type(exc).__name__, error=str(exc))
        )


def _unwrap_wrapper_dict(data: dict[str, Any]) -> dict[str, Any]:
    """payload/structured_request/requests[0] wrapper를 한 겹만 벗겨 낸 flat dict를 만듭니다.

    바깥 dict가 이미 flat이면(즉 wrapper 키가 하나도 없으면) 그대로 반환합니다.
    바깥 키가 안쪽 값과 겹치면 바깥 값이 우선합니다 — 나중에 붙인 source_schedule_id 같은
    보정 값이 예전 payload 안의 값보다 항상 이긴다는 뜻입니다.
    """

    inner: dict[str, Any] | None = None
    if isinstance(data.get("payload"), dict):
        inner = data["payload"]
    elif isinstance(data.get("structured_request"), dict):
        inner = data["structured_request"]
    elif isinstance(data.get("requests"), list) and data["requests"] and isinstance(data["requests"][0], dict):
        inner = data["requests"][0]

    if inner is None:
        return data
    return {**inner, **data}


class SaveStructuredRequestInput(StructuredRequest):
    """SQLite 저장 직전에 검증하는 Week 3 입력 스키마입니다."""

    kind: RequestKind = Field(default="unknown", description="분류된 요청 종류")
    source_schedule_id: str | None = Field(default=None, description="Week 1 임시 일정에서 넘어온 원본 일정 ID")

    @model_validator(mode="before")
    @classmethod
    def unwrap_legacy_payload(cls, value: Any) -> Any:
        """예전 trace의 payload wrapper만 짧게 풀고 실제 검증은 필드 스키마에 맡깁니다."""

        if not isinstance(value, dict):
            return value
        return _unwrap_wrapper_dict(value)


def _save_input_from(value: SaveStructuredRequestInput | StructuredRequest | dict[str, Any] | str) -> SaveStructuredRequestInput:
    """저장 입력을 SaveStructuredRequestInput 하나로 모읍니다."""

    if isinstance(value, SaveStructuredRequestInput):
        return value
    if isinstance(value, StructuredRequest):
        # SaveStructuredRequestInput은 StructuredRequest의 하위 클래스라서 model_validate가
        # 상위 클래스 인스턴스를 그대로 받아들이지 못한다. dict로 한 번 풀어서 넘긴다.
        return SaveStructuredRequestInput.model_validate(value.model_dump())
    if isinstance(value, dict):
        return SaveStructuredRequestInput.model_validate(value)
    if isinstance(value, str):
        try:
            parsed: Any = json.loads(value)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            return SaveStructuredRequestInput.model_validate(parsed)
        # JSON이 아닌 순수 자연어 문자열이다. Week 2의 extract_structured_request가 아직
        # 구현되지 않았다면 여기서 NotImplementedError가 그대로 올라간다 — 이는 Week 3의 책임이
        # 아니라 Week 2 쪽 TODO이므로, 의미를 감추는 대신 원래 예외를 그대로 전파한다.
        structured_request = extract_structured_request(value)
        return SaveStructuredRequestInput.model_validate(structured_request.model_dump())
    raise TypeError(f"_save_input_from은 지원하지 않는 타입을 받았습니다: {type(value).__name__}")


def save_structured_request_payload(
    request: SaveStructuredRequestInput | StructuredRequest | dict[str, Any] | str,
    *,
    store: AppSQLiteStore | None = None,
) -> dict[str, Any]:
    """검증된 structured request를 앱 DB에 저장합니다."""

    validated = _save_input_from(request)
    active_store = store or _store()
    # model_dump()를 그대로 넘기는 이유: AppSQLiteStore.save_structured_request는 모든 필드를
    # payload.get(key)로 읽으므로, 필드가 아예 없는 것과 값이 None인 것을 구분하지 않는다.
    # 따라서 여기서 None 필드를 걸러내도 저장 결과는 달라지지 않는다.
    save_result = active_store.save_structured_request(validated.model_dump())
    return tool_result("save_structured_request", ok=True, **save_result)


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
    kind: RequestKind | None = None
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

    filters: dict[str, Any] = {
        "schedule_ids": schedule_ids,
        "date": date,
        "title": title,
        "start_time": start_time,
        "time_unspecified": time_unspecified,
        "delete_all": delete_all,
    }

    has_condition = delete_all or bool(schedule_ids) or bool(date) or bool(title) or bool(start_time) or time_unspecified
    if not has_condition:
        return {
            "ok": False,
            "error": "삭제 조건이 없습니다. schedule_ids나 날짜/제목/시간 중 하나, 또는 delete_all을 지정하세요.",
            "deleted_count": 0,
            "filters": filters,
            "deleted": [],
        }

    if delete_all:
        deleted_rows = store.delete_all_schedules()
    else:
        deleted_rows = store.delete_schedules_by_filter(
            schedule_ids=schedule_ids,
            date=date,
            title=title,
            start_time=start_time,
            time_unspecified=time_unspecified,
        )

    return {
        "ok": True,
        "deleted_count": len(deleted_rows),
        "filters": filters,
        "deleted": deleted_rows,
    }


def structured_request_from_week01_schedule(schedule: dict[str, Any]) -> SaveStructuredRequestInput:
    """Week 1 임시 일정 dict를 Week 3 저장 입력으로 변환합니다."""

    return SaveStructuredRequestInput(
        kind="personal_schedule",
        title=schedule.get("title"),
        date=schedule.get("date"),
        start_time=schedule.get("start_time"),
        end_time=schedule.get("end_time"),
        members=list(schedule.get("attendees") or []),
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

    def _create_and_save() -> dict[str, Any]:
        week01_result: dict[str, Any] = json.loads(
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
        created_schedule: dict[str, Any] = week01_result["created_schedule"]
        save_input = structured_request_from_week01_schedule(created_schedule)
        sqlite_save = save_structured_request_payload(save_input)
        return {
            "created_schedule": created_schedule,
            "structured_request": save_input.model_dump(),
            "sqlite_save": sqlite_save,
        }

    return _store_call("personal_create_schedule", _create_and_save)


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

    def _save() -> dict[str, Any]:
        save_payload: dict[str, Any] = {
            "kind": kind,
            "title": title,
            "date": date,
            "start_time": start_time,
            "end_time": end_time,
            "members": members or [],
            "priority": priority,
            "reason": reason,
            "original_text": original_text,
            "source_schedule_id": source_schedule_id,
        }
        return _store().save_structured_request(save_payload)

    return _store_call("save_structured_request", _save)


@tool(args_schema=SavedRequestListInput)
def list_saved_requests(
    kind: RequestKind | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> str:
    """SQLite에 저장된 구조화 요청 목록을 조회합니다."""

    def _list() -> dict[str, Any]:
        return {"rows": _store().list_saved_requests(kind=kind, date_from=date_from, date_to=date_to)}

    return _store_call("list_saved_requests", _list)


@tool(args_schema=SavedRequestGetInput)
def get_saved_request(request_id: str) -> str:
    """request_id로 구조화 요청 행 하나를 조회합니다."""

    return _store_call("get_saved_request", lambda: {"row": _store().get_saved_request(request_id)})


@tool(args_schema=SavedScheduleListInput)
def personal_list_saved_schedules(
    limit: int = 50,
    kind: RequestKind | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> str:
    """앱 DB에 저장된 일정 목록을 날짜/종류 필터로 반환합니다. Nana가 조회/수정/삭제 후보를 볼 때 사용합니다."""

    def _list_schedules() -> dict[str, Any]:
        # "내 일정 보여줘" 같은 질문은 보통 개인 일정을 뜻하므로, kind를 안 정하면 group_schedule까지
        # 섞여 나오지 않도록 personal_schedule을 기본값으로 삼는다.
        effective_kind = kind or "personal_schedule"
        filters: dict[str, Any] = {
            "limit": limit,
            "kind": effective_kind,
            "date_from": date_from,
            "date_to": date_to,
        }
        schedules = _store().list_schedules(limit=limit, kind=effective_kind, date_from=date_from, date_to=date_to)
        return {"filters": filters, "schedules": schedules}

    return _store_call("personal_list_saved_schedules", _list_schedules)


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

    def _update() -> dict[str, Any]:
        # None으로 들어온 필드는 "수정하지 않음"을 뜻하고, AppSQLiteStore.update_schedule이 이미
        # 그 규칙(None이면 기존 값 유지)으로 동작하므로 여기서는 그대로 전달하기만 하면 된다.
        update_result = _store().update_schedule(
            schedule_id,
            title=title,
            date=date,
            start_time=start_time,
            end_time=end_time,
            attendees=attendees,
        )
        if update_result is None:
            # DB 오류가 아니라 "그런 일정이 없다"는 정상적인 실패이므로 예외 대신
            # ok=False가 담긴 dict를 그대로 돌려준다 — _store_call은 이걸 그대로 실어 보낸다.
            return {"ok": False, "error": f"schedule_id={schedule_id}에 해당하는 저장 일정을 찾지 못했습니다."}
        return {"updated_schedule": update_result["schedule"], "shared_sync": update_result["shared_sync"]}

    return _store_call("personal_update_saved_schedule", _update)


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

    def _delete() -> dict[str, Any]:
        return _delete_saved_schedules(
            store=_store(),
            schedule_ids=schedule_ids,
            date=date,
            title=title,
            start_time=start_time,
            time_unspecified=time_unspecified,
            delete_all=delete_all,
        )

    return _store_call("personal_delete_saved_schedules", _delete)


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
        f"""
        너는 이제 Week 2의 구조화 결과를 SQLite 기록장에 저장하고 다시 꺼내 볼 수 있는 Week 3 Nana다.
        오늘 날짜는 {current_app_date_iso()}다.
        """,
        SQLITE_MEMORY_PROMPT,
        WEEK03_TOOL_CALL_PROMPT,
        """
        Week 3 tool 선택 기준:
        - 개인 일정을 새로 만들 때는 personal_create_schedule을 호출한다. 이 tool은 Week 1 임시 메모리와
          SQLite 저장을 함께 처리하므로 save_structured_request를 따로 또 호출할 필요는 없다.
        - todo, reminder, group_schedule처럼 personal_create_schedule로 만들 수 없는 요청은
          extract_schedule_request로 구조화한 뒤 save_structured_request로 저장한다.
        - 여러 사람의 일정을 조율하는 요청(Week 4 이후 범위)은 이번 주차에서 다루지 않는다.
        """,
    ]


def build_week03_agent() -> object:
    """Week 1-3 누적 tool 목록을 노출하는 단일 LangChain agent를 만듭니다."""

    if not CONFIG.has_openai_key:
        raise RuntimeError("PROXY_TOKEN이 .env에 필요합니다.")
    global _WEEK03_AGENT
    if _WEEK03_AGENT is None:
        _WEEK03_AGENT = create_agent(
            model=chat_model(),
            tools=week03_tools(),
            system_prompt=week03_system_prompt(),
        )
    return _WEEK03_AGENT


def build_week_agent() -> object:
    """active-week registry가 호출하는 표준 Week agent builder입니다."""

    return build_week03_agent()
