from __future__ import annotations

import json
from typing import Any

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

# Week 1의 임시 메모리와 달리, Week 3부터 일정은 앱 SQLite DB에 남는다는 점을 모델에게 알려준다.
SQLITE_MEMORY_PROMPT = (
    "Week 3부터 일정/할 일/알림은 현재 대화의 임시 메모리가 아니라 앱 SQLite DB에 저장된다. "
    "SQLite에 저장된 기록은 새 대화를 열거나 앱을 다시 시작해도 유지된다. "
    "저장된 일정을 묻는 질문에는 기억이 없다고 답하지 말고, "
    "personal_list_saved_schedules 같은 조회 tool을 호출해 DB에서 확인한 결과로 답한다."
)

# 구조화 → 저장 → 조회/수정/삭제로 이어지는 tool 호출 순서를 고정한다.
WEEK03_TOOL_CALL_PROMPT = (
    "일정/할 일/알림 저장 요청은 먼저 extract_schedule_request(query=사용자 요청 원문)를 호출해 구조화한다. "
    "그 결과 structured_request의 kind/title/date/start_time/end_time/members/priority/reason/original_text를 "
    "save_structured_request 인자로 그대로 전달해 SQLite에 저장한다. "
    "특히 참석자·멤버가 있는 일정은 personal_create_schedule로 만들지 말고 "
    "반드시 extract_schedule_request → save_structured_request 경로로 저장해 group_schedule로 분류되게 한다. "
    "저장된 일정 조회와 수정/삭제 후보 확인은 personal_list_saved_schedules를 쓴다. "
    "kind를 지정하지 않으면 개인 일정과 그룹 일정이 모두 나오므로, 특정 종류만 좁혀 볼 때만 kind를 지정한다. "
    "할 일과 알림까지 포함한 '모든 기록' 질문은 list_saved_requests(kind 미지정)로 확인한다. "
    "수정/삭제는 먼저 personal_list_saved_schedules로 schedule_id 후보를 확인한 뒤 "
    "personal_update_saved_schedule 또는 personal_delete_saved_schedules를 호출한다. "
    "삭제할 때는 schedule_ids나 날짜/제목/시간 필터를 반드시 명시하고, "
    "사용자가 전부 지우라고 분명히 말할 때만 delete_all=True를 쓴다."
)


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


class SaveStructuredRequestInput(StructuredRequest):
    """SQLite 저장 직전에 검증하는 Week 3 입력 스키마입니다."""

    kind: RequestKind = Field(default="unknown", description="분류된 요청 종류")
    source_schedule_id: str | None = Field(default=None, description="Week 1 임시 일정에서 넘어온 원본 일정 ID")

    @model_validator(mode="before")
    @classmethod
    def unwrap_legacy_payload(cls, value: Any) -> Any:
        """예전 trace의 payload wrapper만 짧게 풀고 실제 검증은 필드 스키마에 맡깁니다."""

        # Week 2 StructuredRequest 객체가 그대로 들어오면 dict로 풀어 필드 검증에 넘긴다.
        if isinstance(value, StructuredRequest):
            return value.model_dump()
        # 예전 trace/테스트는 {"payload": {...}}나 {"structured_request": {...}}로 감싸 온다.
        # wrapper 한 겹만 벗겨 내고, 필드 값 검증은 스키마에 맡긴다.
        if isinstance(value, dict):
            inner = value.get("structured_request") or value.get("payload")
            if isinstance(inner, dict):
                return inner
        return value


def _save_input_from(value: SaveStructuredRequestInput | StructuredRequest | dict[str, Any] | str) -> SaveStructuredRequestInput:
    """저장 입력을 SaveStructuredRequestInput 하나로 모읍니다."""

    # 이미 저장 입력이면 그대로 쓴다.
    if isinstance(value, SaveStructuredRequestInput):
        return value
    # StructuredRequest와 dict는 unwrap_legacy_payload가 wrapper를 풀어 주므로 검증만 거치면 된다.
    if isinstance(value, (StructuredRequest, dict)):
        return SaveStructuredRequestInput.model_validate(value)
    if isinstance(value, str):
        text = value.strip()
        # JSON 문자열이면 dict로 파싱해 검증한다.
        if text.startswith("{"):
            try:
                return SaveStructuredRequestInput.model_validate(json.loads(text))
            except json.JSONDecodeError:
                pass
        # JSON이 아닌 자연어는 Week 2 bridge로 먼저 구조화한다.
        return SaveStructuredRequestInput.model_validate(extract_structured_request(text))
    raise RuntimeError(f"저장 입력으로 바꿀 수 없는 값입니다: {type(value)!r}")


def save_structured_request_payload(
    request: SaveStructuredRequestInput | StructuredRequest | dict[str, Any] | str,
    *,
    store: AppSQLiteStore | None = None,
) -> dict[str, Any]:
    """검증된 structured request를 앱 DB에 저장합니다."""

    save_input = _save_input_from(request)
    # 모르는 값(None)은 raw_json에 null로 남기지 않고 아예 뺀다.
    payload = save_input.model_dump(exclude_none=True)
    # original_text의 빈 문자열도 시간 필드의 "미정"처럼 "모르는 값" sentinel이므로 같이 뺀다.
    # (LLM이 extract를 건너뛰고 save를 직접 불러도 ""가 원문 자리에 저장되지 않게 한다.
    #  Week 4부터 raw_json이 검색 대상이라 빈 원문이 저장 품질 문제가 된다.)
    if not str(payload.get("original_text", "")).strip():
        payload.pop("original_text", None)
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

    filters = {
        "schedule_ids": schedule_ids,
        "date": date,
        "title": title,
        "start_time": start_time,
        "time_unspecified": time_unspecified,
        "delete_all": delete_all,
    }
    # 조건이 하나도 없으면 실수로 전체 일정이 지워지는 것을 막는다.
    has_filter = bool(schedule_ids) or bool(date) or bool(title) or bool(start_time) or time_unspecified
    if not delete_all and not has_filter:
        return tool_result(
            "personal_delete_saved_schedules",
            ok=False,
            error="삭제 조건이 없습니다. schedule_ids나 날짜/제목/시간 필터를 지정하거나 delete_all=True를 명시하세요.",
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
    # deleted_count/filters/deleted를 유지해야 trace에서 무엇이 지워졌는지 확인할 수 있다.
    return tool_result(
        "personal_delete_saved_schedules",
        deleted_count=len(deleted),
        filters=filters,
        deleted=deleted,
    )


def _time_or_none(value: str | None) -> str | None:
    """Week 1의 '미정'/빈 문자열 sentinel을 Week 3 스키마 규칙(모르는 값은 None)으로 바꿉니다."""

    if value is None or value.strip() in ("", "미정"):
        return None
    return value


def structured_request_from_week01_schedule(schedule: dict[str, Any]) -> SaveStructuredRequestInput:
    """Week 1 임시 일정 dict를 Week 3 저장 입력으로 변환합니다."""

    attendees = schedule.get("attendees") or []
    return SaveStructuredRequestInput(
        # personal_create_schedule 경로로 만든 일정이므로 kind는 personal_schedule로 고정한다.
        # 참석자 기반 group 분류는 extract_schedule_request → save_structured_request 경로에서 LLM이 정한다.
        kind="personal_schedule",
        title=schedule.get("title"),
        date=schedule.get("date"),
        start_time=_time_or_none(schedule.get("start_time")),
        end_time=_time_or_none(schedule.get("end_time")),
        members=attendees,
        # 원문이 안 넘어온 호출(테스트/직접 invoke)에서 original_text는 비워 둔다.
        # 예전에는 title/date를 담은 JSON을 fallback으로 넣었지만, 그 값들은 이미
        # raw_json(payload 전체 dump)과 title 컬럼에 저장돼 LIKE 검색되므로 새 검색 정보가 없고,
        # 원문 필드에 자연어/JSON 두 성격이 섞이는 비용만 남는다.
        # "모르는 값은 비운다"(시간 필드의 미정 → None과 같은 규칙)로 통일한다.
        # 빈 original_text는 save_structured_request_payload가 저장에서 제외한다.
        source_schedule_id=schedule.get("id"),
    )


@tool("personal_create_schedule")
def personal_create_schedule(
    title: str,
    date: str,
    start_time: str,
    end_time: str = "미정",
    attendees: list[str] | None = None,
    original_text: str = "",
) -> str:
    """Nana의 개인 일정을 생성하고 Week 3+ 앱 SQLite DB에도 저장합니다.

    original_text에는 사용자의 이번 요청 문장을 요약하지 말고 원문 그대로 담습니다.
    """

    # 먼저 Week 1 tool을 그대로 호출해 현재 대화의 임시 메모리에 일정을 만든다.
    raw = week01_personal_create_schedule.invoke(
        {
            "title": title,
            "date": date,
            "start_time": start_time,
            "end_time": end_time,
            "attendees": attendees or [],
        }
    )
    created = json.loads(raw)
    schedule = created.get("created_schedule") or {}
    # 같은 내용을 Week 3 저장 입력으로 바꿔 SQLite에도 저장한다(이중 기록).
    save_input = structured_request_from_week01_schedule(schedule)
    # 사용자 원문이 넘어온 경우에만 채운다. 안 넘어오면 빈 값 그대로 두어 저장에서 제외된다.
    # (extract_schedule_request가 query 인자로 원문을 받는 것과 같은 Week 2 방식으로 두 저장 경로를 통일)
    if original_text:
        save_input.original_text = original_text
    sqlite_save = save_structured_request_payload(save_input)
    return json_payload(
        {
            **created,
            "tool_name": "personal_create_schedule",
            "structured_request": save_input.model_dump(),
            "sqlite_save": sqlite_save,
        }
    )


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

    # 저장 규칙(None 제외, store 호출)은 save_structured_request_payload 한 곳에 모은다.
    # tool은 args_schema가 검증한 인자를 dict로 모아 helper에 넘기는 입구 역할만 한다.
    return json_payload(
        save_structured_request_payload(
            {
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
        )
    )


@tool(args_schema=SavedRequestListInput)
def list_saved_requests(
    kind: RequestKind | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> str:
    """SQLite에 저장된 구조화 요청 목록을 조회합니다."""

    # 결과가 없으면 rows가 빈 list로 나온다. 예외를 던지지 않는다.
    rows = _store().list_saved_requests(kind=kind, date_from=date_from, date_to=date_to)
    return json_payload(tool_result("list_saved_requests", rows=rows))


@tool(args_schema=SavedRequestGetInput)
def get_saved_request(request_id: str) -> str:
    """request_id로 구조화 요청 행 하나를 조회합니다."""

    # 없는 ID여도 예외 대신 row=None으로 답해 agent가 이어서 판단하게 한다.
    row = _store().get_saved_request(request_id)
    return json_payload(tool_result("get_saved_request", request_id=request_id, row=row))


@tool(args_schema=SavedScheduleListInput)
def personal_list_saved_schedules(
    limit: int = 50,
    kind: RequestKind | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> str:
    """앱 DB에 저장된 내 일정 목록을 반환합니다. kind를 지정하지 않으면 개인+그룹 일정을 모두 조회하고,
    특정 종류만 볼 때 kind로 좁힙니다. Nana가 조회/수정/삭제 후보를 볼 때 사용합니다.
    '공유 일정 저장소'의 일정 조회는 이 tool이 아니라 list_shared_schedules가 담당합니다."""

    # kind 미지정이면 개인+그룹 일정을 모두 조회한다. schedules 테이블에는 일정 kind만 들어 있어
    # 할 일/알림이 섞이지 않는다. "그룹은 kind를 바꿔 한 번 더 조회"같은 프롬프트 지시는
    # LLM이 빠뜨릴 수 있으므로, 기본 조회 범위는 tool이 보장한다.
    filters = {"limit": limit, "kind": kind, "date_from": date_from, "date_to": date_to}
    schedules = _store().list_schedules(limit=limit, kind=kind, date_from=date_from, date_to=date_to)
    return json_payload(tool_result("personal_list_saved_schedules", filters=filters, schedules=schedules))


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

    # None 인자는 "수정하지 않음"이라는 뜻이다. store가 None 필드는 기존 값을 유지한다.
    result = _store().update_schedule(
        schedule_id,
        title=title,
        date=date,
        start_time=start_time,
        end_time=end_time,
        attendees=attendees,
    )
    if result is None:
        return json_payload(
            tool_result(
                "personal_update_saved_schedule",
                ok=False,
                schedule_id=schedule_id,
                error="schedule_id에 해당하는 저장 일정이 없습니다.",
            )
        )
    return json_payload(
        tool_result(
            "personal_update_saved_schedule",
            schedule_id=schedule_id,
            updated_schedule=result["schedule"],
            shared_sync=result["shared_sync"],
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

    result = _delete_saved_schedules(
        store=_store(),
        schedule_ids=schedule_ids,
        date=date,
        title=title,
        start_time=start_time,
        time_unspecified=time_unspecified,
        delete_all=delete_all,
    )
    return json_payload(result)


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

    today = current_app_date_iso()
    week03_role = (
        "너는 구조화 결과를 앱 SQLite DB에 기록하고 다시 꺼내 쓰는 3주차 agent다. "
        f"오늘은 {today}이고, 상대 날짜는 이 날짜를 기준으로 해석한다. "
        "Week 2의 'SQLite 저장을 하지 않는다'는 지시는 Week 3부터 적용하지 않는다. "
        "저장 요청은 extract_schedule_request → save_structured_request 순서를 기본으로 하고, "
        "저장된 일정 질문에는 Week 1 임시 메모리 tool이 아니라 personal_list_saved_schedules로 답한다. "
        "최종 답변은 tool 결과를 근거로 한 짧은 자연어 답변이다. "
        "RAG 검색과 여러 멤버의 일정 조율은 아직 Week 3 범위가 아니다."
    )
    return [
        *week02_prompt_parts(),
        SQLITE_MEMORY_PROMPT,
        WEEK03_TOOL_CALL_PROMPT,
        week03_role,
    ]


def build_week03_agent() -> object:
    """Week 1-3 누적 tool 목록을 노출하는 단일 LangChain agent를 만듭니다."""

    if not CONFIG.has_openai_key:
        raise RuntimeError("PROXY_TOKEN이 .env에 필요합니다.")
    global _WEEK03_AGENT
    if _WEEK03_AGENT is None:
        # Week 2와 달리 response_format이 없다. 구조화는 extract_schedule_request tool 안에서 일어나고,
        # 최종 답변은 tool 결과를 근거로 한 자연어이기 때문이다.
        _WEEK03_AGENT = create_agent(
            model=chat_model(),
            tools=week03_tools(),
            system_prompt=week03_system_prompt(),
        )
    return _WEEK03_AGENT


def build_week_agent() -> object:
    """active-week registry가 호출하는 표준 Week agent builder입니다."""

    return build_week03_agent()
