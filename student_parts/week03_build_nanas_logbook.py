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

# Week 1-2는 대화가 끝나면 일정이 사라지는 임시 메모리였다.
# Week 3부터는 SQLite에 영구 저장되므로, 새 대화에서도 조회가 가능하다.
# LLM이 "기억하는 척" 하지 않고 항상 DB에서 실제로 꺼내오도록 지시해야 한다.
SQLITE_MEMORY_PROMPT = """Week 3부터 나나는 임시 메모리 대신 SQLite 기록장을 사용한다.
대화가 끝나거나 새 대화를 열어도 저장된 일정과 할일과 알림은 그대로 남는다.
사용자가 과거 일정을 물어보면 기억하는 척 답하지 말고, 반드시 personal_list_saved_schedules로 DB에서 조회한다."""

# Week 2에서 구조화만 했다면, Week 3는 구조화 → 저장 → 조회/수정/삭제 전체 흐름이다.
# LLM이 extract 후 save를 빠뜨리거나, 조회 전 list 확인을 건너뛰는 실수를 막기 위해 순서를 명시한다.
WEEK03_TOOL_CALL_PROMPT = """Week 3 tool 호출 순서:
1. 저장 요청: extract_schedule_request(query=...) → save_structured_request(kind/title/date/... 직접 전달)
2. 조회 요청: personal_list_saved_schedules 호출
3. 수정 요청: personal_list_saved_schedules로 schedule_id 먼저 확인 → personal_update_saved_schedule 호출
4. 삭제 요청: personal_list_saved_schedules로 후보 확인 → personal_delete_saved_schedules에 schedule_ids 전달
JSON 외 자연어 텍스트는 절대 출력하지 않는다."""


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

        if not isinstance(value, dict):
            return value

        # {"payload": {"kind": "...", ...}} 형태로 들어오는 예전 trace 풀기
        if "payload" in value and isinstance(value["payload"], dict):
            inner = value["payload"]
            # payload 안에 structured_request가 또 중첩된 경우
            if "structured_request" in inner and isinstance(inner["structured_request"], dict):
                return inner["structured_request"]
            return inner

        # {"structured_request": {"kind": "...", ...}} 형태 풀기
        if "structured_request" in value and isinstance(value["structured_request"], dict):
            return value["structured_request"]

        return value


def _save_input_from(
    value: SaveStructuredRequestInput | StructuredRequest | dict[str, Any] | str,
) -> SaveStructuredRequestInput:
    """저장 입력을 SaveStructuredRequestInput 하나로 모읍니다."""

    # 자연어 문자열이 들어오면 Week 2 구조화를 먼저 거쳐야 한다.
    # tool 경로에서는 LLM이 이미 구조화해서 넘기므로 이 분기에 거의 안 들어온다.
    if isinstance(value, str):
        value = extract_structured_request(value)

    # SaveStructuredRequestInput이 아닌 StructuredRequest는 dict로 변환 후 재검증
    if isinstance(value, StructuredRequest) and not isinstance(value, SaveStructuredRequestInput):
        value = value.model_dump()

    if isinstance(value, dict):
        return SaveStructuredRequestInput.model_validate(value)

    return value


def save_structured_request_payload(
    request: SaveStructuredRequestInput | StructuredRequest | dict[str, Any] | str,
    *,
    store: AppSQLiteStore | None = None,
) -> dict[str, Any]:
    """검증된 structured request를 앱 DB에 저장합니다."""

    inp = _save_input_from(request)
    s = store or _store()

    # None 필드를 제외하고 저장한다.
    # 모르는 값을 null로 억지로 넣으면 나중에 필터 조회가 오염될 수 있다.
    payload = {k: v for k, v in inp.model_dump().items() if v is not None}
    result = s.save_structured_request(payload)

    return tool_result("save_structured_request", ok=True, saved=result)


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

    # 아무 조건도 없이 삭제하면 전체가 날아갈 수 있다.
    # delete_all 플래그가 없는 한 최소 하나의 조건이 있어야 한다.
    has_condition = any([schedule_ids, date, title, start_time, time_unspecified, delete_all])
    if not has_condition:
        return tool_result(
            "personal_delete_saved_schedules",
            ok=False,
            error="삭제 조건이 없습니다. schedule_ids 또는 날짜/제목 필터를 지정하세요.",
        )

    filters = {
        "schedule_ids": schedule_ids,
        "date": date,
        "title": title,
        "start_time": start_time,
        "time_unspecified": time_unspecified,
        "delete_all": delete_all,
    }

    # delete_all > schedule_ids > 필터 순서로 처리한다.
    # schedule_ids가 있으면 정확히 그 ID만 지우는 게 안전하다.
    if delete_all:
        deleted = store.delete_all_schedules()
        deleted_list = deleted if isinstance(deleted, list) else []
    elif schedule_ids:
        # AppSQLiteStore에 delete_schedules_by_ids가 없어서 delete_schedule(단수)로 순회한다.
        # 삭제 성공한 것만 리스트에 담아서 반환한다.
        # 없는 ID를 삭제하려 할 때 None이 반환되므로 필터링이 필요하다.
        deleted_list = []
        for sid in schedule_ids:
            result = store.delete_schedule(sid)
            if result:
                deleted_list.append(result)
    else:
        # 날짜/제목/시간 필터로 삭제한다.
        # 필터가 너무 느슨하면 의도치 않은 일정이 삭제될 수 있어서 가이드 주석을 남긴다.
        deleted = store.delete_schedules_by_filter(
            date=date,
            title=title,
            start_time=start_time,
            time_unspecified=time_unspecified,
        )
        deleted_list = deleted if isinstance(deleted, list) else []

    return tool_result(
        "personal_delete_saved_schedules",
        ok=True,
        deleted_count=len(deleted_list),
        filters=filters,
        deleted=deleted_list,
    )


def structured_request_from_week01_schedule(schedule: dict[str, Any]) -> SaveStructuredRequestInput:
    """Week 1 임시 일정 dict를 Week 3 저장 입력으로 변환합니다."""

    # Week 1은 attendees 필드를 썼지만 Week 3는 members를 쓴다.
    # 참석자가 있으면 group_schedule, 없으면 personal_schedule로 자동 분류한다.
    attendees = schedule.get("attendees") or []
    kind: RequestKind = "group_schedule" if attendees else "personal_schedule"

    return SaveStructuredRequestInput(
        kind=kind,
        title=schedule.get("title"),
        date=schedule.get("date"),
        start_time=schedule.get("start_time"),
        end_time=schedule.get("end_time") if schedule.get("end_time") not in (None, "미정") else None,
        members=attendees,
        original_text=schedule.get("title", ""),
        # Week 1 임시 ID를 보존해서 나중에 어디서 왔는지 추적할 수 있게 한다.
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

    # Week 1 임시 저장을 먼저 한다. 기존 동작을 그대로 유지해야 하기 때문이다.
    week1_result_str = week01_personal_create_schedule.invoke({
        "title": title,
        "date": date,
        "start_time": start_time,
        "end_time": end_time,
        "attendees": attendees or [],
    })
    week1_result = json.loads(week1_result_str)
    created_schedule = week1_result.get("created_schedule", {})

    # created_schedule에서 값을 꺼내는 대신 함수 인자를 직접 쓴다.
    # created_schedule 구조가 달라져도 안전하게 저장할 수 있다.
    attendees_list = attendees or []
    kind: RequestKind = "group_schedule" if attendees_list else "personal_schedule"

    payload = {
        "kind": kind,
        "title": title,
        "date": date,
        "start_time": start_time,
        "members": attendees_list,
        "original_text": title,
        "source_schedule_id": created_schedule.get("id"),
    }
    if end_time and end_time != "미정":
        payload["end_time"] = end_time

    store = _store()
    sqlite_result = store.save_structured_request(payload)

    return json_payload({
        "ok": True,
        "tool_name": "personal_create_schedule",
        "created_schedule": created_schedule,
        "structured_request": {k: v for k, v in payload.items() if v is not None},
        "sqlite_save": sqlite_result,
    })


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

    # args_schema가 이미 입력을 검증했으므로 본문은 저장만 담당한다.
    # None 값을 포함하면 DB에 null이 들어가 나중에 필터 조회가 복잡해진다.
    payload = {
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
    payload = {k: v for k, v in payload.items() if v is not None}

    store = _store()
    result = store.save_structured_request(payload)

    return json_payload(tool_result("save_structured_request", ok=True, saved=result))


@tool(args_schema=SavedRequestListInput)
def list_saved_requests(
    kind: RequestKind | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> str:
    """SQLite에 저장된 구조화 요청 목록을 조회합니다."""

    store = _store()
    rows = store.list_saved_requests(kind=kind, date_from=date_from, date_to=date_to)

    return json_payload(tool_result("list_saved_requests", ok=True, rows=rows))


@tool(args_schema=SavedRequestGetInput)
def get_saved_request(request_id: str) -> str:
    """request_id로 구조화 요청 행 하나를 조회합니다."""

    store = _store()
    row = store.get_saved_request(request_id)

    # 결과가 없어도 예외를 던지지 않는다.
    # LLM이 None을 받으면 "해당 요청을 찾을 수 없습니다"로 자연스럽게 답할 수 있다.
    return json_payload(tool_result("get_saved_request", ok=True, row=row))


@tool(args_schema=SavedScheduleListInput)
def personal_list_saved_schedules(
    limit: int = 50,
    kind: RequestKind | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> str:
    """앱 DB에 저장된 일정 목록을 날짜/종류 필터로 반환합니다. Nana가 조회/수정/삭제 후보를 볼 때 사용합니다."""

    # "내 일정 보여줘"는 대부분 개인 일정을 묻는 것이므로 kind 기본값을 personal_schedule로 잡는다.
    # 사용자가 명시적으로 group_schedule을 요청하면 그 값을 쓴다.
    effective_kind = kind or "personal_schedule"

    store = _store()
    schedules = store.list_schedules(
        kind=effective_kind,
        date_from=date_from,
        date_to=date_to,
        limit=limit,
    )

    filters = {
        "kind": effective_kind,
        "date_from": date_from,
        "date_to": date_to,
        "limit": limit,
    }

    return json_payload(tool_result(
        "personal_list_saved_schedules",
        ok=True,
        filters=filters,
        schedules=schedules,
    ))


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

    store = app_store or _store()
    return _delete_saved_schedules(
        store=store,
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

    # None인 필드는 "수정하지 않음"을 의미한다.
    # 모든 필드가 None이면 사실상 변경사항이 없는데, 이 경우도 store에 위임해서 처리한다.
    updates = {k: v for k, v in {
        "title": title,
        "date": date,
        "start_time": start_time,
        "end_time": end_time,
        "attendees": attendees,
    }.items() if v is not None}

    store = _store()
    result = store.update_schedule(schedule_id=schedule_id, **updates)

    if not result:
        return json_payload(tool_result(
            "personal_update_saved_schedule",
            ok=False,
            error=f"schedule_id={schedule_id}를 찾을 수 없습니다.",
        ))

    return json_payload(tool_result(
        "personal_update_saved_schedule",
        ok=True,
        # store.update_schedule()이 실제로 수행한 공유 일정 동기화 결과를 그대로 반환한다.
        updated_schedule=result["schedule"],
        shared_sync=result["shared_sync"],
    ))


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

    store = _store()
    result = _delete_saved_schedules(
        store=store,
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
        personal_create_schedule if _tool_name(item) == "personal_create_schedule" else item
        for item in week01_tools()
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
        SQLITE_MEMORY_PROMPT,
        WEEK03_TOOL_CALL_PROMPT,
        f"""오늘은 {current_app_date_iso()}이다.
Week 3 tool 선택 기준:
- 저장 요청: extract_schedule_request → save_structured_request
- 조회 요청: personal_list_saved_schedules
- 수정 요청: personal_list_saved_schedules로 schedule_id 확인 → personal_update_saved_schedule
- 삭제 요청: personal_list_saved_schedules로 후보 확인 → personal_delete_saved_schedules
Week 3에서는 MCP, RAG, 외부 멤버 일정 조율, Supervisor/Sub-agent를 사용하지 않는다.""",
    ]


def build_week03_agent() -> object:
    """Week 1-3 누적 tool 목록을 노출하는 단일 LangChain agent를 만듭니다."""

    if not CONFIG.has_openai_key:
        raise RuntimeError("PROXY_TOKEN이 .env에 필요합니다.")
    global _WEEK03_AGENT
    if _WEEK03_AGENT is None:
        # Week 2와 달리 response_format을 쓰지 않는다.
        # Week 3는 tool 호출 결과가 주 반환값이므로 자유로운 tool calling 흐름을 유지한다.
        _WEEK03_AGENT = create_agent(
            model=chat_model(),
            tools=week03_tools(),
            system_prompt=week03_system_prompt(),
        )
    return _WEEK03_AGENT


def build_week_agent() -> object:
    """active-week registry가 호출하는 표준 Week agent builder입니다."""

    return build_week03_agent()