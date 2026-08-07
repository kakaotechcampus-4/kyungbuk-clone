from __future__ import annotations

import json
from typing import Any

from langchain.agents import create_agent
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from fixed.app_store import AppSQLiteStore
from fixed.config import CONFIG
from fixed.external_mcp import call_external_tool_payload
from fixed.external_people_store import (
    external_schedule_summary,
    normalize_external_member_names,
    normalize_external_schedule_date_bounds,
    strip_parenthetical_text,
)
from fixed.llm import chat_model
from fixed.mcp_client import (
    call_local_mcp_tool,
    call_local_mcp_tool_sync,
    load_local_mcp_tools,
    load_local_mcp_tools_sync,
)
from fixed.runtime_clock import current_app_date_iso
from fixed.session_scope import DEFAULT_SESSION_SCOPE, current_session_scope
from student_parts.week01_wake_up_nana import PERSONAL_SCHEDULES, join_system_prompt
from student_parts.week02_structure_natural_language_requests import StructuredRequest
from student_parts.week03_build_nanas_logbook import tool_result
from student_parts.week04_retrieve_nanas_memory import week04_prompt_parts, week04_tools


_WEEK05_AGENT: Any | None = None

call_mcp_tool = call_local_mcp_tool
call_mcp_tool_sync = call_local_mcp_tool_sync
load_langchain_mcp_tools = load_local_mcp_tools
load_langchain_mcp_tools_sync = load_local_mcp_tools_sync

def _schedule_scope(schedule: dict[str, Any]) -> str:
    return str(schedule.get("session_id") or DEFAULT_SESSION_SCOPE)


def _personal_schedules_for_current_scope() -> list[dict[str, Any]]:
    """SQLite 저장 일정과 현재 대화의 임시 일정만 group 조율 후보로 사용합니다."""

    saved_schedules = AppSQLiteStore(CONFIG.app_db_path).list_schedules(limit=200)
    saved_schedule_ids = {
        str(schedule.get("schedule_id"))
        for schedule in saved_schedules
        if schedule.get("schedule_id")
    }

    session_id = current_session_scope()
    pending_schedules = [
        schedule
        for schedule in PERSONAL_SCHEDULES
        if _schedule_scope(schedule) == session_id
        and str(schedule.get("id") or schedule.get("schedule_id") or "") not in saved_schedule_ids
    ]

    return [*saved_schedules, *pending_schedules]


def json_payload(payload: dict[str, Any]) -> str:
    """도구 반환용 dict를 한글이 깨지지 않는 JSON 문자열로 변환합니다."""

    return json.dumps(payload, ensure_ascii=False)


class SearchPreviousConversationsInput(BaseModel):
    """외부 이전 대화 검색 입력입니다."""

    query: str
    member_names: list[str] | None = None
    limit: int = Field(default=5, ge=1, le=50)


class LoadConversationMessagesInput(BaseModel):
    """외부 대화 메시지 조회 입력입니다."""

    conversation_id: str


class ExtractSchedulesFromHistoryInput(BaseModel):
    """외부 멤버 일정 추출 입력입니다."""

    member_names: list[str]
    date_from: str
    date_to: str


class CreateSharedScheduleInput(BaseModel):
    """공유 일정 생성 입력입니다."""

    member_name: str
    title: str
    date: str
    start_time: str
    end_time: str = "미정"
    notes: str | None = None
    source_conversation_id: str | None = None
    schedule_id: str | None = None


class DeleteSharedScheduleInput(BaseModel):
    """공유 일정 삭제 입력입니다."""

    schedule_id: str | None = None
    source_conversation_id: str | None = None


class ListSharedSchedulesInput(BaseModel):
    """공유 일정 조회 입력입니다."""

    member_names: list[str] | None = None
    date_from: str | None = None
    date_to: str | None = None
    source_conversation_id: str | None = None
    limit: int = Field(default=50, ge=1, le=200)


class CollectMemberSchedulesInput(BaseModel):
    """내 일정과 외부 멤버 busy-time 수집 입력입니다."""

    member_names: list[str]
    date_from: str
    date_to: str


def _structured_request_from_schedule_row(row: dict[str, Any]) -> StructuredRequest:
    """앱 일정 row를 Week 2 StructuredRequest 기준으로 읽습니다.

    SQLite row는 `request_kind`로 개인/그룹을 구분합니다. Week 1 임시 일정 row에는
    이 값이 없으므로 개인 일정으로 봅니다.
    """

    return StructuredRequest(
        kind="group_schedule" if row.get("request_kind") == "group_schedule" else "personal_schedule",
        title=row.get("title"),
        date=row.get("date"),
        start_time=row.get("start_time"),
        end_time=row.get("end_time"),
        members=row.get("attendees") or row.get("members") or [],
        original_text=str(row.get("title") or ""),
    )


def _my_schedule_notes(request: StructuredRequest) -> str:
    """내 일정 row가 개인 일정인지, 참석자가 있는 그룹 일정인지 설명합니다."""

    if request.kind != "group_schedule":
        return "Nana 개인 일정"
    members = [str(member).strip() for member in (request.members or []) if str(member).strip()]
    return f"Nana 그룹 일정 · 참석자: {', '.join(members)}" if members else "Nana 그룹 일정"


def _dedupe_schedule_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """같은 일정이 앱 DB와 공유 저장소 양쪽에서 들어와도 한 번만 남깁니다.

    앱 DB에 저장된 내 일정은 공유 저장소에도 자동 동기화되므로, member_names에 "나"가
    들어온 호출에서는 같은 일정이 두 경로로 들어옵니다. 앞에 오는 앱 DB row를 남깁니다.

    두 경로가 같은 일정을 서로 다르게 다듬기 때문에 값을 그대로 비교하면 안 됩니다.
      - 공유 저장소는 제목에서 소괄호를 지우고 공백을 하나로 줄입니다. 앱 DB는 원문을 둡니다.
      - 앱 DB 경로만 end_time "미정"을 "18:00"으로 바꿉니다. 그래서 end_time은 키에서 뺍니다.
        같은 사람이 같은 날 같은 시각에 시작하는 같은 제목의 일정은 하나로 봅니다.
      - start_time이 비어 있으면 공유 저장소는 "미정"으로 저장하므로 같은 값으로 맞춥니다.
    """

    deduped: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row.get("member_name") or "").strip(),
            str(row.get("date") or "").strip(),
            str(row.get("start_time") or "").strip() or "미정",
            strip_parenthetical_text(str(row.get("title") or "")),
        )
        deduped.setdefault(key, row)
    return list(deduped.values())


def _collect_member_schedules(
    *,
    member_names: list[str],
    date_from: str,
    date_to: str,
    personal_schedules: list[dict[str, Any]],
) -> dict[str, Any]:
    """내 일정과 외부 멤버 일정을 같은 row 구조로 합칩니다."""

    normalized_member_names = normalize_external_member_names(member_names)
    normalized_date_from, normalized_date_to = normalize_external_schedule_date_bounds(
        member_names,
        date_from,
        date_to,
    )

    my_rows: list[dict[str, Any]] = []
    for schedule in personal_schedules:
        request = _structured_request_from_schedule_row(schedule)
        schedule_date = str(request.date or "")
        if normalized_date_from and schedule_date < normalized_date_from:
            continue
        if normalized_date_to and schedule_date > normalized_date_to:
            continue
        end_time = request.end_time or "미정"
        base_notes = schedule.get("notes") or _my_schedule_notes(request)
        notes = (
            f"{base_notes} (종료 시간 미정 - 확정 전까지는 해당 날짜 전체가 예약된 것으로 간주)"
            if end_time == "미정"
            else base_notes
        )

        my_rows.append(
            {
                "member_name": "나",
                "title": request.title or "제목 없음",
                "date": schedule_date,
                "start_time": request.start_time or "미정",
                "end_time": end_time,
                "notes": notes,
            }
        )

    external_member_names = [name for name in normalized_member_names if name != "나"]
    external_rows: list[dict[str, Any]] = []
    if external_member_names:
        payload = json.loads(
            call_mcp_tool_sync(
                "extract_schedules_from_history",
                {
                    "member_names": external_member_names,
                    "date_from": normalized_date_from,
                    "date_to": normalized_date_to,
                },
            )
        )
        external_rows = payload.get("rows") or []

    # my_rows를 external_rows보다 먼저 두어야 중복이 걸러질 때 앱 DB row의 notes가 남습니다.
    rows = _dedupe_schedule_rows([*my_rows, *external_rows])
    my_rows = [row for row in rows if row.get("member_name") == "나"]
    external_rows = [row for row in rows if row.get("member_name") != "나"]

    return {
        "member_names": normalized_member_names,
        "date_from": normalized_date_from,
        "date_to": normalized_date_to,
        "rows": rows,
        "my_schedule_count": len(my_rows),
        "external_schedule_count": len(external_rows),
        "schedule_summary": external_schedule_summary(rows),
    }


@tool(args_schema=SearchPreviousConversationsInput)
def search_previous_conversations(
    query: str,
    member_names: list[str] | None = None,
    limit: int = 5,
) -> str:
    """외부 SQLite 데이터베이스에 저장된 이전 대화를 검색합니다. query에는 LLM이 고른 짧은 핵심 명사나 구를 넣습니다."""

    return call_mcp_tool_sync(
        "search_previous_conversations",
        {
            "query": query,
            "member_names": member_names,
            "limit": limit,
        },
    )


@tool(args_schema=LoadConversationMessagesInput)
def load_conversation_messages(conversation_id: str) -> str:
    """외부 SQLite 데이터베이스에서 특정 이전 대화의 모든 메시지를 불러옵니다."""

    payload = call_external_tool_payload(
        "load_conversation_messages",
        {"conversation_id": conversation_id},
    )
    return json_payload(payload)


@tool(args_schema=ExtractSchedulesFromHistoryInput)
def extract_schedules_from_history(member_names: list[str], date_from: str, date_to: str) -> str:
    """외부 SQLite 이전 대화에서 멤버별 일정을 추출합니다."""

   
    return call_mcp_tool_sync(
        "extract_schedules_from_history",
        {
            "member_names": member_names,
            "date_from": date_from,
            "date_to": date_to,
        },
    )


@tool(args_schema=CreateSharedScheduleInput)
def create_shared_schedule(
    member_name: str,
    title: str,
    date: str,
    start_time: str,
    end_time: str = "미정",
    notes: str | None = None,
    source_conversation_id: str | None = None,
    schedule_id: str | None = None,
) -> str:
    """외부 MCP 공유 일정 저장소에 일정을 등록하거나 갱신합니다."""

    return call_mcp_tool_sync(
        "create_shared_schedule",
        {
            "member_name": member_name,
            "title": title,
            "date": date,
            "start_time": start_time,
            "end_time": end_time,
            "notes": notes,
            "source_conversation_id": source_conversation_id,
            "schedule_id": schedule_id,
        },
    )


@tool(args_schema=DeleteSharedScheduleInput)
def delete_shared_schedule(
    schedule_id: str | None = None,
    source_conversation_id: str | None = None,
) -> str:
    """외부 MCP 공유 일정 저장소에서 일정을 삭제합니다."""
    return call_mcp_tool_sync(
        "delete_shared_schedule",
        {
            "schedule_id": schedule_id,
            "source_conversation_id": source_conversation_id,
        },
    )


@tool(args_schema=ListSharedSchedulesInput)
def list_shared_schedules(
    member_names: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    source_conversation_id: str | None = None,
    limit: int = 50,
) -> str:
    """외부 MCP 공유 일정 저장소에 등록된 일정을 조회합니다. 필터가 없으면 기본 공유 일정을 반환합니다."""

    return call_mcp_tool_sync(
        "list_shared_schedules",
        {
            "member_names": member_names,
            "date_from": date_from,
            "date_to": date_to,
            "source_conversation_id": source_conversation_id,
            "limit": limit,
        },
    )


@tool(args_schema=CollectMemberSchedulesInput)
def collect_member_schedules(member_names: list[str], date_from: str, date_to: str) -> str:
    """내 일정과 다른 사람들의 일정을 MCP SQLite 기록에서 모읍니다."""

    payload = _collect_member_schedules(
        member_names=member_names,
        date_from=date_from,
        date_to=date_to,
        personal_schedules=_personal_schedules_for_current_scope(),
    )
    return json_payload(tool_result("collect_member_schedules", ok=True, **payload))


def week05_tools() -> list[Any]:
    """4주차까지의 도구에 외부 SQLite/MCP 일정 도구를 누적한 목록입니다."""

    return [
        *week04_tools(),
        search_previous_conversations,
        load_conversation_messages,
        extract_schedules_from_history,
        create_shared_schedule,
        delete_shared_schedule,
        list_shared_schedules,
        collect_member_schedules,
    ]


def week05_system_prompt() -> str:
    """5주차 단일 agent가 따르는 시스템 프롬프트입니다."""

    return join_system_prompt(week05_prompt_parts())


def week05_prompt_parts() -> list[str]:
    """1~5주차 system prompt 조각을 누적합니다."""

    return [
        *week04_prompt_parts(),
        (
            "[Week 5 외부 대화·공유 일정 MCP]\n"
            f"오늘 날짜는 {current_app_date_iso()}이며, tool에 넘기는 날짜는 항상 YYYY-MM-DD로 계산한다.\n"
            "내 개인 참고자료, 내 저장 일정, 앱 대화 기록은 Week 1-4 tool로 처리한다. "
            "외부 멤버(예: 철수, 영희)의 대화와 일정은 DB를 직접 읽지 않고 MCP wrapper tool로만 조회한다.\n"
            "외부 멤버의 이전 대화나 그 대화 속 일정을 물으면 search_previous_conversations를 먼저 호출해 "
            "관련 대화와 conversation_id를 확인한 뒤, extract_schedules_from_history로 일정 rows를 가져온다.\n"
            "search_previous_conversations를 호출할 때 사람 이름은 member_names에 넣고, query에는 조사를 뗀 "
            "한두 단어짜리 대화 주제 핵심어(예: 일정, 회의, 인터뷰)만 넣는다. 사람 이름이나 "
            "'전체 메시지', '원문'처럼 요청 방식을 가리키는 말은 query에 넣지 않는다.\n"
            "대화 원문이나 전체 메시지를 보여 달라는 요청이면 검색으로 찾은 conversation_id로 "
            "load_conversation_messages를 호출해 rows의 sender/content/created_at 순서를 그대로 근거로 삼는다.\n"
            "예: '철수 대화 원문 전체를 보여줘' 요청은 "
            "search_previous_conversations(query='일정', member_names=['철수'])로 conversation_id를 얻은 뒤 "
            "load_conversation_messages(conversation_id=그 값)를 호출한다.\n"
            "날짜 범위와 멤버가 이미 분명한 일정 조회는 extract_schedules_from_history를 member_names/date_from/date_to와 함께 호출한다.\n"
            "나와 다른 사람의 일정을 함께 비교해야 하면 collect_member_schedules를 호출한다. "
            "member_names에는 확인할 사람 이름을, date_from/date_to에는 확인할 날짜 범위를 넣는다.\n"
            "공유 일정 저장소에 등록된 row 자체를 확인해야 하면 list_shared_schedules를 호출하고, "
            "내 일정이 공유 저장소에 동기화됐는지 확인할 때는 member_names에 '나'를 넣어 호출한다. "
            "공유 일정 row를 직접 등록하거나 삭제해야 하면 create_shared_schedule / delete_shared_schedule을 호출한다.\n"
            "일정을 추측하지 않고 tool 결과의 rows와 schedule_summary만 근거로 답하며, "
            "rows가 비어 있으면 기록을 찾지 못했다고 답하고 어떤 핵심어로 검색했는지 함께 알려 준다.\n"
            "여러 사람의 최종 회의 시간 하나를 확정하는 일은 Week 6 범위이므로, 이번 주에는 모은 일정 근거를 정리해 알려 준다."
        ),
    ]


def build_week05_agent() -> object:
    """Week 1-5 누적 tool 목록을 노출하는 단일 LangChain agent를 만듭니다."""

    if not CONFIG.has_openai_key:
        raise RuntimeError("PROXY_TOKEN이 .env에 필요합니다.")
    global _WEEK05_AGENT
    if _WEEK05_AGENT is None:
        _WEEK05_AGENT = create_agent(
            model=chat_model(),
            tools=week05_tools(),
            system_prompt=week05_system_prompt(),
        )
    return _WEEK05_AGENT


def build_week_agent() -> object:
    """active-week registry가 호출하는 표준 Week agent builder입니다."""

    return build_week05_agent()
