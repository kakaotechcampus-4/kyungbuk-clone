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
from student_parts.week04_retrieve_nanas_memory import (
    WEEK04_SEARCH_SOURCES,
    week04_prompt_parts,
    week04_tools,
)


_WEEK05_AGENT: Any | None = None


# [5주차 수강생 구현 가이드]
#
# 목표
#   외부 SQLite/MCP 서버에 있는 Kana의 이전 대화와 공유 일정을 LangChain agent가 사용할 수 있게 감쌉니다.
#   학생이 직접 SQL을 작성하는 주차가 아니라, MCP tool을 호출하고 그 결과를 agent용 JSON으로 전달하는
#   wrapper tool을 만드는 주차입니다.
#
# 과제 구성
#   - 메인과제: 외부 SQLite/MCP 서버의 이전 대화를 검색·로드하고 그 대화에서 일정을 추출하는
#     MCP wrapper 세로 슬라이스에 더해, 공유 일정 조회(list_shared_schedules)와
#     내 일정·외부 멤버 busy-time을 한 rows로 합치는 collect_member_schedules까지 완성합니다.
#     이 두 tool은 Week 6 Kana 하위 agent가 그대로 재사용하는 연결 지점이라 메인과제입니다.
#   - 추가 과제: 공유 일정 저장소에 row를 직접 등록·삭제하는 create_shared_schedule/delete_shared_schedule
#     wrapper를 확장합니다. 구현하지 않으려면 week05_tools() 목록에서 이 두 tool을 빼면 됩니다.
#
# 구현 위치와 사용할 코드
#   - 이 파일(student_parts/week05_load_kanas_past_conversations.py)의 @tool wrapper 함수들을 구현합니다.
#   - 실제 외부 SQLite/MCP tool 구현은 mcp_server/sqlite_mcp_server.py에 있으며, 학생은 이 파일을 직접 수정하지 않습니다.
#   - MCP 호출은 fixed/mcp_client.py의 call_local_mcp_tool_sync를 이 파일에서 별칭으로 둔
#     call_mcp_tool_sync(tool_name, args)를 사용합니다.
#   - load_conversation_messages는 fixed/external_mcp.py의 call_external_tool_payload(...)를 사용해
#     외부 tool payload를 dict로 받은 뒤 json_payload()로 감쌉니다.
#   - 멤버 이름/날짜 정규화와 요약은 fixed/external_people_store.py의
#     normalize_external_member_names(), normalize_external_schedule_date_bounds(),
#     external_schedule_summary()를 사용합니다.
#   - 내 일정 수집은 _personal_schedules_for_current_scope()에서 처리합니다. 이 helper는
#     fixed/app_store.py의 AppSQLiteStore(CONFIG.app_db_path).list_schedules(...)와
#     student_parts/week01_wake_up_nana.py의 PERSONAL_SCHEDULES 중 현재 대화 범위 row를 합칩니다.
#   - Week 3+ AppSQLiteStore는 개인/그룹 일정을 저장할 때 공유 일정 저장소에 자동 동기화할 수 있습니다.
#     list_shared_schedules wrapper(메인)는 공유 저장소 row를 직접 확인할 때,
#     create/delete_shared_schedule wrapper(추가)는 row를 직접 등록/삭제해 보정할 때 사용합니다.
#   - week05_tools()는 student_parts/week04_retrieve_nanas_memory.py의 week04_tools() 위에
#     Week 5 MCP wrapper tool들을 누적해 Week 5 단일 agent에 공개합니다.
#     추가 과제(create/delete_shared_schedule)를 구현하지 않으려면 week05_tools() 목록에서 해당 tool을 빼면 됩니다.
#
# 메인과제 구현 대상
#   1. search_previous_conversations
#      - query, member_names, limit를 받습니다.
#      - 이 파일의 call_mcp_tool_sync("search_previous_conversations", args)를 호출하고 결과 문자열을 그대로 반환합니다.
#      - 멤버 이름 정규화는 외부 SQLite store/MCP 경계에서 한 번만 처리하므로 wrapper에서 중복 변환하지 않습니다.
#
#   2. load_conversation_messages
#      - conversation_id로 외부 SQLite/MCP helper에서 이전 대화 메시지를 조회합니다.
#      - call_external_tool_payload("load_conversation_messages", {"conversation_id": conversation_id})를 사용합니다.
#      - 대화 메시지의 sender/content/created_at 순서가 보존되도록 결과를 가공하지 않습니다.
#
#   3. extract_schedules_from_history
#      - member_names, date_from, date_to를 받습니다.
#      - call_mcp_tool_sync("extract_schedules_from_history", args)를 호출합니다.
#      - 날짜 형식 정리는 외부 SQLite store/MCP 경계에서 한 번만 처리합니다.
#      - 결과 rows는 member_name/title/date/start_time/end_time/notes 필드를 유지해야 합니다.
#
#   4. list_shared_schedules
#      - call_mcp_tool_sync("list_shared_schedules", args)를 호출해 공유 일정 저장소 row를 조회합니다.
#      - 공유 저장소 자체를 확인할 때는 "나"를 포함한 등록 row를 조회합니다.
#      - 필터 없이 호출하면 외부 실습용 기본 공유 일정 row가 우선 반환될 수 있습니다.
#      - Week 6 Kana 하위 agent가 공유 저장소 row 조회에 그대로 사용하는 tool입니다.
#
#   5. collect_member_schedules
#      - 3주차 이후 저장된 내 일정은 앱 SQLite에서 읽고, 현재 대화의 임시 일정만 추가로 합칩니다.
#      - 외부 멤버 일정은 call_mcp_tool_sync("extract_schedules_from_history", args) 결과를 이 tool 안에서 읽습니다.
#      - 두 출처를 member_name/title/date/start_time/end_time/notes가 있는 rows 배열로 직접 합칩니다.
#      - schedule_summary도 함께 반환해 LLM이 바쁜 시간을 자연어로 설명할 수 있게 합니다.
#      - PERSONAL_SCHEDULES는 현재 대화 범위의 아직 DB에 없는 임시 일정만 합치고, SQLite에 이미 저장된 일정과 중복하지 않습니다.
#      - Week 6 추가 과제(find_common_available_slots)가 이 tool의 rows를 busy_rows 근거로 사용합니다.
#
# 추가 과제 구현 대상 (구현하지 않으려면 week05_tools() 목록에서 해당 tool을 제거)
#   1. create_shared_schedule / delete_shared_schedule
#      - 각각 call_mcp_tool_sync("create_shared_schedule" / "delete_shared_schedule", args)를 호출합니다.
#      - 공유 일정 저장소 row를 생성/삭제할 때 MCP tool 결과를 그대로 전달합니다.
#      - schedule_id 또는 source_conversation_id를 보존해야 나중에 수정/삭제 동기화가 가능합니다.
#
# 책임 경계
#   mcp_server/sqlite_mcp_server.py의 @mcp.tool 구현은 학생 구현 대상이 아닙니다.
#   이 파일의 wrapper tool은 직접 SQL이나 중복 정규화 helper를 두지 않고 store/MCP helper의 결과 JSON을 전달합니다.
#   week05_tools()는 Week 1-4 도구에 외부 SQLite/MCP 일정 도구를 누적합니다.
#   외부 멤버 busy-time 조회와 공유 저장소 row 조회는 Week 5 범위지만, 여러 사람의 최종 회의 시간 선택은 Week 6 범위입니다.
#
# 검증 방법
#   - 메인과제: ./run.sh --week5에서 외부 팀원 일정 조회 요청을 입력하고, trace에서
#     search_previous_conversations, load_conversation_messages, extract_schedules_from_history 중
#     어떤 tool이 어떤 순서로 호출됐는지 확인합니다.
#     collect_member_schedules 결과 rows에 "나"와 외부 멤버 일정이 같은 구조로 들어 있고,
#     list_shared_schedules 결과에 rows와 schedule_summary가 유지되는지 확인합니다.
#   - 추가 과제: create_shared_schedule로 등록한 row가 list_shared_schedules 조회에 나타나고
#     delete_shared_schedule로 삭제되는지 확인합니다.
#
# 함수별 동작 설명 ([메인]/[추가]/[공통]은 각 함수가 속한 과제 티어입니다)
#   - [메인] _schedule_scope(schedule)
#     Week 1 임시 일정이 어느 대화 범위에 속하는지 읽습니다. session_id가 없으면 기본 scope로 처리합니다.
#
#   - [메인] _personal_schedules_for_current_scope()
#     Week 3 이후 SQLite에 저장된 내 일정과 현재 대화에만 남아 있는 Week 1 임시 일정을 합칩니다.
#     이미 SQLite에 저장된 일정과 임시 일정이 중복되지 않도록 schedule_id/id를 기준으로 한 번 걸러냅니다.
#
#   - [공통] json_payload(payload)
#     외부 MCP 결과나 내부 helper 결과 dict를 한글이 보존되는 JSON 문자열로 바꿉니다.
#
#   - [메인] SearchPreviousConversationsInput / LoadConversationMessagesInput / ExtractSchedulesFromHistoryInput
#     외부 이전 대화 검색, 대화 메시지 로드, 외부 대화에서 일정 추출 tool의 입력 스키마입니다.
#
#   - [메인] ListSharedSchedulesInput / CollectMemberSchedulesInput
#     공유 일정 저장소 row 조회와, 내 일정·외부 멤버 busy-time을 같은 rows 배열로 합치는 tool의 입력 스키마입니다.
#
#   - [추가] CreateSharedScheduleInput / DeleteSharedScheduleInput
#     외부 공유 일정 저장소에 row를 생성, 삭제할 때 쓰는 입력 스키마입니다.
#
#   - [메인] _structured_request_from_schedule_row(row)
#     SQLite schedule row나 Week 1 임시 schedule row를 Week 2 StructuredRequest 모양으로 읽습니다.
#     뒤에서 내 일정 row를 외부 멤버 row와 같은 구조로 맞출 때 사용합니다.
#
#   - [메인] _collect_member_schedules(...)
#     내 일정과 외부 멤버 일정을 같은 member_name/title/date/start_time/end_time/notes row 구조로 합칩니다.
#     외부 멤버 이름과 날짜 범위는 fixed/external_people_store.py helper로 정규화합니다.
#
#   - [메인] search_previous_conversations(...)
#     외부 SQLite/MCP 서버에 저장된 과거 대화를 검색합니다. wrapper는 query/member_names/limit를 넘기고 결과 문자열을 그대로 반환합니다.
#
#   - [메인] load_conversation_messages(conversation_id)
#     검색으로 찾은 특정 외부 대화의 전체 메시지를 불러옵니다. sender/content/created_at 순서를 보존합니다.
#
#   - [메인] extract_schedules_from_history(...)
#     외부 멤버의 이전 대화에서 일정 또는 바쁜 시간 row를 추출합니다.
#
#   - [메인] list_shared_schedules(...)
#     공유 일정 저장소 row를 조회하는 MCP wrapper입니다. Week 6 Kana 하위 agent도 그대로 사용합니다.
#
#   - [메인] collect_member_schedules(...)
#     내 일정과 외부 멤버 busy-time을 한 번에 모으는 Week 5 핵심 tool입니다.
#     Week 6의 공통 가능 시간 결정 tool(추가 과제)이 이 rows를 busy_rows 근거로 사용합니다.
#
#   - [추가] create_shared_schedule(...) / delete_shared_schedule(...)
#     공유 일정 저장소에 row를 등록/삭제하는 MCP wrapper입니다. source_conversation_id와 schedule_id를 보존해 동기화 근거로 씁니다.
#
#   - [공통] week05_tools()
#     Week 4까지의 tool에 외부 대화/MCP/공유 일정 tool을 누적합니다.
#
#   - [공통] week05_system_prompt() / week05_prompt_parts()
#     개인 저장/RAG는 이전 주차 도구로, 외부 멤버 대화와 일정은 MCP wrapper로 처리하도록 agent 역할을 설명합니다.
#
#   - [공통] build_week05_agent() / build_week_agent()
#     Week 1~5 tool을 가진 agent를 한 번만 만들고 재사용합니다.


call_mcp_tool = call_local_mcp_tool
call_mcp_tool_sync = call_local_mcp_tool_sync
load_langchain_mcp_tools = load_local_mcp_tools
load_langchain_mcp_tools_sync = load_local_mcp_tools_sync


def _schedule_scope(schedule: dict[str, Any]) -> str:
    return str(schedule.get("session_id") or DEFAULT_SESSION_SCOPE)


def _personal_schedules_for_current_scope() -> list[dict[str, Any]]:
    """SQLite 저장 일정과 현재 대화의 임시 일정만 group 조율 후보로 사용합니다."""

    # Week 3+ 저장 일정은 앱 SQLite가 원본이다. 날짜 필터는 _collect_member_schedules가
    # 외부 멤버 일정과 같은 기준으로 한 번에 적용하므로 여기서는 넉넉히 읽기만 한다.
    saved = AppSQLiteStore(CONFIG.app_db_path).list_schedules(limit=200)
    saved_ids = {row.get("schedule_id") for row in saved if row.get("schedule_id")}
    scope = current_session_scope()
    # Week 1 임시 일정은 현재 대화 범위 것만 더한다. 다른 대화의 임시 일정이
    # 이 대화의 조율 후보로 새어 들어오지 않게 하기 위해서다.
    # personal_create_schedule 호환 경로는 임시 일정 id를 schedule_id로 그대로 써서
    # SQLite에도 저장하므로, 이미 저장된 임시 일정은 id 기준으로 한 번 걸러 중복을 막는다.
    extras = [
        schedule
        for schedule in PERSONAL_SCHEDULES
        if _schedule_scope(schedule) == scope and schedule.get("id") not in saved_ids
    ]
    return [*saved, *extras]


def json_payload(payload: dict[str, Any]) -> str:
    """도구 반환용 dict를 한글이 깨지지 않는 JSON 문자열로 변환합니다."""

    return json.dumps(payload, ensure_ascii=False)


def _call_mcp_or_soft_fail(tool_name: str, args: dict[str, Any]) -> str:
    """외부 MCP tool을 호출하고, 실패는 예외 대신 실패 payload로 반환합니다.

    MCP 명세의 tool execution error 방식(soft-fail): 프로토콜 예외로 터뜨리면
    agent 루프가 실행 자체를 멈추지만, isError성 payload로 돌려주면 모델이
    실패 내용을 읽고 스스로 재시도하거나 다른 tool로 우회할 수 있다.
    fixed/external_mcp.py의 sync_personal_schedule_to_shared가 쓰는 것과 같은 규칙이다.
    """

    try:
        return call_mcp_tool_sync(tool_name, args)
    except Exception as exc:
        return json_payload(
            {
                "ok": False,
                "tool_name": tool_name,
                "error": f"외부 MCP 호출에 실패했습니다: {type(exc).__name__}: {exc}",
            }
        )


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
    """앱 일정 row를 Week 2 StructuredRequest 기준으로 읽습니다."""

    return StructuredRequest(
        kind="personal_schedule",
        title=row.get("title"),
        date=row.get("date"),
        start_time=row.get("start_time"),
        end_time=row.get("end_time"),
        members=row.get("attendees") or row.get("members") or [],
        original_text=str(row.get("title") or ""),
    )


def _collect_member_schedules(
    *,
    member_names: list[str],
    date_from: str,
    date_to: str,
    personal_schedules: list[dict[str, Any]],
) -> dict[str, Any]:
    """내 일정과 외부 멤버 일정을 같은 row 구조로 합칩니다."""

    # 외부 저장소 규칙으로 멤버 이름(별칭 통일)과 날짜 범위(ISO datetime → 날짜)를 정규화한다.
    # "나"는 외부 멤버가 아니므로 외부 조회 명단에서는 뺀다.
    external_members = normalize_external_member_names(
        [name for name in member_names if str(name).strip() not in ("나", "")]
    )
    normalized_from, normalized_to = normalize_external_schedule_date_bounds(
        external_members, date_from, date_to
    )

    # 내 일정은 member_names에 "나"가 있을 때만 합친다. 앱 검증에서 "철수랑 하린 일정
    # 뽑아줘"에 요청하지 않은 내 일정까지 rows로 섞여 나오는 문제가 재현돼서,
    # 누구 일정을 모을지는 인자가 정하고 tool이 임의로 늘리지 않게 했다.
    include_me = any(str(name).strip() == "나" for name in member_names)

    # 내 일정을 외부 멤버 row와 같은 member_name/title/date/... 구조로 맞춘다.
    # Week 2 StructuredRequest를 기준 모양으로 쓰므로 SQLite row와 임시 row가 같은 형태가 된다.
    rows: list[dict[str, Any]] = []
    for schedule in personal_schedules if include_me else []:
        request = _structured_request_from_schedule_row(schedule)
        # 날짜가 조율 범위 밖이거나 아예 없는 일정은 busy-time 근거가 못 되므로 제외한다.
        if not request.date:
            continue
        if (normalized_from and request.date < normalized_from) or (
            normalized_to and request.date > normalized_to
        ):
            continue
        rows.append(
            {
                "member_name": "나",
                "title": request.title or "제목 없음",
                "date": request.date,
                "start_time": request.start_time,
                "end_time": request.end_time,
                "notes": "앱에 저장된 내 일정",
            }
        )

    # 외부 멤버 busy-time은 MCP tool 결과를 이 tool 안에서 직접 읽어 합친다.
    external_error: str | None = None
    if external_members:
        external_payload = json.loads(
            _call_mcp_or_soft_fail(
                "extract_schedules_from_history",
                {
                    "member_names": external_members,
                    "date_from": normalized_from,
                    "date_to": normalized_to,
                },
            )
        )
        rows.extend(external_payload.get("rows") or [])
        if not external_payload.get("ok", False):
            # 외부 조회가 실패해도 내 일정 rows는 유효하므로 실패 사실만 함께 알린다.
            external_error = external_payload.get("error")

    result: dict[str, Any] = {
        "member_names": [*( ["나"] if include_me else [] ), *external_members],
        "date_from": normalized_from,
        "date_to": normalized_to,
        "rows": rows,
        # 요약을 같이 주면 LLM이 바쁜 시간을 자연어로 설명하기 쉽다.
        "schedule_summary": external_schedule_summary(rows),
    }
    if external_error:
        result["external_error"] = external_error
    return result


@tool(args_schema=SearchPreviousConversationsInput)
def search_previous_conversations(
    query: str,
    member_names: list[str] | None = None,
    limit: int = 5,
) -> str:
    """외부 SQLite 데이터베이스에 저장된 다른 멤버들의 이전 대화를 검색합니다.

    다른 멤버(철수/하린 등)가 과거에 말한 내용을 찾을 때 사용합니다.
    나와 Nana가 나눈 대화는 search_conversation_messages가 담당합니다.
    LIKE 검색이므로 query에는 핵심 명사 하나만 넣습니다(여러 단어를 이으면 0건이 되기 쉽습니다).
    특정 멤버의 대화 목록이 목적이면 query를 빈 문자열로 두고 member_names만 넣습니다.
    대화 원문이 필요하면 이 검색 rows의 conversation_id로 load_conversation_messages를 호출합니다.
    """

    # 멤버 이름 정규화는 외부 store/MCP 경계에서 한 번만 처리한다. wrapper는 인자를 그대로 전달한다.
    result_text = _call_mcp_or_soft_fail(
        "search_previous_conversations",
        {"query": query, "member_names": member_names, "limit": limit},
    )
    # 앱 검증에서 재현된 실패: 다단어 query가 LIKE에서 0건 → 그대로 "없다"로 종료.
    # Week 4의 coverage와 같은 방식으로, 0건 결과 안에 교정 방법을 데이터로 실어 보낸다.
    payload = json.loads(result_text)
    if payload.get("ok") and not payload.get("rows"):
        payload["retry_hint"] = (
            "LIKE 검색이라 여러 단어 query는 0건이 되기 쉽습니다. 핵심 단어 하나로 줄이거나, "
            "query를 빈 문자열로 두고 member_names만으로 다시 검색하세요. "
            "대화 원문이 목적이면 재검색 rows의 conversation_id로 load_conversation_messages를 호출하세요."
        )
        return json_payload(payload)
    return result_text


@tool(args_schema=LoadConversationMessagesInput)
def load_conversation_messages(conversation_id: str) -> str:
    """외부 SQLite 데이터베이스에서 특정 이전 대화의 모든 메시지를 불러옵니다."""

    try:
        # dict payload helper를 쓰고, sender/content/created_at 순서가 보존되도록 결과를 가공하지 않는다.
        payload = call_external_tool_payload(
            "load_conversation_messages", {"conversation_id": conversation_id}
        )
    except Exception as exc:
        return json_payload(
            {
                "ok": False,
                "tool_name": "load_conversation_messages",
                "error": f"외부 MCP 호출에 실패했습니다: {type(exc).__name__}: {exc}",
            }
        )
    return json_payload(payload)


@tool(args_schema=ExtractSchedulesFromHistoryInput)
def extract_schedules_from_history(member_names: list[str], date_from: str, date_to: str) -> str:
    """외부 SQLite 이전 대화에서 다른 멤버들의 일정만 추출합니다.

    내 일정은 포함되지 않습니다. 나를 포함해 여러 사람의 일정을 한 번에 모아야
    하면 collect_member_schedules를 사용합니다.
    """

    # 날짜 형식 정리는 외부 store/MCP 경계에서 한 번만 처리한다.
    return _call_mcp_or_soft_fail(
        "extract_schedules_from_history",
        {"member_names": member_names, "date_from": date_from, "date_to": date_to},
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

    # schedule_id/source_conversation_id를 보존해 넘겨야 나중에 같은 복사본을 수정/삭제할 수 있다.
    return _call_mcp_or_soft_fail(
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

    return _call_mcp_or_soft_fail(
        "delete_shared_schedule",
        {"schedule_id": schedule_id, "source_conversation_id": source_conversation_id},
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

    return _call_mcp_or_soft_fail(
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
    """member_names에 적힌 사람들의 일정을 한 번에 모읍니다.

    내 일정은 member_names에 '나'가 있을 때만 포함됩니다. 다른 멤버들만의
    일정이 목적이면 extract_schedules_from_history를 사용합니다.
    """

    # 합치는 규칙은 helper 한 곳에 두고, tool은 검증된 인자와 내 일정 목록을 넘기는 입구 역할만 한다.
    result = _collect_member_schedules(
        member_names=member_names,
        date_from=date_from,
        date_to=date_to,
        personal_schedules=_personal_schedules_for_current_scope(),
    )
    return json_payload({"ok": True, "tool_name": "collect_member_schedules", **result})


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


# Week 5 라우팅 검증에서 재현된 실패: "하린이 전에 뭐 있다고 했지?" 같은 남의 대화 질문이
# 내 기억 tool로 가서 0건 후 종료됐다. 출처 커버리지 표가 내 기억 3개만 알아서,
# 0건 안내(other_sources)가 내 기억 안에서만 맴돌게 유도했기 때문이다.
# Week 5의 외부 검색 출처를 표에 등록해 coverage 안내가 외부 tool까지 가리키게 한다.
# 이 등록은 week5 모듈이 import될 때(=Week 5 agent가 활성일 때)만 일어나므로
# Week 4 단독 실행의 동작은 바뀌지 않는다.
WEEK04_SEARCH_SOURCES["search_previous_conversations"] = {
    "what": "외부 MCP 저장소에 있는 다른 멤버들의 이전 대화",
    "when": "다른 멤버(철수/하린 등)가 과거에 말한 내용이나 멤버들의 대화 기록을 찾을 때",
}
WEEK04_SEARCH_SOURCES["extract_schedules_from_history"] = {
    "what": "외부 MCP 저장소에 있는 다른 멤버들의 일정(busy-time)",
    "when": "다른 멤버들의 일정이나 바쁜 시간을 확인할 때",
}

# Week 4까지의 검색은 전부 "내" 기억이었다. Week 5부터는 다른 멤버들의 기록이 외부
# MCP 저장소에 있으므로, 질문의 주어(나 vs 다른 멤버)로 tool 계열을 가르는 규칙을 준다.
WEEK05_EXTERNAL_SOURCE_PROMPT = (
    "Week 5부터 다른 멤버들의 이전 대화와 일정은 내 기억이 아니라 외부 MCP 저장소에 있다. "
    "내 취향/기록/지난 대화는 Week 4까지의 검색 tool로 찾고, "
    "다른 멤버의 과거 대화나 일정은 외부 MCP tool로 찾는다. "
    "다른 멤버가 과거에 무엇을 말했는지는 search_previous_conversations로 대화를 찾고, "
    "사용자가 대화 원문을 요청하면 검색 요약으로 대신하지 말고 반드시 검색 rows의 "
    "conversation_id로 load_conversation_messages를 호출해 원문으로 답한다. "
    "search_previous_conversations의 query에는 핵심 단어 하나만 넣고, "
    "특정 멤버의 대화 목록이 목적이면 query를 빈 문자열로 두고 member_names만 넣는다. "
    "다른 멤버들만의 일정은 extract_schedules_from_history로 추출하고, "
    "내 일정까지 함께 모아야 할 때만 collect_member_schedules에 '나'를 포함한 member_names로 요청한다. "
    "공유 일정 저장소에 등록된 row 자체를 확인할 때는 list_shared_schedules를 사용한다. "
    "외부 tool 결과의 rows와 schedule_summary를 근거로만 답하고, "
    "외부 기록에 없는 멤버 일정을 지어내지 않는다. "
    "여러 사람의 최종 회의 시간을 고르는 조율은 아직 Week 5 범위가 아니다."
)


def week05_prompt_parts() -> list[str]:
    """1~5주차 system prompt 조각을 누적합니다."""

    return [
        *week04_prompt_parts(),
        WEEK05_EXTERNAL_SOURCE_PROMPT,
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
