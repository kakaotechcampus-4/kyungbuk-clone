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
    PERSONAL_SHARED_MEMBER_NAME,
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
from student_parts.week04_retrieve_nanas_memory import week04_prompt_parts, week04_tools


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


def _personal_schedules_for_current_scope(date_from: str, date_to: str) -> list[dict[str, Any]]:
    """SQLite 저장 일정과 현재 대화의 임시 일정만 group 조율 후보로 사용합니다."""

    date_start, date_end = normalize_external_schedule_date_bounds(None, date_from, date_to)

    store = AppSQLiteStore(CONFIG.app_db_path)
    db_schedules = store.list_schedules(date_from=date_start, date_to=date_end, limit=200)

    saved_schedule_ids = {row["schedule_id"] for row in db_schedules}
    for s in PERSONAL_SCHEDULES:
        if _schedule_scope(s) == current_session_scope():
            if s.get("id") not in saved_schedule_ids:
                db_schedules.append(s)
    return db_schedules


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


def _has_time(value: Any) -> bool:
    """'14:00'처럼 실제 시각이 있으면 True, None/""/"미정"이면 False입니다.

    앱 일정은 결측을 None으로, 외부 일정은 "미정"으로 표현하므로 둘을 같은 기준으로 봅니다.
    fixed/schedule_decision.py의 parse_time_minutes도 같은 두 값을 결측으로 취급합니다.
    """

    return bool(value) and value != "미정"

def _collect_member_schedules(
    *,
    member_names: list[str],
    date_from: str,
    date_to: str,
    personal_schedules: list[dict[str, Any]],
) -> dict[str, Any]:
    """내 일정과 외부 멤버 일정을 같은 row 구조로 합칩니다."""

    normalized_names = normalize_external_member_names(member_names)

    seen = set()
    unique_member_names = []
    for n in normalized_names:
        if n not in seen:
            seen.add(n)
            unique_member_names.append(n)

    external_member_names = [n for n in unique_member_names if n != PERSONAL_SHARED_MEMBER_NAME]
    date_start, date_end = normalize_external_schedule_date_bounds(member_names, date_from, date_to)

    external_rows = []
    lookup_error = None

    try:
        history_result = call_mcp_tool_sync("extract_schedules_from_history", {
            "member_names": external_member_names,
            "date_from": date_start,
            "date_to": date_end,
        })
        history_payload = json.loads(history_result)
        if not isinstance(history_payload, dict):
            lookup_error = {"reason": "외부 응답이 dict가 아닙니다."}
        elif not history_payload.get("ok"):
            lookup_error = {"reason": history_payload.get("error") or "외부 일정 조회에 실패했습니다."}
        else:
            rows = history_payload.get("rows")
            if isinstance(rows, list):
                collected_rows = []
                for r in rows:
                    new_row = dict(r)
                    new_row["time_complete"] = (
                        _has_time(new_row.get("start_time")) and _has_time(new_row.get("end_time"))
                    )
                    collected_rows.append(new_row)
                external_rows = collected_rows
            else:
                lookup_error = {"reason": "외부 응답의 rows가 list가 아닙니다."}
    except Exception as exc:

        lookup_error = {
            "reason": f"외부 일정 조회 중 오류가 발생했습니다: {exc}",
            "error_type": type(exc).__name__,
        }

    my_rows = []
    for schedule in personal_schedules:
        sr = _structured_request_from_schedule_row(schedule)
        if sr.date is not None:
            if date_start and sr.date < date_start:
                continue
            if date_end and sr.date > date_end:
                continue
        my_rows.append({
            "member_name": PERSONAL_SHARED_MEMBER_NAME, "title": sr.title, "date": sr.date,
            "start_time": sr.start_time, "end_time": sr.end_time, "notes": _my_schedule_notes(sr),
            "time_complete": _has_time(sr.start_time) and _has_time(sr.end_time),
        })

    all_rows = _dedupe_schedule_rows(my_rows + external_rows)
    summary = external_schedule_summary(all_rows)

    if any(not row["time_complete"] for row in all_rows):
        summary = "시간 정보가 불완전한 일정(종료 시간 미정 등)이 있습니다. " \
                  "이 일정은 하루 끝까지 바쁜 것으로 계산되므로, 공통 시간을 단정하기 전에 " \
                  "사용자에게 정확한 시간을 확인하시오.\n" + summary

    if lookup_error is not None:
        summary = "외부 멤버 일정 조회에 실패했습니다. 아래 목록에는 외부 멤버 일정이 빠져 있으므로, " \
                  "외부 멤버에게 일정이 없다거나 한가하다고 답하지 마시오.\n" + summary

    return {
        "rows": all_rows,
        "schedule_summary": summary,
        "external_lookup": {"ok": lookup_error is None, "error": lookup_error},
    }


@tool(args_schema=SearchPreviousConversationsInput)
def search_previous_conversations(
    query: str,
    member_names: list[str] | None = None,
    limit: int = 5,
) -> str:
    """외부 SQLite 데이터베이스에 저장된 이전 대화를 검색합니다.

    대화 내용 자체를 찾는 요청("무슨 얘기 했었지?", "예전에 뭐라고 했어?")에 사용합니다.
    query에는 LLM이 고른 짧은 핵심 명사나 구를 넣습니다.
    찾은 대화의 전체 메시지가 필요하면 load_conversation_messages를 이어서 호출합니다.
    멤버의 일정을 묻는 요청이라면 이 tool 결과만으로 답을 끝내지 말고
    extract_schedules_from_history까지 호출해야 합니다.
    """

    return call_mcp_tool_sync("search_previous_conversations", {"query": query, "member_names": member_names, "limit": limit})


@tool(args_schema=LoadConversationMessagesInput)
def load_conversation_messages(conversation_id: str) -> str:
    """외부 SQLite 데이터베이스에서 특정 이전 대화의 모든 메시지를 불러옵니다.

    search_previous_conversations로 찾은 conversation_id의 대화 전문을 확인할 때 사용합니다.
    sender/content/created_at 순서가 그대로 유지됩니다.
    """

    payload = call_external_tool_payload("load_conversation_messages", {"conversation_id": conversation_id})
    return json_payload(payload)


@tool(args_schema=ExtractSchedulesFromHistoryInput)
def extract_schedules_from_history(member_names: list[str], date_from: str, date_to: str) -> str:
    """외부 SQLite 이전 대화에서 멤버별 일정을 추출합니다.

    특정 멤버의 일정 자체를 알려달라는 요청(그 사람이 언제 바쁜지만 알려주면 되는 경우)에 사용합니다.
    "철수 일정 알려줘", "민준이 언제 바빠?"처럼 사람 이름과 일정을 함께 묻고 앱 저장을 언급하지 않은
    요청이 여기에 해당합니다. 이런 요청을 앱 저장 일정 조회(personal_list_saved_schedules)로 처리하면
    엉뚱한 저장소를 보고 "없습니다"라고 답하게 됩니다.
    "이전 대화에서 일정을 찾아줘"처럼 대화를 근거로 말하더라도 일정을 묻는 것이므로 이 tool을 호출합니다.
    이 tool을 호출하지 않은 채 "일정을 찾을 수 없다"고 답해서는 안 됩니다.
    나와 함께 가능한 시간을 찾는 요청이라면 이 tool 대신 collect_member_schedules를 사용합니다.

    사용자가 날짜를 말하지 않았다면 date_from/date_to를 오늘 하루나 이번 달로 좁히지 마십시오.
    외부 기록은 지난 대화에서 추출한 것이라 대부분 과거 날짜에 있고, 오늘이나 이번 달에는
    아무 것도 없을 수 있습니다. 날짜 언급이 없으면 최근 두세 달 전부터 다음 달까지처럼
    과거를 포함한 넉넉한 범위를 넣고, 그 결과가 0건이면 범위를 더 넓혀 한 번 더 조회한 뒤에
    답하십시오. 좁은 범위 한 번만 조회하고 "일정이 없다"고 답하면 실제로 있는 일정을
    없다고 말하게 됩니다.
    조회 결과가 비어 있을 때는 어떤 날짜 범위를 조회했는지 반드시 답변에 밝히십시오.
    """

    return call_mcp_tool_sync("extract_schedules_from_history", {"member_names": member_names, "date_from": date_from, "date_to": date_to})


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

    # TODO: call_mcp_tool_sync("create_shared_schedule", args)로 공유 일정 row를 생성/갱신하세요.
    ...


@tool(args_schema=DeleteSharedScheduleInput)
def delete_shared_schedule(
    schedule_id: str | None = None,
    source_conversation_id: str | None = None,
) -> str:
    """외부 MCP 공유 일정 저장소에서 일정을 삭제합니다."""

    # TODO: call_mcp_tool_sync("delete_shared_schedule", args)로 공유 일정을 삭제하세요.
    ...


@tool(args_schema=ListSharedSchedulesInput)
def list_shared_schedules(
    member_names: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    source_conversation_id: str | None = None,
    limit: int = 50,
) -> str:
    """외부 MCP 공유 일정 저장소에 등록된 일정을 조회합니다.

    공유 일정 저장소를 확인하는 요청("공유 일정 보여줘", "공유 저장소에 뭐가 등록돼 있어?")은
    이 tool로 지원되므로 거절하지 말고 호출한 뒤 rows와 schedule_summary를 근거로 답합니다.
    필터가 없으면 외부 실습용 기본 공유 일정을 반환합니다.
    """

    return call_mcp_tool_sync("list_shared_schedules", {"member_names": member_names, "date_from": date_from, "date_to": date_to, "source_conversation_id": source_conversation_id, "limit": limit})


@tool(args_schema=CollectMemberSchedulesInput)
def collect_member_schedules(member_names: list[str], date_from: str, date_to: str) -> str:
    """내 일정과 다른 사람들의 일정을 하나의 rows로 모읍니다.

    나와 한 명 이상의 외부 멤버가 함께 가능한 시간을 찾는 요청(약속·미팅 조율)에 사용합니다.
    언급된 외부 멤버가 한 명뿐이어도 조율 대상에는 항상 "나"가 포함되므로 이 tool을 사용하고,
    내 일정을 빼고 답해서는 안 됩니다. 특정 멤버의 일정만 알려주면 되는 요청이라면
    대신 extract_schedules_from_history를 사용합니다.

    사용자가 날짜를 말하지 않았다면 date_from/date_to를 오늘 하루나 이번 달로 좁히지 말고,
    최근 두세 달 전부터 다음 달까지처럼 과거를 포함한 넉넉한 범위를 넣으십시오.
    결과가 0건이면 범위를 넓혀 다시 조회하고, 결과가 비어 있을 때는 어떤 날짜 범위를
    조회했는지 반드시 답변에 밝히십시오.

    반환값 해석:
    - schedule_summary 맨 앞에 경고 문장이 붙어 있으면 그 지시를 우선 따릅니다.
    - external_lookup.ok가 false이면 외부 조회가 실패한 것이므로, rows에 그 멤버가 없다는 이유로
      한가하다고 답하지 말고 조회에 실패했다는 사실을 사용자에게 알립니다.
    - row의 time_complete가 false이면 시작·종료 시간이 불완전해 하루 끝까지 바쁜 것으로 계산된
      일정이므로, 그 시간대를 단정하지 말고 사용자에게 정확한 시간을 확인합니다.
    """

    result = _collect_member_schedules(
        member_names=member_names,
        date_from=date_from,
        date_to=date_to,
        personal_schedules=_personal_schedules_for_current_scope(date_from, date_to),
    )

    return json_payload(result)


def week05_tools() -> list[Any]:
    """4주차까지의 도구에 외부 SQLite/MCP 일정 도구를 누적한 목록입니다."""

    return [
        *week04_tools(),
        search_previous_conversations,
        load_conversation_messages,
        extract_schedules_from_history,
        # create_shared_schedule, delete_shared_schedule: 추가과제
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
        "도구는 사람 이름이 언급됐는지가 아니라, 사용자가 찾는 데이터가 어디에 저장돼 있는지로 고르시오. "
        "Week 3~4 도구(personal_list_saved_schedules, search_saved_requests, search_personal_references)는 "
        "사용자가 앱에 저장한 데이터를 찾는다고 명시할 때만 쓰시오. "
        "즉 \"내가 저장한\", \"앱에 등록한\", \"앱에 저장해 둔\"처럼 앱 저장소를 직접 가리키는 표현이 요청에 있어야 한다. "
        "그런 표현이 없으면 다른 사람의 일정·대화를 묻는 요청으로 보고 이번 주차의 외부 SQLite/MCP 도구를 사용하시오. "
        "사람 이름이 나왔다는 이유만으로 앱 저장 데이터를 조회해서는 안 된다. "
        "단 \"내 일정이랑 철수 일정 맞춰줘\"처럼 나와 다른 사람의 시간을 맞추는 요청은 "
        "앱 저장을 가리키는 표현이 함께 있어도 조율 요청이므로 collect_member_schedules를 쓰시오. "
        "같은 이름이 나와도 다음 두 요청은 서로 다른 도구를 쓴다. "
        "\"철수 일정 알려줘\"는 앱 저장을 언급하지 않았으므로 extract_schedules_from_history이고, "
        "\"앱에 저장한 철수와의 회의 보여줘\"는 앱 저장을 명시했으므로 personal_list_saved_schedules다. "
        "각 도구를 언제 쓰는지는 그 도구의 설명에 적혀 있으니 그 기준을 따르시오.",
        "특정 멤버의 일정이 없다고 답하려면, 그 멤버 이름을 member_names에 넣어 실제로 조회한 결과가 비어 있을 때만 그렇게 답하시오. "
        "조회하지 않은 멤버에 대해 일정이 없다고 단정하지 마시오.",
        "공유 일정을 새로 '등록'하거나 '삭제'해달라는 요청은 이번 주차에서 지원하지 않는다고 답하시오. "
        "공유 일정 '조회'는 list_shared_schedules로 지원하므로 조회 요청까지 거절하지 마시오.",
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
