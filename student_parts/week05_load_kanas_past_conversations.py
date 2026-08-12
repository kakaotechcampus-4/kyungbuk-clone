from __future__ import annotations

import json
from typing import Any

from langchain.agents import create_agent
from langchain_core.tools import tool
from pydantic import BaseModel, Field

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
from fixed.session_scope import DEFAULT_SESSION_SCOPE, current_session_scope
from student_parts.week01_wake_up_nana import PERSONAL_SCHEDULES, join_system_prompt
from student_parts.week02_structure_natural_language_requests import StructuredRequest
from student_parts.week04_retrieve_nanas_memory import SQLITE_STORE, week04_prompt_parts, week04_tools


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

    # TODO: SQLite 저장 일정과 현재 대화의 임시 일정을 합쳐 반환하세요.
    saved_schedules = SQLITE_STORE.list_schedules(limit=200)
    saved_ids = {row.get("schedule_id") for row in saved_schedules}

    current_session_id = current_session_scope()
    pending_schedules = [
        schedule for schedule in PERSONAL_SCHEDULES if _schedule_scope(schedule) == current_session_id and schedule.get("id") not in saved_ids
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

    # TODO: 내 SQLite/임시 일정과 외부 MCP 일정 rows를 같은 구조로 합치세요.
    rows: list[dict[str, Any]] = []

    for schedule in personal_schedules:
        req = _structured_request_from_schedule_row(schedule)
        rows.append({
            "member_name": "나",
            "title": req.title,
            "date": req.date,
            "start_time": req.start_time,
            "end_time": req.end_time,
            "kind" : req.kind,
            "members" : req.members,
        })

    external_payload = json.loads(
        call_mcp_tool_sync(
            "extract_schedules_from_history",
            {
                "member_names": member_names,
                "date_from": date_from,
                "date_to": date_to,
            }
        )
    )

    external_rows = external_payload.get("rows", [])

    rows.extend(external_rows)
    rows = _dedupe_schedule_rows(rows)
    final_summary = external_schedule_summary(rows)

    return {"rows": rows, "schedule_summary": final_summary}


@tool(args_schema=SearchPreviousConversationsInput)
def search_previous_conversations(
    query: str,
    member_names: list[str] | None = None,
    limit: int = 5,
) -> str:
    """나 이외의 외부 팀원과 예전에 나눈 대화를 검색합니다.

    특정 팀원과 과거에 어떤 대화를 나눴는지, 그런 대화가 있었는지 자체를 확인해야 할 때
    가장 먼저 호출하세요. 이 도구를 호출해 확인하기 전에는 "그런 대화가 없다"고 답하지
    않습니다.

    - query: 대화 제목/본문에서 찾을 짧은 핵심 명사나 구(예: "일정", "회의"). 빈 문자열이면
      필터 없이 최신순으로 반환합니다.
    - member_names: 대화 상대로 좁힐 팀원 이름 목록입니다. None이면 모든 팀원의 대화를
      대상으로 하고, 빈 리스트를 주면 아무 것도 반환되지 않습니다.
    - limit: 반환할 최대 대화 개수(기본 5, 최대 50)입니다.

    반환값은 conversation_id/member_name/title/content/created_at 필드를 가진 대화 rows를
    담은 JSON 문자열입니다. 여기서 얻은 conversation_id로 해당 대화의 전체 메시지를 순서대로
    조회할 수 있습니다."""

    # TODO: call_mcp_tool_sync("search_previous_conversations", args)를 호출하고 결과 문자열을 반환하세요.
    return call_mcp_tool_sync("search_previous_conversations", {"query": query, "member_names": member_names, "limit": limit})


@tool(args_schema=LoadConversationMessagesInput)
def load_conversation_messages(conversation_id: str) -> str:
    """검색으로 찾은 특정 외부 대화 하나의 전체 메시지를 순서대로 불러옵니다.

    대화 목록의 요약(title/content 일부)만으로는 부족하고, 실제 발화 전체를 근거로 답해야 할
    때 사용하세요. sender/content/created_at이 작성 시각 오름차순으로 정렬된 메시지 목록을
    그대로 반환합니다.

    - conversation_id: 반드시 이전 대화 검색 결과에 실제로 존재했던 값을 그대로 넣어야 하며,
      임의로 추측해서 만든 값을 넣지 않습니다."""

    # TODO: call_external_tool_payload("load_conversation_messages", {"conversation_id": ...}) 결과를 JSON으로 반환하세요.
    payload = call_external_tool_payload("load_conversation_messages", {"conversation_id": conversation_id})
    return json_payload(payload)


@tool(args_schema=ExtractSchedulesFromHistoryInput)
def extract_schedules_from_history(member_names: list[str], date_from: str, date_to: str) -> str:
    """외부 팀원의 이전 대화 기록에서 실제 일정/바쁜 시간 row를 추출합니다.

    특정 팀원(들)이 주어진 기간 동안 어떤 일정이 있는지, 언제 바쁜지 물어볼 때 사용하세요.
    대화 내용을 직접 읽고 요약하는 대신, 이미 정리된 busy-time 기록을 조회합니다. 나
    자신("나")의 일정은 여기 포함되지 않으므로, 내 일정까지 함께 필요한 경우에는 이 도구
    하나만으로는 부족합니다.

    - member_names: 조회할 외부 팀원 이름 목록(필수).
    - date_from / date_to: 조회할 날짜 범위(YYYY-MM-DD 형식). 시간까지 포함된 문자열이 와도
      날짜 부분만 사용됩니다.

    반환값은 member_name/title/date/start_time/end_time/notes 필드를 가진 rows를 담은 JSON
    문자열입니다."""

    # TODO: call_mcp_tool_sync("extract_schedules_from_history", args)를 호출해 외부 멤버 busy-time rows를 반환하세요.
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
    """공유 일정 저장소에 새 일정을 등록하거나 같은 schedule_id의 기존 일정을 갱신합니다.

    회의 시간이 확정되어 나를 포함한 팀 전체가 볼 수 있는 공유 저장소에 반영해야 할 때
    사용하세요. schedule_id를 지정하면 해당 row를 덮어써 갱신하고, 지정하지 않으면 새 row로
    등록됩니다. source_conversation_id를 남겨 두면 같은 요청에서 만든 row를 나중에 찾아
    수정/삭제할 수 있습니다."""

    # TODO: call_mcp_tool_sync("create_shared_schedule", args)로 공유 일정 row를 생성/갱신하세요.
    args = {
        "member_name": member_name,
        "title": title,
        "date": date,
        "start_time": start_time,
        "end_time": end_time,
        "notes": notes,
        "source_conversation_id": source_conversation_id,
        "schedule_id": schedule_id,
    }
    return call_mcp_tool_sync("create_shared_schedule", args)


@tool(args_schema=DeleteSharedScheduleInput)
def delete_shared_schedule(
    schedule_id: str | None = None,
    source_conversation_id: str | None = None,
) -> str:
    """공유 일정 저장소에서 특정 row를 삭제합니다.

    schedule_id 또는 해당 row를 만들 때 남겨둔 source_conversation_id 중 하나로 삭제 대상을
    지정합니다. 둘 다 비어 있으면 아무 것도 삭제되지 않습니다. 삭제 후에는 공유 일정 조회
    결과에서 해당 row가 더 이상 보이지 않아야 합니다."""

    # TODO: call_mcp_tool_sync("delete_shared_schedule", args)로 공유 일정을 삭제하세요.
    args = {
        "schedule_id": schedule_id,
        "source_conversation_id": source_conversation_id,
    }
    return call_mcp_tool_sync("delete_shared_schedule", args)


@tool(args_schema=ListSharedSchedulesInput)
def list_shared_schedules(
    member_names: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    source_conversation_id: str | None = None,
    limit: int = 50,
) -> str:
    """여러 사람이 함께 보는 공유 일정 저장소의 row를 조회합니다.

    개인 일정이 아니라 팀 전체가 등록해 둔 공유 일정 자체를 확인하고 싶을 때 사용하세요.
    member_names/date_from/date_to/source_conversation_id로 결과를 좁힐 수 있고, 아무
    필터도 주지 않으면 실습용 기본 공유 일정이 우선 반환될 수 있습니다.

    반환값은 schedule_id/member_name/title/date/start_time/end_time/notes/
    source_conversation_id 필드를 가진 rows와, 사람이 읽기 쉬운 요약을 함께 담은 JSON
    문자열입니다."""

    # TODO: call_mcp_tool_sync("list_shared_schedules", args)로 공유 일정 저장소 rows를 조회하세요.
    return call_mcp_tool_sync("list_shared_schedules", {"member_names": member_names, "date_from": date_from, "date_to": date_to, "source_conversation_id": source_conversation_id, "limit": limit})


@tool(args_schema=CollectMemberSchedulesInput)
def collect_member_schedules(member_names: list[str], date_from: str, date_to: str) -> str:
    """내 일정과 다른 팀원들의 바쁜 시간을 같은 형식의 rows로 한 번에 모읍니다.

    나를 포함한 여러 사람의 일정을 한 번에 비교하거나, 함께 가능한 시간을 찾기 위한 근거가
    필요할 때 사용하세요. 내 일정과 외부 팀원 일정을 각각 따로 조회해서 직접 합치지 말고 이
    도구 하나로 처리합니다.

    - member_names: 나 이외에 함께 조회할 팀원 이름 목록(필수).
    - date_from / date_to: 조회할 날짜 범위(YYYY-MM-DD 형식).

    반환값은 member_name/title/date/start_time/end_time/notes 필드를 가진 rows(내 일정은
    member_name이 "나")와, 바쁜 시간을 자연어로 설명하는 schedule_summary를 함께 담은 JSON
    문자열입니다."""

    # TODO: 내 일정과 외부 멤버 busy-time rows를 모아 JSON 문자열로 반환하세요.
    result = _collect_member_schedules(
        member_names=member_names,
        date_from=date_from,
        date_to=date_to,
        personal_schedules=_personal_schedules_for_current_scope(),
    )
    return json_payload(result)


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
        """
        나 이외의 외부 팀원의 이전 대화나 일정을 물어보면, 내 개인 기록을 다루는 도구가 아니라
        외부 팀원 데이터를 다루는 도구로 먼저 확인하세요. 그런 대화나 일정이 없다고 답하기 전에
        반드시 관련 도구를 호출해 실제 결과를 확인해야 하며, 도구를 호출하지 않고 추측해서
        답하지 않습니다. 어떤 도구를 언제 쓸지는 각 도구 설명에 안내된 용도를 따르세요.
        """,
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
