from __future__ import annotations

import json
from typing import Any

from langchain.agents import create_agent
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from fixed.app_store import AppSQLiteStore
from fixed.config import CONFIG
from fixed.external_mcp import PERSONAL_SHARED_MEMBER_NAME, call_external_tool_payload
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


# list_schedules의 기본 limit은 12라서 그대로 쓰면 조율 후보가 조용히 잘립니다.
# 날짜 범위 필터는 _collect_member_schedules에서 걸리므로 여기서는 넉넉히 읽어 둡니다.
PERSONAL_SCHEDULE_SCAN_LIMIT = 200

# 내 일정 row의 notes에 출처를 남기면 external_schedule_summary()가 그 값을 요약 줄 끝에
# 그대로 출력해서, LLM 근거와 trace 디버깅에 같은 표시가 함께 남습니다.
PERSONAL_SOURCE_NOTES = {
    "app_sqlite": "앱 저장 일정",
    "session_temp": "현재 대화 임시 일정",
}

# "외부 tool"이라고만 쓰면 Week 1-5 tool이 한 목록으로 들어오는 agent가 어디까지를 외부로 볼지
# 알 수 없습니다. 남의 기록만 보는 tool과 공유 저장소/양쪽을 함께 보는 tool을 이름으로 구분합니다.
EXTERNAL_SOURCE_PROMPT = (
    "Week 5부터 일정 근거는 두 저장소로 나뉜다. "
    "1) 내 일정(개인·그룹)은 앱 SQLite에 있고 Week 1-4 도구로 저장·조회한다. "
    "2) 다른 사람의 이전 대화와 바쁜 시간은 외부 SQLite에 있고, "
    "search_previous_conversations, load_conversation_messages, extract_schedules_from_history "
    "세 tool로만 볼 수 있다. Week 1-4 도구는 외부 SQLite를 보지 못한다. "
    "3) 외부 SQLite에는 앱이 저장할 때 동기화한 내 일정 복사본도 함께 들어 있다. "
    'extract_schedules_from_history나 list_shared_schedules에 "나"를 넣으면 그 복사본이 돌아오지만, '
    "복사본은 중복이거나 앱 SQLite와 어긋날 수 있으므로 내 일정의 근거로 쓰지 않는다. "
    "공유 저장소에 실제로 어떤 row가 등록됐는지 확인할 때만 list_shared_schedules로 조회한다. "
    "4) collect_member_schedules는 앱 SQLite의 내 일정과 외부 멤버의 바쁜 시간을 한 번에 모으는 tool이다. "
    '중복을 막으려고 외부 조회에서만 "나"를 빼고, 내 일정은 앱 SQLite에서 항상 함께 읽는다. '
    "그래서 결과가 비어 있어도 개인 일정 조회 tool을 다시 부르지 않는다."
)

WEEK05_TOOL_CALL_PROMPT = """Week 5 tool 호출 규칙:
- "예전에 철수랑 무슨 얘기 했지"처럼 남의 과거 대화를 물으면 search_previous_conversations를 쓴다.
  query에는 문장이나 조사를 넣지 말고 "일정", "회의"처럼 짧은 핵심 명사만 넣는다.
  외부 서버는 query를 토큰화하지 않고 원문 부분일치로만 찾기 때문이다.
  특정 대화의 전문이 필요하면 그 결과의 conversation_id로 load_conversation_messages를 이어서 호출한다.
- 특정 멤버의 바쁜 시간만 필요하면 extract_schedules_from_history를 쓴다.
- "나까지 포함해서 언제 바쁜지"처럼 여러 사람을 함께 봐야 하면 collect_member_schedules를 쓴다.
  이 tool은 member_names에 "나"를 넣지 않아도 내 일정을 항상 함께 모은다.
- 공유 일정 저장소에 실제로 등록된 row를 확인할 때는 list_shared_schedules를 쓴다.
  내 일정 복사본까지 보려면 member_names에 "나"를 명시해야 한다. 필터를 비우면 실습용 기본 일정만 돌아온다.
- date_from/date_to는 항상 YYYY-MM-DD로 넘긴다.
- 조회 결과가 0건이면 "일정이 없다"고 단정하지 않는다. tool 결과의 note를 읽고 날짜 범위를 다시 확인한다.
- 저장과 조율을 먼저 구분한다. 사용자가 날짜와 시간을 이미 정해 등록·저장을 요청하면
  ("8월 2일 10시에 철수와 회의 등록해줘") 이것은 조율이 아니라 저장이다.
  Week 3 규칙대로 extract_schedule_request → save_structured_request로 처리하고,
  collect_member_schedules나 extract_schedules_from_history는 호출하지 않는다.
- 시간이 아직 정해지지 않아 언제가 좋을지 찾아야 하는 요청만 조율이다
  ("다음 주에 철수와 언제 회의하면 좋을지 알아봐 줘"). 이때 collect_member_schedules로 바쁜 시간 근거를 모은다.
- 조율에서 바쁜 시간 근거를 정리하는 것까지가 Week 5다. 여러 사람의 최종 회의 시간을 직접 확정하지는 않는다."""


def _schedule_scope(schedule: dict[str, Any]) -> str:
    return str(schedule.get("session_id") or DEFAULT_SESSION_SCOPE)


def _store() -> AppSQLiteStore:
    """호출 시점에 store를 만들어 CONFIG.app_db_path를 바꾼 테스트도 같은 경로를 보게 합니다."""

    return AppSQLiteStore(CONFIG.app_db_path)


def _personal_schedules_for_current_scope() -> list[dict[str, Any]]:
    """SQLite 저장 일정과 현재 대화의 임시 일정만 group 조율 후보로 사용합니다."""

    saved = _store().list_schedules(limit=PERSONAL_SCHEDULE_SCAN_LIMIT)
    saved_ids = {str(row.get("schedule_id")) for row in saved if row.get("schedule_id")}
    session_id = current_session_scope()
    pending = [
        schedule
        for schedule in PERSONAL_SCHEDULES
        if _schedule_scope(schedule) == session_id and str(schedule.get("id")) not in saved_ids
    ]
    return [*saved, *pending]


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


def _personal_schedule_source(schedule: dict[str, Any]) -> str:
    """내 일정 row가 앱 SQLite에서 왔는지 현재 대화 임시 저장소에서 왔는지 구분합니다."""

    return "app_sqlite" if schedule.get("schedule_id") else "session_temp"


def _busy_row_within_dates(row: dict[str, Any], date_from: str, date_to: str) -> bool:
    """조회 범위 안의 일정만 busy-time 근거로 남깁니다.

    날짜가 없는 일정은 "언제 바쁜지"를 말할 수 없으므로 제외합니다.
    """

    date_text = str(row.get("date") or "").strip()
    if not date_text:
        return False
    if date_from and date_text < date_from:
        return False
    if date_to and date_text > date_to:
        return False
    return True


def _personal_busy_row(schedule: dict[str, Any]) -> dict[str, Any]:
    """내 일정 row를 외부 멤버 row와 같은 필드 구조로 맞춥니다."""

    request = _structured_request_from_schedule_row(schedule)
    source = _personal_schedule_source(schedule)
    return {
        "member_name": PERSONAL_SHARED_MEMBER_NAME,
        "title": request.title or "제목 없음",
        "date": request.date,
        "start_time": request.start_time or "미정",
        "end_time": request.end_time or "미정",
        "notes": PERSONAL_SOURCE_NOTES[source],
        "source": source,
    }


def _external_busy_row(row: dict[str, Any]) -> dict[str, Any]:
    """외부 MCP 일정 row에서 busy-time 판단에 필요한 필드만 남깁니다."""

    return {
        "member_name": row.get("member_name"),
        "title": row.get("title"),
        "date": row.get("date"),
        "start_time": row.get("start_time"),
        "end_time": row.get("end_time"),
        "notes": row.get("notes"),
        "source": "external_mcp",
    }


def _busy_row_sort_key(row: dict[str, Any]) -> tuple[str, str, str]:
    """합친 rows와 요약이 날짜·시간 순으로 읽히도록 정렬 기준을 만듭니다."""

    return (
        str(row.get("date") or ""),
        str(row.get("start_time") or ""),
        str(row.get("member_name") or ""),
    )


def _empty_busy_rows_note(date_from: str, date_to: str) -> str:
    """0건일 때 LLM이 "일정이 없다"로 끝내지 않도록 다음 행동을 결과에 실어 보냅니다.

    이 tool의 0건은 "일정이 없다"가 아니라 "이 날짜 범위에 기록이 없다"입니다.
    Week 4에서 확인했듯 시스템 프롬프트 지시는 확률이지만 방금 받은 tool 결과 속
    지시는 훨씬 잘 지켜지므로, 기준 날짜와 다음 행동을 의사결정 지점에 함께 넘깁니다.
    """

    return (
        f"{date_from}~{date_to} 범위에 기록된 일정이 없습니다(오늘은 {current_app_date_iso()}). "
        "'일정이 없다'고 단정하지 말고, 확인할 날짜 범위를 사용자에게 되묻거나 "
        "search_previous_conversations로 해당 멤버의 이전 대화를 먼저 찾아 날짜를 확인하세요."
    )


def _no_personal_rows_note(date_from: str, date_to: str, *, has_rows: bool) -> str:
    """내 일정이 0건일 때 개인 일정 조회 tool을 다시 부르지 않도록 결과에 근거를 실어 보냅니다.

    LLM E2E에서 이 tool이 내 일정 0건을 반환하자, agent가 곧바로
    personal_list_saved_schedules를 같은 날짜 범위로 다시 호출하는 중복이 관찰됐습니다.
    "내 일정도 함께 모은다"는 system prompt 지시로는 막히지 않았으므로,
    이미 조회했다는 사실을 tool 결과에 실어 의사결정 지점에서 끊습니다.

    전체 0건일 때는 _empty_busy_rows_note()가 뒤에 이어 붙으므로,
    "이 결과로 답하라"로 끝내면 날짜 재확인 안내와 충돌합니다. 그래서 마지막 문장만 나눕니다.
    """

    tail = (
        "같은 범위를 개인 일정 조회 tool로 다시 확인하지 말고, 이 결과로 답하세요."
        if has_rows
        else "같은 범위를 개인 일정 조회 tool로 다시 확인해도 결과는 같으니 재호출하지 마세요."
    )
    return (
        "내 일정은 앱 SQLite와 현재 대화 임시 일정에서 이미 함께 조회했고 "
        f"{date_from}~{date_to} 범위에는 0건입니다. " + tail
    )


def _collect_member_schedules(
    *,
    member_names: list[str],
    date_from: str,
    date_to: str,
    personal_schedules: list[dict[str, Any]],
) -> dict[str, Any]:
    """내 일정과 외부 멤버 일정을 같은 row 구조로 합칩니다."""

    normalized_date_from, normalized_date_to = normalize_external_schedule_date_bounds(
        member_names, date_from, date_to
    )
    requested_member_names = normalize_external_member_names(member_names)
    # 앱이 개인/그룹 일정을 저장할 때 공유 저장소에도 "나" 복사본을 남기므로(fixed/external_mcp.py),
    # "나"를 외부 조회에 그대로 넘기면 내 일정이 앱 DB와 공유 저장소에서 두 번 들어옵니다.
    external_member_names = [
        name for name in requested_member_names if name != PERSONAL_SHARED_MEMBER_NAME
    ]

    rows = [
        _personal_busy_row(schedule)
        for schedule in personal_schedules
        if _busy_row_within_dates(schedule, normalized_date_from, normalized_date_to)
    ]
    personal_row_count = len(rows)
    # 조회할 외부 멤버가 없으면 MCP 호출마다 subprocess가 새로 뜨는 비용만 남습니다.
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
        rows.extend(_external_busy_row(row) for row in payload.get("rows", []))
    rows.sort(key=_busy_row_sort_key)

    return {
        "member_names": requested_member_names,
        "external_member_names": external_member_names,
        "date_from": normalized_date_from,
        "date_to": normalized_date_to,
        "personal_row_count": personal_row_count,
        "external_row_count": len(rows) - personal_row_count,
        "rows": rows,
        "schedule_summary": external_schedule_summary(rows),
    }


@tool(args_schema=SearchPreviousConversationsInput)
def search_previous_conversations(
    query: str,
    member_names: list[str] | None = None,
    limit: int = 5,
) -> str:
    """외부 SQLite 데이터베이스에 저장된 이전 대화를 검색합니다. query에는 LLM이 고른 짧은 핵심 명사나 구를 넣습니다."""

    # member_names는 None(전체 멤버)과 빈 list(멤버 지정 없음 → 0건)의 뜻이 다르므로 그대로 넘깁니다.
    return call_mcp_tool_sync(
        "search_previous_conversations",
        {"query": query, "member_names": member_names, "limit": limit},
    )


@tool(args_schema=LoadConversationMessagesInput)
def load_conversation_messages(conversation_id: str) -> str:
    """외부 SQLite 데이터베이스에서 특정 이전 대화의 모든 메시지를 불러옵니다."""

    # sender/content/created_at 순서를 그대로 유지해야 하므로 payload를 가공하지 않고 다시 감싸기만 합니다.
    payload = call_external_tool_payload(
        "load_conversation_messages",
        {"conversation_id": conversation_id},
    )
    return json_payload(payload)


@tool(args_schema=ExtractSchedulesFromHistoryInput)
def extract_schedules_from_history(member_names: list[str], date_from: str, date_to: str) -> str:
    """외부 SQLite 이전 대화에서 멤버별 일정을 추출합니다."""

    # 이름/날짜 정규화와 schedule_summary는 store와 MCP 경계에서 이미 처리하므로 여기서 다시 하지 않습니다.
    return call_mcp_tool_sync(
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

    # schedule_id와 source_conversation_id를 보존해야 나중에 같은 row를 찾아 갱신/삭제할 수 있습니다.
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

    # None 필터도 그대로 넘겨야 store의 "필터가 없으면 실습용 기본 공유 일정" 분기가 그대로 동작합니다.
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
    result = {
        "ok": True,
        "tool_name": "collect_member_schedules",
        "source": "sqlite:schedules+mcp:external_schedules",
        **payload,
    }
    # 전체 0건은 "내 일정 0건"이기도 하므로 두 note를 배타적으로 쓰면 재조회 방지 근거가 사라집니다.
    notes: list[str] = []
    if not payload["personal_row_count"]:
        notes.append(
            _no_personal_rows_note(
                payload["date_from"],
                payload["date_to"],
                has_rows=bool(payload["rows"]),
            )
        )
    if not payload["rows"]:
        notes.append(_empty_busy_rows_note(payload["date_from"], payload["date_to"]))
    if notes:
        result["note"] = " ".join(notes)
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
        """당신은 Week 5 외부 기록 조율 agent입니다. Week 4까지의 저장·조회·RAG 위에
외부 SQLite/MCP 서버에 있는 다른 사람의 이전 대화와 바쁜 시간을 함께 봅니다.
Week 4의 '외부 멤버 일정 조율은 하지 않는다'는 지시는 Week 5에서는 적용하지 않습니다.""",
        EXTERNAL_SOURCE_PROMPT,
        WEEK05_TOOL_CALL_PROMPT,
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
