from __future__ import annotations

import json
from typing import Any

from langchain.agents import create_agent
from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator

from fixed.app_store import AppSQLiteStore
from fixed.config import CONFIG
from fixed.external_mcp import call_external_tool_payload
from fixed.external_people_store import (
    PERSONAL_SHARED_MEMBER_NAME,
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

# 내 일정을 busy-time 후보로 모을 때 SQLite에서 한 번에 훑는 최대 row 수입니다.
SAVED_SCHEDULE_SCAN_LIMIT = 200

# 시간을 모를 때 저장 경로가 넣는 자리 표시 문자열입니다. 실제 시간으로 보지 않습니다.
UNSET_TIME_TEXTS = {"미정", "시간 미정", "없음"}

WEEK05_MCP_HISTORY_PROMPT = (
    "Week 5부터 Kana는 다른 사람의 과거 대화와 일정을 앱 DB가 아니라 외부 SQLite MCP 서버에서 가져온다. "
    "이 데이터는 agent가 직접 SQL로 읽지 않고 반드시 MCP wrapper tool을 통해서만 접근한다. "
    "외부 멤버가 과거에 무슨 말을 했는지 찾을 때는 search_previous_conversations로 대화를 먼저 검색해 "
    "conversation_id를 확보하고, 그 대화의 원문을 그대로 확인해야 할 때 load_conversation_messages를 부른다. "
    "대화 목록 검색과 메시지 로드는 다른 tool이므로, 어떤 대화가 있는지만 알면 되는 질문은 검색 결과만으로 답한다. "
    "다만 사용자가 '원문', '전체 대화', '뭐라고 했는지 그대로' 처럼 대화 내용 자체를 요구하면 "
    "검색으로 얻은 conversation_id를 load_conversation_messages에 넘겨 반드시 원문을 불러온 뒤 답한다. "
    "외부 멤버만의 바쁜 시간을 볼 때는 extract_schedules_from_history에 멤버 이름과 날짜 범위를 넘겨 추출한다. "
    "'누가 언제 바쁜지', '회의 시간을 잡아 보자'처럼 나와 다른 사람 일정을 함께 봐야 하는 조율 질문은 "
    "extract_schedules_from_history와 개인 일정 조회 tool을 따로 부르지 않고 collect_member_schedules 하나로 모은다. "
    "collect_member_schedules가 앱 SQLite에 저장된 내 일정까지 외부 멤버 일정과 같은 rows 구조로 합쳐 주기 때문에, "
    "임시 메모리만 읽는 personal_list_schedules로는 내 일정 근거가 빠질 수 있다. "
    "공유 일정 저장소에 실제로 어떤 row가 등록돼 있는지 확인할 때는 list_shared_schedules를 쓴다. "
    "Week 4의 search_conversation_messages는 '나와 Kana가 이 앱에서 나눈 대화'를 찾는 tool이고, "
    "search_previous_conversations는 '외부 멤버의 지난 대화'를 찾는 tool이므로 질문 대상에 맞는 쪽을 고른다. "
    "답변 근거는 tool 결과의 rows와 schedule_summary에 실제로 있는 값만 사용하고, 없는 일정을 추측해서 만들지 않는다. "
    "collect_member_schedules 결과의 external_status가 ok가 아니면 외부 멤버 일정을 가져오지 못한 상태다. "
    "이때는 내 일정만 근거로 남았다고 먼저 밝히고, 외부 멤버가 한가하다고 단정하지 않는다. "
    "row의 time_status가 complete가 아닌 일정은 시작·종료 시간이 불완전하므로 "
    "'그날 일정이 있다'는 근거로만 쓰고 구체적인 시간 겹침 계산에는 쓰지 않는다. "
    "이번 주차 범위는 후보 수집까지다. 여러 사람의 최종 회의 시간을 혼자 확정해 저장하지 않고, "
    "누가 언제 바쁘고 어느 시간대가 비어 보이는지 근거와 함께 설명한 뒤 사용자에게 확인을 받는다."
)


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
#   - [메인] _personal_schedules_for_current_scope(date_from, date_to)
#     Week 3 이후 SQLite에 저장된 내 일정과 현재 대화에만 남아 있는 Week 1 임시 일정을 합칩니다.
#     이미 SQLite에 저장된 일정과 임시 일정이 중복되지 않도록 schedule_id/id를 기준으로 한 번 걸러냅니다.
#     날짜 범위는 파이썬 필터가 아니라 DB 조회 인자로 넘겨 limit 안에서 일정이 밀려 누락되지 않게 합니다.
#
#   - [메인] _dedupe_preserving_order(names) / _busy_time_text(value) / _schedule_time_status(start, end)
#     alias 정규화 뒤 중복 멤버 이름 제거, "미정" 자리 표시 시간 걸러내기,
#     시간 정보 완성도(complete/start_only/date_only) 분류를 담당하는 작은 helper들입니다.
#
#   - [메인] _external_busy_rows(...)
#     외부 MCP busy-time 조회를 감싸 rows와 external_status를 함께 돌려줍니다.
#     외부 조회가 실패해도 내 일정 근거는 남기는 부분 성공 정책을 여기서 구현합니다.
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


def _dedupe_preserving_order(names: list[str]) -> list[str]:
    """alias 변환 뒤 같은 이름이 두 번 남지 않도록 첫 등장 순서를 유지하며 중복을 제거합니다."""

    seen: set[str] = set()
    unique: list[str] = []
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        unique.append(name)
    return unique


def _personal_schedules_for_current_scope(
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[dict[str, Any]]:
    """SQLite 저장 일정과 현재 대화의 임시 일정만 group 조율 후보로 사용합니다.

    날짜 범위는 DB 조회 단계에서 좁힌다. limit을 먼저 걸고 파이썬에서 범위를 필터링하면
    저장 일정이 SAVED_SCHEDULE_SCAN_LIMIT보다 많아졌을 때 요청한 날짜의 일정이
    조회 결과 밖으로 밀려 실제로 존재하는 busy-time이 누락될 수 있다.
    """

    saved_schedules = AppSQLiteStore(CONFIG.app_db_path).list_schedules(
        limit=SAVED_SCHEDULE_SCAN_LIMIT,
        date_from=date_from or None,
        date_to=date_to or None,
    )
    saved_ids = {str(row.get("schedule_id")) for row in saved_schedules if row.get("schedule_id")}

    session_id = current_session_scope()
    schedules = list(saved_schedules)
    for schedule in PERSONAL_SCHEDULES:
        # 다른 대화에서 만든 임시 일정은 그 대화 안에서만 살아 있어야 하므로 조율 후보로 쓰지 않는다.
        # session_id가 없는 임시 row는 DEFAULT_SESSION_SCOPE로 보고, 기본 scope가 아닌 대화에서는 제외한다.
        # PERSONAL_SCHEDULES는 프로세스 메모리라 앱을 다시 켜면 사라지고, Week 1 생성 경로는 항상
        # session_id를 채운다. 그래서 scope 없는 row는 남겨 둘 가치가 낮고, 다른 대화의 일정이
        # 조율 근거로 새는 쪽이 더 위험하다고 보아 기존 데이터 호환성보다 대화 격리를 우선했다.
        if _schedule_scope(schedule) != session_id:
            continue
        # Week 3 호환 personal_create_schedule은 임시 일정 id를 그대로 SQLite schedule_id로 저장한다.
        # 그래서 같은 일정이 임시/저장 양쪽에 있으면 DB에 남은 쪽만 남기고 중복을 걸러낸다.
        if str(schedule.get("id")) in saved_ids:
            continue
        schedules.append(schedule)
    return schedules


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

    conversation_id: str = Field(
        min_length=1,
        description="search_previous_conversations 결과 row의 conversation_id입니다. 빈 값은 허용하지 않습니다.",
    )

    @field_validator("conversation_id")
    @classmethod
    def _reject_blank_conversation_id(cls, value: str) -> str:
        """공백만 있는 conversation_id를 tool 실행 전에 막습니다.

        min_length=1은 "   "를 통과시키므로 strip 검증을 함께 둔다. 이렇게 하면
        "조회 결과가 없음"과 "conversation_id 없이 잘못 호출함"이 trace에서 구분된다.
        """

        stripped = value.strip()
        if not stripped:
            raise ValueError(
                "conversation_id가 비어 있습니다. search_previous_conversations로 대화를 먼저 검색한 뒤 "
                "결과 row의 conversation_id를 넘기세요."
            )
        return stripped


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


def _busy_time_text(value: Any) -> str:
    """시간 문자열에서 "시간이 실제로 있는 값"만 남깁니다.

    외부 store와 Week 3 저장 경로는 시간을 모를 때 None 대신 "미정"을 넣기도 한다.
    "미정"은 truthy라서 그대로 두면 시간이 있는 일정으로 오해된다.
    """

    text = str(value or "").strip()
    return "" if text in UNSET_TIME_TEXTS else text


def _schedule_time_status(start_time: str, end_time: str) -> str:
    """일정 row의 시간 정보가 얼마나 채워졌는지 분류합니다.

    busy-time 계산에 쓸 수 있는 row와 "그날 일정이 있다"까지만 말할 수 있는 row를
    구분한다. 날짜가 없는 row는 애초에 rows에 넣지 않으므로 여기서는 다루지 않는다.
      - complete   : 시작·종료 시간이 모두 있어 시간 겹침 계산에 쓸 수 있다.
      - start_only : 시작 시간만 있어 언제 시작하는지까지만 말할 수 있다.
      - date_only  : 날짜만 있어 그날 일정이 있다는 근거로만 쓴다.
    """

    if start_time and end_time:
        return "complete"
    if start_time:
        return "start_only"
    return "date_only"


def _external_busy_rows(
    *,
    external_member_names: list[str],
    date_from: str,
    date_to: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """외부 MCP busy-time 조회 결과를 rows와 조회 상태로 나눠 반환합니다.

    외부 조회가 실패해도 내 일정 근거는 남기는 부분 성공 정책을 쓴다. collect_member_schedules는
    Week 6 조율의 입구라서, 외부 호출 하나가 깨졌다고 이미 확보한 내 일정까지 버리면
    "왜 근거가 없는지"를 trace에서 읽을 수 없다. 그래서 실패 원인은 payload에 남기고
    external_status로 알린다.
    """

    if not external_member_names:
        return [], {"external_status": "skipped", "external_error": None}

    try:
        payload_text = call_mcp_tool_sync(
            "extract_schedules_from_history",
            {
                "member_names": external_member_names,
                "date_from": date_from,
                "date_to": date_to,
            },
        )
    except Exception as exc:
        return [], {"external_status": "failed", "external_error": f"{type(exc).__name__}: {exc}"}

    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        return [], {"external_status": "failed", "external_error": f"JSON이 아닌 MCP 응답입니다: {exc}"}

    if not isinstance(payload, dict):
        return [], {"external_status": "failed", "external_error": "MCP 응답이 dict가 아닙니다."}
    if payload.get("ok") is False:
        return [], {"external_status": "failed", "external_error": str(payload.get("error") or "MCP tool이 ok=False를 반환했습니다.")}

    rows = payload.get("rows")
    if not isinstance(rows, list):
        return [], {"external_status": "failed", "external_error": f"rows가 list가 아닙니다: {type(rows).__name__}"}

    return [dict(row) for row in rows if isinstance(row, dict)], {"external_status": "ok", "external_error": None}


def _collect_member_schedules(
    *,
    member_names: list[str],
    date_from: str,
    date_to: str,
    personal_schedules: list[dict[str, Any]],
) -> dict[str, Any]:
    """내 일정과 외부 멤버 일정을 같은 row 구조로 합칩니다.

    date_from/date_to는 호출자가 이미 정규화한 값을 받습니다.
    """

    # alias와 실제 이름이 함께 들어오면(["A", "철수"] → ["철수", "철수"]) 정규화 결과가 겹친다.
    # SQL IN 조회 결과는 중복되지 않으므로, 반환 payload의 member_names도 같은 계약을 갖도록
    # 입력 순서를 유지하면서 중복을 제거한다.
    normalized_members = _dedupe_preserving_order(normalize_external_member_names(member_names))
    # "나"는 앱 SQLite에서 직접 읽는다. 개인 일정은 공유 저장소에도 복사되므로,
    # 외부 조회 대상에 "나"를 남겨 두면 같은 일정이 두 번 rows에 들어간다.
    external_member_names = [name for name in normalized_members if name != PERSONAL_SHARED_MEMBER_NAME]

    personal_rows: list[dict[str, Any]] = []
    for schedule in personal_schedules:
        request = _structured_request_from_schedule_row(schedule)
        date = str(request.date or "")
        # 날짜 없는 할 일/알림 row는 "언제 바쁜지"의 근거가 될 수 없어서 제외한다.
        if not date:
            continue
        if date_from and date < date_from:
            continue
        if date_to and date > date_to:
            continue
        start_time = _busy_time_text(request.start_time)
        end_time = _busy_time_text(request.end_time)
        time_status = _schedule_time_status(start_time, end_time)
        notes = "앱에 저장된 내 일정"
        if time_status != "complete":
            notes += " · 시간이 불완전해 그날 일정이 있다는 근거로만 사용"
        personal_rows.append(
            {
                "member_name": PERSONAL_SHARED_MEMBER_NAME,
                "title": request.title or "제목 없음",
                "date": date,
                "start_time": start_time or "미정",
                "end_time": end_time or "미정",
                "time_status": time_status,
                "notes": notes,
            }
        )

    external_rows, external_state = _external_busy_rows(
        external_member_names=external_member_names,
        date_from=date_from,
        date_to=date_to,
    )
    # 외부 row도 "미정" 시간이 섞여 오므로 내 일정과 같은 기준으로 time_status를 붙인다.
    for row in external_rows:
        row["time_status"] = _schedule_time_status(
            _busy_time_text(row.get("start_time")),
            _busy_time_text(row.get("end_time")),
        )

    # 두 출처를 합친 뒤 날짜/시작시간 순으로 세워 둔다. LLM이 "이 시간대는 겹친다"를 읽기 쉬워진다.
    rows = [*personal_rows, *external_rows]
    rows.sort(key=lambda row: (str(row.get("date") or ""), str(row.get("start_time") or "")))
    schedule_summary = external_schedule_summary(rows)
    if external_state["external_status"] == "failed":
        schedule_summary = (
            f"외부 멤버 일정 조회에 실패했습니다({external_state['external_error']}). "
            "아래 근거는 앱에 저장된 내 일정뿐입니다.\n" + schedule_summary
        )
    return {
        "member_names": [PERSONAL_SHARED_MEMBER_NAME, *external_member_names],
        "date_from": date_from,
        "date_to": date_to,
        "personal_rows": personal_rows,
        "external_rows": external_rows,
        "rows": rows,
        "schedule_summary": schedule_summary,
        **external_state,
    }


@tool(args_schema=SearchPreviousConversationsInput)
def search_previous_conversations(
    query: str,
    member_names: list[str] | None = None,
    limit: int = 5,
) -> str:
    """외부 SQLite 데이터베이스에 저장된 이전 대화를 검색합니다. query에는 LLM이 고른 짧은 핵심 명사나 구를 넣습니다."""

    # 멤버 이름 정규화는 외부 store/MCP 경계에서 한 번만 하므로 wrapper에서 다시 변환하지 않는다.
    return call_mcp_tool_sync(
        "search_previous_conversations",
        {"query": query, "member_names": member_names, "limit": limit},
    )


@tool(args_schema=LoadConversationMessagesInput)
def load_conversation_messages(conversation_id: str) -> str:
    """외부 SQLite 데이터베이스에서 특정 이전 대화의 모든 메시지를 불러옵니다."""

    # 원문 확인용 tool이라 rows의 sender/content/created_at 순서를 건드리지 않고 payload를 그대로 넘긴다.
    payload = call_external_tool_payload(
        "load_conversation_messages",
        {"conversation_id": conversation_id},
    )
    return json_payload(payload)


@tool(args_schema=ExtractSchedulesFromHistoryInput)
def extract_schedules_from_history(member_names: list[str], date_from: str, date_to: str) -> str:
    """외부 SQLite 이전 대화에서 멤버별 일정을 추출합니다."""

    # 날짜 형식 정리도 외부 store/MCP 경계 담당이라 wrapper는 인자를 그대로 전달한다.
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

    # schedule_id와 source_conversation_id를 그대로 넘겨야 나중에 같은 row를 찾아 갱신/삭제할 수 있다.
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

    # 두 식별자 모두 비어 있으면 store가 아무 것도 지우지 않는다. wrapper에서 전체 삭제로 넓히지 않는다.
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

    # 필터를 None 그대로 넘긴다. store가 "필터 없음"을 실습용 기본 공유 일정 조회로 구분하기 때문이다.
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

    # 날짜 범위는 여기서 한 번만 정규화하고, 같은 값을 SQLite 조회와 외부 MCP 조회에 함께 넘긴다.
    # 그래야 "요청한 날짜 범위"라는 계약이 두 출처에서 동일하게 지켜진다.
    normalized_date_from, normalized_date_to = normalize_external_schedule_date_bounds(
        member_names,
        date_from,
        date_to,
    )
    result = _collect_member_schedules(
        member_names=member_names,
        date_from=normalized_date_from,
        date_to=normalized_date_to,
        personal_schedules=_personal_schedules_for_current_scope(
            date_from=normalized_date_from,
            date_to=normalized_date_to,
        ),
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


def week05_prompt_parts() -> list[str]:
    """1~5주차 system prompt 조각을 누적합니다."""

    return [
        *week04_prompt_parts(),
        WEEK05_MCP_HISTORY_PROMPT,
        f"외부 멤버 일정을 조회할 때 오늘 날짜는 {current_app_date_iso()}이며, "
        "date_from/date_to는 항상 이 날짜를 기준으로 계산한 YYYY-MM-DD 문자열로 넘긴다.",
        # 누적 prompt에서는 뒤에 있는 지시가 우선한다. Week 1/3의 "개인 일정은 조회 tool로 확인한다"가
        # 조율 질문에서도 따라붙어 collect_member_schedules 뒤에 personal_list_schedules를 덧붙이는
        # trace가 반복 관찰됐다. 그래서 이 예외를 마지막 조각으로 따로 못박는다.
        "조율 질문에서 collect_member_schedules를 이미 호출했다면 그 결과 rows에 내 일정(member_name='나')과 "
        "외부 멤버 일정이 모두 들어 있다. 같은 요청에 personal_list_schedules나 personal_list_saved_schedules를 "
        "이어서 호출하지 않는다. 내 일정 근거는 collect_member_schedules 결과 rows에서만 읽는다.",
        # "사용자에게 확인을 받는다"는 최종 회의 시간 확정에만 적용되는 규칙인데, LLM이 이를 조회까지
        # 넓혀서 "검색해 드릴까요?"로 되묻고 tool을 하나도 부르지 않는 trace가 반복 관찰됐다.
        # 그래서 확인이 필요한 범위를 마지막 조각에서 좁혀 준다.
        "search_previous_conversations, load_conversation_messages, extract_schedules_from_history, "
        "list_shared_schedules, collect_member_schedules는 읽기 전용 조회라서 사용자에게 되묻지 않고 바로 실행한다. "
        "확인을 받아야 하는 것은 여러 사람의 최종 회의 시간을 확정하거나 공유 저장소에 row를 쓰는 경우뿐이다. "
        "사용자가 '원문', '전체 대화', '그대로'를 요구하면 같은 턴에서 search_previous_conversations로 "
        "conversation_id를 얻고 이어서 load_conversation_messages까지 호출한 뒤 답한다. "
        "검색만 하고 멈추거나 conversation_id를 사용자에게 물어보지 않는다.",
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
