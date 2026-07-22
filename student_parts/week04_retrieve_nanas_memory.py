from __future__ import annotations

import json
from typing import Any

from langchain.agents import create_agent
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from fixed.config import CONFIG
from fixed.conversation_rag_store import ConversationRAGStore
from fixed.llm import chat_model
from fixed.runtime_clock import current_app_date_iso
from fixed.app_store import AppSQLiteStore
from fixed.reference_store import PersonalReferenceStore
from fixed.session_scope import DEFAULT_SESSION_SCOPE, current_session_scope
from student_parts.week01_wake_up_nana import join_system_prompt
from student_parts.week03_build_nanas_logbook import week03_prompt_parts, week03_tools


REFERENCE_STORE = PersonalReferenceStore(CONFIG.chroma_dir)
SQLITE_STORE = AppSQLiteStore(CONFIG.app_db_path)
CONVERSATION_RAG_STORE = ConversationRAGStore(CONFIG.chroma_dir)
_WEEK04_AGENT: Any | None = None

# 저장된 기억이 출처별로 나뉘어 있고, 질문 성격에 따라 검색 tool을 골라야 함을 모델에게 알려준다.
WEEK04_RAG_SOURCE_PROMPT = (
    "Week 4부터 Nana는 저장된 기억을 출처별로 검색한다. "
    "취향/선호/메모 같은 자연어 참고자료 질문은 search_personal_references(ChromaDB 참고자료)로 찾는다. "
    "저장된 일정/할 일/알림 기록에서 근거를 찾을 때는 search_saved_requests(SQLite 구조화 기록)로 찾는다. "
    "일정으로 저장하지 않고 채팅으로만 말했던 내용은 search_conversation_messages(지난 대화 RAG)로 찾는다. "
    "사용자가 기억해 달라는 취향/메모는 add_personal_reference로 참고자료에 저장한다. "
    "단, 일정/할 일/알림 저장은 Week 3의 extract_schedule_request → save_structured_request 경로를 그대로 쓴다."
)

# 검색 결과를 근거로만 답하게 해서, 기록에 없는 내용을 지어내는 답변을 막는다.
WEEK04_RAG_ANSWER_PROMPT = (
    "검색 tool의 query에는 사용자 질문에서 고른 짧은 핵심 명사나 구를 넣는다. "
    "검색 결과 hits/rows가 비어 있으면 관련 기록이 없다고 답하고 내용을 지어내지 않는다. "
    "지난 대화 검색 결과에서는 assistant 발화만으로 사실을 확정하지 말고 user 발화를 우선 근거로 삼는다. "
    "최종 답변에는 어떤 출처의 기록을 근거로 했는지 짧게 언급한다."
)


# [4주차 수강생 구현 가이드]
#
# 목표
#   Nana가 "내가 적어 둔 참고자료", "SQLite에 저장된 일정/할 일 기록",
#   "앱에 저장된 일반 채팅 발화"를 구분해서 검색하게 합니다.
#   Week 4의 핵심은 RAG를 하나의 마법 함수로 보지 않고, 데이터 출처별 검색 tool을 분리하는 것입니다.
#
# 과제 구성
#   - 메인과제: 개인 참고자료를 추가하고, 참고자료와 SQLite 저장 기록을 출처별로 검색하는
#     RAG 세로 슬라이스를 완성합니다.
#   - 추가 과제: 앱 대화 발화를 ChromaDB에 lazy sync해 검색하는 agentic RAG와
#     이전 버전 호환 통합 검색까지 확장합니다.
#
# 구현 위치와 사용할 코드
#   - 이 파일(student_parts/week04_retrieve_nanas_memory.py)의 개인 참고자료/RAG tool을 구현합니다.
#   - 개인 참고자료 저장소는 fixed/reference_store.py의 PersonalReferenceStore이며,
#     이 파일 상단의 REFERENCE_STORE가 CONFIG.chroma_dir 기준 인스턴스입니다.
#   - SQLite 저장 요청 검색은 fixed/app_store.py의 AppSQLiteStore를 사용하고,
#     이 파일 상단의 SQLITE_STORE가 CONFIG.app_db_path 기준 인스턴스입니다.
#   - 일반 채팅 발화 검색은 fixed/conversation_rag_store.py의 ConversationRAGStore를 사용하고,
#     이 파일 상단의 CONVERSATION_RAG_STORE가 CONFIG.chroma_dir 기준 인스턴스입니다.
#   - 각 tool 입력은 Pydantic args_schema로 검증하고,
#     search_personal_reference_hits(), search_saved_request_rows(), search_conversation_message_rows()에서 조회 결과를 정리합니다.
#   - tool 함수 add_personal_reference/search_personal_references/search_saved_requests/search_conversation_messages는
#     위 helper 결과를 json_payload()로 감싼 JSON 문자열로 반환합니다.
#   - top_k/limit 보정은 이 파일의 safe_limit()를 사용해 tool 안에서 처리합니다.
#   - week04_tools()는 student_parts/week03_build_nanas_logbook.py의 week03_tools() 위에
#     Week 4 RAG tool을 누적해 agent에 공개합니다.
#
# 메인과제 구현 대상
#   1. add_personal_reference
#      - title/content/tags를 REFERENCE_STORE.add_personal_reference에 넘깁니다.
#      - tags가 None이면 빈 list로 바꿉니다.
#      - 이 tool 안에서 reference_backend와 reference가 있는 JSON payload를 완성합니다.
#
#   2. search_personal_references
#      - query와 top_k로 ChromaDB 개인 참고자료를 검색합니다.
#      - top_k는 이 tool 안에서 안전한 범위로 정리합니다.
#      - course repo 기준 계약에 맞게 top-level {"hits": [...]} JSON을 반환합니다.
#      - hit에는 id, content, distance, metadata(title/tags)가 들어가야 답변 근거로 쓰기 쉽습니다.
#
#   3. search_saved_requests
#      - SQLITE_STORE.search_saved_requests(query, limit)를 호출합니다.
#      - top_k는 이 tool 안에서 안전한 범위로 정리합니다.
#      - 검색 결과가 없으면 rows=[]를 그대로 반환합니다.
#      - course repo 기준 계약에 맞게 top-level {"rows": [...]} JSON을 반환합니다.
#
# 추가 과제 구현 대상
#   1. search_conversation_messages
#      - SQLite에 저장된 앱 대화 메시지를 ConversationRAGStore.sync_from_sqlite(...)로 ChromaDB에 lazy sync합니다.
#      - conversation_id를 명시하지 않으면 현재 대화 범위는 검색에서 제외해 "방금 한 말"이 과거 검색처럼 섞이지 않게 합니다.
#      - 반환 JSON에는 hits와 rows에 같은 결과를 넣고, context/rag_backend/sync도 함께 둡니다.
#      - hit에는 conversation_id, role, content 등 대화 근거가 있어야 하며, assistant 발화만으로 사실을 확정하지 않습니다.
#
# 출처 구분
#   search_personal_references는 ChromaDB + OpenAI embedding 기반 reference 검색입니다.
#   search_saved_requests는 SQLite structured_requests/schedules 계열 기록 검색입니다.
#   search_conversation_messages는 SQLite conversations/messages를 대화 단위 청크로 sync해 검색하는 agentic RAG입니다.
#   LLM이 질문 성격에 따라 둘 중 하나 또는 둘 다 선택하도록 prompt가 준비되어 있습니다.
#
# 참고 코드
#   search_nana_memory는 reference_backend와 context를 함께 확인하는 compatibility helper입니다.
#   학생 핵심 구현 대상은 add_personal_reference, search_personal_references,
#   search_saved_requests, search_conversation_messages 4개입니다.
#   week04_tools()는 Week 1-3 도구에 이 RAG 도구들을 누적합니다.
#
# 검증 방법
#   - 메인과제: 참고자료를 추가한 뒤 관련 질문을 입력하고 trace에서 search_personal_references 호출을 확인합니다.
#     저장된 일정/할 일 질문은 search_saved_requests가 호출되는지, 결과 JSON top-level 키가 각각 hits, rows인지 확인합니다.
#   - 추가 과제: 일반 채팅 발화 질문은 search_conversation_messages가 호출되고 현재 대화가 제외되는지 확인합니다.
#
# 함수별 동작 설명 ([메인]/[추가]/[공통]은 각 함수가 속한 과제 티어입니다)
#   - [공통] _decode_attendees(raw_attendees)
#     SQLite row의 attendees_json 문자열을 list로 바꿉니다. 깨진 JSON이나 list가 아닌 값은 빈 list로 처리합니다.
#
#   - [공통] json_payload(payload)
#     tool 응답 dict를 한글이 보존되는 JSON 문자열로 바꿉니다.
#
#   - [공통] safe_limit(limit, default, maximum)
#     LLM이나 사용자가 넘긴 limit/top_k 값을 int로 바꾸고 1 이상 maximum 이하로 제한합니다.
#
#   - [메인] AddPersonalReferenceInput / SearchPersonalReferencesInput / SearchSavedRequestsInput
#     개인 참고자료 추가, 개인 참고자료 검색, SQLite 저장 요청 검색 tool의 입력 스키마입니다.
#
#   - [추가] SearchConversationMessagesInput / SearchNanaMemoryInput
#     앱 대화 RAG 검색과 기존 호환용 통합 검색 tool의 입력 스키마입니다.
#
#   - [메인] add_personal_reference_dict(...)
#     PersonalReferenceStore에 참고자료를 저장하고, 어떤 backend에 저장됐는지와 저장된 reference row를 dict로 반환합니다.
#
#   - [메인] search_personal_reference_hits(...)
#     vector store 검색 결과를 id/content/distance/metadata 구조로 정리합니다. tool은 이 list를 hits로 감싸 반환합니다.
#
#   - [메인] search_saved_request_rows(...)
#     AppSQLiteStore의 저장 요청 검색 결과를 rows 배열로 반환합니다. 일정/할 일/알림 구조화 기록을 찾을 때 사용합니다.
#
#   - [추가] search_conversation_messages_dict(...)
#     SQLite 대화 기록을 ConversationRAGStore에 lazy sync한 뒤 ChromaDB 검색을 수행합니다.
#     현재 대화는 기본적으로 제외해 "방금 한 말"이 과거 검색 결과처럼 섞이지 않게 합니다.
#
#   - [추가] search_conversation_message_rows(...)
#     search_conversation_messages_dict(...)에서 hits만 꺼내는 내부 helper입니다.
#
#   - [메인] add_personal_reference(...)
#     참고자료 추가 tool입니다. title/content/tags를 받아 vector store에 저장하고 JSON 문자열을 반환합니다.
#
#   - [메인] search_personal_references(...)
#     개인 참고자료 전용 검색 tool입니다. top-level hits 키를 반환하므로 LLM이 근거 문서를 바로 읽을 수 있습니다.
#
#   - [메인] search_saved_requests(...)
#     SQLite에 저장된 structured request/schedule 기록 검색 tool입니다. top-level rows 키를 반환합니다.
#
#   - [추가] search_conversation_messages(...)
#     앱에 저장된 일반 대화 발화를 검색하는 RAG tool입니다. 일정 DB 검색과 다른 출처임을 context/rag_backend/sync로 함께 보여줍니다.
#
#   - [추가] search_nana_memory(...)
#     이전 버전 호환용 통합 검색 tool입니다. 개인 참고자료 hit와 SQLite 일정 chunk를 한 번에 묶어 context 문자열을 만듭니다.
#
#   - [공통] week04_tools()
#     Week 3까지의 tool에 Week 4 RAG tool들을 누적해 agent에 공개합니다.
#
#   - [공통] week04_system_prompt() / week04_prompt_parts()
#     질문 성격에 따라 reference, saved request, conversation RAG 중 맞는 tool을 고르도록 system prompt를 만듭니다.
#
#   - [공통] build_week04_agent() / build_week_agent()
#     Week 1~4 tool을 가진 agent를 만들고 재사용합니다.


def _decode_attendees(raw_attendees: str | None) -> list[str]:
    try:
        decoded = json.loads(raw_attendees or "[]")
    except Exception:
        return []
    return decoded if isinstance(decoded, list) else []


def json_payload(payload: dict[str, Any]) -> str:
    """도구 반환용 dict를 한글이 깨지지 않는 JSON 문자열로 변환합니다."""

    return json.dumps(payload, ensure_ascii=False)


def safe_limit(limit: int, default: int = 5, maximum: int = 50) -> int:
    """사용자/LLM이 넘긴 limit 값을 안전한 양의 정수 범위로 보정합니다."""

    try:
        value = int(limit)
    except (TypeError, ValueError):
        value = default
    return max(1, min(value, maximum))


class AddPersonalReferenceInput(BaseModel):
    """개인 참고자료 추가 입력입니다."""

    title: str
    content: str
    tags: list[str] | None = None


class SearchPersonalReferencesInput(BaseModel):
    """개인 참고자료 검색 입력입니다."""

    query: str
    top_k: int = Field(default=2, ge=1, le=20)


class SearchSavedRequestsInput(BaseModel):
    """SQLite 저장 요청 검색 입력입니다."""

    query: str
    top_k: int = Field(default=3, ge=1, le=50)


class SearchConversationMessagesInput(BaseModel):
    """앱 대화 RAG 검색 입력입니다."""

    query: str
    top_k: int = Field(default=5, ge=1, le=50)
    conversation_id: str | None = None


class SearchNanaMemoryInput(BaseModel):
    """Week 4 호환 통합 검색 입력입니다."""

    query: str
    date_from: str | None = None
    date_to: str | None = None
    attendee: str | None = None
    limit: int = Field(default=5, ge=1, le=20)


def add_personal_reference_dict(
    reference_store: PersonalReferenceStore,
    *,
    title: str,
    content: str,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """개인 참고자료를 vector store에 추가하고 backend 정보를 반환합니다."""

    saved = reference_store.add_personal_reference(title, content, tags or [])
    # store가 돌려준 dict에서 backend 설명을 분리해, 저장된 reference row와 나란히 보여준다.
    backend = saved.pop("backend", None) or reference_store.backend_info()
    return {"reference_backend": backend, "reference": saved}


def search_personal_reference_hits(
    reference_store: PersonalReferenceStore,
    *,
    query: str,
    top_k: int = 2,
) -> list[dict[str, Any]]:
    """ChromaDB 검색 결과를 tool이 바로 반환하기 쉬운 hit 구조로 정리합니다."""

    hits: list[dict[str, Any]] = []
    for found in reference_store.search_personal_references(query, limit=top_k):
        # store는 title/tags를 hit 옆에 펼쳐 주지만, course 계약은 metadata 아래로 모은
        # id/content/distance/metadata 구조라서 여기서 모양을 맞춘다.
        hits.append(
            {
                "id": found.get("id"),
                "content": found.get("content"),
                "distance": found.get("distance"),
                "metadata": {"title": found.get("title", ""), "tags": found.get("tags", "")},
            }
        )
    return hits


def search_saved_request_rows(
    sqlite_store: AppSQLiteStore,
    *,
    query: str,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """SQLite 저장 요청을 검색하고 실제 검색 결과만 반환합니다."""

    # LIKE 검색과 정렬은 store가 담당한다. 결과가 없으면 빈 list가 그대로 나온다.
    return sqlite_store.search_saved_requests(query, limit=top_k)


def search_conversation_messages_dict(
    sqlite_store: AppSQLiteStore,
    conversation_rag_store: ConversationRAGStore,
    *,
    query: str,
    top_k: int = 5,
    conversation_id: str | None = None,
) -> dict[str, Any]:
    """SQLite 대화 목록을 lazy sync한 뒤 ChromaDB conversation RAG 결과를 반환합니다."""

    # 검색 직전에 SQLite 대화를 ChromaDB로 동기화한다(lazy sync).
    # source_hash가 같은 대화는 건너뛰므로 새로 생기거나 바뀐 대화만 upsert된다.
    sync = conversation_rag_store.sync_from_sqlite(sqlite_store)

    # conversation_id를 명시하지 않으면 현재 대화는 검색에서 뺀다.
    # 방금 한 말이 "과거 기억"인 것처럼 검색 결과에 섞이는 것을 막기 위해서다.
    exclude_conversation_id = None
    if not conversation_id:
        scope = current_session_scope()
        # 직접 호출/테스트 기본값(DEFAULT_SESSION_SCOPE)은 실제 대화가 아니므로 제외 대상이 없다.
        if scope != DEFAULT_SESSION_SCOPE:
            exclude_conversation_id = scope

    hits = conversation_rag_store.search(
        query=query,
        top_k=top_k,
        exclude_conversation_id=exclude_conversation_id,
        conversation_id=conversation_id,
    )
    return {
        "hits": hits,
        # 이전 버전 trace/테스트는 rows 키를 보므로 같은 결과를 rows에도 둔다.
        "rows": hits,
        "context": conversation_rag_store.context_from_hits(hits),
        "rag_backend": conversation_rag_store.backend_info(),
        "sync": sync,
        # trace에서 "현재 대화가 제외됐는지"를 바로 확인할 수 있게 남긴다.
        "excluded_conversation_id": exclude_conversation_id,
    }


def search_conversation_message_rows(
    sqlite_store: AppSQLiteStore,
    *,
    query: str,
    top_k: int = 5,
    conversation_id: str | None = None,
) -> list[dict[str, Any]]:
    """앱 SQLite에 저장된 일반 채팅 대화 청크를 RAG 검색합니다."""

    result = search_conversation_messages_dict(
        sqlite_store,
        CONVERSATION_RAG_STORE,
        query=query,
        top_k=top_k,
        conversation_id=conversation_id,
    )
    return result["hits"]


@tool(args_schema=AddPersonalReferenceInput)
def add_personal_reference(title: str, content: str, tags: list[str] | None = None) -> str:
    """개인 참고자료를 ChromaDB에 추가합니다."""

    # 저장 로직은 helper 한 곳에 두고, tool은 args_schema가 검증한 인자를 넘기는 입구 역할만 한다.
    payload = add_personal_reference_dict(REFERENCE_STORE, title=title, content=content, tags=tags)
    return json_payload({"ok": True, "tool_name": "add_personal_reference", **payload})


@tool(args_schema=SearchPersonalReferencesInput)
def search_personal_references(query: str, top_k: int = 2) -> str:
    """개인 참고자료를 ChromaDB와 OpenAI embedding 기반으로 검색합니다."""

    top_k = safe_limit(top_k, default=2, maximum=20)
    hits = search_personal_reference_hits(REFERENCE_STORE, query=query, top_k=top_k)
    # course 계약대로 top-level hits 키를 유지한다. 결과가 없어도 hits=[]로 답한다.
    return json_payload({"ok": True, "tool_name": "search_personal_references", "query": query, "top_k": top_k, "hits": hits})


@tool(args_schema=SearchSavedRequestsInput)
def search_saved_requests(query: str, top_k: int = 3) -> str:
    """SQLite에 저장된 구조화 일정/할 일/알림 row를 검색합니다. query에는 LLM이 고른 일정/할 일/알림 핵심어를 넣습니다."""

    top_k = safe_limit(top_k, default=3, maximum=50)
    rows = search_saved_request_rows(SQLITE_STORE, query=query, top_k=top_k)
    # course 계약대로 top-level rows 키를 유지한다. 결과가 없어도 rows=[]로 답한다.
    return json_payload({"ok": True, "tool_name": "search_saved_requests", "query": query, "top_k": top_k, "rows": rows})


@tool(args_schema=SearchConversationMessagesInput)
def search_conversation_messages(
    query: str,
    top_k: int = 5,
    conversation_id: str | None = None,
) -> str:
    """앱 SQLite 대화 목록을 대화 단위 ChromaDB RAG로 검색합니다. query에는 LLM이 고른 짧은 핵심 명사나 구를 넣습니다."""

    top_k = safe_limit(top_k, default=5, maximum=50)
    result = search_conversation_messages_dict(
        SQLITE_STORE,
        CONVERSATION_RAG_STORE,
        query=query,
        top_k=top_k,
        conversation_id=conversation_id,
    )
    return json_payload({"ok": True, "tool_name": "search_conversation_messages", "query": query, "top_k": top_k, **result})


def search_nana_memory_dict(
    reference_store: PersonalReferenceStore,
    sqlite_store: AppSQLiteStore,
    *,
    query: str,
    date_from: str | None = None,
    date_to: str | None = None,
    attendee: str | None = None,
    limit: int = 5,
) -> dict[str, Any]:
    """개인 참고자료 hit와 SQLite 일정 chunk를 한 번에 모으는 compatibility 검색입니다."""

    reference_hits = search_personal_reference_hits(reference_store, query=query, top_k=limit)

    # attendee 필터는 store 조회 조건에 없어서 여기서 거른다.
    # 거른 뒤에도 limit개가 남을 수 있게 attendee가 있으면 넉넉히 가져온다.
    schedules = sqlite_store.list_schedules(
        limit=limit * 5 if attendee else limit,
        date_from=date_from,
        date_to=date_to,
    )
    if attendee:
        schedules = [
            schedule
            for schedule in schedules
            if attendee in (schedule.get("attendees") or _decode_attendees(schedule.get("attendees_json")))
        ]
    schedules = schedules[:limit]

    # 일정 row를 "날짜 시간 | 제목 | 참석" 한 줄 chunk로 만들어 답변 근거로 바로 쓰기 쉽게 한다.
    schedule_chunks: list[dict[str, Any]] = []
    for schedule in schedules:
        attendees = schedule.get("attendees") or _decode_attendees(schedule.get("attendees_json"))
        when = " ".join(part for part in (schedule.get("date"), schedule.get("start_time")) if part) or "날짜 미정"
        parts = [when, str(schedule.get("title") or "")]
        if attendees:
            parts.append("참석: " + ", ".join(attendees))
        schedule_chunks.append(
            {
                "schedule_id": schedule.get("schedule_id"),
                "chunk": " | ".join(part for part in parts if part),
                "schedule": schedule,
            }
        )

    # 두 출처의 결과를 사람이 읽는 context 문자열 하나로도 묶어 준다(이전 버전 계약).
    context_lines = ["[개인 참고자료]"]
    if reference_hits:
        context_lines.extend(f"- {hit.get('content')}" for hit in reference_hits)
    else:
        context_lines.append("- 검색된 참고자료가 없습니다.")
    context_lines.append("[저장된 일정]")
    if schedule_chunks:
        context_lines.extend(f"- {chunk['chunk']}" for chunk in schedule_chunks)
    else:
        context_lines.append("- 검색된 저장 일정이 없습니다.")

    return {
        "query": query,
        "filters": {"date_from": date_from, "date_to": date_to, "attendee": attendee, "limit": limit},
        "reference_backend": reference_store.backend_info(),
        "reference_hits": reference_hits,
        "schedule_chunks": schedule_chunks,
        "context": "\n".join(context_lines),
    }


@tool(args_schema=SearchNanaMemoryInput)
def search_nana_memory(
    query: str,
    date_from: str | None = None,
    date_to: str | None = None,
    attendee: str | None = None,
    limit: int = 5,
) -> str:
    """개인 참고자료와 SQLite 저장 일정을 한 번에 검색하고 일정 chunk를 반환합니다."""

    limit = safe_limit(limit, default=5, maximum=20)
    result = search_nana_memory_dict(
        REFERENCE_STORE,
        SQLITE_STORE,
        query=query,
        date_from=date_from,
        date_to=date_to,
        attendee=attendee,
        limit=limit,
    )
    return json_payload({"ok": True, "tool_name": "search_nana_memory", **result})

def week04_tools() -> list[Any]:
    """3주차까지의 도구에 4주차 RAG 도구를 누적한 목록입니다."""

    return [
        *week03_tools(),
        add_personal_reference,
        search_personal_references,
        search_saved_requests,
        search_conversation_messages,
    ]


def week04_system_prompt() -> str:
    """4주차 단일 agent가 따르는 시스템 프롬프트입니다."""

    return join_system_prompt(week04_prompt_parts())


def week04_prompt_parts() -> list[str]:
    """1~4주차 system prompt 조각을 누적합니다."""

    today = current_app_date_iso()
    week04_role = (
        "너는 참고자료·SQLite 저장 기록·지난 대화를 출처별로 검색해 답하는 4주차 agent다. "
        f"오늘은 {today}이고, 상대 날짜는 이 날짜를 기준으로 해석한다. "
        "Week 3의 'RAG 검색은 아직 범위가 아니다'는 지시는 Week 4부터 적용하지 않는다. "
        "질문 성격에 맞는 검색 tool을 고르고, 출처가 애매하면 두 출처를 모두 확인한 뒤 근거를 요약해 답한다."
    )
    return [
        *week03_prompt_parts(),
        WEEK04_RAG_SOURCE_PROMPT,
        WEEK04_RAG_ANSWER_PROMPT,
        week04_role,
    ]


def build_week04_agent() -> object:
    """Week 1-4 누적 tool 목록을 노출하는 단일 LangChain agent를 만듭니다."""

    if not CONFIG.has_openai_key:
        raise RuntimeError("PROXY_TOKEN이 .env에 필요합니다.")
    global _WEEK04_AGENT
    if _WEEK04_AGENT is None:
        _WEEK04_AGENT = create_agent(
            model=chat_model(),
            tools=week04_tools(),
            system_prompt=week04_system_prompt(),
        )
    return _WEEK04_AGENT


def build_week_agent() -> object:
    """active-week registry가 호출하는 표준 Week agent builder입니다."""

    return build_week04_agent()
