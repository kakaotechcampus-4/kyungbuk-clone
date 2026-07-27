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


# def _decode_attendees(raw_attendees: str | None) -> list[str]:
#     try:
#         decoded = json.loads(raw_attendees or "[]")
#     except Exception:
#         return []
#     return decoded if isinstance(decoded, list) else []


def json_payload(payload: dict[str, Any]) -> str:
    """도구 반환용 dict를 한글이 깨지지 않는 JSON 문자열로 변환합니다."""

    return json.dumps(payload, ensure_ascii=False)


def tool_result(tool_name: str, *, ok: bool = True, **payload: Any) -> dict[str, Any]:
    """Week 4 tool들이 공통으로 쓰는 JSON payload 껍데기를 만듭니다."""

    return {"ok": ok, "tool_name": tool_name, **payload}


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

    # PersonalReferenceStore.add_personal_reference(...)로 개인 참고자료를 저장
        # PersonalReferenceStore.add_personal_reference() 반환 dict값
        # {
        #     "reference_id": reference_id,
        #     "title": title,
        #     "content": content,
        #     "tags": tags or [],
        #     "backend": self.backend_info(),
        # }
    # 어떤 backend에 저장되었는지와 저장된 reference row를 dict로 반환

    saved = reference_store.add_personal_reference(
        title=title,
        content=content,
        tags=tags
    )

    return {
        "reference_backend": saved["backend"],
        "reference": saved
    }


def search_personal_reference_hits(
    reference_store: PersonalReferenceStore,
    *,
    query: str,
    top_k: int = 2,
) -> list[dict[str, Any]]:
    """ChromaDB 검색 결과를 tool이 바로 반환하기 쉬운 hit 구조로 정리합니다."""

    # 개인 참고자료 검색 결과를 id/content/distance/metadata 구조로 정리 후 반환
        # reference_store.search_personal_references() 리턴 구조
        # [{
        #     "id": result.get("ids", [[]])[0][index],
        #     "title": metadata.get("title", ""),
        #     "content": document,
        #     "tags": metadata.get("tags", ""),
        #     "distance": distance,
        # }, {}, ... ]

    raw_hits = reference_store.search_personal_references(
        query=query,
        limit=safe_limit(
            limit=top_k,
            default=2,
            maximum=20
        )
    )

    hits = []
    for hit in raw_hits:
        hits.append({
            "id": hit["id"],
            "content": hit["content"],
            "distance": hit["distance"],
            "metadata": {
                "title": hit["title"],
                "tags": hit["tags"], 
            }
        })
    
    return hits


def search_saved_request_rows(
    sqlite_store: AppSQLiteStore,
    *,
    query: str,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """SQLite 저장 요청을 검색하고 실제 검색 결과만 반환합니다."""

    # AppSQLiteStore.search_saved_requests(...)로 저장된 요청을 검색 및 반환

    rows = sqlite_store.search_saved_requests(
        query=query,
        limit=safe_limit(
            limit=top_k,
            default=3,
            maximum=50
        )
    )

    return rows

# 추가 과제
def search_conversation_messages_dict(
    sqlite_store: AppSQLiteStore,
    conversation_rag_store: ConversationRAGStore,
    *,
    query: str,
    top_k: int = 5,
    conversation_id: str | None = None,
) -> dict[str, Any]:
    """SQLite 대화 목록을 lazy sync한 뒤 ChromaDB conversation RAG 결과를 반환합니다."""

    # TODO: SQLite 대화 기록을 ConversationRAGStore에 lazy sync한 뒤 현재 대화를 제외하고 검색하세요.
    ...

# 추가 과제
def search_conversation_message_rows(
    sqlite_store: AppSQLiteStore,
    *,
    query: str,
    top_k: int = 5,
    conversation_id: str | None = None,
) -> list[dict[str, Any]]:
    """앱 SQLite에 저장된 일반 채팅 대화 청크를 RAG 검색합니다."""

    # TODO: search_conversation_messages_dict(...) 결과에서 hits만 반환하세요.
    ...


@tool(args_schema=AddPersonalReferenceInput)
def add_personal_reference(title: str, content: str, tags: list[str] | None = None) -> str:
    """개인 참고자료를 ChromaDB에 추가합니다."""

    # 개인 참고자료를 저장하고 JSON 문자열로 반환
    # helper: add_personal_reference_dict()

    if tags == None: tags = []
    saved_reference = add_personal_reference_dict(
        reference_store=REFERENCE_STORE,
        title=title,
        content=content,
        tags=tags
    )

    return json_payload(tool_result(
        tool_name="add_personal_reference",
        ok=True,
        **saved_reference
    ))


@tool(args_schema=SearchPersonalReferencesInput)
def search_personal_references(query: str, top_k: int = 2) -> str:
    """개인 참고자료를 ChromaDB와 OpenAI embedding 기반으로 검색합니다."""

    # query/top_k로 개인 참고자료 vector store를 검색하고 top-level hits를 반환
    # helper: search_personal_reference_hits()

    hits = search_personal_reference_hits(
        reference_store=REFERENCE_STORE,
        query=query,
        top_k=top_k
    )
    return json_payload(tool_result(
        tool_name="search_personal_references",
        ok=True,
        hits=hits
    ))


@tool(args_schema=SearchSavedRequestsInput)
def search_saved_requests(query: str, top_k: int = 3) -> str:
    """SQLite에 저장된 구조화 일정/할 일/알림 row를 검색합니다. query에는 LLM이 고른 일정/할 일/알림 핵심어를 넣습니다."""

    # AppSQLiteStore.search_saved_requests(...)로 저장 요청을 검색하고 top-level rows를 반환
    # helper: search_saved_request_rows()

    rows = search_saved_request_rows(
        sqlite_store=SQLITE_STORE,
        query=query,
        top_k=top_k
    )
    return json_payload(tool_result(
        tool_name="search_saved_requests",
        ok=True,
        rows=rows
    ))

# 추가 과제
@tool(args_schema=SearchConversationMessagesInput)
def search_conversation_messages(
    query: str,
    top_k: int = 5,
    conversation_id: str | None = None,
) -> str:
    """앱 SQLite 대화 목록을 대화 단위 ChromaDB RAG로 검색합니다. query에는 LLM이 고른 짧은 핵심 명사나 구를 넣습니다."""

    # TODO: 앱 SQLite 대화 목록을 대화 단위 ChromaDB RAG로 검색하고 JSON 문자열로 반환하세요.
    ...

# 추가 과제
@tool(args_schema=SearchNanaMemoryInput)
def search_nana_memory(
    query: str,
    date_from: str | None = None,
    date_to: str | None = None,
    attendee: str | None = None,
    limit: int = 5,
) -> str:
    """개인 참고자료와 SQLite 저장 일정을 한 번에 검색하고 일정 chunk를 반환합니다."""

    # TODO: compatibility 통합 검색이 필요하면 개인 참고자료와 SQLite 일정 chunk를 함께 구성하세요.
    ...

def week04_tools() -> list[Any]:
    """3주차까지의 도구에 4주차 RAG 도구를 누적한 목록입니다."""

    return [
        *week03_tools(),
        add_personal_reference,
        search_personal_references,
        search_saved_requests,
        # 추가 과제
        # search_conversation_messages,
    ]


def week04_system_prompt() -> str:
    """4주차 단일 agent가 따르는 시스템 프롬프트입니다."""

    return join_system_prompt(week04_prompt_parts())


def week04_prompt_parts() -> list[str]:
    """1~4주차 system prompt 조각을 누적합니다."""

    return [
        *week03_prompt_parts(),
        """
        [개인 참고자료 지침]
        - 사용자가 "기억해줘/참고해줘/이건 내 취향이야"처럼 앞으로도 참고할 만한 선호·습관·맥락을 알려주면
          add_personal_reference로 저장한다. 일회성 일정/할 일 요청은 여기 저장하지 않고 기존 저장 tool을 쓴다.
        - 답변하기 전에 사용자의 취향/맥락이 근거가 될 수 있는 질문이면 search_personal_references를 먼저 호출해
          확인하고, 확인 없이 추측으로 답하지 않는다. 이 확인은 personal_list_saved_schedules 같은 일정 목록
          조회로 대체되지 않는다 — 목록 조회는 "무엇이 저장돼 있나"를, 참고자료 검색은 "사용자가 무엇을
          선호하나"를 답하는 서로 다른 근거이므로 둘 다 필요하면 둘 다 호출한다.

        [출처 선택 기준]
        - "내가 뭘 좋아해/선호해/메모해둔 거 있나" 류 질문 → search_personal_references
        - "저번에 이런 내용으로 저장했었나?/그 기록에 뭐라고 적었었지?"처럼 특정 키워드로 과거 저장 기록의
          세부 내용을 찾는 질문 → search_saved_requests
        - "일정 보여줘/확인해줘"처럼 목록 자체를 보고 싶어하는 질문에는 search_saved_requests 대신 기존
          personal_list_saved_schedules(일정)나 list_saved_requests(할 일/알림 포함 전체)를 그대로 사용한다.
        - 회의/일정 시간을 새로 정하거나 조율하는 질문(예: "언제가 좋을까", "시간 다시 잡자")에는
          search_saved_requests(기존 기록)와 search_personal_references(개인 선호)를 둘 다 호출해 근거를
          모은 뒤 답한다. 둘 중 하나만 호출하고 끝내지 않는다.

        [personal_list_saved_schedules의 한계 — 반드시 지킬 것]
        - personal_list_saved_schedules가 반환하는 필드는 schedule_id/title/date/start_time/end_time/
          attendees/source뿐이며 reason(저장 당시 이유·맥락)을 절대 포함하지 않는다.
        - "왜 저장했는지/이유가 뭐였는지/무슨 맥락이었는지"를 묻는 질문은 personal_list_saved_schedules로는
          구조적으로 답할 수 없다. 이런 질문에는 이 tool을 호출하지 말고 반드시 search_saved_requests로
          해당 기록을 찾아 reason 필드를 확인한다.

        [search_saved_requests 검색어(query) 작성 지침]
        - 이 tool은 저장된 내용과 겹치는 부분 문자열만 찾는 단순 LIKE 검색이며, 질문 문장을 토큰화하거나
          조사/어미를 걸러주지 않는다.
        - query에는 "이유가 뭐야/왜/뭐라고 적었지" 같은 질문 표현이나 조사를 넣지 않고, 저장된 내용에
          실제로 등장했을 이벤트/제목 핵심 명사만 넣는다 (예: "팀 회의", "킥오프", "치과").
        - 첫 검색에서 이미 답에 쓸 수 있는 관련 결과를 찾았다면 같은 tool을 반복 호출하지 않는다. 결과가
          비어 있을 때만 다른 핵심 명사로 한 번 더 검색해본다.
        - search_saved_requests 결과의 reason이 null이거나 비어 있으면, 저장 당시 이유가 따로 기록되지
          않았다는 뜻이다. 이유를 지어내지 않고 "그때 이유는 따로 적어두지 않았다"고 솔직하게 답한다.

        [Week 4 범위 제한]
        - 지금 진행 중인 이 대화의 이전 메시지는 별도 tool 없이도 항상 볼 수 있으므로 그대로 참고해도 된다.
        - [미구현 안내] 하지만 이 대화가 아닌 다른 이전 대화(세션)에서 나눈 일반 잡담 내용을 검색하는 기능은
          아직 tool로 제공되지 않는다. 사용자가 "저번에/예전에 다른 대화에서 무슨 얘기 했었지?"처럼 물으면
          tool을 호출하지 말고 아직 지원되지 않는다고 솔직하게 답한다.
        """,
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
