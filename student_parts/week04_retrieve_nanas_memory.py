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

# Week 4 설계의 핵심: Nana의 기억은 세 출처로 나뉘어 있고, tool 하나는 그중 하나만 본다.
# 이 표가 "어떤 tool이 무엇을 보는가"의 단일 출처이며, 빈손일 때 남은 출처를 안내하는 데도 쓴다.
#
# `when`(어떤 질문일 때 쓰는 출처인가)이 특히 중요하다. 남은 출처를 이름만 나열했더니
# LLM이 목록의 첫 번째를 집고 멈추는 회귀가 실측됐다(12회 중 3회 오답).
# 남은 출처는 "무엇을 보는가"가 아니라 "어떤 질문에 답하는가"로 알려 줘야 제대로 라우팅된다.
MEMORY_SOURCES = {
    "search_personal_references": {
        "what": "내가 적어 둔 개인 참고자료(메모·선호·원칙)",
        "when": "'내가 ~를 좋아한다고 적어 뒀지', '내 선호가 뭐였지'처럼 적어 둔 메모나 선호를 물을 때",
    },
    "search_saved_requests": {
        "what": "SQLite에 저장된 일정/할 일/알림 기록",
        "when": "'저장한 일정/할 일 중에 ~ 있었나'처럼 저장된 구조화 기록을 키워드로 물을 때",
    },
    "search_conversation_messages": {
        "what": "앱에 남아 있는 이전 채팅 대화",
        "when": "'예전 대화에서', '전에 말했잖아'처럼 지난 채팅에서 오간 말을 물을 때",
    },
}

RAG_SOURCE_PROMPT = (
    "Week 4부터 Nana의 기억은 세 개의 서로 다른 저장소에 나뉘어 있고, tool도 출처별로 분리되어 있다. "
    "1) 개인 참고자료(내가 적어 둔 선호·원칙·메모)는 ChromaDB에 있고 search_personal_references로 찾는다. "
    "2) 저장된 일정/할 일/알림 구조화 기록은 SQLite structured_requests에 있고 search_saved_requests로 찾는다. "
    "3) 앱에 남은 이전 채팅 발화는 SQLite conversations/messages를 대화 단위로 sync한 ChromaDB에 있고 "
    "search_conversation_messages로 찾는다. "
    "'내가 뭐 좋아한다고 했지'는 1번, '저장한 일정 중에 회의 있었나'는 2번, "
    "'예전 대화에서 뭐라고 했지'는 3번이다. "
    "하나의 tool이 세 출처를 모두 보지는 못하므로 질문 성격에 맞는 tool을 고르고, "
    "필요하면 여러 개를 각각 호출해 근거를 합친다."
)

WEEK04_TOOL_CALL_PROMPT = """Week 4 tool 호출 규칙:
- "기억해 둬", "메모해 둬"처럼 참고자료로 남겨 달라는 요청은 add_personal_reference에 title/content/tags를 넣어 저장합니다.
  일정/할 일/알림 저장 요청은 Week 3 경로(extract_schedule_request → save_structured_request)를 그대로 씁니다.
- 저장된 일정/할 일/알림의 "목록"을 보여 달라는 요청에는 Week 3 조회 tool을 씁니다.
  (일정은 personal_list_saved_schedules, 할 일·알림은 list_saved_requests)
  search_saved_requests는 제목/사유/원문 키워드로 기록을 되짚어 찾아야 할 때만 씁니다.
- search_personal_references 결과는 top-level hits, search_saved_requests 결과는 top-level rows에 들어 있습니다.
  절대 내용을 지어내지 않습니다.
- hits/rows가 비어 있어도 그것만으로 "기억에 없다"고 결론짓지 않습니다.
  각 tool은 세 출처 중 하나만 보므로 0건은 "이 출처에 없다"는 뜻일 뿐입니다.
  0건 응답에는 아직 확인하지 않은 출처를 "어떤 질문일 때 쓰는지"와 함께 알려 주는 note가 옵니다.
  note가 오면 목록 순서가 아니라 사용자 질문 유형에 맞는 출처를 골라 한 번 더 검색합니다.
  특히 "예전 대화에서", "전에 말했잖아" 유형은 반드시 search_conversation_messages까지 확인합니다.
  해당하는 출처를 모두 확인한 뒤에야 "그 기억은 없다"고 답합니다.
  같은 tool을 같은 query로 반복 호출하지는 않습니다.
- 검색 결과가 질문과 상관없는 내용이면 억지로 근거로 쓰지 않습니다.
  벡터 검색은 관련 없어도 top_k개를 항상 돌려주므로, distance가 멀거나 주제가 다르면
  "그 내용은 질문과 무관하다"고 보고 다른 출처를 확인합니다.
- search_conversation_messages는 현재 대화를 검색에서 제외합니다.
  방금 사용자가 한 말은 이 tool로 찾지 말고 지금 대화 맥락에서 바로 답합니다.
- 대화 검색 결과 중 assistant 발화만 있는 내용은 사실로 확정하지 않습니다.
  사용자 발화나 저장 기록(rows)으로 뒷받침될 때만 단정하고, 아니면 "이전 답변에서 그렇게 말한 적이 있다"까지만 말합니다.
- 최종 답변에는 어느 출처(참고자료 / 저장 기록 / 이전 대화)에서 찾았는지 밝힙니다."""


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


def empty_result_note(tool_name: str) -> str:
    """검색 결과가 0건일 때, 아직 확인하지 않은 나머지 출처를 알려 주는 힌트를 만듭니다.

    Week 4 tool은 각자 한 출처만 봅니다. 그래서 이 tool의 0건은 "그런 기억이 없다"가 아니라
    "이 출처에는 없다"입니다. 그런데 실측에서 그 구분이 사라진 채 곧장 "없습니다"라는 최종
    답변으로 끝나 버리는 조용한 실패가 나왔습니다(search_saved_requests 8회 중 1회).
    시스템 프롬프트로 막으면 확률이지만 방금 받은 tool 결과 속 지시는 훨씬 잘 지켜지므로,
    남은 출처를 결과 JSON에 실어 의사결정 지점에서 바로잡습니다.

    남은 출처는 이름만이 아니라 "어떤 질문일 때 쓰는지"까지 적습니다. 이름만 나열했을 때
    LLM이 목록 첫 번째를 집고 멈추는 오답이 실측됐기 때문입니다.
    """

    return f"'{MEMORY_SOURCES[tool_name]['what']}'에는 없습니다. {_other_sources_hint(tool_name)}"


def relevance_check_note(tool_name: str, hits: list[dict[str, Any]]) -> str:
    """검색 결과가 있지만 질문과 무관할 수 있을 때, 내용으로 판단하라고 안내합니다.

    ChromaDB 벡터 검색에는 임계값이 없어서 질문과 관련이 적어도 top_k개를 항상 돌려줍니다.
    그래서 empty_result_note는 hits가 빌 때만 나오는데, 참고자료 검색에서는 hits가 거의
    비지 않아 사실상 발동하지 않습니다(멘토 리뷰 [P5]).

    distance로 잘라내거나 라우팅을 강제하는 방식은 실측에서 반례가 있었습니다.
    관련 질의(1.04)와 무관 질의(1.24)의 거리 구간이 겹쳐서 임계값으로 가를 수 없었습니다.
    그래서 결과를 자르지 않고 그대로 두되, (1) 가장 가까운 distance를 눈에 띄게 노출하고
    (2) "벡터 검색은 무관해도 결과를 준다"는 사실과 함께 내용으로 판단하라고 안내만 합니다.
    최종 판단은 숫자가 아니라 LLM이 hit 내용과 질문을 견줘 내리게 합니다.
    """

    return (
        "아래 hits는 벡터 검색이 질문과의 거리순으로 돌려준 것입니다. "
        "벡터 검색은 관련이 적어도 top_k개를 항상 반환하므로, 결과가 있다는 사실만으로 근거가 되지 않습니다. "
        "각 hit의 content가 사용자 질문에 실제로 답하는지 직접 확인하세요(distance는 참고용이며 클수록 멉니다). "
        f"어느 hit도 질문 내용과 맞지 않으면 이 출처에는 답이 없는 것으로 보고, {_other_sources_hint(tool_name)}"
    )


def _other_sources_hint(tool_name: str) -> str:
    """이 tool이 못 찾았을 때 확인할 나머지 두 출처를, '어떤 질문일 때 쓰는지'와 함께 안내합니다.

    남은 출처를 이름만 나열했더니 LLM이 목록 첫 번째를 집고 멈추는 오답이 실측됐습니다.
    그래서 이름이 아니라 "어떤 질문에 답하는 출처인가"(when)로 라우팅 단서를 줍니다.
    """

    remaining = " / ".join(
        f"{source['when']} → {name}" for name, source in MEMORY_SOURCES.items() if name != tool_name
    )
    return (
        f"Nana의 기억은 출처가 셋으로 나뉘어 있고 이 tool은 그중 하나만 봅니다. "
        f"아직 확인하지 않은 출처 — {remaining}. "
        f"사용자 질문이 위 둘 중 하나에 해당하면 그 tool로 한 번 더 찾아본 뒤에 답하세요. "
        f"해당하는 출처를 모두 확인했는데도 근거가 없으면 그때는 없다고 답해도 됩니다."
    )


def nearest_distance(hits: list[dict[str, Any]]) -> float | None:
    """hit 목록에서 가장 가까운(작은) distance를 꺼냅니다. 값이 없으면 None."""

    distances = [hit["distance"] for hit in hits if hit.get("distance") is not None]
    return min(distances) if distances else None


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

    saved = reference_store.add_personal_reference(title=title, content=content, tags=tags or [])
    # store는 backend 정보를 reference row 안에 넣어 준다. tool payload에서는 출처를 한눈에 보도록 최상위로 분리한다.
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
    for raw_hit in reference_store.search_personal_references(query, limit=top_k):
        # ChromaDB metadata는 tags를 콤마 문자열 하나로 보관하므로 list로 되돌린다.
        tags = [tag.strip() for tag in str(raw_hit.get("tags") or "").split(",") if tag.strip()]
        hits.append(
            {
                "id": raw_hit.get("id", ""),
                "content": raw_hit.get("content", ""),
                "distance": raw_hit.get("distance"),
                "metadata": {"title": raw_hit.get("title", ""), "tags": tags},
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

    # 결과가 없으면 빈 list 그대로 돌려 agent가 "기록 없음"을 그대로 읽게 한다.
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

    # 검색 직전에만 sync한다(lazy). source_hash가 같은 대화는 store가 알아서 건너뛴다.
    sync = conversation_rag_store.sync_from_sqlite(sqlite_store)

    # conversation_id를 콕 집어 준 경우가 아니면 "방금 한 말"이 과거 검색 결과처럼 섞이지 않게 현재 대화를 뺀다.
    # 직접 tool 호출(DEFAULT_SESSION_SCOPE)에는 제외할 대화가 없다.
    active_scope = current_session_scope()
    exclude_conversation_id = None
    if conversation_id is None and active_scope != DEFAULT_SESSION_SCOPE:
        exclude_conversation_id = active_scope

    hits = conversation_rag_store.search(
        query=query,
        top_k=top_k,
        exclude_conversation_id=exclude_conversation_id,
        conversation_id=conversation_id,
    )
    return {
        # 이전 버전 호환을 위해 같은 결과를 hits/rows 두 키로 노출한다.
        "hits": hits,
        "rows": hits,
        "context": conversation_rag_store.context_from_hits(hits),
        "rag_backend": conversation_rag_store.backend_info(),
        "sync": sync,
        "conversation_id": conversation_id,
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

    return search_conversation_messages_dict(
        sqlite_store,
        CONVERSATION_RAG_STORE,
        query=query,
        top_k=top_k,
        conversation_id=conversation_id,
    )["hits"]


@tool(args_schema=AddPersonalReferenceInput)
def add_personal_reference(title: str, content: str, tags: list[str] | None = None) -> str:
    """개인 참고자료를 ChromaDB에 추가합니다."""

    payload = add_personal_reference_dict(REFERENCE_STORE, title=title, content=content, tags=tags or [])
    return json_payload({"ok": True, "tool_name": "add_personal_reference", **payload})


@tool(args_schema=SearchPersonalReferencesInput)
def search_personal_references(query: str, top_k: int = 2) -> str:
    """개인 참고자료를 ChromaDB와 OpenAI embedding 기반으로 검색합니다."""

    resolved_top_k = safe_limit(top_k, default=2, maximum=20)
    hits = search_personal_reference_hits(REFERENCE_STORE, query=query, top_k=resolved_top_k)
    result = {
        "ok": True,
        "tool_name": "search_personal_references",
        "query": query,
        "top_k": resolved_top_k,
        "reference_backend": REFERENCE_STORE.backend_info(),
        # 벡터 검색은 무관해도 top_k개를 돌려주므로, LLM이 관련성을 판단할 수 있게
        # 가장 가까운 거리를 최상위로 끌어올려 눈에 띄게 노출한다.
        "nearest_distance": nearest_distance(hits),
        "hits": hits,
    }
    # 참고자료 검색은 hits가 거의 비지 않는다(멘토 리뷰 [P5]). 그래서 0건 안내만으로는
    # "관련 없는 결과를 근거로 쓰는" 실패를 못 막는다. 결과가 있으면 자르지 않고 그대로 두되,
    # 내용으로 관련성을 판단하라는 안내를 붙인다. distance로 강제 필터링하지는 않는다.
    result["note"] = (
        empty_result_note("search_personal_references")
        if not hits
        else relevance_check_note("search_personal_references", hits)
    )
    return json_payload(result)


@tool(args_schema=SearchSavedRequestsInput)
def search_saved_requests(query: str, top_k: int = 3) -> str:
    """SQLite에 저장된 구조화 일정/할 일/알림 row를 검색합니다. query에는 LLM이 고른 일정/할 일/알림 핵심어를 넣습니다."""

    resolved_top_k = safe_limit(top_k, default=3, maximum=50)
    rows = search_saved_request_rows(SQLITE_STORE, query=query, top_k=resolved_top_k)
    result = {
        "ok": True,
        "tool_name": "search_saved_requests",
        "query": query,
        "top_k": resolved_top_k,
        "source": "sqlite:structured_requests",
        "rows": rows,
    }
    if not rows:
        result["note"] = empty_result_note("search_saved_requests")
    return json_payload(result)


@tool(args_schema=SearchConversationMessagesInput)
def search_conversation_messages(
    query: str,
    top_k: int = 5,
    conversation_id: str | None = None,
) -> str:
    """앱 SQLite 대화 목록을 대화 단위 ChromaDB RAG로 검색합니다. query에는 LLM이 고른 짧은 핵심 명사나 구를 넣습니다."""

    resolved_top_k = safe_limit(top_k, default=5, maximum=50)
    payload = search_conversation_messages_dict(
        SQLITE_STORE,
        CONVERSATION_RAG_STORE,
        query=query,
        top_k=resolved_top_k,
        conversation_id=conversation_id,
    )
    result = {
        "ok": True,
        "tool_name": "search_conversation_messages",
        "query": query,
        "top_k": resolved_top_k,
        "source": "sqlite:conversations+messages",
        **payload,
    }
    if not payload["hits"]:
        result["note"] = empty_result_note("search_conversation_messages")
    return json_payload(result)


def _schedule_chunk(schedule: dict[str, Any]) -> dict[str, Any]:
    """저장 일정 row를 참고자료 hit와 나란히 읽을 수 있는 chunk로 만듭니다."""

    attendees = schedule.get("attendees")
    if not isinstance(attendees, list):
        # list_schedules를 거치지 않은 raw row가 들어와도 attendees_json에서 복원한다.
        attendees = _decode_attendees(schedule.get("attendees_json"))
    date = str(schedule.get("date") or "날짜 미정")
    start_time = str(schedule.get("start_time") or "시간 미정")
    end_time = str(schedule.get("end_time") or "미정")
    title = str(schedule.get("title") or "제목 없음")
    return {
        "schedule_id": schedule.get("schedule_id", ""),
        "title": title,
        "date": date,
        "start_time": start_time,
        "end_time": end_time,
        "attendees": attendees,
        "text": f"{date} {start_time}~{end_time} | {title} | 참석자: {', '.join(attendees) or '없음'}",
    }


def _memory_context(hits: list[dict[str, Any]], schedule_chunks: list[dict[str, Any]]) -> str:
    """참고자료 hit와 저장 일정 chunk를 하나의 근거 문자열로 묶습니다."""

    lines = ["[개인 참고자료]"]
    if hits:
        for index, hit in enumerate(hits, start=1):
            metadata = hit.get("metadata") or {}
            lines.append(f"[{index}] {metadata.get('title', '')} | {hit.get('content', '')}")
    else:
        lines.append("- 검색된 참고자료가 없습니다.")

    lines.append("[SQLite 저장 일정]")
    if schedule_chunks:
        for index, chunk in enumerate(schedule_chunks, start=1):
            lines.append(f"[{index}] {chunk['text']}")
    else:
        lines.append("- 검색된 저장 일정이 없습니다.")
    return "\n".join(lines)


@tool(args_schema=SearchNanaMemoryInput)
def search_nana_memory(
    query: str,
    date_from: str | None = None,
    date_to: str | None = None,
    attendee: str | None = None,
    limit: int = 5,
) -> str:
    """개인 참고자료와 SQLite 저장 일정을 한 번에 검색하고 일정 chunk를 반환합니다."""

    resolved_limit = safe_limit(limit, default=5, maximum=20)
    hits = search_personal_reference_hits(REFERENCE_STORE, query=query, top_k=resolved_limit)
    schedules = SQLITE_STORE.list_schedules(limit=resolved_limit, date_from=date_from, date_to=date_to)
    chunks = [_schedule_chunk(schedule) for schedule in schedules]
    if attendee:
        chunks = [chunk for chunk in chunks if attendee in chunk["attendees"]]
    return json_payload(
        {
            "ok": True,
            "tool_name": "search_nana_memory",
            "query": query,
            "base_date": current_app_date_iso(),
            "filters": {
                "date_from": date_from,
                "date_to": date_to,
                "attendee": attendee,
                "limit": resolved_limit,
            },
            "reference_backend": REFERENCE_STORE.backend_info(),
            "hits": hits,
            "schedule_chunks": chunks,
            "context": _memory_context(hits, chunks),
        }
    )


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

    return [
        *week03_prompt_parts(),
        """당신은 Week 4 기억 검색 agent입니다. Week 3까지의 저장/조회 기능 위에
개인 참고자료 RAG와 이전 대화 RAG로 근거를 찾아 답합니다.
Week 3의 'RAG를 하지 않는다'는 지시는 Week 4에서는 적용하지 않습니다.
다만 외부 멤버 일정 조율은 Week 4에서도 하지 않습니다.""",
        RAG_SOURCE_PROMPT,
        WEEK04_TOOL_CALL_PROMPT,
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
