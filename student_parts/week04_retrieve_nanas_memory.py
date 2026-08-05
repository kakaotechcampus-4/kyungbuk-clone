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
from fixed.reference_store import OpenAIEmbeddingFunction, PersonalReferenceStore
from fixed.session_scope import DEFAULT_SESSION_SCOPE, current_session_scope
from student_parts.week01_wake_up_nana import join_system_prompt
from student_parts.week03_build_nanas_logbook import week03_prompt_parts, week03_tools


REFERENCE_STORE = PersonalReferenceStore(CONFIG.chroma_dir)
SQLITE_STORE = AppSQLiteStore(CONFIG.app_db_path)
CONVERSATION_RAG_STORE = ConversationRAGStore(CONFIG.chroma_dir)
_WEEK04_AGENT: Any | None = None

# 검색 tool 세 개가 각자 어떤 출처를 보고 어떤 질문에 맞는지 정리한 표다.
# 2차 리뷰 반영: 같은 라우팅 정보가 프롬프트/coverage/tool description 세 곳에 흩어져
# 한 곳만 고치면 나머지가 조용히 어긋나는 문제가 있어, 이 표를 단일 출처로 두고
# 라우팅 프롬프트와 tool의 LLM용 description을 여기서 만든다. 검색 방식별 query 작성법
# 같은 tool 고유 지침과 행동 규칙(조언 질문 등)은 표에 안 어울려서 손으로 유지한다.
# 각 출처의 주어가 "나"임을 명시한다. Week 5부터 다른 멤버들의 외부 저장소가 출처로
# 추가되는데, 주어가 없으면 "하린이 전에 뭐라고 했지?" 같은 남의 대화 질문까지
# 내 대화 검색으로 빨려 들어가는 실패가 재현됐다(Week 5 라우팅 검증).
WEEK04_SEARCH_SOURCES = {
    "search_personal_references": {
        "what": "내 개인 참고자료(취향·선호·원칙 메모)",
        "when": "내가 적어 둔 취향/선호/원칙을 묻거나 일정·시간 조언에 선호 근거가 필요할 때",
    },
    "search_saved_requests": {
        "what": "SQLite에 저장된 내 일정/할 일/알림 기록",
        "when": "저장된 내 일정/할 일/알림 기록을 키워드로 되짚어 찾을 때",
    },
    "search_conversation_messages": {
        "what": "나와 Nana가 나눈 예전 채팅 대화",
        "when": "'전에 말했잖아', '내가 ~라고 했지'처럼 내가 지난 채팅에서 한 말을 찾을 때",
    },
}


def _routing_rules() -> str:
    """출처 표를 '언제 → 어느 tool' 라우팅 문장들로 바꿉니다."""

    return " ".join(
        f"{guide['when']}는 {name}({guide['what']})로 찾는다."
        for name, guide in WEEK04_SEARCH_SOURCES.items()
    )


def _search_tool_description(tool_name: str, usage_hint: str) -> str:
    """검색 tool의 LLM용 description을 출처 표에서 만듭니다.

    출처가 무엇이고 언제 쓰는지는 표에서 오고, 검색 방식별 query 작성법 같은
    tool 고유 지침만 usage_hint로 덧붙입니다.
    """

    guide = WEEK04_SEARCH_SOURCES[tool_name]
    return f"검색 대상: {guide['what']}. {guide['when']} 사용합니다. {usage_hint}"


# 저장된 기억이 출처별로 나뉘어 있고, 질문 성격에 따라 검색 tool을 골라야 함을 모델에게 알려준다.
# 출처별 라우팅 문장은 위 표에서 생성하고, 행동 규칙만 손으로 쓴다.
WEEK04_RAG_SOURCE_PROMPT = (
    "Week 4부터 Nana는 저장된 기억을 출처별로 검색한다. "
    + _routing_rules() + " "
    "'언제 ~하는 게 좋을까', '~해도 괜찮을까'처럼 일정·회의·시간을 어떻게 할지 조언이나 추천을 요청받으면, "
    "일반 상식으로 바로 답하지 말고 먼저 search_personal_references로 저장된 선호를 확인해 그 결과를 우선 근거로 답한다. "
    "사용자가 기억해 달라는 취향/메모는 add_personal_reference로 참고자료에 저장한다. "
    "단, 일정/할 일/알림 저장은 Week 3의 extract_schedule_request → save_structured_request 경로를 그대로 쓴다."
)

# 검색 결과를 근거로만 답하게 해서, 기록에 없는 내용을 지어내는 답변을 막는다.
# query 지침은 검색 방식별로 다르다. 처음엔 모든 tool에 "짧은 핵심어"를 일괄 지시했는데,
# 벡터 검색은 구체적인 문장이 유리해서(짧은 query가 관련 선호를 recall에서 밀어냄) 분리했다.
WEEK04_RAG_ANSWER_PROMPT = (
    "query는 검색 방식에 맞게 만든다. search_saved_requests는 키워드(LIKE) 검색이므로 "
    "제목/원문에 있을 법한 핵심 키워드만 넣고, 벡터 검색인 search_personal_references와 "
    "search_conversation_messages에는 사용자 질문을 그대로 넣거나 충분히 구체적인 문구를 넣는다. "
    "선호처럼 관련 기록이 여러 개일 수 있는 검색은 top_k를 3 이상으로 요청한다. "
    "검색 결과가 비어 있는 것은 '그 출처에 없다'는 뜻일 뿐이므로, "
    "질문에 해당하는 다른 출처까지 확인한 뒤에만 관련 기록이 없다고 답하고 내용을 지어내지 않는다. "
    "검색 결과가 질문과 무관한 내용이면 억지로 근거로 쓰지 않는다. "
    "지난 대화 검색 결과에서는 assistant 발화만으로 사실을 확정하지 말고 user 발화를 우선 근거로 삼는다. "
    "최종 답변에는 어떤 출처의 기록을 근거로 했는지 짧게 언급한다."
)

# 처음에는 긴 산문 note 하나였는데, "어느 출처를 봤고 어느 출처가 남았는지"는 지시가 아니라
# 데이터라서 멘토 리뷰 제안대로 구조 필드로 승격하고 지시는 next_step 한 문장만 남겼다.
# 필드 이름은 제안받은 searched/unsearched 대신 searched/other_sources를 쓴다 —
# tool은 무상태라 agent가 이전 턴에 무엇을 검색했는지 모르는데, 이미 검색한 tool이
# "unsearched"로 표기되면 대화 관점에서 거짓이 되어 재검색을 부추길 수 있다.
# other_sources("이 tool이 안 보는 다른 출처")는 항상 참이고, 재검색 판단은
# 호출 이력을 아는 agent 몫으로 next_step에 명시한다.
def source_coverage(
    tool_name: str,
    *,
    status: str,
    min_distance: float | None = None,
    far_threshold: float | None = None,
) -> dict[str, Any]:
    """검색 결과 JSON에 실어 보내는 출처 커버리지 데이터를 만듭니다.

    tool마다 한 출처만 보므로 0건은 "기억이 없다"가 아니라 "이 출처에는 없다"입니다.
    앱 검증에서 agent가 한 출처 0건만 보고 기억이 없다고 단정하는 실패가 반복 재현돼서,
    시스템 프롬프트(모든 턴에 멀리 있음)보다 잘 지켜지는 tool 결과 안에 안내를 심습니다.
    """

    searched: dict[str, Any] = {
        "tool": tool_name,
        "covers": WEEK04_SEARCH_SOURCES[tool_name]["what"],
        "status": status,
    }
    # 참고자료 벡터 검색의 "전부 무관" 판정 근거를 데이터로 같이 남긴다.
    if min_distance is not None:
        searched["min_distance"] = round(min_distance, 3)
    if far_threshold is not None:
        searched["far_threshold"] = round(far_threshold, 3)
    return {
        "searched": searched,
        "other_sources": [
            {"tool": name, "covers": guide["what"], "use_when": guide["when"]}
            for name, guide in WEEK04_SEARCH_SOURCES.items()
            if name != tool_name
        ],
        "next_step": (
            "이 출처만 보고 기억이 없다고 단정하지 않는다. 사용자 질문이 other_sources의 "
            "use_when에 해당하고 이 대화에서 아직 검색하지 않은 tool이면 검색한 뒤 답한다. "
            "이미 검색한 tool은 다시 검색하지 않고, 확인한 출처에 모두 없을 때만 "
            "관련 기록이 없다고 답한다."
        ),
    }


def empty_result_fields(tool_name: str) -> dict[str, Any]:
    """검색 0건일 때 tool 결과 dict에 병합할 source_coverage 필드를 만듭니다."""

    return {"source_coverage": source_coverage(tool_name, status="no_results")}


# far 기준의 안전망 값. 앱 trace 실측(관련 hit distance 1.0~1.3 vs 무관 1.6 이상)의 중간이다.
# 멘토 리뷰 지적대로 이 상수는 임베딩 모델·거리척도에 종속된 값이라, 아래 프로브 보정이
# 실패했을 때(키 없음/네트워크 오류)만 fallback으로 쓴다.
REFERENCE_FAR_DISTANCE_FALLBACK = 1.45

# 프로브 보정용 고정 문장 쌍. store 내용과 무관한 예시로, "관련 있는 질문-메모"와
# "무관한 질문-메모"가 현재 임베딩 모델에서 어느 거리 스케일에 놓이는지 실행 시점에 잰다.
# 임베딩 모델이 바뀌면 임계값도 같이 다시 계산되므로 고정 상수의 모델 종속 문제가 사라진다.
# (검증: text-embedding-3-small에서 보정값 1.444 — 실측으로 정한 1.45와 사실상 일치)
# store seed 참고자료를 관련 프로브로 쓰지 않는 이유(2차 리뷰 질문): 검색 대상 자체로 잰
# 임계값이 그 대상의 판정에 다시 쓰이는 순환이 생기고, 참고자료가 추가/삭제될 때마다
# 임계값이 store 내용에 끌려다닌다. "이 6쌍이 도메인을 대표한다"는 가정은 남지만,
# 기준이 데이터와 독립적으로 고정된다는 성질이 더 중요하다고 판단했다.
FAR_PROBE_PAIRS_RELEVANT = [
    ("점심시간에 약속 잡아도 될까?", "점심시간에는 약속을 잡지 않고 쉬는 것을 선호한다."),
    ("아침 운동은 언제 하는 게 좋아?", "아침 7시에 가볍게 운동하는 습관이 있다."),
    ("보고서는 언제까지 내야 하지?", "보고서 마감은 매달 마지막 금요일이다."),
]
FAR_PROBE_PAIRS_IRRELEVANT = [
    ("점심시간에 약속 잡아도 될까?", "고양이는 하루 대부분을 잠으로 보낸다."),
    ("아침 운동은 언제 하는 게 좋아?", "전세 계약은 2년 단위로 갱신된다."),
    ("보고서는 언제까지 내야 하지?", "제주도는 겨울에도 비교적 따뜻하다."),
]
_FAR_DISTANCE_CACHE: float | None = None


def reference_far_distance() -> tuple[float, bool]:
    """참고자료 hit를 "전부 무관"으로 볼 far 기준을 프로브 보정으로 계산합니다.

    관련 프로브 쌍의 최대 거리와 무관 프로브 쌍의 최소 거리의 중간값을 기준으로 잡고,
    성공한 보정값만 프로세스당 한 번 캐시합니다. 반환은 (기준값, 보정 성공 여부)입니다.
    Chroma 기본 l2(제곱 유클리드) + 단위 벡터 임베딩이라 distance = 2 - 2*cos유사도이며,
    프로브 거리도 같은 방식으로 계산해 스케일을 맞춥니다.
    """

    global _FAR_DISTANCE_CACHE
    if _FAR_DISTANCE_CACHE is not None:
        return _FAR_DISTANCE_CACHE, True
    try:
        embed = OpenAIEmbeddingFunction(
            api_key=CONFIG.proxy_token,
            base_url=CONFIG.embedding_proxy_url,
            model=CONFIG.openai_embedding_model,
        )
        pairs = FAR_PROBE_PAIRS_RELEVANT + FAR_PROBE_PAIRS_IRRELEVANT
        # 한 번의 API 호출로 모든 프로브 문장을 임베딩한다(질문/메모 순서 유지).
        vectors = embed([text for pair in pairs for text in pair])
        distances = [
            sum((a - b) ** 2 for a, b in zip(vectors[i * 2], vectors[i * 2 + 1]))
            for i in range(len(pairs))
        ]
        relevant = distances[: len(FAR_PROBE_PAIRS_RELEVANT)]
        irrelevant = distances[len(FAR_PROBE_PAIRS_RELEVANT):]
        _FAR_DISTANCE_CACHE = (max(relevant) + min(irrelevant)) / 2
        return _FAR_DISTANCE_CACHE, True
    except Exception:
        # 보정 실패는 검색 자체를 막을 일이 아니므로 실측 fallback으로 이번 판정만 동작한다.
        # 실패를 캐시하면 첫 호출의 일시적 네트워크 오류가 프로세스 끝까지 fallback을
        # 고정시키므로(2차 리뷰 지적) 캐시하지 않고 다음 호출에서 다시 보정을 시도한다.
        return REFERENCE_FAR_DISTANCE_FALLBACK, False


def reference_hits_fields(hits: list[dict[str, Any]]) -> dict[str, Any]:
    """참고자료 검색 결과가 없거나 전부 질문과 멀 때 병합할 coverage 필드를 만듭니다.

    벡터 검색은 관련이 없어도 항상 top_k개를 돌려주므로, "0건"만 보면
    무관한 hit를 받고도 안내가 빠진다. 그래서 hit가 전부 먼 경우까지 같이 본다.
    검색 결과를 잘라내는 필터가 아니라 안내만 붙이므로, 기준이 조금 어긋나도 답이 사라지지는 않는다.
    """

    if not hits:
        return empty_result_fields("search_personal_references")
    distances = [hit["distance"] for hit in hits if isinstance(hit.get("distance"), (int, float))]
    far_threshold, calibrated = reference_far_distance()
    if distances and min(distances) > far_threshold:
        coverage = source_coverage(
            "search_personal_references",
            status="all_hits_far",
            min_distance=min(distances),
            far_threshold=far_threshold,
        )
        # 판정에 쓴 기준이 프로브 보정값인지 fallback 상수인지 사후에 되짚을 수 있게 남긴다.
        coverage["searched"]["calibrated"] = calibrated
        return {"source_coverage": coverage}
    return {}


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


@tool(
    args_schema=SearchPersonalReferencesInput,
    description=_search_tool_description(
        "search_personal_references",
        "일정·회의·시간을 어떻게 할지 조언·추천이 필요한 질문에도 일반 상식으로 답하기 전에 먼저 사용합니다. "
        "벡터 검색이므로 query에는 사용자 질문을 그대로 넣거나 충분히 구체적인 문구를 넣고, "
        "관련 선호가 여러 개일 수 있으면 top_k를 3 이상으로 요청합니다.",
    ),
)
def search_personal_references(query: str, top_k: int = 2) -> str:
    """개인 참고자료를 ChromaDB 벡터 검색으로 찾습니다. (LLM용 description은 출처 표에서 생성)"""

    top_k = safe_limit(top_k, default=2, maximum=20)
    hits = search_personal_reference_hits(REFERENCE_STORE, query=query, top_k=top_k)
    # course 계약대로 top-level hits 키를 유지한다. 결과가 없어도 hits=[]로 답한다.
    result = {"ok": True, "tool_name": "search_personal_references", "query": query, "top_k": top_k, "hits": hits}
    # 0건이거나 hit가 전부 질문과 먼 경우, 남은 출처 데이터를 실어 한 출처만 보고 단정하는 것을 막는다.
    result.update(reference_hits_fields(hits))
    return json_payload(result)


@tool(
    args_schema=SearchSavedRequestsInput,
    description=_search_tool_description(
        "search_saved_requests",
        "일정으로 저장하지 않고 채팅으로만 말한 내용은 search_conversation_messages가 담당합니다. "
        "LIKE 검색이므로 query에는 문장이 아니라 제목/원문에 있을 법한 핵심 키워드를 넣습니다.",
    ),
)
def search_saved_requests(query: str, top_k: int = 3) -> str:
    """SQLite 구조화 기록을 키워드로 검색합니다. (LLM용 description은 출처 표에서 생성)"""

    top_k = safe_limit(top_k, default=3, maximum=50)
    rows = search_saved_request_rows(SQLITE_STORE, query=query, top_k=top_k)
    # course 계약대로 top-level rows 키를 유지한다. 결과가 없어도 rows=[]로 답한다.
    result = {"ok": True, "tool_name": "search_saved_requests", "query": query, "top_k": top_k, "rows": rows}
    if not rows:
        result.update(empty_result_fields("search_saved_requests"))
    return json_payload(result)


@tool(
    args_schema=SearchConversationMessagesInput,
    description=_search_tool_description(
        "search_conversation_messages",
        "일정·참고자료로 저장하지 않은 일상 대화 내용은 이 tool로만 찾을 수 있습니다. "
        "벡터 검색이므로 query에는 사용자 질문을 그대로 넣거나 충분히 구체적인 문구를 넣습니다.",
    ),
)
def search_conversation_messages(
    query: str,
    top_k: int = 5,
    conversation_id: str | None = None,
) -> str:
    """예전 채팅 대화를 대화 단위 ChromaDB RAG로 검색합니다. (LLM용 description은 출처 표에서 생성)"""

    top_k = safe_limit(top_k, default=5, maximum=50)
    result = search_conversation_messages_dict(
        SQLITE_STORE,
        CONVERSATION_RAG_STORE,
        query=query,
        top_k=top_k,
        conversation_id=conversation_id,
    )
    payload = {"ok": True, "tool_name": "search_conversation_messages", "query": query, "top_k": top_k, **result}
    if not result["hits"]:
        payload.update(empty_result_fields("search_conversation_messages"))
    return json_payload(payload)


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
