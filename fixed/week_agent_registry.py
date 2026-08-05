from __future__ import annotations

"""현재 활성 주차에 맞는 student_parts agent를 찾아 실행하는 registry입니다.

각 주차 파일은 같은 이름의 `build_week_agent()`를 제공하고, 이 모듈은 설정값이나
UI 입력으로 받은 week 번호를 실제 Python 모듈로 매핑합니다. 앱 런타임은 개별 주차의
구현을 직접 import하지 않고 이 registry만 사용합니다.
"""

import importlib
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from fixed.config import CONFIG
from fixed.langchain_trace import (
    extract_agent_events,
    extract_final_text,
    extract_langchain_trace as extract_common_langchain_trace,
    message_tool_call_names,
    stream_chunk_messages,
)


WEEK_AGENT_MODULES = {
    1: "student_parts.week01_wake_up_nana",
    2: "student_parts.week02_structure_natural_language_requests",
    3: "student_parts.week03_build_nanas_logbook",
    4: "student_parts.week04_retrieve_nanas_memory",
    5: "student_parts.week05_load_kanas_past_conversations",
    6: "student_parts.week06_kanamate_decides_schedule",
}


@dataclass
class ActiveWeekAgentResult:
    """주차별 agent 실행 결과와 trace를 함께 담는 값 객체입니다."""

    answer: str
    trace: dict[str, Any]


@dataclass
class ActiveWeekAgentStreamEvent:
    """stream 실행 중 진행 문구 또는 최종 결과를 표현합니다."""

    status_text: str | None = None
    result: ActiveWeekAgentResult | None = None


def normalize_active_week(active_week: int | str | None) -> int:
    """입력된 주차 값을 1~6 사이 정수로 정규화하고 범위를 검증합니다."""

    try:
        week = int(active_week or 1)
    except (TypeError, ValueError):
        week = 1
    if week not in WEEK_AGENT_MODULES:
        raise ValueError("active_week은 1부터 6 사이여야 합니다.")
    return week


def _extract_trace(module: Any, result: dict[str, Any]) -> dict[str, Any]:
    """주차 모듈 전용 trace extractor가 있으면 사용하고, 없으면 공통 extractor를 씁니다."""

    extractor = getattr(module, "extract_langchain_trace", extract_common_langchain_trace)
    trace = extractor(result)
    if isinstance(trace, dict):
        return trace
    return extract_common_langchain_trace(result)


def _structured_response_from_stream_chunk(chunk: Any) -> Any | None:
    """LangChain stream update chunk에서 structured_response 값을 찾습니다."""

    if not isinstance(chunk, dict):
        return None
    if "structured_response" in chunk:
        return chunk["structured_response"]
    for value in chunk.values():
        if isinstance(value, dict) and "structured_response" in value:
            return value["structured_response"]
    return None


def run_active_week_agent(active_week: int | str | None, messages: list[dict[str, str]]) -> ActiveWeekAgentResult:
    """선택된 주차의 student_parts agent를 실행하고 UI trace payload로 변환합니다.

    PROXY_TOKEN이 없으면 LangChain agent를 만들지 않고 안내 payload를 반환합니다.
    예외가 나도 앱 화면이 깨지지 않도록 오류를 answer/trace에 담아 반환합니다.
    """

    week = normalize_active_week(active_week)
    if not CONFIG.has_openai_key:
        return ActiveWeekAgentResult(
            answer=(
                f"Week {week} 프롬프트 기반 에이전트 실행에는 .env의 PROXY_TOKEN이 필요합니다. "
                "키를 추가하면 선택한 주차의 agent가 prompt와 tool을 직접 선택해 실행합니다."
            ),
            trace={
                "mode": "active_week_agent",
                "active_week": week,
                "error": "missing_proxy_token",
                "events": [],
            },
        )

    try:
        module = importlib.import_module(WEEK_AGENT_MODULES[week])
        builder = getattr(module, "build_week_agent")
        agent = builder()
        result = agent.invoke({"messages": messages})
        trace = _extract_trace(module, result)
        trace["mode"] = "active_week_agent"
        trace["active_week"] = week
        return ActiveWeekAgentResult(answer=extract_final_text(result), trace=trace)
    except Exception as exc:
        return ActiveWeekAgentResult(
            answer=f"Week {week} agent 실행 중 오류가 발생했습니다: {type(exc).__name__}: {exc}",
            trace={
                "mode": "active_week_agent",
                "active_week": week,
                "events": [],
                "error": str(exc),
                "error_type": type(exc).__name__,
            },
        )


def stream_active_week_agent(
    active_week: int | str | None,
    messages: list[dict[str, str]],
) -> Iterator[ActiveWeekAgentStreamEvent]:
    """선택된 주차 agent를 stream으로 실행하며 UI progress 이벤트를 함께 반환합니다.

    LangChain update chunk에서 tool call 이름을 추출해 "현재 X 실행 중" 상태를 만들고,
    마지막에는 모은 메시지들로 최종 답변과 trace를 구성합니다.
    """

    week = normalize_active_week(active_week)
    yield ActiveWeekAgentStreamEvent(status_text="답변을 진행중입니다")
    if not CONFIG.has_openai_key:
        yield ActiveWeekAgentStreamEvent(result=_missing_openai_key_result(week))
        return

    collected_messages: list[Any] = []
    structured_response: Any | None = None
    try:
        module = importlib.import_module(WEEK_AGENT_MODULES[week])
        builder = getattr(module, "build_week_agent")
        agent = builder()
        for chunk in agent.stream({"messages": messages}, stream_mode="updates"):
            chunk_structured_response = _structured_response_from_stream_chunk(chunk)
            if chunk_structured_response is not None:
                structured_response = chunk_structured_response
            for message in stream_chunk_messages(chunk):
                collected_messages.append(message)
                for tool_name in message_tool_call_names(message):
                    yield ActiveWeekAgentStreamEvent(status_text=f"현재 {tool_name} 실행 중")

        result = {"messages": collected_messages}
        if structured_response is not None:
            result["structured_response"] = structured_response
        trace = _extract_trace(module, result)
        trace["mode"] = "active_week_agent"
        trace["active_week"] = week
        yield ActiveWeekAgentStreamEvent(result=ActiveWeekAgentResult(answer=extract_final_text(result), trace=trace))
    except Exception as exc:
        yield ActiveWeekAgentStreamEvent(
            result=ActiveWeekAgentResult(
                answer=f"Week {week} agent 실행 중 오류가 발생했습니다: {type(exc).__name__}: {exc}",
                trace={
                    "mode": "active_week_agent",
                    "active_week": week,
                    "events": extract_agent_events({"messages": collected_messages}),
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
            )
        )


def _missing_openai_key_result(week: int) -> ActiveWeekAgentResult:
    """PROXY_TOKEN이 없을 때 모든 실행 경로에서 공통으로 쓰는 실패 결과를 만듭니다."""

    return ActiveWeekAgentResult(
        answer=(
            f"Week {week} 프롬프트 기반 에이전트 실행에는 .env의 PROXY_TOKEN이 필요합니다. "
            "키를 추가하면 선택한 주차의 agent가 prompt와 tool을 직접 선택해 실행합니다."
        ),
        trace={
            "mode": "active_week_agent",
            "active_week": week,
            "error": "missing_proxy_token",
            "events": [],
        },
    )
