from __future__ import annotations

"""LangChain 실행 결과를 앱 trace JSON으로 바꾸는 함수들입니다.

LangChain 메시지 객체는 버전과 실행 방식에 따라 dict, message 객체, stream chunk 등
형태가 조금씩 다릅니다. 이 모듈은 앱 UI와 테스트가 같은 구조의 trace를 볼 수 있도록
그 차이를 흡수합니다.
"""

import json
from typing import Any


def to_jsonable(value: Any) -> Any:
    """Pydantic model 등 LangChain 결과 객체를 JSON 표시 가능한 값으로 바꿉니다."""

    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    return str(value)


def structured_response_to_text(value: Any) -> str:
    """structured_response를 최종 답변에 표시할 class/JSON 텍스트로 변환합니다."""

    if value is None:
        return ""
    if hasattr(value, "model_dump"):
        return repr(value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def message_content_to_text(message: Any) -> str:
    """LangChain message나 dict payload에서 최종 답변 텍스트를 꺼냅니다."""

    if isinstance(message, dict):
        content = message.get("content", "")
    else:
        content = getattr(message, "content", message)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part).strip()
    return str(content).strip()


def normalize_messages_value(value: Any) -> list[Any]:
    """LangChain stream chunk의 messages 값을 항상 list로 정규화합니다."""

    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if value is None:
        return []
    return [value]


def stream_chunk_messages(chunk: Any) -> list[Any]:
    """LangChain stream update chunk에서 message 목록만 추출합니다."""

    if not isinstance(chunk, dict):
        return []
    if "messages" in chunk:
        return normalize_messages_value(chunk["messages"])

    messages: list[Any] = []
    for value in chunk.values():
        if isinstance(value, dict) and "messages" in value:
            messages.extend(normalize_messages_value(value["messages"]))
    return messages


def message_tool_call_names(message: Any) -> list[str]:
    """진행 상태 표시에 사용할 tool call 이름을 추출합니다."""

    tool_calls = getattr(message, "tool_calls", None) or []
    names: list[str] = []
    for call in tool_calls:
        if isinstance(call, dict) and call.get("name"):
            names.append(str(call["name"]))
    return names


def extract_final_text(result: dict[str, Any]) -> str:
    """LangChain 실행 결과의 마지막 비어 있지 않은 메시지를 최종 답변으로 사용합니다."""

    if isinstance(result, dict) and result.get("structured_response") is not None:
        text = structured_response_to_text(result["structured_response"])
        if text:
            return text

    messages = result.get("messages", []) if isinstance(result, dict) else []
    for message in reversed(messages):
        text = message_content_to_text(message)
        if text:
            return text
    return "응답을 생성하지 못했습니다."


def extract_agent_events(result: dict[str, Any]) -> list[dict[str, Any]]:
    """LangChain tool call/tool result 메시지를 trace 이벤트 배열로 변환합니다."""

    events: list[dict[str, Any]] = []
    messages = result.get("messages", []) if isinstance(result, dict) else []
    for message in messages:
        tool_calls = getattr(message, "tool_calls", None) or []
        for call in tool_calls:
            events.append(
                {
                    "event": "tool_call",
                    "tool_name": call.get("name"),
                    "arguments": call.get("args"),
                    "id": call.get("id"),
                }
            )
        if getattr(message, "type", "") == "tool":
            content = getattr(message, "content", "")
            parsed_content: Any = content
            try:
                parsed_content = json.loads(content)
            except Exception:
                pass
            events.append(
                {
                    "event": "tool_result",
                    "tool_name": getattr(message, "name", None),
                    "content": parsed_content,
                    "id": getattr(message, "tool_call_id", None),
                }
            )
    return events


def extract_langchain_trace(result: dict[str, Any]) -> dict[str, Any]:
    """Week 1-5가 공통으로 쓰는 기본 trace payload를 만듭니다."""

    trace = {"events": extract_agent_events(result)}
    if isinstance(result, dict) and result.get("structured_response") is not None:
        trace["structured_response"] = to_jsonable(result["structured_response"])
    return trace
