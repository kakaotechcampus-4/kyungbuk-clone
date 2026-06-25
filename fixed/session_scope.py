from __future__ import annotations

"""현재 agent 실행이 속한 채팅 대화 범위를 보관합니다.

Week 1-2의 인메모리 도구는 SQLite 영구 저장소가 아니므로, 앱의 새 대화가
이전 대화에서 만든 임시 값을 보지 않도록 현재 conversation_id를 기준으로 범위를 나눕니다.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar


DEFAULT_SESSION_SCOPE = "__direct_tool_call__"
_ACTIVE_CONVERSATION_ID: ContextVar[str | None] = ContextVar("kanana_active_conversation_id", default=None)


def current_session_scope() -> str:
    """현재 agent 실행의 대화 범위를 반환합니다."""

    return _ACTIVE_CONVERSATION_ID.get() or DEFAULT_SESSION_SCOPE


@contextmanager
def conversation_session_scope(conversation_id: str | None) -> Iterator[None]:
    """tool 실행 중 참조할 현재 conversation_id를 임시로 설정합니다."""

    token = _ACTIVE_CONVERSATION_ID.set(conversation_id or None)
    try:
        yield
    finally:
        _ACTIVE_CONVERSATION_ID.reset(token)
