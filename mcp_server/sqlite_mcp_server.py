from __future__ import annotations

"""외부 멤버 대화/일정 SQLite DB를 MCP tool로 노출하는 stdio 서버입니다.

LangChain agent는 이 파일을 직접 import하지 않고 MCP adapter를 통해 subprocess로 실행합니다.
각 `@mcp.tool` 함수는 JSON 문자열을 반환합니다. 이렇게 하면 LangChain tool result,
테스트, 앱 trace가 모두 같은 payload 구조를 읽을 수 있습니다.
"""

import json
import os
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from fixed.config import CONFIG
from fixed.external_people_store import ExternalPeopleSQLiteStore, external_schedule_summary


DB_PATH = Path(os.getenv("KANANA_EXTERNAL_DB_PATH", str(CONFIG.external_db_path)))
STORE = ExternalPeopleSQLiteStore(DB_PATH)
mcp = FastMCP("kanana-sqlite-history")


@mcp.tool()
def search_previous_conversations(query: str, member_names: list[str] | None = None, limit: int = 5) -> str:
    """외부 Kanana SQLite 데이터베이스에서 이전 대화를 검색합니다.

    `query`에는 LLM이 직접 고른 핵심 검색 문자열을 넣습니다. 서버는 query를
    토큰화하거나 조사/불용어 처리를 하지 않습니다.
    `member_names`가 None이면 모든 멤버를 대상으로 검색하고, 빈 list면 명시된
    멤버가 없는 요청으로 보아 빈 rows를 반환합니다. 반환 rows는 conversation_id와 메시지 content를 포함합니다.
    """

    rows = STORE.search_previous_conversations(query=query, member_names=member_names, limit=limit)
    return json.dumps({"ok": True, "tool_name": "search_previous_conversations", "rows": rows}, ensure_ascii=False)


@mcp.tool()
def load_conversation_messages(conversation_id: str) -> str:
    """특정 이전 대화의 모든 메시지를 시간순으로 불러옵니다."""

    rows = STORE.load_conversation_messages(conversation_id=conversation_id)
    return json.dumps({"ok": True, "tool_name": "load_conversation_messages", "rows": rows}, ensure_ascii=False)


@mcp.tool()
def extract_schedules_from_history(member_names: list[str], date_from: str, date_to: str) -> str:
    """이전 대화 기록에서 멤버별 일정을 추출하고 날짜/시간 포함 요약을 반환합니다.

    현재 수업 fixture에서는 실제 자연어 추출 대신 seed된 external_schedules 테이블을
    조회합니다. store가 멤버 이름과 날짜 범위를 정규화하므로 tool은 인자를 그대로 넘깁니다.
    """

    rows = STORE.extract_schedules_from_history(
        member_names=member_names,
        date_from=date_from,
        date_to=date_to,
    )
    return json.dumps(
        {
            "ok": True,
            "tool_name": "extract_schedules_from_history",
            "rows": rows,
            "schedule_summary": external_schedule_summary(rows),
        },
        ensure_ascii=False,
    )


@mcp.tool()
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
    """공유 일정 저장소에 일정을 등록하거나 같은 schedule_id의 일정을 갱신합니다.

    앱 내부 개인 일정이 저장될 때도 이 tool을 통해 `"나"` 일정 복사본이 만들어집니다.
    `source_conversation_id`는 나중에 앱 원본 request_id 기준으로 삭제/갱신할 때 사용됩니다.
    """

    row = STORE.create_shared_schedule(
        member_name=member_name,
        title=title,
        date=date,
        start_time=start_time,
        end_time=end_time,
        notes=notes,
        source_conversation_id=source_conversation_id,
        schedule_id=schedule_id,
    )
    return json.dumps(
        {
            "ok": True,
            "tool_name": "create_shared_schedule",
            "shared_schedule": row,
        },
        ensure_ascii=False,
    )


@mcp.tool()
def delete_shared_schedule(
    schedule_id: str | None = None,
    source_conversation_id: str | None = None,
) -> str:
    """공유 일정 저장소에서 schedule_id 또는 앱 원본 request_id로 연결된 일정을 삭제합니다."""

    deleted = STORE.delete_shared_schedules(
        schedule_id=schedule_id,
        source_conversation_id=source_conversation_id,
    )
    return json.dumps(
        {
            "ok": True,
            "tool_name": "delete_shared_schedule",
            "deleted_count": len(deleted),
            "deleted": deleted,
        },
        ensure_ascii=False,
    )


@mcp.tool()
def list_shared_schedules(
    member_names: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    source_conversation_id: str | None = None,
    limit: int = 50,
) -> str:
    """공유 일정 저장소에 등록된 일정을 멤버/날짜/source 기준으로 조회합니다.

    필터가 없으면 오래된 앱 동기화 row 대신 기본 공유 일정을 반환합니다.
    """

    rows = STORE.list_shared_schedules(
        member_names=member_names,
        date_from=date_from,
        date_to=date_to,
        source_conversation_id=source_conversation_id,
        limit=limit,
    )
    return json.dumps(
        {
            "ok": True,
            "tool_name": "list_shared_schedules",
            "rows": rows,
            "schedule_summary": external_schedule_summary(rows),
        },
        ensure_ascii=False,
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
