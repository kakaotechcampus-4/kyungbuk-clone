from __future__ import annotations

"""로컬 MCP 서버를 LangChain 도구처럼 불러오고 호출하는 헬퍼입니다.

Week 5/6은 외부 멤버 일정 데이터를 직접 SQL로 읽지 않고 MCP 도구로 접근합니다.
이 모듈은 `mcp_server/sqlite_mcp_server.py`를 stdio subprocess로 띄운 뒤,
LangChain MCP adapter가 반환한 tool 객체를 실행합니다.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from threading import Thread
from typing import Any

from fixed.config import CONFIG, PACKAGE_ROOT


def _mcp_result_to_text(result: Any) -> str:
    """MCP/LangChain tool 결과를 앱에서 다루기 쉬운 문자열로 정규화합니다.

    adapter 버전에 따라 result가 문자열, content list, 객체 list 등으로 올 수 있어서
    text/content 필드를 우선 모으고, 마지막에는 JSON 문자열로 직렬화합니다.
    """

    if isinstance(result, str):
        return result
    if isinstance(result, list):
        parts: list[str] = []
        for item in result:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
            else:
                text = getattr(item, "text", None) or getattr(item, "content", None)
            if text:
                parts.append(str(text))
        if parts:
            return "\n".join(parts)
    return json.dumps(result, ensure_ascii=False)


def _run_coroutine_sync(coro: Any) -> Any:
    """async MCP 호출을 동기 함수에서 사용할 수 있게 실행합니다.

    이미 event loop가 돌고 있는 환경(예: 테스트나 Gradio 내부)에서는 `asyncio.run`을
    직접 부를 수 없으므로 별도 thread에서 새 event loop를 열어 실행합니다.
    """

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: list[Any] = []
    errors: list[BaseException] = []

    def runner() -> None:
        """실행 중인 event loop와 충돌하지 않도록 별도 thread에서 coroutine을 완료합니다."""

        try:
            result.append(asyncio.run(coro))
        except BaseException as exc:
            errors.append(exc)

    thread = Thread(target=runner, daemon=True)
    thread.start()
    thread.join()
    if errors:
        raise errors[0]
    return result[0]


async def load_local_mcp_tools(db_path: str | Path | None = None) -> list[Any]:
    """로컬 SQLite MCP 서버의 tool 목록을 LangChain tool 객체로 불러옵니다.

    `db_path`가 들어오면 테스트나 임시 실행용 DB를 MCP subprocess 환경 변수로 전달합니다.
    없으면 `.env` 또는 기본 외부 DB 경로를 사용합니다.
    """

    from langchain_mcp_adapters.client import MultiServerMCPClient

    server_path = PACKAGE_ROOT / "mcp_server" / "sqlite_mcp_server.py"
    env = os.environ.copy()
    selected_db_path = db_path or env.get("KANANA_EXTERNAL_DB_PATH") or CONFIG.external_db_path
    env["KANANA_EXTERNAL_DB_PATH"] = str(selected_db_path)
    client = MultiServerMCPClient(
        {
            "kanana_sqlite": {
                "transport": "stdio",
                "command": sys.executable,
                "args": [str(server_path)],
                "env": env,
            }
        }
    )
    return await client.get_tools()


def load_local_mcp_tools_sync(db_path: str | Path | None = None) -> list[Any]:
    """동기 코드에서 MCP tool 목록을 불러오는 wrapper입니다."""

    return _run_coroutine_sync(load_local_mcp_tools(db_path=db_path))


async def call_local_mcp_tool(tool_name: str, args: dict[str, Any], db_path: str | Path | None = None) -> str:
    """이름으로 MCP tool 하나를 찾아 실행하고 결과를 문자열로 반환합니다."""

    tools = {item.name: item for item in await load_local_mcp_tools(db_path=db_path)}
    if tool_name not in tools:
        available = ", ".join(sorted(tools))
        raise ValueError(f"Unknown MCP tool {tool_name!r}. Available tools: {available}")
    return _mcp_result_to_text(await tools[tool_name].ainvoke(args))


def call_local_mcp_tool_sync(tool_name: str, args: dict[str, Any], db_path: str | Path | None = None) -> str:
    """동기 코드에서 MCP tool 하나를 호출하는 wrapper입니다."""

    return _run_coroutine_sync(call_local_mcp_tool(tool_name=tool_name, args=args, db_path=db_path))
