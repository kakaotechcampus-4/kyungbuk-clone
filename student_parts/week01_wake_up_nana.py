from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any

from langchain.agents import create_agent
from langchain.tools import tool

from fixed.config import CONFIG
from fixed.langchain_trace import (
    extract_agent_events,
    extract_final_text,
    extract_langchain_trace,
    message_content_to_text,
    message_tool_call_names,
    normalize_messages_value,
    stream_chunk_messages,
)
from fixed.llm import chat_model
from fixed.runtime_clock import current_app_date_iso, next_weekday_iso
from fixed.session_scope import DEFAULT_SESSION_SCOPE, current_session_scope


PERSONAL_SCHEDULES: list[dict[str, Any]] = []
_WEEK01_AGENT: Any | None = None

# 현재 채팅 안에서 만든 개인 일정만 기억한다는 공통 system prompt
CHAT_MEMORY_PROMPT = """
너는 Nana라는 개인 일정 비서다.
현재 Week 1에서는 앱 DB나 SQLite를 사용하지 않고,
현재 대화 세션 안의 임시 메모리 PERSONAL_SCHEDULES에만 개인 일정을 저장한다.

사용자가 일정 생성, 조회, 삭제를 요청하면 반드시 제공된 tool을 사용한다.
사용자가 단순 인사나 설명을 요청하면 tool 없이 자연스럽게 답한다.

현재 대화 세션의 일정만 다룬다.
다른 대화 세션의 일정은 조회하거나 삭제하지 않는다.
"""


def join_system_prompt(parts: list[str]) -> str:
    """주차별 prompt 조각을 읽기 쉬운 누적 system prompt로 합칩니다."""

    header = (
        "아래 system prompt는 주차별로 누적된 안내다. "
        "같은 주제의 지시가 여러 번 나오면 더 높은 주차 또는 더 뒤에 있는 지시를 우선한다."
    )
    return "\n\n".join([header, *[part.strip() for part in parts if part.strip()]])


def _json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="microseconds")


def _new_personal_id() -> str:
    return f"personal_{uuid.uuid4().hex[:10]}"


def _schedule_scope(schedule: dict[str, Any]) -> str:
    """기존 직접 tool 호출 row는 기본 scope로 취급합니다."""

    return str(schedule.get("session_id") or DEFAULT_SESSION_SCOPE)


def _current_session_schedules() -> list[dict[str, Any]]:
    session_id = current_session_scope()
    return [schedule for schedule in PERSONAL_SCHEDULES if _schedule_scope(schedule) == session_id]


@tool
def personal_create_schedule(
    title: str,
    date: str,
    start_time: str,
    end_time: str = "미정",
    attendees: list[str] | None = None,
) -> str:
    """Nana의 개인 일정을 현재 대화의 임시 메모리에 생성합니다."""

    schedule = {
        "id": _new_personal_id(),
        "title": title,
        "date": date,
        "start_time": start_time,
        "end_time": end_time,
        "attendees": attendees if attendees is not None else [],
        "created_at": _now_iso(),
        "session_id": current_session_scope(),
    }

    PERSONAL_SCHEDULES.append(schedule)

    return _json(
        {
            "ok": True,
            "tool_name": "personal_create_schedule",
            "created_schedule": schedule,
        }
    )


@tool
def personal_list_schedules(date_from: str | None = None, date_to: str | None = None) -> str:
    """선택한 시작일과 종료일 범위에 포함되는 Nana의 개인 일정을 조회합니다."""

    schedules = _current_session_schedules()

    if date_from is not None:
        schedules = [
            schedule
            for schedule in schedules
            if schedule.get("date", "") >= date_from
        ]

    if date_to is not None:
        schedules = [
            schedule
            for schedule in schedules
            if schedule.get("date", "") <= date_to
        ]

    return _json(
        {
            "ok": True,
            "tool_name": "personal_list_schedules",
            "schedules": schedules,
        }
    )


@tool
def personal_delete_schedule(schedule_id: str) -> str:
    """일정 ID에 해당하는 개인 일정을 삭제합니다."""

    session_id = current_session_scope()
    before_count = len(PERSONAL_SCHEDULES)

    PERSONAL_SCHEDULES[:] = [
        schedule
        for schedule in PERSONAL_SCHEDULES
        if not (
            str(schedule.get("id")) == str(schedule_id)
            and _schedule_scope(schedule) == session_id
        )
    ]

    after_count = len(PERSONAL_SCHEDULES)
    deleted = before_count != after_count

    return _json(
        {
            "ok": True,
            "tool_name": "personal_delete_schedule",
            "schedule_id": schedule_id,
            "deleted": deleted,
        }
    )


def week01_tools() -> list[Any]:
    """1주차에서 직접 구현한 개인 일정 CRUD 도구 목록입니다."""

    return [personal_create_schedule, personal_list_schedules, personal_delete_schedule]


def week01_system_prompt() -> str:
    """1주차 단일 Nana agent가 따르는 시스템 프롬프트입니다."""

    return join_system_prompt(week01_prompt_parts())


def week01_prompt_parts() -> list[str]:
    """1주차부터 누적되는 system prompt 조각입니다."""

    today = current_app_date_iso()

    return [
        CHAT_MEMORY_PROMPT,
        f"""
오늘 날짜는 {today}이다.

너는 Kanana Schedule Agent의 Week 1 실습용 개인 일정 비서 Nana다.

사용자가 개인 일정을 만들어 달라고 하면 personal_create_schedule tool을 사용한다.
사용자가 개인 일정을 보여 달라고 하면 personal_list_schedules tool을 사용한다.
사용자가 개인 일정을 지워 달라고 하면 personal_delete_schedule tool을 사용한다.

일정 생성 시 필요한 값은 다음과 같다.
- title: 일정 제목
- date: YYYY-MM-DD 형식의 날짜
- start_time: HH:MM 형식의 시작 시간
- end_time: 종료 시간이 없으면 "미정"
- attendees: 참석자가 없으면 빈 리스트 []

사용자가 "내일", "다음 주", "오늘"처럼 상대 날짜를 말하면
오늘 날짜를 기준으로 가능한 한 YYYY-MM-DD 형식으로 바꾸어 tool에 전달한다.

조회할 때 사용자가 날짜 범위를 말하면 date_from, date_to를 채운다.
날짜 조건이 없으면 현재 대화 세션의 전체 개인 일정을 조회한다.

삭제할 때는 먼저 일정 목록을 확인해 삭제할 schedule_id를 찾고,
그 ID를 personal_delete_schedule에 전달한다.
삭제할 일정을 특정할 수 없으면 사용자에게 어떤 일정을 삭제할지 물어본다.

tool 실행 결과를 사용자에게 설명할 때는 JSON을 그대로 길게 읽지 말고,
일정 제목, 날짜, 시간 중심으로 자연스럽게 요약한다.
""",
    ]


def build_week01_agent() -> object:
    """Week 1 tool 목록만 노출하는 단일 LangChain agent를 만듭니다."""

    if not CONFIG.has_openai_key:
        raise RuntimeError("PROXY_TOKEN이 .env에 필요합니다.")
    global _WEEK01_AGENT
    if _WEEK01_AGENT is None:
        _WEEK01_AGENT = create_agent(
            model=chat_model(),
            tools=week01_tools(),
            system_prompt=week01_system_prompt(),
        )
    return _WEEK01_AGENT


def build_week_agent() -> object:
    """active-week registry가 호출하는 표준 Week agent builder입니다."""

    return build_week01_agent()


def list_personal_schedule_dicts(date_from: str | None = None, date_to: str | None = None) -> list[dict[str, Any]]:
    """개인 일정 dict 목록이 필요한 내부 코드에서 사용하는 비-도구 헬퍼입니다."""

    schedules = json.loads(personal_list_schedules.invoke({"date_from": date_from, "date_to": date_to}))
    return schedules["schedules"]


def ensure_demo_personal_schedule() -> None:
    if PERSONAL_SCHEDULES:
        return
    personal_create_schedule.invoke(
        {
            "title": "개인 집중 작업",
            "date": next_weekday_iso(2),
            "start_time": "09:00",
            "end_time": "10:00",
            "attendees": [],
        }
    )
