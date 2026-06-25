from __future__ import annotations

"""Week 6 그룹 일정 조율에서 쓰는 tool-description payload 함수 모음입니다.

가능 시간 후보와 최종 선택은 LangChain tool description을 읽은 LLM agent가 직접 고릅니다.
이 모듈은 LangChain tool을 직접 알지 않고, agent가 넘긴 payload를 검증하고 정규화합니다.
"""

from datetime import date, timedelta
from typing import Any

from pydantic import BaseModel, Field, ValidationError


class CommonSlotCandidate(BaseModel):
    """LLM이 제안하는 공통 가능 시간 후보입니다."""

    date: str = Field(description="YYYY-MM-DD 형식 날짜")
    start_time: str = Field(description="HH:MM 24시간 형식 시작 시간")
    end_time: str = Field(description="HH:MM 24시간 형식 종료 시간")
    duration_minutes: int = Field(default=60, description="회의 길이(분)")
    reason: str = Field(default="", description="이 시간이 적절하다고 판단한 짧은 근거")


def parse_time_minutes(value: str | None, fallback: int) -> int:
    """`HH:MM` 문자열을 자정 기준 분으로 변환합니다.

    값이 비어 있거나 `"미정"`이면 caller가 정한 fallback을 사용합니다.
    """

    if not value or value == "미정":
        return fallback
    try:
        hour_text, minute_text = value.split(":", 1)
        return int(hour_text) * 60 + int(minute_text)
    except (AttributeError, ValueError):
        return fallback


def format_time_minutes(minutes: int) -> str:
    """자정 기준 분 값을 `HH:MM` 문자열로 바꿉니다."""

    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def normalize_date_bound(value: str) -> str:
    """ISO datetime 또는 문자열에서 날짜 부분만 남깁니다."""

    return str(value).split("T", 1)[0].strip()


def date_range(date_from: str, date_to: str) -> list[str]:
    """양 끝 날짜를 포함하는 YYYY-MM-DD 목록을 반환합니다.

    범위가 거꾸로 들어와도 start/end를 바꿔 안전하게 계산합니다.
    """

    start = date.fromisoformat(normalize_date_bound(date_from))
    end = date.fromisoformat(normalize_date_bound(date_to))
    if end < start:
        start, end = end, start
    days: list[str] = []
    current = start
    while current <= end:
        days.append(current.isoformat())
        current += timedelta(days=1)
    return days


def busy_rows_overlap(rows: list[dict[str, Any]], day: str, start_minutes: int, end_minutes: int) -> list[dict[str, Any]]:
    """후보 시간과 겹치는 busy row 목록을 찾습니다."""

    blockers: list[dict[str, Any]] = []
    for row in rows:
        if row.get("date") != day:
            continue
        busy_start = parse_time_minutes(row.get("start_time"), 0)
        busy_end = parse_time_minutes(row.get("end_time"), 24 * 60)
        if start_minutes < busy_end and busy_start < end_minutes:
            blockers.append(row)
    return blockers


def normalize_llm_candidate_slots(
    *,
    candidate_slots: list[Any] | None,
    llm_reason: str | None = None,
    date_from: str,
    date_to: str,
    busy_rows: list[dict[str, Any]],
    duration_minutes: int,
    workday_start: str,
    workday_end: str,
    limit: int,
) -> list[dict[str, Any]]:
    """LLM 후보를 앱 payload에 맞게 정리하고 명백히 불가능한 후보만 제외합니다."""

    valid_days = set(date_range(date_from, date_to))
    work_start = parse_time_minutes(workday_start, 9 * 60)
    work_end = parse_time_minutes(workday_end, 18 * 60)
    requested_duration = max(30, int(duration_minutes or 60))

    slots: list[dict[str, Any]] = []
    for candidate in candidate_slots or []:
        try:
            if isinstance(candidate, CommonSlotCandidate):
                slot = candidate.model_dump()
            elif hasattr(candidate, "model_dump"):
                slot = CommonSlotCandidate.model_validate(candidate.model_dump()).model_dump()
            else:
                slot = CommonSlotCandidate.model_validate(candidate).model_dump()
        except (TypeError, ValidationError, ValueError):
            continue

        day = normalize_date_bound(slot.get("date", ""))
        start_minutes = parse_time_minutes(slot.get("start_time"), -1)
        end_minutes = parse_time_minutes(slot.get("end_time"), -1)
        if day not in valid_days:
            continue
        if start_minutes < work_start or end_minutes > work_end or end_minutes <= start_minutes:
            continue
        if end_minutes - start_minutes < requested_duration:
            continue
        if busy_rows_overlap(busy_rows, day, start_minutes, end_minutes):
            continue

        slots.append(
            {
                "date": day,
                "start_time": format_time_minutes(start_minutes),
                "end_time": format_time_minutes(end_minutes),
                "duration_minutes": end_minutes - start_minutes,
                "reason": slot.get("reason") or llm_reason or "LLM이 tool description 계약에 따라 고른 공통 가능 시간입니다.",
            }
        )
        if len(slots) >= limit:
            break
    return slots


def slot_to_text(slot: Any) -> str:
    """후보 slot dict 또는 문자열을 사용자 답변용 시간 문자열로 바꿉니다."""

    if isinstance(slot, str):
        return slot
    if not isinstance(slot, dict):
        return str(slot)
    date_text = slot.get("date") or "날짜 미정"
    start_time = slot.get("start_time") or "시간 미정"
    end_time = slot.get("end_time")
    return f"{date_text} {start_time}-{end_time}" if end_time else f"{date_text} {start_time}"


def find_common_available_slots_payload(
    *,
    member_names: list[str],
    date_from: str,
    date_to: str,
    busy_rows: list[dict[str, Any]],
    duration_minutes: int = 60,
    workday_start: str = "09:00",
    workday_end: str = "18:00",
    limit: int = 5,
    candidate_slots: list[Any] | None = None,
    llm_reason: str | None = None,
) -> dict[str, Any]:
    """LLM이 tool 인자로 넘긴 공통 가능 시간 후보 payload를 검증해 기록합니다."""

    normalized_candidate_slots = normalize_llm_candidate_slots(
        candidate_slots=candidate_slots,
        llm_reason=llm_reason,
        date_from=date_from,
        date_to=date_to,
        busy_rows=busy_rows,
        duration_minutes=duration_minutes,
        workday_start=workday_start,
        workday_end=workday_end,
        limit=limit,
    )
    return {
        "ok": True,
        "tool_name": "find_common_available_slots",
        "members": member_names,
        "busy_rows": busy_rows,
        "candidate_slots": normalized_candidate_slots,
        "slot_source": "llm",
        "payload_source": "tool_description",
        "llm_reason": llm_reason or "",
    }


def decide_final_slot_payload(
    *,
    candidate_slots: list[Any] | None = None,
    selected_slot: Any | None = None,
    selected_index: int | None = None,
    member_names: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    duration_minutes: int = 60,
    final_slot: str | None = None,
    needs_agent_selection: bool | None = None,
    reason: str | None = None,
    busy_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """LLM이 tool 인자로 넘긴 최종 회의 시간 결정을 payload로 기록합니다."""

    slots = list(candidate_slots or [])
    candidates = [slot_to_text(slot) for slot in slots]
    selected = selected_slot
    invalid_selection = False
    if selected is None and selected_index is not None:
        try:
            index = int(selected_index)
        except (TypeError, ValueError):
            invalid_selection = True
        else:
            if 0 <= index < len(slots):
                selected = slots[index]
            else:
                invalid_selection = True

    resolved_final_slot = final_slot or (slot_to_text(selected) if selected is not None else None)
    if reason:
        final_reason = reason
    elif invalid_selection:
        final_reason = "선택한 후보 번호가 후보 목록 범위를 벗어났습니다."
    elif isinstance(selected, dict) and selected.get("reason"):
        final_reason = str(selected["reason"])
    elif resolved_final_slot:
        final_reason = "LLM이 tool description 계약에 따라 후보를 최종 시간으로 선택했습니다."
    elif candidates:
        final_reason = "후보는 전달됐지만 final_slot 또는 selected_index가 없어 최종 확정하지 않았습니다."
    else:
        final_reason = "공통 가능 시간을 찾지 못했습니다."

    if needs_agent_selection is None:
        needs_agent_selection = resolved_final_slot is None

    payload: dict[str, Any] = {
        "final_slot": resolved_final_slot,
        "reason": final_reason,
        "candidates": candidates,
        "needs_agent_selection": needs_agent_selection,
    }
    if selected_index is not None:
        payload["selected_index"] = selected_index
    if selected is not None:
        payload["selected_slot"] = selected
    if member_names is not None:
        payload["members"] = member_names
    if date_from is not None:
        payload["date_from"] = normalize_date_bound(date_from)
    if date_to is not None:
        payload["date_to"] = normalize_date_bound(date_to)
    if busy_rows is not None:
        payload["busy_rows"] = busy_rows
    if slots and any(isinstance(slot, dict) for slot in slots):
        payload["candidate_slots"] = slots
    return payload
