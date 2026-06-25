from __future__ import annotations

"""앱 내부 일정과 외부 공유 일정 MCP 저장소를 동기화하는 헬퍼입니다.

AppSQLiteStore는 개인 일정을 앱 DB에 저장하지만, Week 5/6의 그룹 조율은 외부 멤버
일정 MCP 도구를 통해 busy time을 모읍니다. 그래서 개인 일정 저장/수정/삭제 시
이 모듈을 통해 외부 공유 저장소의 "나" 일정 복사본도 맞춰 둡니다.
"""

import json
from typing import Any

from fixed.mcp_client import call_local_mcp_tool_sync


PERSONAL_SHARED_MEMBER_NAME = "나"


def call_external_tool_payload(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    """외부 SQLite MCP tool을 동기로 호출하고 JSON 문자열 결과를 dict로 파싱합니다."""

    payload_text = call_local_mcp_tool_sync(tool_name, args)
    return json.loads(payload_text)


def sync_personal_schedule_to_shared(schedule: dict[str, Any]) -> dict[str, Any]:
    """앱 DB의 개인 일정 하나를 외부 공유 일정 저장소에 생성/갱신합니다.

    `source_conversation_id`에는 앱 request_id를 넣어 두어 나중에 삭제할 때 같은 복사본을
    찾을 수 있게 합니다. 외부 MCP 호출 실패는 예외로 전파하지 않고 실패 payload로 반환해
    앱 DB 저장 자체가 깨지지 않게 합니다.
    """

    if not schedule.get("date"):
        return {
            "ok": False,
            "status": "skipped",
            "reason": "공유 일정 등록에는 날짜가 필요합니다.",
        }
    try:
        payload = call_external_tool_payload(
            "create_shared_schedule",
            {
                "member_name": PERSONAL_SHARED_MEMBER_NAME,
                "title": schedule.get("title") or "제목 없음",
                "date": schedule.get("date"),
                "start_time": schedule.get("start_time") or "미정",
                "end_time": schedule.get("end_time") or "미정",
                "notes": "앱 개인 일정 자동 동기화",
                "source_conversation_id": f"app:{schedule['request_id']}",
                "schedule_id": f"shared_{schedule['schedule_id']}",
            },
        )
        shared = payload.get("shared_schedule", {})
        return {
            "ok": True,
            "status": shared.get("sync_status", "synced"),
            "tool_name": "create_shared_schedule",
            "shared_schedule": shared,
        }
    except Exception as exc:
        return {
            "ok": False,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def sync_group_schedule_to_shared(schedule: dict[str, Any]) -> dict[str, Any]:
    """앱 DB의 확정 그룹 일정을 참석자별 공유 busy-time row로 동기화합니다."""

    if not schedule.get("date"):
        return {
            "ok": False,
            "status": "skipped",
            "reason": "공유 일정 등록에는 날짜가 필요합니다.",
        }

    members = [str(member).strip() for member in (schedule.get("attendees") or []) if str(member).strip()]
    if not members:
        return {
            "ok": True,
            "status": "skipped",
            "reason": "공유할 참석자가 없습니다.",
            "shared_schedules": [],
        }

    shared_schedules: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    attendee_text = ", ".join(members)
    for index, member_name in enumerate(members):
        try:
            payload = call_external_tool_payload(
                "create_shared_schedule",
                {
                    "member_name": member_name,
                    "title": schedule.get("title") or "제목 없음",
                    "date": schedule.get("date"),
                    "start_time": schedule.get("start_time") or "미정",
                    "end_time": schedule.get("end_time") or "미정",
                    "notes": f"앱 그룹 일정 자동 동기화 · 참석자: {attendee_text}",
                    "source_conversation_id": f"group:{schedule['request_id']}:{member_name}",
                    "schedule_id": f"shared_{schedule['schedule_id']}_{index}",
                },
            )
            shared_schedules.append(payload.get("shared_schedule", {}))
        except Exception as exc:
            errors.append(
                {
                    "member_name": member_name,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    return {
        "ok": not errors,
        "status": "failed" if errors else "synced",
        "tool_name": "create_shared_schedule",
        "shared_schedules": shared_schedules,
        "errors": errors,
    }


def delete_personal_schedule_from_shared(request_id: str) -> dict[str, Any]:
    """앱 request_id에 연결된 외부 공유 일정 복사본을 삭제합니다."""

    try:
        payload = call_external_tool_payload(
            "delete_shared_schedule",
            {"source_conversation_id": f"app:{request_id}"},
        )
        return {
            "ok": True,
            "tool_name": "delete_shared_schedule",
            "deleted": payload.get("deleted", []),
        }
    except Exception as exc:
        return {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def delete_group_schedule_from_shared(schedule: dict[str, Any]) -> dict[str, Any]:
    """앱 그룹 일정 request_id와 참석자 기준으로 공유 일정 복사본을 삭제합니다."""

    request_id = str(schedule.get("request_id") or "").strip()
    if not request_id:
        return {"ok": True, "tool_name": "delete_shared_schedule", "deleted": []}

    deleted: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    members = [str(member).strip() for member in (schedule.get("attendees") or []) if str(member).strip()]
    for member_name in members:
        try:
            payload = call_external_tool_payload(
                "delete_shared_schedule",
                {"source_conversation_id": f"group:{request_id}:{member_name}"},
            )
            deleted.extend(payload.get("deleted", []))
        except Exception as exc:
            errors.append(
                {
                    "member_name": member_name,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    return {
        "ok": not errors,
        "tool_name": "delete_shared_schedule",
        "deleted": deleted,
        "errors": errors,
    }
