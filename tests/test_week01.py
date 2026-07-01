import json
import pytest

from fixed.session_scope import conversation_session_scope
from student_parts.week01_wake_up_nana import (
    PERSONAL_SCHEDULES,
    personal_create_schedule,
    personal_delete_schedule,
    personal_list_schedules,
)


@pytest.fixture(autouse=True)
def clear_schedules():
    """각 테스트 전후로 PERSONAL_SCHEDULES를 비웁니다."""
    PERSONAL_SCHEDULES.clear()
    yield
    PERSONAL_SCHEDULES.clear()


# ── personal_create_schedule ──────────────────────────────────────────────────

def test_create_returns_ok():
    with conversation_session_scope("test-session"):
        result = json.loads(personal_create_schedule.invoke({
            "title": "팀 미팅",
            "date": "2026-07-01",
            "start_time": "10:00",
        }))
    assert result["ok"] is True


def test_create_payload_keys():
    with conversation_session_scope("test-session"):
        result = json.loads(personal_create_schedule.invoke({
            "title": "팀 미팅",
            "date": "2026-07-01",
            "start_time": "10:00",
        }))
    assert "tool_name" in result
    assert "created_schedule" in result


def test_create_schedule_fields():
    with conversation_session_scope("test-session"):
        result = json.loads(personal_create_schedule.invoke({
            "title": "팀 미팅",
            "date": "2026-07-01",
            "start_time": "10:00",
            "end_time": "11:00",
            "attendees": ["철수", "영희"],
        }))
    schedule = result["created_schedule"]
    assert schedule["title"] == "팀 미팅"
    assert schedule["date"] == "2026-07-01"
    assert schedule["start_time"] == "10:00"
    assert schedule["end_time"] == "11:00"
    assert schedule["attendees"] == ["철수", "영희"]


def test_create_attendees_none_becomes_empty_list():
    with conversation_session_scope("test-session"):
        result = json.loads(personal_create_schedule.invoke({
            "title": "혼자 작업",
            "date": "2026-07-01",
            "start_time": "09:00",
        }))
    assert result["created_schedule"]["attendees"] == []


def test_create_appends_to_store():
    with conversation_session_scope("test-session"):
        personal_create_schedule.invoke({
            "title": "팀 미팅",
            "date": "2026-07-01",
            "start_time": "10:00",
        })
    assert len(PERSONAL_SCHEDULES) == 1


# ── personal_list_schedules ───────────────────────────────────────────────────

def test_list_returns_created_schedule():
    with conversation_session_scope("test-session"):
        personal_create_schedule.invoke({
            "title": "팀 미팅",
            "date": "2026-07-01",
            "start_time": "10:00",
        })
        result = json.loads(personal_list_schedules.invoke({}))
    assert result["ok"] is True
    assert len(result["schedules"]) == 1
    assert result["schedules"][0]["title"] == "팀 미팅"


def test_list_date_from_filter():
    with conversation_session_scope("test-session"):
        personal_create_schedule.invoke({"title": "과거 일정", "date": "2026-06-01", "start_time": "10:00"})
        personal_create_schedule.invoke({"title": "미래 일정", "date": "2026-08-01", "start_time": "10:00"})
        result = json.loads(personal_list_schedules.invoke({"date_from": "2026-07-01"}))
    assert len(result["schedules"]) == 1
    assert result["schedules"][0]["title"] == "미래 일정"


def test_list_date_to_filter():
    with conversation_session_scope("test-session"):
        personal_create_schedule.invoke({"title": "과거 일정", "date": "2026-06-01", "start_time": "10:00"})
        personal_create_schedule.invoke({"title": "미래 일정", "date": "2026-08-01", "start_time": "10:00"})
        result = json.loads(personal_list_schedules.invoke({"date_to": "2026-07-01"}))
    assert len(result["schedules"]) == 1
    assert result["schedules"][0]["title"] == "과거 일정"


def test_list_isolates_other_sessions():
    with conversation_session_scope("session-A"):
        personal_create_schedule.invoke({"title": "A의 일정", "date": "2026-07-01", "start_time": "10:00"})
    with conversation_session_scope("session-B"):
        result = json.loads(personal_list_schedules.invoke({}))
    assert len(result["schedules"]) == 0


# ── personal_delete_schedule ──────────────────────────────────────────────────

def test_delete_removes_schedule():
    with conversation_session_scope("test-session"):
        created = json.loads(personal_create_schedule.invoke({
            "title": "지울 일정",
            "date": "2026-07-01",
            "start_time": "10:00",
        }))
        schedule_id = created["created_schedule"]["id"]
        result = json.loads(personal_delete_schedule.invoke({"schedule_id": schedule_id}))
    assert result["deleted"] is True
    assert len(PERSONAL_SCHEDULES) == 0


def test_delete_returns_false_for_missing_id():
    with conversation_session_scope("test-session"):
        result = json.loads(personal_delete_schedule.invoke({"schedule_id": "personal_없는id"}))
    assert result["deleted"] is False


def test_delete_does_not_cross_sessions():
    with conversation_session_scope("session-A"):
        created = json.loads(personal_create_schedule.invoke({
            "title": "A의 일정",
            "date": "2026-07-01",
            "start_time": "10:00",
        }))
        schedule_id = created["created_schedule"]["id"]
    with conversation_session_scope("session-B"):
        result = json.loads(personal_delete_schedule.invoke({"schedule_id": schedule_id}))
    assert result["deleted"] is False
    assert len(PERSONAL_SCHEDULES) == 1
