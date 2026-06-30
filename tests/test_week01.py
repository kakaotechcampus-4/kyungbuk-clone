import json

import pytest

from student_parts.week01_wake_up_nana import (
    PERSONAL_SCHEDULES,
    personal_create_schedule,
    personal_delete_schedule,
    personal_list_schedules,
)


@pytest.fixture(autouse=True)
def clear_schedules():
    PERSONAL_SCHEDULES.clear()
    yield
    PERSONAL_SCHEDULES.clear()


def _create(title="테스트 회의", date="2026-07-01", start_time="10:00", **kwargs):
    return json.loads(personal_create_schedule.invoke({"title": title, "date": date, "start_time": start_time, **kwargs}))


def _list(**kwargs):
    return json.loads(personal_list_schedules.invoke(kwargs))


def _delete(schedule_id):
    return json.loads(personal_delete_schedule.invoke({"schedule_id": schedule_id}))


def test_create_returns_created_schedule():
    result = _create()
    assert result["ok"] is True
    assert "created_schedule" in result
    cs = result["created_schedule"]
    assert cs["title"] == "테스트 회의"
    assert cs["date"] == "2026-07-01"
    assert cs["start_time"] == "10:00"


def test_create_none_attendees_becomes_empty_list():
    result = _create()
    assert result["created_schedule"]["attendees"] == []


def test_create_appends_to_store():
    before = len(PERSONAL_SCHEDULES)
    _create()
    assert len(PERSONAL_SCHEDULES) == before + 1


def test_list_returns_current_session_only():
    _create(title="내 일정")
    PERSONAL_SCHEDULES.append({
        "id": "other_01",
        "title": "다른 세션 일정",
        "date": "2026-07-01",
        "start_time": "11:00",
        "session_id": "other_session",
    })
    result = _list()
    titles = [s["title"] for s in result["schedules"]]
    assert "내 일정" in titles
    assert "다른 세션 일정" not in titles


def test_list_date_from_filter():
    _create(title="이전 일정", date="2026-07-05")
    _create(title="이후 일정", date="2026-07-15")
    result = _list(date_from="2026-07-10")
    titles = [s["title"] for s in result["schedules"]]
    assert "이후 일정" in titles
    assert "이전 일정" not in titles


def test_list_date_to_filter():
    _create(title="이전 일정", date="2026-07-05")
    _create(title="이후 일정", date="2026-07-15")
    result = _list(date_to="2026-07-10")
    titles = [s["title"] for s in result["schedules"]]
    assert "이전 일정" in titles
    assert "이후 일정" not in titles


def test_list_does_not_mutate_store():
    _create()
    _create()
    before = len(PERSONAL_SCHEDULES)
    _list()
    assert len(PERSONAL_SCHEDULES) == before


def test_delete_removes_correct_schedule():
    created = _create()
    schedule_id = created["created_schedule"]["id"]
    result = _delete(schedule_id)
    assert result["deleted"] is True
    assert all(s["id"] != schedule_id for s in PERSONAL_SCHEDULES)


def test_delete_wrong_session_does_nothing():
    other_id = "personal_othersession01"
    PERSONAL_SCHEDULES.append({
        "id": other_id,
        "title": "다른 세션 일정",
        "date": "2026-07-01",
        "start_time": "10:00",
        "session_id": "other_session",
    })
    before = len(PERSONAL_SCHEDULES)
    result = _delete(other_id)
    assert result["deleted"] is False
    assert len(PERSONAL_SCHEDULES) == before


def test_delete_nonexistent_id():
    before = len(PERSONAL_SCHEDULES)
    result = _delete("personal_nonexistent")
    assert result["deleted"] is False
    assert len(PERSONAL_SCHEDULES) == before


# --- 리뷰 후 추가된 케이스 ---

def test_list_date_range_combined():
    _create(title="범위 이전", date="2026-07-01")
    _create(title="범위 내", date="2026-07-10")
    _create(title="범위 이후", date="2026-07-20")
    result = _list(date_from="2026-07-05", date_to="2026-07-15")
    titles = [s["title"] for s in result["schedules"]]
    assert "범위 내" in titles
    assert "범위 이전" not in titles
    assert "범위 이후" not in titles


def test_list_date_from_boundary_inclusive():
    _create(title="경계 당일", date="2026-07-10")
    result = _list(date_from="2026-07-10")
    titles = [s["title"] for s in result["schedules"]]
    assert "경계 당일" in titles


def test_list_date_to_boundary_inclusive():
    _create(title="경계 당일", date="2026-07-10")
    result = _list(date_to="2026-07-10")
    titles = [s["title"] for s in result["schedules"]]
    assert "경계 당일" in titles


def test_create_tool_name():
    result = _create()
    assert result["tool_name"] == "personal_create_schedule"


def test_list_tool_name():
    result = _list()
    assert result["tool_name"] == "personal_list_schedules"


def test_delete_tool_name():
    created = _create()
    schedule_id = created["created_schedule"]["id"]
    result = _delete(schedule_id)
    assert result["tool_name"] == "personal_delete_schedule"


def test_delete_then_list_e2e():
    created = _create()
    schedule_id = created["created_schedule"]["id"]
    _delete(schedule_id)
    result = _list()
    assert all(s["id"] != schedule_id for s in result["schedules"])


def test_create_id_has_personal_prefix():
    result = _create()
    assert result["created_schedule"]["id"].startswith("personal_")


def test_create_has_session_id():
    result = _create()
    assert "session_id" in result["created_schedule"]


def test_create_with_attendees():
    result = _create(attendees=["홍길동", "김철수"])
    assert result["created_schedule"]["attendees"] == ["홍길동", "김철수"]


def test_list_empty_store():
    result = _list()
    assert result["ok"] is True
    assert result["schedules"] == []
