import json
from unittest.mock import patch
from student_parts.week06_kanamate_decides_schedule import find_common_available_slots_dict

FAKE_BUSY_ROWS = [
    {"member": "죠르디", "title": "회의", "start_time": "2026-08-06T10:00:00", "end_time": "2026-08-06T11:00:00"}
]

def test_find_slots_with_busy_rows():
    """busy_rows가 있을 때 '나'가 포함되어 정상 동작하는지 검증"""
    result = find_common_available_slots_dict(
        member_names=["죠르디"],
        date_from="2026-08-06",
        date_to="2026-08-06",
        busy_rows=FAKE_BUSY_ROWS
    )
    
    assert isinstance(result, dict)
    assert "나" in result["members"]
    assert "죠르디" in result["members"]

@patch("student_parts.week06_kanamate_decides_schedule.collect_member_schedules")
def test_find_slots_when_busy_rows_is_none(mock_tool):
    """busy_rows가 없을 때 스스로 데이터를 채우는지 검증"""
    
    mock_tool.invoke.return_value = json.dumps({"ok": True, "rows": FAKE_BUSY_ROWS})
    
    result = find_common_available_slots_dict(
        member_names=["죠르디"],
        date_from="2026-08-06",
        date_to="2026-08-06",
        busy_rows=None 
    )
    
    mock_tool.invoke.assert_called_once()
    assert isinstance(result, dict)
