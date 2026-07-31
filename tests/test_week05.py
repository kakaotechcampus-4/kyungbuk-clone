import json
from unittest.mock import patch

from student_parts.week05_load_kanas_past_conversations import _collect_member_schedules


def test_collect_member_schedules_comprehensive():
    """Week 5 핵심 로직인 _collect_member_schedules의 전반적인 기능을 검증합니다.
    - 날짜 필터링 정상 작동 여부
    - 시간 데이터 부재 시 None 유지 여부
    - 외부 데이터와의 병합 구조 및 메타데이터(ok, tool_name 등) 포함 여부
    """

    # 1. 가짜 내 일정 준비 (정상 일정, 범위 밖 일정, 시간 없는 일정)
    fake_personal_schedules = [
        {
            "kind": "personal_schedule",
            "title": "정상적인 내 일정",
            "date": "2024-11-20",
            "start_time": "10:00",
            "end_time": "11:00",
        },
        {
            "kind": "personal_schedule",
            "title": "범위 밖 일정 (필터링되어야 함)",
            "date": "2024-11-25",
            "start_time": "10:00",
            "end_time": "11:00",
        },
        {
            "kind": "personal_schedule",
            "title": "시간 미정 일정",
            "date": "2024-11-20",
            # start_time, end_time 필드 없음
        },
    ]

    # 2. 가짜 MCP 외부 응답 준비
    fake_external_response = {
        "rows": [
            {
                "member_name": "팀원A",
                "title": "팀 회의",
                "date": "2024-11-20",
                "start_time": "14:00",
                "end_time": "15:00",
                "notes": "회의록 지참",
            }
        ]
    }

    # call_mcp_tool_sync 함수를 가로채서(Mocking) 가짜 응답을 반환하도록 설정
    with patch(
        "student_parts.week05_load_kanas_past_conversations.call_mcp_tool_sync"
    ) as mock_call:
        mock_call.return_value = json.dumps(fake_external_response)

        # 함수 실행 (검색 범위: 11월 19일 ~ 11월 21일)
        result = _collect_member_schedules(
            member_names=["팀원A", "팀원a"],  # 중복/대소문자 정규화 테스트
            date_from="2024-11-19",
            date_to="2024-11-21",
            personal_schedules=fake_personal_schedules,
        )

        # 3. 메타데이터 구조 검증
        assert result["ok"] is True
        assert result["tool_name"] == "collect_member_schedules"
        assert "schedule_summary" in result

        # 중복/대소문자 처리되어 멤버가 ['나', '팀원A'] 이어야 함
        assert "팀원A" in result["members"]
        assert "나" in result["members"]

        # 4. 데이터 병합 및 필터링 검증
        # 범위 밖 일정(11월 25일)은 제거되고, 내 일정 2개 + 외부 일정 1개 = 총 3개여야 함
        assert len(result["rows"]) == 3

        my_schedules = [row for row in result["rows"] if row["member_name"] == "나"]
        assert len(my_schedules) == 2

        # 시간 미정 데이터가 None으로 잘 유지되었는지 검증
        empty_time_schedule = next(s for s in my_schedules if s["title"] == "시간 미정 일정")
        assert empty_time_schedule["start_time"] is None
        assert empty_time_schedule["end_time"] is None

        # 외부 일정이 정상적으로 합쳐졌는지 검증
        external_schedules = [row for row in result["rows"] if row["member_name"] == "팀원A"]
        assert len(external_schedules) == 1
        assert external_schedules[0]["start_time"] == "14:00"
