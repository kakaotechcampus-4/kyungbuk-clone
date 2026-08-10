from __future__ import annotations

import pytest

from fixed.config import CONFIG
from fixed.week_agent_registry import run_active_week_agent

pytestmark = pytest.mark.skipif(
    not CONFIG.has_openai_key, reason="PROXY_TOKEN이 .env에 없어 실제 LLM 호출 테스트를 건너뜁니다."
)

# 개인 일정 질문을 supervisor가 nana_agent로 위임하는지 체크
def test_personal_schedule_query_delegates_to_nana():
    result = run_active_week_agent(6, [{"role": "user", "content": "내 일정 보여줘"}])
    assert result.trace["supervisor_selected_agent"] == "nana_agent"

# 외부 멤버 일정 질문을 superviosr가 kana_agent로 위임하고 extract_schedule툴 사용하는지 확인
def test_external_member_query_delegates_to_kana():
    result = run_active_week_agent(
        6, [{"role": "user", "content": "철수 7월 7일부터 7월 17일까지 일정 알려줘"}]
    )
    assert result.trace["supervisor_selected_agent"] == "kana_agent"
    assert "extract_schedules_from_history" in result.trace["inner_tool_names"]

# 회의 시간 선호도 질문은 nana_agent로 위임되고 search_personal_references를 호출하는지 확인
def test_meeting_time_preference_query_uses_personal_references():
    result = run_active_week_agent(
        6, [{"role": "user", "content": "나는 회의 시간을 언제쯤으로 잡는 걸 선호했었지?"}]
    )
    assert result.trace["supervisor_selected_agent"] == "nana_agent"
    assert "search_personal_references" in result.trace["inner_tool_names"]

# 공유 일정 저장소 조회 질문은 kana_agent로 위임되고 list_shared_schedules를 호출하는지 확인
def test_shared_schedule_store_query_uses_list_shared_schedules():
    result = run_active_week_agent(
        6, [{"role": "user", "content": "공유 일정 저장소에 등록된 일정 목록 보여줘"}]
    )
    assert result.trace["supervisor_selected_agent"] == "kana_agent"
    assert "list_shared_schedules" in result.trace["inner_tool_names"]
