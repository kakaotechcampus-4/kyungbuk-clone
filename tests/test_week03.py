import json
from types import SimpleNamespace

import pytest

import student_parts.week03_build_nanas_logbook as w3
from fixed.runtime_clock import current_app_date_iso
from student_parts.week01_wake_up_nana import PERSONAL_SCHEDULES, week01_tools
from student_parts.week02_structure_natural_language_requests import week02_prompt_parts
from student_parts.week03_build_nanas_logbook import (
    SaveStructuredRequestInput,
    build_week03_agent,
    build_week_agent,
    get_saved_request,
    list_saved_requests,
    personal_list_saved_schedules,
    save_structured_request,
    week03_prompt_parts,
    week03_system_prompt,
    week03_tools,
)


# ============================================================
# 수동 E2E 테스트 체크리스트 (./run.sh --week3) — 메인과제만
# ============================================================
# 아래는 pytest가 검증하지 않는 "실제 LLM tool 호출 순서/품질"을 사람이 눈으로
# 확인하기 위한 시나리오입니다. Gradio 채팅창에 문장을 입력한 뒤 상세 trace 패널의
# events(tool_call/tool_result)를 함께 확인하세요. 확인했으면 [ ]를 [x]로 바꾸고
# 날짜/결과를 적어두세요.
#
# [x] 시나리오 1 — 자연어 저장 → 조회 → 영속성 (2026-07-15 확인 완료, 메인과제 필수 검증)
#     입력 1: "내일 10시 개인 코칭 저장해줘"
#     확인:
#       - trace events에 extract_schedule_request → save_structured_request 순서로 tool_call 발생
#       - save_structured_request 결과의 saved_rows에 structured_requests/schedules 두 row가 있음
#     입력 2 (같은 대화 또는 새 대화): "내 일정 보여줘"
#     확인:
#       - personal_list_saved_schedules tool_call 발생, 방금 저장한 일정이 schedules에 포함됨
#     확인 (영속성): 앱을 재시작하거나 새 대화를 열고 "내 일정 보여줘"를 다시 입력해도
#       같은 일정이 그대로 보임
#     결과: 정상 동작 확인. (build_week03_agent로 직접 돌려본 사전 스모크 테스트에서도
#     extract_schedule_request -> save_structured_request -> personal_list_saved_schedules
#     순서로 호출되고 저장된 일정이 그대로 재조회되는 것을 확인함.)
#
# [ ] 시나리오 2 (선택/보너스, 필수 아님) — 할 일(todo) 저장 후 list_saved_requests로 조회
#     이 파일의 메인과제 4개 tool 중 list_saved_requests/get_saved_request는 시나리오 1에서
#     검증되지 않는다. 이 둘의 실제 동작은 이미 pytest(test_list_saved_requests_filters_by_kind 등)로
#     직접 확인됐으므로, 아래는 "LLM이 자연어만으로 이 tool을 스스로 골라 부르는지"를 추가로 보고 싶을
#     때만 시도하는 선택 시나리오다. LLM이 personal_list_saved_schedules만 계속 시도하고
#     list_saved_requests를 안 부르더라도 실패가 아니다(프롬프트에 명시적 라우팅 규칙이 없음).
#     입력 1: "보고서 제출하는 할일 추가해줘"
#     확인: save_structured_request가 kind="todo"로 호출됨 (schedules에는 안 들어감)
#     입력 2: "내가 저장한 요청 기록들 보여줘"
#     확인 (되면 좋음, 안 돼도 무방): list_saved_requests tool_call 발생, 방금 저장한 요청이 rows에 포함됨
#     결과:
# ============================================================
#
# 참고: 일정 수정(personal_update_saved_schedule)/삭제(personal_delete_saved_schedules)/
# Week 1 호환 생성(personal_create_schedule)/레거시 payload 정규화는 추가과제이며
# 아직 TODO 스텁 상태입니다. week03_tools()에는 여전히 노출되어 있으니, LLM이 수정/삭제를
# 시도하면 정상 동작하지 않는 것이 현재는 정상입니다(추가과제 구현 전까지).


@pytest.fixture(autouse=True)
def _reset_week03_state(monkeypatch):
    monkeypatch.setattr("fixed.app_store.sync_personal_schedule_to_shared", lambda schedule: {"ok": True, "status": "stubbed"})
    monkeypatch.setattr("fixed.app_store.sync_group_schedule_to_shared", lambda schedule: {"ok": True, "status": "stubbed"})
    monkeypatch.setattr("fixed.app_store.delete_personal_schedule_from_shared", lambda request_id: {"ok": True, "deleted": []})
    monkeypatch.setattr("fixed.app_store.delete_group_schedule_from_shared", lambda schedule: {"ok": True, "deleted": []})
    PERSONAL_SCHEDULES.clear()
    w3._WEEK03_AGENT = None
    yield
    PERSONAL_SCHEDULES.clear()
    w3._WEEK03_AGENT = None


@pytest.fixture
def use_temp_config(tmp_path, monkeypatch):
    """@tool 함수가 내부에서 호출하는 _store()가 임시 SQLite 파일을 쓰도록 CONFIG를 교체합니다."""

    monkeypatch.setattr(
        w3,
        "CONFIG",
        SimpleNamespace(app_db_path=tmp_path / "tool_app.sqlite3", has_openai_key=True),
    )


# --- SaveStructuredRequestInput (스키마 자체는 메인과제 save_structured_request가 바로 사용) ---

def test_save_structured_request_input_inherits_structured_request_fields():
    save_input = SaveStructuredRequestInput(kind="todo", title="장보기")
    assert save_input.title == "장보기"
    assert save_input.members == []
    assert save_input.source_schedule_id is None


def test_save_structured_request_input_kind_defaults_to_unknown():
    assert SaveStructuredRequestInput().kind == "unknown"


# --- save_structured_request tool (메인과제) ---

def test_save_structured_request_tool_invoke_saves_to_sqlite(use_temp_config):
    raw = save_structured_request.invoke(
        {"kind": "personal_schedule", "title": "회의", "date": "2026-07-17", "start_time": "15:00"}
    )
    result = json.loads(raw)
    assert result["ok"] is True
    assert result["tool_name"] == "save_structured_request"
    assert any(row["table"] == "schedules" for row in result["saved_rows"])


def test_save_structured_request_tool_defaults_members_to_empty_list(use_temp_config):
    raw = save_structured_request.invoke({"kind": "todo", "title": "청소"})
    result = json.loads(raw)
    assert result["ok"] is True


# --- list_saved_requests / get_saved_request (메인과제) ---

def test_list_saved_requests_filters_by_kind(use_temp_config):
    save_structured_request.invoke({"kind": "todo", "title": "장보기"})
    save_structured_request.invoke({"kind": "reminder", "title": "약 먹기"})
    result = json.loads(list_saved_requests.invoke({"kind": "todo"}))
    assert len(result["rows"]) == 1
    assert result["rows"][0]["title"] == "장보기"


def test_get_saved_request_returns_row_for_existing_id(use_temp_config):
    saved = json.loads(save_structured_request.invoke({"kind": "todo", "title": "빨래"}))
    result = json.loads(get_saved_request.invoke({"request_id": saved["request_id"]}))
    assert result["row"]["title"] == "빨래"


def test_get_saved_request_returns_none_for_missing_id(use_temp_config):
    result = json.loads(get_saved_request.invoke({"request_id": "req_does_not_exist"}))
    assert result["row"] is None


# --- personal_list_saved_schedules (메인과제) ---

def test_personal_list_saved_schedules_defaults_kind_to_personal_schedule(use_temp_config):
    save_structured_request.invoke({"kind": "personal_schedule", "title": "코칭", "date": "2026-07-16", "start_time": "10:00"})
    save_structured_request.invoke(
        {"kind": "group_schedule", "title": "동아리", "date": "2026-07-18", "start_time": "15:00", "members": ["철수"]}
    )
    result = json.loads(personal_list_saved_schedules.invoke({}))
    assert result["filters"]["kind"] == "personal_schedule"
    assert len(result["schedules"]) == 1
    assert result["schedules"][0]["title"] == "코칭"


def test_personal_list_saved_schedules_respects_limit(use_temp_config):
    for i in range(3):
        save_structured_request.invoke(
            {"kind": "personal_schedule", "title": f"일정{i}", "date": "2026-07-16", "start_time": "10:00"}
        )
    result = json.loads(personal_list_saved_schedules.invoke({"limit": 2}))
    assert len(result["schedules"]) == 2


# --- week03_tools (공통, 이미 구현되어 있음) ---

def test_week03_tools_includes_main_task_sqlite_tools():
    names = {w3._tool_name(t) for t in week03_tools()}
    for expected in [
        "extract_schedule_request",
        "save_structured_request",
        "list_saved_requests",
        "get_saved_request",
        "personal_list_saved_schedules",
    ]:
        assert expected in names


# --- week03_prompt_parts / week03_system_prompt (공통, 메인과제 흐름 안내) ---

def test_week03_prompt_parts_extends_week02_prompt_parts():
    week02_parts = week02_prompt_parts()
    week03_parts = week03_prompt_parts()
    assert week03_parts[: len(week02_parts)] == week02_parts


def test_week03_prompt_parts_mentions_sqlite_and_tool_order():
    combined = "\n".join(week03_prompt_parts())
    assert "SQLite" in combined
    assert "extract_schedule_request" in combined
    assert "save_structured_request" in combined


def test_week03_prompt_parts_mentions_today_date():
    combined = "\n".join(week03_prompt_parts())
    assert current_app_date_iso() in combined


def test_week03_system_prompt_includes_prompt_parts_text():
    system_prompt = week03_system_prompt()
    for part in week03_prompt_parts():
        assert part in system_prompt


# --- build_week03_agent (공통, 이미 구현되어 있음) ---

def test_build_week03_agent_raises_without_proxy_token(monkeypatch):
    monkeypatch.setattr(w3, "CONFIG", SimpleNamespace(has_openai_key=False))
    with pytest.raises(RuntimeError, match="PROXY_TOKEN이 .env에 필요합니다."):
        build_week03_agent()


def test_build_week03_agent_returns_singleton_instance():
    first = build_week03_agent()
    second = build_week03_agent()
    assert first is second


def test_build_week_agent_delegates_to_week03():
    assert build_week_agent() is build_week03_agent()
