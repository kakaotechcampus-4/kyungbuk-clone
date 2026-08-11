# Kanana Schedule Agent — CLAUDE.md

## 프로젝트 목적

카카오 카테캠 강의용 LangChain 실습 프로젝트입니다.
학생이 `student_parts/` 안의 `# TODO` 함수를 직접 구현하며 LangChain tool과
에이전트 실행 흐름을 배우는 것이 목표입니다.

## 구현 범위

- **구현 대상 (활성 주차)**: `student_parts/week06_kanamate_decides_schedule.py` 안의 `# TODO` 함수들
- **읽기 전용**: `fixed/` 디렉터리(구조 이해용 참고 코드 — 특히 SQLite 접근은 `fixed/app_store.py`의
  `AppSQLiteStore` 메서드로만 하고, 개인 참고자료 벡터 검색은 `fixed/reference_store.py`의
  `PersonalReferenceStore` 메서드로만 하고, 대화 RAG는 `fixed/conversation_rag_store.py`의
  `ConversationRAGStore` 메서드로만 하고, 외부 멤버 대화/공유 일정은 `fixed/mcp_client.py`의
  `call_local_mcp_tool_sync`(이 파일 별칭 `call_mcp_tool_sync`)와 `fixed/external_mcp.py`의
  `call_external_tool_payload`로만 합니다 — 직접 SQL이나 ChromaDB 호출, MCP 서버
  (`mcp_server/sqlite_mcp_server.py`) 수정을 새로 하지 않습니다, 공통 가능 시간 겹침 검증/최종
  확정 payload 생성은 `fixed/schedule_decision.py`의 `find_common_available_slots_payload()` /
  `decide_final_slot_payload()` / `normalize_date_bound()`로만 하고, agent 실행 결과에서 trace를
  뽑는 것은 `fixed/langchain_trace.py`의 `extract_agent_events()` / `extract_final_text()`로만
  합니다),
  `student_parts_baseline/`(이전 주차 정답 코드 — 참고용이며 그대로 복사해 붙이지 않습니다)
- **수정 금지**: `app.py`, `run.sh`, `pyproject.toml`, `uv.lock`

## 완료된 주차

- **Week 1** (`student_parts/week01_wake_up_nana.py`): 임시 메모리(`PERSONAL_SCHEDULES`) CRUD — 완료
- **Week 2** (`student_parts/week02_structure_natural_language_requests.py`): 자연어 →
  `StructuredRequest` 구조화 — 완료
- **Week 3** (`student_parts/week03_build_nanas_logbook.py`): `StructuredRequest`를 SQLite에
  저장/조회/수정/삭제 — 완료
- **Week 4** (`student_parts/week04_retrieve_nanas_memory.py`): 개인 참고자료/SQLite 기록/대화
  RAG 검색 tool 분리 — 완료
- **Week 5** (`student_parts/week05_load_kanas_past_conversations.py`): 외부 대화/공유 일정
  MCP wrapper tool + `collect_member_schedules` busy-time 수집 — 메인·추가과제 모두 완료.
  Week 6에서 `search_previous_conversations` / `extract_schedules_from_history` /
  `list_shared_schedules` / `collect_member_schedules` / `load_conversation_messages`를 그대로
  import해 재사용합니다.

## Week 6 도움 방식 (힌트 우선, 단계별, 메인과제 → 추가과제 순)

`week06_kanamate_decides_schedule.py` 안에 이미 `[6주차 수강생 구현 가이드]` 주석(목표·과제
구성·메인/추가 티어·함수별 설명)이 있습니다. 정답을 알려주지 말고, 아래 순서대로 하나씩 힌트를
먼저 주고 학생이 시도한 뒤 피드백합니다. 메인과제를 먼저 끝낸 뒤에만 추가과제로 넘어갑니다.

### 메인과제 (권장 순서)

**1. `week06_prompt_parts` / `nana_prompt_parts` / `kana_prompt_parts` / `supervisor_system_prompt`**
— 이후 모든 agent 동작을 좌우하는 뼈대
- supervisor는 직접 업무를 처리하지 않고 반드시 `nana_agent` 또는 `kana_agent` 중 하나로
  위임해야 한다는 점, 어떤 요청이 Nana(개인 일정/저장/RAG) 담당이고 어떤 요청이 Kana(외부 멤버
  일정/공통 시간 조율) 담당인지 판단 기준을 prompt에 적어야 하는 이유를 설명합니다.
- `week06_prompt_parts`는 `week05_prompt_parts()`를, `nana_prompt_parts`는
  `week04_prompt_parts()`를 누적하지만 `kana_prompt_parts`는 누적 없이 시작하므로 Kana 역할을
  처음부터 다 써야 하는 이유를 설명합니다.
- 하위 에이전트는 supervisor prompt를 공유하지 않으므로 각자 필요한 지시(위임 대상이 아닌
  요청은 짧게 알리기 등)를 스스로 갖고 있어야 한다는 점을 짚어줍니다.

**2. `nana_agent`**
- `_NANA_SUBAGENT`가 None일 때만 `create_agent(model=chat_model(), tools=week04_tools(),
  system_prompt=nana_system_prompt())`로 만들고 이후에는 재사용해야 하는 이유(매 호출마다 새로
  만들면 비용/일관성 문제)를 설명합니다.
- query를 user 메시지로 invoke한 뒤 `extract_agent_events(...)`/`extract_final_text(...)`로
  trace와 answer를 뽑고, answer/trace/inner_tool_names를 JSON으로 반환하도록 안내합니다.

**3. `kana_agent` — 가장 복잡, 마지막**
- `_KANA_SUBAGENT`를 `kana_tools()`와 `kana_system_prompt()`로 한 번만 만들고 재사용하도록
  안내합니다.
- 하위 trace의 event content를 훑어 `final_slot`이 들어 있는 dict를 `final_slot_payload`로,
  `final_decision` 값을 `final_decision_payload`로 끌어올려야 하는 이유(추가과제 tool을 아직
  안 붙였으면 자연히 `None`으로 남는다는 점)를 설명합니다.
- answer, trace, inner_tool_names, final_slot_payload, final_decision_payload를 JSON으로
  반환하도록 안내합니다.

### 추가과제 (메인과제 완료 후)

**1. `FIND_COMMON_AVAILABLE_SLOTS_DESCRIPTION` / `DECIDE_FINAL_SLOT_DESCRIPTION`**
- 이 Python tool들이 후보나 최종 시간을 대신 계산해주지 않는다는 점, agent가 `busy_rows`를
  읽고 `candidate_slots`/`selected_index`/`final_slot`을 직접 골라 argument로 넘겨야 한다는
  계약을 description에 명시해야 하는 이유를 설명합니다. 이 계약이 description에 없으면 agent가
  tool에 계산을 떠넘긴다는 점도 짚어줍니다.
- `candidate_slots` 항목 형식(date, start_time, end_time, duration_minutes, reason)과
  `final_slot` 형식(`'YYYY-MM-DD HH:MM-HH:MM'`)을 description에 명시해야 하는 이유를 설명합니다.

**2. `find_common_available_slots_dict` + `find_common_available_slots` + `decide_final_slot`**
1. `find_common_available_slots_dict`가 `normalize_external_member_names()`로 멤버 이름을,
   `normalize_date_bound()`로 날짜를 정규화하고, `busy_rows`가 `None`이면
   `collect_member_schedules.invoke({...})`를 호출해 rows를 채워야 하는 이유를 설명합니다. 내
   일정도 근거이므로 `member_names`에 `"나"`를 함께 포함해야 한다는 점도 짚어줍니다.
2. 실제 겹침 검증/payload 정리는 `find_common_available_slots_payload(...)`에 맡기고, 이 함수가
   Python 룰이나 nested LLM으로 candidate를 새로 계산하지 않는다는 원칙을 설명합니다.
3. `decide_final_slot`도 nested LLM 없이 Kana agent가 넘긴 `final_slot`/`selected_index`/
   `needs_agent_selection`/`reason`을 그대로 `decide_final_slot_payload(...)`에 넘겨 course
   repo JSON 계약(top-level `final_slot`/`reason`/`candidates`)에 맞춰 기록해야 하는 이유를
   설명합니다. `selected_index`나 `selected_slot`이 없으면 `final_slot`을 자동으로 고르지 말고
   `needs_agent_selection=True` 상태를 유지해야 한다는 점도 짚어줍니다.

## Week 6 핵심 제약 (구현 시 반드시 지킬 것)

- supervisor가 볼 수 있는 tool은 `supervisor_tools()`의 `nana_agent`/`kana_agent` 두 개뿐입니다.
  supervisor에 다른 tool을 새로 노출하지 않습니다.
- Nana 하위 agent의 tool은 `week04_tools()`를 그대로 재사용합니다. 새로 만들지 않습니다.
- Kana 하위 agent의 tool은 `kana_tools()`에서 Week 2/Week 5 tool과
  `find_common_available_slots`/`decide_final_slot`을 조립합니다. 추가과제를 구현하지 않기로
  하면 `kana_tools()` 목록과 `kana_prompt_parts()`에서 두 tool 언급을 함께 지웁니다.
- 공통 가능 시간 겹침 검증/최종 확정 payload 생성은 `fixed/schedule_decision.py`의
  `find_common_available_slots_payload()` / `decide_final_slot_payload()` /
  `normalize_date_bound()`로만 합니다. 겹침 계산 로직을 이 파일에 새로 작성하지 않습니다.
- `find_common_available_slots`/`decide_final_slot`은 Python이 후보나 최종 시간을 자동으로
  고르지 않습니다. Kana agent가 tool description을 읽고 직접 고른 값을 argument로 넘겨야
  합니다.
- trace 정리는 `fixed/langchain_trace.py`의 `extract_agent_events()` / `extract_final_text()`로
  만 합니다.
- `_NANA_SUBAGENT` / `_KANA_SUBAGENT` / `_SUPERVISOR_AGENT`는 한 번만 만들고 재사용합니다 (매
  호출마다 새로 만들지 않습니다).
- Week 1~5 구현은 다시 작성하지 않고 import해서 역할별로 재조립합니다.
- `propose_group_schedule`은 이전 실습 흐름과의 호환을 위해 이미 구현되어 있고 `kana_tools()`에
  포함되지 않습니다. 새로 손대지 않습니다.

## 검증 방법

앱을 실행하고 상세 Trace 탭에서 확인합니다.

```bash
./run.sh --week6
```

자동화 테스트 없음 — Trace JSON이 기대한 키와 값을 가지는지 눈으로 확인합니다.

메인과제 확인 순서:
1. 개인 일정 요청을 입력 → supervisor trace에서 `nana_agent`가 선택되고, Nana 하위 trace에
   `personal_list_saved_schedules` 등 Week 4 tool 호출이 남는지 확인합니다.
2. 외부 멤버/그룹 조율 요청을 입력 → supervisor trace에서 `kana_agent`가 선택되는지 확인합니다.
3. 위임이 엉뚱한 agent로 가면 tool 구현이 아니라 prompt의 판단 기준을 먼저 고칩니다.
4. 추가과제를 아직 구현하지 않았다면 `kana_tools()`에서 `find_common_available_slots`와
   `decide_final_slot`을 빼고 Kana prompt에서도 두 tool 언급을 지운 뒤 위임 흐름만 확인합니다.

추가과제 확인 순서: 그룹 일정 조율 요청에서 하위 trace에 `search_previous_conversations` →
`extract_schedules_from_history` 또는 `collect_member_schedules` → `find_common_available_slots`
→ `decide_final_slot`이 순서대로 이어지고, `final_slot_payload`가 최종 답변과 일치하는지
확인합니다.

## 주차 경계

Week 6은 "한 agent가 모두 처리"하던 구조를 supervisor + Nana/Kana 하위 agent로 나누고, 여러
사람의 공통 가능 시간 계산·최종 확정까지 다룹니다. Week 1~5 구현은 다시 작성하지 않고 import해서
그대로 재사용합니다. Week 7 이후 개념을 Week 6 코드에 미리 추가하지 않습니다.
`mcp_server/sqlite_mcp_server.py`의 `@mcp.tool` 구현과 `fixed/schedule_decision.py`의 겹침
검증 로직은 학생 구현 대상이 아니므로 직접 수정하지 않습니다.
