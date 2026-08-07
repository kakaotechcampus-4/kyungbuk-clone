# 6주차. 카나메이트가 약속을 결정하다

**부제:** Supervisor/Sub-agent 최종 일정 결정

## 학습 목표

- `nana_agent`, `kana_agent`, supervisor의 책임을 구분한다.
- supervisor가 업무 tool을 직접 호출하지 않고 delegate tool만 호출하는 구조를 설명한다.
- 팀원 A/B/C의 이전 일정과 개인 일정을 종합해 최종 회의 시간을 결정하는 trace를 검증한다.

## 핵심 개념

6주차는 최종 일정 결정 구조다. `nana_agent`는 각 대화에서 내 개인 일정 생성/조회/삭제, 구조화 저장, 개인 RAG 검색을 담당한다. `kana_agent`는 MCP SQLite에서 다른 사람들의 일정과 이전 대화 기록을 불러와 모든 개인 일정을 종합한다.

Supervisor는 직접 업무 tool을 호출하지 않는다. `nana_agent`, `kana_agent` delegate tool만 호출하고, 각 sub-agent의 내부 trace를 근거로 최종 답변을 만든다.

## 실습 흐름

1. `notebook/6주차_카나메이트가_약속을_결정하다.ipynb`에서 최종 golden case를 확인한다.
2. Nana payload에서 내 개인 일정 조회 trace를 확인한다.
3. Kana payload에서 MCP SQLite 이전 대화 검색과 후보 비교 trace를 확인한다.
4. Supervisor trace가 `nana_agent`, `kana_agent` delegate tool만 호출하는지 본다.
5. “팀원 A/B/C와 다음 주 회의 시간을 잡아줘” 요청에서 가능한 시간 후보와 최종 선택 이유를 검증한다.

## 관찰할 trace/payload

- supervisor delegate tool call
- `nana_agent.trace`
- `kana_agent.trace`
- `search_previous_conversations`
- `extract_schedules_from_history`
- `decide_final_slot`
- 최종 `final_slot`과 `reason`

## 확인 질문

1. Supervisor가 직접 개인 일정 tool이나 MCP 검색 tool을 호출하지 않는 이유는 무엇인가?
2. Nana와 Kana의 책임은 어떤 기준으로 나뉘는가?
3. 최종 일정 결정에서 가능한 시간 후보와 선택 이유를 함께 보여줘야 하는 이유는 무엇인가?

## 작은 응용 과제

팀원 한 명이 선택된 시간에 불가능하다고 가정하고, 최종 후보 비교 trace가 어떻게 바뀌어야 하는지 작성한다.
