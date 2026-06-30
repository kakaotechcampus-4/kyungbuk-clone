# CLAUDE.md — Kanana Schedule Agent (kyungpook-clone)

## 행동 규칙

**내 허락 없이는 어떠한 작업도 진행하지 않는다.**
코드 수정, 파일 생성, 명령 실행 등 모든 작업은 사용자의 명시적 허락을 받은 후에만 진행한다.

**모든 구현은 단위별로 끊어서 진행한다.**
한 번에 전체를 구현하지 않는다. 함수 하나, 기능 하나 단위로 사용자와 확인하며 진행한다.

**작업 전 반드시 무엇을 바꾸는지 먼저 설명하고, 허락을 받은 후에만 작업을 진행한다.**

---

## 프로젝트 개요

카카오테크캠퍼스 강의용 일정 Agent 실습 프로젝트 (Week 1).
LLM(GPT-4.1-mini)이 LangChain tool을 직접 골라 호출하는 AI 일정 관리 챗봇.
UI는 Gradio로 구성되어 있으며 `http://127.0.0.1:7860`에서 실행된다.

---

## 파일 구조

```
kyungpook-clone/
├── app.py                        # Gradio 채팅 UI + 상세 trace 화면
├── run.sh                        # 설치 및 실행 runner (uv 기반)
├── .env                          # API 키 등 환경변수 (git 제외)
├── .env.example                  # .env 템플릿
├── student_parts/
│   └── week01_wake_up_nana.py    # ★ 학생 구현 파일 (과제 범위)
├── fixed/                        # 기준 코드 (수정 금지)
│   ├── agent_runtime.py          # 사용자 메시지 처리 런타임
│   ├── week_agent_registry.py    # build_week_agent() 호출 및 주차 관리
│   ├── session_scope.py          # 대화 범위(session_id) 관리
│   ├── llm.py                    # LLM 연결 (chat_model())
│   ├── langchain_trace.py        # trace 추출 헬퍼
│   ├── runtime_clock.py          # 날짜/시각 헬퍼
│   ├── config.py                 # .env 설정 로드
│   └── ...                       # 기타 store, trace 관련 파일
├── mcp_server/                   # MCP 서버 관련
└── static/                       # UI 스타일, 이미지
```

---

## 과제 범위

**수정 대상: `student_parts/week01_wake_up_nana.py` 한 파일만.**

구현해야 할 함수 3개:

| 함수 | 역할 | 반환 JSON 필수 키 |
|---|---|---|
| `personal_create_schedule` | 개인 일정 생성 | `ok`, `tool_name`, `created_schedule` |
| `personal_list_schedules` | 일정 조회 (날짜 필터) | `ok`, `tool_name`, `schedules` |
| `personal_delete_schedule` | 일정 삭제 | `ok`, `tool_name`, `deleted` |

선택 구현:
- `CHAT_MEMORY_PROMPT` — 공통 system prompt
- `week01_prompt_parts()` — Nana agent 성격/규칙 정의

---

## 핵심 구현 규칙

- **저장소**: 파일 상단 `PERSONAL_SCHEDULES` 리스트 (Python 메모리, DB 없음)
- **대화 격리**: 일정 dict에 `session_id = current_session_scope()` 포함, 조회/삭제 시 같은 session_id만 처리
- **반환**: 항상 `_json(dict)` 사용 (문자열 반환)
- **ID 생성**: `_new_personal_id()`
- **시각**: `_now_iso()`
- **삭제**: `PERSONAL_SCHEDULES[:] = [...]` 방식 (리스트 객체 참조 유지)
- `fixed/` 폴더는 수정하지 않는다

---

## 실행 방법

```bash
# Git Bash에서
./run.sh --install   # 최초 1회
./run.sh --week1     # 앱 실행
```

---

## 사용자 배경

- Python 학습 단계 (익숙하지 않음)
- 코드 설명 시 한 줄씩 쉽게 풀어서 설명한다
- 코드는 사용자가 직접 타이핑하며 따라가는 방식으로 진행한다

---

## 환경

- Python: 3.11 (uv로 가상환경 `.venv` 생성)
- 패키지: LangChain, LangGraph, Gradio, OpenAI
- API: Kakao Tech Campus Proxy (`mlapi.run`)
- `.env`의 `PROXY_TOKEN`이 없으면 LLM 호출 불가
