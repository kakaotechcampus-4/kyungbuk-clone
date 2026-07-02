# Kanana Schedule Agent — 프로젝트 분석 문서

> 대상: FastAPI 경험 있음, LangChain 미경험 CS 4학년
> 목적: 6주 학습 전 전체 구조 파악

---

## 1. 한 줄 요약

**"채팅창에 일정을 말하면, LLM이 알아서 적절한 도구를 선택해 일정을 관리해주는 AI 에이전트 앱"**

FastAPI 앱에서 라우터가 요청을 받아 비즈니스 로직을 호출하는 것처럼,  
이 앱에서는 **LangChain Agent**가 채팅 입력을 받아 적절한 **Tool 함수**를 선택·실행한다.

---

## 2. 기술 스택 한눈에 보기

| 역할 | 기술 | FastAPI 비유 |
|------|------|------------|
| UI | Gradio | Jinja2 템플릿 |
| AI 오케스트레이션 | LangChain Agent | 라우터(Router) |
| LLM | GPT-4.1-mini (proxy) | 비즈니스 로직 |
| Tool 정의 | `@tool` 데코레이터 | `@app.post()` 라우터 |
| 영구 저장소 | SQLite | DB 세션 |
| 외부 데이터 | MCP (stdio 서브프로세스) | 외부 API 클라이언트 |
| 벡터 검색 | ChromaDB | 검색 인덱스 |
| 설정 관리 | python-dotenv + dataclass | pydantic Settings |

---

## 3. 디렉터리 구조

```
kyungpook-clone/
│
├── app.py                          # Gradio UI 진입점
│
├── fixed/                          # 수정 금지 — 프레임워크 코드
│   ├── config.py                   # .env 로딩, CONFIG 싱글턴
│   ├── agent_runtime.py            # UI ↔ LangChain 연결 어댑터
│   ├── week_agent_registry.py      # 주차별 에이전트 디스패처
│   ├── app_store.py                # SQLite CRUD (대화, 일정, 구조화 데이터)
│   ├── store_base.py               # SQLite 공통 유틸 (ID 생성, 시간 포맷 등)
│   ├── session_scope.py            # 대화별 메모리 격리 (ContextVar)
│   ├── llm.py                      # ChatOpenAI 팩토리
│   ├── langchain_trace.py          # LangChain 실행 결과 → UI 트레이스 변환
│   ├── mcp_client.py               # 외부 MCP 서버 호출 클라이언트
│   ├── external_mcp.py             # 내 일정 → 공유 일정 DB 동기화
│   ├── external_people_store.py    # 팀원 대화/일정 픽스처 데이터
│   ├── conversation_rag_store.py   # ChromaDB 벡터 검색
│   ├── reference_store.py          # 임베딩 함수
│   ├── schedule_decision.py        # 일정 결정 로직
│   ├── runtime_clock.py            # 앱 시작 시각 고정 헬퍼
│   └── trace.py                    # 비-LangChain용 간단 트레이스 수집기
│
├── student_parts/                  # 학생 구현 영역
│   └── week01_wake_up_nana.py      # ★ Week 1 과제 파일
│
├── mcp_server/
│   └── sqlite_mcp_server.py        # 팀원 데이터 노출 MCP 서버 (stdio)
│
├── static/                         # UI 에셋 (CSS, 이미지)
├── data/                           # 런타임 SQLite DB 파일
├── .env                            # API 키 등 환경 변수
├── pyproject.toml                  # 프로젝트 의존성
└── run.sh                          # 설치/실행 스크립트
```

---

## 4. 핵심 개념: LangChain Agent란?

LangChain Agent를 모르면 코드 흐름이 이해되지 않는다. 먼저 개념을 잡자.

### FastAPI와 비교

```
[FastAPI 방식]
사용자 요청 → 라우터가 URL 기반으로 핸들러 결정 → 핸들러 실행 → 응답

[LangChain Agent 방식]
사용자 채팅 → LLM이 의도 파악 후 Tool 선택 → Tool 실행 → LLM이 결과 해석 → 응답
```

핵심 차이: **무엇을 실행할지 결정하는 주체가 LLM**이다.

### ReAct 루프 (이 프로젝트의 동작 방식)

```
사용자: "내일 오후 2시에 팀 회의 일정 추가해줘"

LLM 판단: "personal_create_schedule을 써야겠다"
  → Tool 호출: personal_create_schedule(title="팀 회의", date="2026-07-02", start_time="14:00")
  → Tool 결과: {"ok": true, "created_schedule": {...}}
LLM 판단: "결과를 받았으니 응답을 생성하면 된다"
  → 최종 응답: "내일 오후 2시 팀 회의 일정을 추가했습니다."
```

---

## 5. 전체 코드 흐름 트리

채팅창에 메시지를 입력하면 벌어지는 일을 파일·함수 단위로 추적한 트리다.

```
[사용자가 채팅창에 메시지 입력 후 전송]
│
├── app.py
│   └── queue_user_message()          # 사용자 메시지 UI에 즉시 표시
│       └── finish_agent_response()   # 에이전트 실행 & 스트리밍 처리
│           │
│           └── fixed/agent_runtime.py
│               └── AgentRuntime.stream_agent()
│                   ├── app_store.py → append_message()     # 사용자 메시지 DB 저장
│                   ├── app_store.py → load_conversation()  # 대화 히스토리 로드
│                   │
│                   └── fixed/week_agent_registry.py
│                       └── stream_active_week_agent()
│                           │
│                           ├── fixed/llm.py
│                           │   └── chat_model()            # ChatOpenAI 인스턴스 생성
│                           │
│                           ├── student_parts/week01_wake_up_nana.py
│                           │   └── build_week_agent()      # LangChain 에이전트 빌드
│                           │       ├── week01_tools()      # Tool 목록 반환
│                           │       │   ├── personal_create_schedule  ← 학생 구현
│                           │       │   ├── personal_list_schedules   ← 학생 구현
│                           │       │   └── personal_delete_schedule  ← 학생 구현
│                           │       └── week01_system_prompt()        # 시스템 프롬프트 구성
│                           │
│                           └── [LangChain Agent 실행 루프]
│                               ├── LLM이 Tool 선택
│                               ├── Tool 함수 실행 (위 3개 중 하나)
│                               │   └── fixed/session_scope.py
│                               │       └── current_session_scope()  # 현재 대화 ID 확인
│                               ├── Tool 결과를 LLM에 전달
│                               └── 최종 텍스트 응답 생성
│
├── fixed/agent_runtime.py
│   └── (스트리밍 이벤트 수신 중)
│       ├── "현재 personal_create_schedule 실행 중" 상태 텍스트 yield
│       └── 최종 응답 수신
│           ├── app_store.py → append_message()   # 어시스턴트 응답 DB 저장
│           └── RuntimeResult 반환 (answer + trace)
│
└── app.py
    └── finish_agent_response() 스트리밍 루프 종료
        ├── 채팅 메시지 업데이트
        └── 트레이스 JSON 업데이트 (상세 탭)
```

---

## 6. 주요 파일별 역할 상세

### 6-1. [app.py](app.py) — UI 레이어

Gradio로 만든 웹 UI. 크게 두 탭으로 구성된다.

- **채팅 탭**: 메시지 입력/출력, 대화 목록 사이드바, 저장된 일정 표시
- **상세 탭**: 에이전트가 어떤 Tool을 어떤 인자로 호출했는지 JSON 트레이스

FastAPI의 `@app.post()` 핸들러에 해당하는 중심 함수:
- `queue_user_message()` → 사용자 메시지 처리 시작
- `finish_agent_response()` → 에이전트 실행 결과 스트리밍

### 6-2. [fixed/agent_runtime.py](fixed/agent_runtime.py) — 어댑터 레이어

UI와 LangChain 사이의 번역기 역할.

```python
# 핵심 데이터 클래스
@dataclass
class RuntimeResult:
    answer: str          # 최종 텍스트 응답
    trace: dict          # 트레이스 (어떤 Tool이 실행됐는지)
    conversation_id: str

@dataclass
class RuntimeStreamEvent:
    status_text: str | None  # "현재 OOO 실행 중" 진행 상태
    result: RuntimeResult | None  # 완료 시 최종 결과
```

### 6-3. [fixed/week_agent_registry.py](fixed/week_agent_registry.py) — 디스패처

`KANANA_ACTIVE_WEEK` 환경 변수를 보고 주차별 학생 파일을 import해서 에이전트를 실행한다.

```python
# 이 함수가 student_parts/week01_wake_up_nana.py를 호출함
def stream_active_week_agent(messages, conversation_id):
    module = importlib.import_module("student_parts.week01_wake_up_nana")
    agent = module.build_week_agent(messages)
    for chunk in agent.stream(...):
        yield 진행상태_이벤트
    yield 최종결과_이벤트
```

### 6-4. [student_parts/week01_wake_up_nana.py](student_parts/week01_wake_up_nana.py) — ★ 학생 구현 파일

Week 1 과제의 전부. 3개의 `@tool` 함수를 구현한다.

```python
# 전역 메모리 저장소 (DB 없음, 현재 프로세스 내 메모리에만 존재)
PERSONAL_SCHEDULES: list[dict] = []

@tool
def personal_create_schedule(title, date, start_time, end_time="미정", attendees=None):
    """LLM이 일정 생성 요청을 받으면 자동으로 이 함수를 호출"""
    ...

@tool
def personal_list_schedules(date_from=None, date_to=None):
    """LLM이 일정 조회 요청을 받으면 자동으로 이 함수를 호출"""
    ...

@tool
def personal_delete_schedule(schedule_id):
    """LLM이 일정 삭제 요청을 받으면 자동으로 이 함수를 호출"""
    ...
```

`build_week_agent()`가 이 Tool들을 LangChain Agent에 바인딩한다.

### 6-5. [fixed/app_store.py](fixed/app_store.py) — 영구 저장소

SQLite 기반 CRUD. 주요 테이블:

```
conversations  → 대화방 목록
messages       → 각 대화방의 메시지 (role: user/assistant)
schedules      → DB에 저장된 일정 (Week 2 이후 사용)
structured_requests → LLM이 파싱한 구조화 데이터
```

Week 1에서는 **일정이 DB에 저장되지 않고** `PERSONAL_SCHEDULES` 리스트(메모리)에만 저장된다.  
대화 메시지는 DB에 저장된다.

### 6-6. [fixed/session_scope.py](fixed/session_scope.py) — 대화 격리

```python
# 왜 필요한가?
# PERSONAL_SCHEDULES는 전역 변수 → 모든 대화방이 같은 메모리를 공유함
# → 대화방 A의 일정이 대화방 B에서 보이면 안 됨
# → ContextVar로 "지금 어느 대화방인지"를 Thread/async safe하게 추적
current_session_scope()  # 현재 대화 ID 반환
```

### 6-7. [mcp_server/sqlite_mcp_server.py](mcp_server/sqlite_mcp_server.py) — 외부 데이터 서버

팀원(철수, 영희, 민준 등)의 대화·일정 데이터를 MCP(stdio) 프로토콜로 제공하는 서버.  
`mcp_client.py`가 이 서버를 서브프로세스로 실행해서 도구처럼 호출한다.  
(Week 1에서는 직접 사용하지 않음 — 후반 주차에서 팀 일정 조율 시 활용)

---

## 7. 데이터 흐름 다이어그램

```
사용자 입력 → [app.py] → [agent_runtime.py] → [week_agent_registry.py]
                                                        ↓
                                           [week01_wake_up_nana.py]
                                           build_week_agent() 호출
                                                        ↓
                                        ┌───────────────────────────┐
                                        │    LangChain Agent 루프    │
                                        │                           │
                                        │  GPT-4.1-mini             │
                                        │      ↓ Tool 선택           │
                                        │  personal_create_schedule │
                                        │  personal_list_schedules  │
                                        │  personal_delete_schedule │
                                        │      ↓ 실행 결과           │
                                        │  GPT-4.1-mini → 최종 응답  │
                                        └───────────────────────────┘
                                                        ↓
                                        스트리밍 이벤트 yield
                                                        ↓
                                        [agent_runtime.py] DB 저장
                                                        ↓
                                        [app.py] UI 업데이트
```

---

## 8. Week 1 과제 포인트

과제 목표: `personal_create_schedule`, `personal_list_schedules`, `personal_delete_schedule` 3개 구현

| 함수 | 입력 | 해야 할 일 | 반환 형식 |
|------|------|-----------|----------|
| `personal_create_schedule` | title, date, start_time, end_time, attendees | dict 생성 후 PERSONAL_SCHEDULES에 추가 | `{"ok": true, "tool_name": "...", "created_schedule": {...}}` |
| `personal_list_schedules` | date_from, date_to | 현재 세션 일정 필터링 후 반환 | `{"ok": true, "tool_name": "...", "schedules": [...]}` |
| `personal_delete_schedule` | schedule_id | 현재 세션에서 id 일치 항목 제거 | `{"ok": true, "tool_name": "...", "deleted": bool}` |

**중요**: `_current_session_schedules()`를 반드시 사용해야 대화방 간 격리가 된다.

---

## 9. 설정값 흐름

```
.env 파일
  PROXY_TOKEN=<토큰>
  KANANA_ACTIVE_WEEK=1
        ↓
fixed/config.py → load_config() → CONFIG 싱글턴
        ↓
fixed/llm.py → chat_model() → ChatOpenAI(base_url=CONFIG.chat_proxy_url, ...)
        ↓
week_agent_registry.py → normalize_active_week() → "1" → week01 실행
```

---

## 10. 학습 순서 권장

1. **[student_parts/week01_wake_up_nana.py](student_parts/week01_wake_up_nana.py)** — 과제 파일 전체 읽기 (주석 포함)
2. **[fixed/week_agent_registry.py](fixed/week_agent_registry.py)** — 내 파일이 어떻게 호출되는지 확인
3. **[fixed/session_scope.py](fixed/session_scope.py)** — 대화 격리 원리 이해
4. **[fixed/agent_runtime.py](fixed/agent_runtime.py)** — UI ↔ 에이전트 연결 이해
5. **[app.py](app.py)** — UI 레이어 (필요할 때만)
6. **[fixed/app_store.py](fixed/app_store.py)** — DB 구조 (Week 2 이후 본격 사용)

---

## 11. 자주 헷갈리는 포인트

**Q. `@tool`이 `@app.post()`와 다른 점?**  
A. FastAPI는 URL 기반으로 핸들러를 결정하지만, LangChain `@tool`은 LLM이 함수 docstring을 읽고 어떤 함수를 쓸지 스스로 결정한다. docstring이 곧 API 명세서다.

**Q. Week 1에서 일정이 앱을 재시작하면 사라지는 이유?**  
A. `PERSONAL_SCHEDULES`가 전역 리스트(메모리)이기 때문. DB에 저장하는 코드는 아직 없다. Week 2~3에서 DB 연동을 구현한다.

**Q. MCP가 뭔가?**  
A. Model Context Protocol. LLM이 외부 데이터 소스(여기서는 팀원 일정 DB)를 표준화된 방식으로 호출할 수 있게 해주는 프로토콜. 이 프로젝트에서는 `mcp_server/sqlite_mcp_server.py`를 서브프로세스로 실행해서 stdio로 통신한다.

**Q. `fixed/` 폴더는 왜 건드리면 안 되나?**  
A. 이후 주차 학생들의 코드와 충돌 없이 합칠 수 있도록 프레임워크 코드를 분리해둔 것. 학생 구현은 항상 `student_parts/` 안에서만 한다.
