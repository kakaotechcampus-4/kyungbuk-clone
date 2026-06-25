# Kanana 프로젝트 구조 안내

이 문서는 학생들이 Week 1 main 브랜치를 처음 열었을 때 전체 흐름을 빠르게 파악하기 위한 지도입니다. 실행 방법은 [README.md](README.md)를 기준으로 보고, 이 문서는 "어느 파일이 어떤 역할을 하는지"를 이해하는 데 집중합니다.

Week 1-6 전체 프로젝트는 `week_1_to_6f` 브랜치에 보존되어 있습니다.

## 30초 요약

| 경로 | 역할 |
| --- | --- |
| `app.py` | Gradio 채팅 UI와 상세 trace 화면을 구성합니다. |
| `student_parts/week01_wake_up_nana.py` | 학생이 Week 1 핵심 tool 함수를 구현하는 파일입니다. |
| `fixed/` | 설정, 런타임, 저장소, trace, LLM 연결 등 기준 코드입니다. |
| `static/` | UI 스타일과 Kanana 브랜드 이미지입니다. |
| `run.sh` | 설치와 Week 1 앱 실행을 담당하는 runner입니다. |

## 전체 실행 흐름

1. 사용자가 `./run.sh` 또는 `./run.sh --week1`으로 앱을 실행합니다.
2. `app.py`가 Gradio UI를 띄우고 `fixed/agent_runtime.py`가 사용자 메시지를 저장합니다.
3. `fixed/week_agent_registry.py`가 `student_parts/week01_wake_up_nana.py`의 `build_week_agent()`를 호출합니다.
4. Week 1 agent가 prompt와 tool 목록을 보고 필요한 tool을 호출합니다.
5. tool call/result는 상세 탭의 trace JSON으로 표시됩니다.

초기 배포 상태에서 학생 구현 대상 tool 본문은 `# TODO`와 빈칸으로 남아 있습니다. 학생은 가이드를 읽고 함수 본문을 하나씩 완성한 뒤 trace에서 입력값과 결과 payload를 확인합니다.

## Week 1 학습 흐름

| 주차 | 파일 | 핵심 개념 | 구현 포인트 |
| --- | --- | --- | --- |
| Week 1 | `student_parts/week01_wake_up_nana.py` | LangChain tool 기초 | 현재 대화 전용 개인 일정 생성/조회/삭제 |

## 추천 탐색 순서

1. [README.md](README.md)로 실행 방법과 `.env` 설정을 확인합니다.
2. `student_parts/week01_wake_up_nana.py`를 열어 `[수강생 구현 가이드]`를 읽습니다.
3. `week01_tools()`가 어떤 tool을 agent에 공개하는지 확인합니다.
4. `./run.sh --week1`로 앱을 실행하고 샘플 요청을 입력합니다.
5. 상세 trace에서 호출된 `tool_name`, 입력값, 결과 payload를 확인합니다.
6. 해당 함수 본문을 구현한 뒤 다시 실행해 trace 결과가 어떻게 바뀌는지 비교합니다.

## 자주 쓰는 명령

```bash
./run.sh --install
```

처음 의존성을 설치하고 Week 1 앱을 실행합니다.

```bash
./run.sh --week1
```

Week 1 agent로 앱을 실행합니다.

```bash
./run.sh --help
```

runner에서 지원하는 옵션을 확인합니다.

## 읽는 팁

- 학생 구현 범위는 `student_parts/week01_wake_up_nana.py` 파일 상단의 `[수강생 구현 가이드]`가 기준입니다.
- `fixed/`는 기준 구현을 이해하기 위한 참고 코드이며, 수업에서 별도 지시가 없으면 수정하지 않습니다.
- 앱 화면에서 어떤 tool이 호출됐는지 궁금하면 상세 탭의 trace를 확인하세요.
