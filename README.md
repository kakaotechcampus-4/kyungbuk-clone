# Kanana Schedule Agent

Kanana 강의용 일정 Agent 실습 프로젝트의 Week 1 main 브랜치입니다. 학생들은 `student_parts/week01_wake_up_nana.py`를 열고, 파일 상단의 `[수강생 구현 가이드]`가 지정한 함수와 tool 본문을 직접 완성합니다.

처음 구조를 볼 때는 [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)를 먼저 읽고, 수업 흐름은 [CURRICULUM.md](CURRICULUM.md)를 기준으로 따라가면 됩니다.

## 실행

기본 Python 패키지 관리는 `uv`를 사용합니다.

```bash
cd kakao_clone_coding_projects_q
./run.sh --install
```

설치 후에는 아래 명령으로 Week 1 앱을 실행합니다.

```bash
./run.sh
```

명시적으로 Week 1을 선택할 수도 있습니다.

```bash
./run.sh --week1
```

`.env`는 repo 루트의 파일을 읽습니다. `.env.example`을 복사해 개인 키를 채워 넣으세요.

```bash
PROXY_TOKEN=여기에 api key 입력
CHAT_PROXY_URL=https://mlapi.run/4bbd0c4d-bf02-4e59-a635-457b1c30c56a/v1
EMBEDDING_PROXY_URL=https://mlapi.run/b54ff33e-6d14-42df-93f9-0f1132160ee8/v1
OPENAI_MODEL=openai/gpt-4.1-mini
OPENAI_EMBEDDING_MODEL=openai/text-embedding-3-small
KANANA_ACTIVE_WEEK=1
KANANA_USE_LLM=1
KANANA_LLM_ASSIST=1
```

`PROXY_TOKEN`이 없으면 프롬프트 기반 agent는 실행되지 않고 안내 메시지가 표시됩니다. 키를 넣으면 Week 1 agent가 prompt와 tool 목록을 보고 직접 tool을 고릅니다.

### Conda fallback

conda 환경이 필요한 경우 `environment.yml` 기반 runner를 사용할 수 있습니다.

```bash
./run.sh --conda --install
./run.sh --conda
```

## Week 1 구현 포인트

- 파일: `student_parts/week01_wake_up_nana.py`
- 구현 대상: `personal_create_schedule`, `personal_list_schedules`, `personal_delete_schedule`
- 목표: 현재 대화 전용 임시 개인 일정 CRUD tool을 완성합니다.

## 구현 확인

이 학생용 repo에는 자동 테스트 하네스가 포함되어 있지 않습니다. 앱을 실행한 뒤 채팅을 입력하고, 화면의 상세 trace에서 어떤 tool이 호출됐는지와 tool 결과 JSON에 어떤 값이 들어왔는지 확인하세요.

초기 배포 상태의 구현 대상 함수 본문에는 `# TODO`와 빈칸이 들어 있습니다. 학생이 함수를 완성하면 상세 trace에서 실제 결과 JSON을 확인할 수 있어야 합니다.

## 패키지 관리

새 의존성의 기준 파일은 `pyproject.toml`과 `uv.lock`입니다. `requirements.txt`와 `environment.yml`은 기존 수강생 환경을 위한 fallback 파일입니다.

```bash
uv add "package-name>=1.0"
uv remove package-name
uv lock
```
