# AI 활용 학습 노트

AI(Claude)와 함께 과제를 하면서 반복적으로 배운 패턴을 정리합니다. 매 주차 PR "AI 활용 내역"에는
그 주차에 있었던 구체적인 사례를, 여기에는 여러 주차에 걸쳐 재사용 가능한 일반적인 교훈을 남깁니다.

## 1. 짐작하지 말고 실제 소스로 검증하기
AI가 준 코드가 함수 시그니처나 반환 모양을 짐작해서 짜는 경우가 종종 있었습니다.
- 예: `normalize_external_schedule_date_bounds(date_from, date_to)`처럼 인자 2개로 짐작해서 호출했는데,
  실제로는 `fixed/external_people_store.py`에 `member_names`까지 3개를 받는 함수였습니다.
- 대응: 에러가 나거나 확신이 안 서면 항상 `fixed/` 원본 코드를 열어서 실제 시그니처/반환 형태를 확인한 뒤 코드를 확정합니다.
- 실행 후 trace 페이로드로 실제 tool_call/tool_result를 직접 눈으로 확인하는 것도 같은 이유입니다.

## 2. tool 라우팅은 프롬프트의 "예시"로 고친다
LLM이 여러 tool 중 잘못된 것을 고르는 문제는 tool description(docstring)만 고쳐서는 해결 안 될 때가 많았습니다.
- Week4: "예전에 ~ 얘기한 적 있어?" 질문이 계속 저장된 일정 검색 tool로 잘못 라우팅됨.
- Week6: "민준이랑 서연 일정 확인해줘"가 supervisor에서 nana_agent로 잘못 위임됨.
- 두 경우 모두 추상적인 역할 설명만으로는 부족했고, **구체적인 예문**(이런 문장이 오면 이 tool/agent를 써라)을
  시스템 프롬프트에 직접 넣었을 때 해결됐습니다.
- Week6: supervisor가 "다른 사람 이름이 하나라도 등장하면 nana_agent로 보내지 않는다"는 규칙을 썼는데,
  "내일 10시에 철수랑 개발 미팅 잡아줘"처럼 참석자가 있어도 결국 내 일정을 만드는 요청까지
  kana_agent로 잘못 보내는 문제가 있었습니다. 판단 기준을 표면적 특징("이름이 있냐")이 아니라
  진짜 의도("다른 사람의 일정을 확인/조율하는 것이냐, 참석자가 있어도 내 일정을 관리하는 것이냐")로
  바꿔야 한다는 걸 배웠습니다.

## 3. "agent가 agent를 tool처럼 부르는" 구조 (Week6)
Week6은 한 agent가 모든 tool을 갖고 있던 구조에서, supervisor가 `nana_agent`/`kana_agent`라는
tool 뒤에 완전히 다른 system prompt와 tool 목록을 가진 하위 agent를 숨겨두는 구조로 바뀌었습니다.
- supervisor, Nana, Kana는 프롬프트를 공유하지 않으므로, 각 역할이 "나는 무엇을 담당하고 무엇은 아닌지"를
  스스로의 프롬프트에 갖고 있어야 합니다.
- 하위 agent 내부 tool 호출은 supervisor 입장에서 블랙박스라서, `inner_tool_names`처럼 내부 trace를
  끌어올리는 코드가 디버깅에 필수적이었습니다.

## 4. 작업 습관
- 커밋은 Conventional Commit(`feat`, `fix`, `test`, `docs`)으로 나눠서 올린다.
- 순수 로직 함수(예: `_tool_call_names`)는 pytest로 짧게라도 테스트한다.

## 5. LLM 호출 함수도 monkeypatch로 테스트 가능하다
`nana_agent`/`kana_agent`처럼 내부에서 실제 LLM을 호출하는 함수는 "그냥 테스트 불가능하다"고
단정했었는데, "LLM 호출"과 "결과를 받아서 trace를 파싱하고 JSON으로 조립하는 로직"을 분리해서
생각하면 후자는 충분히 테스트할 수 있습니다.
- 전역 변수에 캐싱해둔 하위 agent(`_NANA_SUBAGENT`)를 `monkeypatch`로 가짜 객체(`.invoke()`가
  미리 정해둔 결과를 돌려주는 객체)로 바꿔치기하면, 실제 API 호출 없이 파싱/조립 로직만 검증할 수 있습니다.
- 가짜 메시지 객체를 만들 때는 실제 LangChain 메시지의 속성(`.type`, `.content`, `.tool_calls`,
  `.name`, `.tool_call_id`)을 그대로 흉내내야 하고, 이를 위해 `fixed/langchain_trace.py`의 실제
  파싱 함수(`message_content_to_text` 등) 구현을 먼저 확인했습니다.