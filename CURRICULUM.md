# Kanana Week 1 커리큘럼

이 문서는 Kanana Schedule Agent Week 1 수업에서 다루는 미션을 정리한 운영안입니다. Week 1-6 전체 커리큘럼은 `week_1_to_6f` 브랜치에 보존되어 있습니다.

## 운영 기준

- 수업은 앱 실행, 채팅 입력, 상세 trace 확인, 함수 구현, 재실행 순서로 진행합니다.
- 초기 배포 상태의 구현 대상 함수 본문은 `# TODO`와 빈칸으로 남아 있습니다.
- 학생은 `received` 입력값과 기대 payload 키를 trace에서 확인한 뒤 실제 구현으로 바꿉니다.
- prompt, tool-list, agent builder 기준 구현은 연결 구조를 읽는 참고 코드입니다.

## Week 1 · 개인 일정 CRUD tool

파일: `student_parts/week01_wake_up_nana.py`

미션은 `personal_create_schedule`, `personal_list_schedules`, `personal_delete_schedule`을 구현하는 것입니다. 현재 대화 범위의 임시 메모리인 `PERSONAL_SCHEDULES`를 사용하고, tool 결과는 JSON 문자열로 반환합니다.

확인 포인트는 상세 trace에서 세 tool 중 어떤 tool이 호출됐는지, `created_schedule`, `schedules`, `deleted` payload가 기대한 모양으로 바뀌었는지 보는 것입니다.

## 진행 템플릿

1. Week 1 파일의 `[수강생 구현 가이드]`를 읽습니다.
2. 앱을 실행합니다.
3. 샘플 프롬프트를 입력하고 상세 trace에서 호출된 tool과 입력값을 봅니다.
4. 구현 대상 함수 본문만 수정합니다.
5. 다시 실행해 trace payload가 실제 결과로 바뀌었는지 확인합니다.

## 멘토 확인 기준

- 구현 대상 함수가 가이드의 책임 범위 안에서 완성됐는지 확인합니다.
- tool 결과 JSON이 prompt에서 기대하는 top-level 키를 유지하는지 확인합니다.
- trace에서 LLM이 고른 tool과 tool result가 설명 가능한지 확인합니다.
- 학생이 직접 실행한 프롬프트와 관찰한 trace를 바탕으로 구현 과정을 설명할 수 있는지 확인합니다.
