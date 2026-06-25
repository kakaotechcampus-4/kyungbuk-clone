from __future__ import annotations

"""앱 실행 시점을 기준으로 한 날짜/시간 헬퍼입니다.

테스트와 수업 데모에서 "오늘", "다음 주 화요일" 같은 상대 날짜가 실행 중에
흔들리지 않도록 import 시점의 OS 날짜를 고정해서 사용합니다.
"""

from datetime import date, datetime, timedelta


APP_STARTED_AT = datetime.now().astimezone()
APP_TODAY = APP_STARTED_AT.date()


def current_app_date() -> date:
    """프로그램 시작 시 OS에서 읽은 현재 날짜를 반환합니다."""

    return APP_TODAY


def current_app_date_iso() -> str:
    """프로그램 시작 시 OS에서 읽은 현재 날짜를 YYYY-MM-DD로 반환합니다."""

    return current_app_date().isoformat()


def app_started_at_iso() -> str:
    """프로그램 시작 시각을 OS 타임존이 포함된 ISO 문자열로 반환합니다."""

    return APP_STARTED_AT.isoformat(timespec="seconds")


def next_weekday_date(weekday: int) -> date:
    """프로그램 시작일 기준 다음 주의 특정 요일 날짜를 반환합니다.

    weekday는 datetime.date.weekday()와 같은 규칙을 사용합니다.
    월요일은 0, 화요일은 1, 수요일은 2입니다.
    """

    next_monday = current_app_date() + timedelta(days=7 - current_app_date().weekday())
    return next_monday + timedelta(days=weekday)


def next_weekday_iso(weekday: int) -> str:
    """프로그램 시작일 기준 다음 주의 특정 요일 날짜를 YYYY-MM-DD로 반환합니다."""

    return next_weekday_date(weekday).isoformat()
