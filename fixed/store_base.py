from __future__ import annotations

"""SQLite 저장소들이 공통으로 쓰는 작은 유틸리티 모음입니다."""

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


def now_iso() -> str:
    """현재 시각을 마이크로초까지 포함한 ISO 문자열로 반환합니다.

    SQLite row 정렬과 trace 표시가 안정적으로 되도록 모든 저장소에서 같은 형식을 씁니다.
    """

    return datetime.now().astimezone().isoformat(timespec="microseconds")


def new_id(prefix: str) -> str:
    """읽기 쉬운 접두어가 붙은 짧은 UUID 기반 ID를 만듭니다."""

    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def decode_schedule_row(row: dict[str, Any]) -> dict[str, Any]:
    """DB의 `attendees_json` 문자열을 Python list인 `attendees`로 복원합니다."""

    decoded = dict(row)
    raw_attendees = decoded.pop("attendees_json", "[]") or "[]"
    try:
        decoded["attendees"] = json.loads(raw_attendees)
    except Exception:
        decoded["attendees"] = []
    return decoded


SCHEDULE_COLUMNS = (
    "schedule_id, request_id, owner, title, date, start_time, end_time, "
    "attendees_json, source, created_at"
)
SCHEDULE_COLUMNS_WITH_KIND = (
    f"{SCHEDULE_COLUMNS}, "
    "(SELECT kind FROM structured_requests WHERE request_id = schedules.request_id) AS request_kind"
)


class SQLiteFileStore:
    """파일 기반 SQLite 저장소가 공유하는 경로 준비와 연결 설정입니다."""

    def __init__(self, path: Path):
        """SQLite 파일 경로를 보관하고 부모 디렉터리를 미리 생성합니다."""

        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        """row를 dict처럼 읽을 수 있는 SQLite 연결을 새로 엽니다."""

        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn
