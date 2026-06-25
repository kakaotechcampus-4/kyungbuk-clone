from __future__ import annotations

"""외부 멤버의 과거 대화와 공유 일정을 담는 SQLite 저장소입니다.

수업에서는 실제 카카오톡/캘린더 데이터 대신 이 저장소를 외부 시스템처럼 사용합니다.
Week 5 MCP 서버는 이 저장소를 tool로 감싸고, Week 6은 여기서 나온 busy time과
앱 내부 개인 일정을 합쳐 공통 가능 시간을 계산합니다.
"""

import os
import re
from pathlib import Path
from typing import Any

from fixed.config import CONFIG
from fixed.runtime_clock import app_started_at_iso
from fixed.store_base import SQLiteFileStore, new_id


EXTERNAL_MEMBER_ALIAS: dict[str, str] = {}
PERSONAL_SHARED_MEMBER_NAME = "나"
PARENTHETICAL_TEXT_RE = re.compile(r"\s*[\(（][^()（）]*[\)）]")
USER_VISIBLE_TEXT_FIELDS = {"member_name", "title", "sender", "content", "notes"}
JULY_PRACTICE_DATE_FROM = "2026-07-07"
JULY_PRACTICE_DATE_TO = "2026-07-17"
JULY_PRACTICE_MEMBER_NAMES = ["철수", "영희", "민준", "서연", "지훈", "하린"]
JULY_PRACTICE_CONVERSATIONS = [
    (
        "ext_cs",
        "철수",
        "철수의 일정 공유",
        "철수: 7월 7일 10시는 API 연동 실습, 7월 9일 14시는 고객 인터뷰, 7월 15일 16시는 QA 리뷰가 있어요.",
    ),
    (
        "ext_yh",
        "영희",
        "영희의 일정 공유",
        "영희: 7월 7일 13시는 디자인 피드백, 7월 10일 10시는 콘텐츠 점검, 7월 16일 15시는 발표 리허설입니다.",
    ),
    (
        "ext_mj",
        "민준",
        "민준의 일정 공유",
        "민준: 7월 8일 9시 30분은 데이터 정리, 7월 9일 11시는 백엔드 리뷰, 7월 14일 15시는 운영 회의가 있어요.",
    ),
    (
        "ext_sy",
        "서연",
        "서연의 일정 공유",
        "서연: 7월 8일 13시 30분은 사용자 테스트, 7월 10일 16시는 보고서 정리, 7월 15일 10시는 UX 워크숍입니다.",
    ),
    (
        "ext_jh",
        "지훈",
        "지훈의 일정 공유",
        "지훈: 7월 7일 15시는 모델 평가, 7월 14일 10시는 보안 점검, 7월 16일 13시는 릴리즈 회의가 있습니다.",
    ),
    (
        "ext_hr",
        "하린",
        "하린의 일정 공유",
        "하린: 7월 8일 11시는 온보딩 세션, 7월 10일 14시는 파트너 콜, 7월 17일 9시는 회고 준비가 있어요.",
    ),
]
JULY_PRACTICE_SCHEDULES = [
    ("extsch_july_cs_1", "철수", "API 연동 실습", "2026-07-07", "10:00", "11:00", "ext_cs"),
    ("extsch_july_cs_2", "철수", "고객 인터뷰", "2026-07-09", "14:00", "15:30", "ext_cs"),
    ("extsch_july_cs_3", "철수", "QA 리뷰", "2026-07-15", "16:00", "17:00", "ext_cs"),
    ("extsch_july_yh_1", "영희", "디자인 피드백", "2026-07-07", "13:00", "14:00", "ext_yh"),
    ("extsch_july_yh_2", "영희", "콘텐츠 점검", "2026-07-10", "10:00", "11:30", "ext_yh"),
    ("extsch_july_yh_3", "영희", "발표 리허설", "2026-07-16", "15:00", "16:00", "ext_yh"),
    ("extsch_july_mj_1", "민준", "데이터 정리", "2026-07-08", "09:30", "10:30", "ext_mj"),
    ("extsch_july_mj_2", "민준", "백엔드 리뷰", "2026-07-09", "11:00", "12:00", "ext_mj"),
    ("extsch_july_mj_3", "민준", "운영 회의", "2026-07-14", "15:00", "16:30", "ext_mj"),
    ("extsch_july_sy_1", "서연", "사용자 테스트", "2026-07-08", "13:30", "14:30", "ext_sy"),
    ("extsch_july_sy_2", "서연", "보고서 정리", "2026-07-10", "16:00", "17:00", "ext_sy"),
    ("extsch_july_sy_3", "서연", "UX 워크숍", "2026-07-15", "10:00", "11:00", "ext_sy"),
    ("extsch_july_jh_1", "지훈", "모델 평가", "2026-07-07", "15:00", "16:00", "ext_jh"),
    ("extsch_july_jh_2", "지훈", "보안 점검", "2026-07-14", "10:00", "11:30", "ext_jh"),
    ("extsch_july_jh_3", "지훈", "릴리즈 회의", "2026-07-16", "13:00", "14:00", "ext_jh"),
    ("extsch_july_hr_1", "하린", "온보딩 세션", "2026-07-08", "11:00", "12:00", "ext_hr"),
    ("extsch_july_hr_2", "하린", "파트너 콜", "2026-07-10", "14:00", "15:00", "ext_hr"),
    ("extsch_july_hr_3", "하린", "회고 준비", "2026-07-17", "09:00", "10:00", "ext_hr"),
]
JULY_PRACTICE_CONVERSATION_IDS = [conversation[0] for conversation in JULY_PRACTICE_CONVERSATIONS]


def strip_parenthetical_text(value: Any) -> Any:
    """사용자에게 보이는 외부 데이터 문자열에서 소괄호와 그 안의 내용을 제거합니다."""

    if not isinstance(value, str):
        return value
    cleaned = PARENTHETICAL_TEXT_RE.sub("", value)
    return " ".join(cleaned.split())


def strip_external_row_parentheticals(row: dict[str, Any]) -> dict[str, Any]:
    """외부 tool 결과 row에서 사용자-facing 텍스트 필드의 소괄호 내용을 제거합니다."""

    return {
        key: strip_parenthetical_text(value) if key in USER_VISIBLE_TEXT_FIELDS else value
        for key, value in row.items()
    }


def external_db_path_from_env() -> Path:
    """환경 변수 또는 기본 설정에서 외부 멤버 SQLite DB 경로를 얻습니다."""

    return Path(os.getenv("KANANA_EXTERNAL_DB_PATH", str(CONFIG.external_db_path)))


def normalize_external_member_names(member_names: list[str] | None) -> list[str]:
    """외부 저장소에서 쓰는 멤버 이름 목록으로 정규화합니다."""

    return [
        EXTERNAL_MEMBER_ALIAS.get(str(name).strip(), str(name).strip())
        for name in (member_names or [])
        if str(name).strip()
    ]


def normalize_external_schedule_date_bounds(
    member_names: list[str] | None,
    date_from: str,
    date_to: str,
) -> tuple[str, str]:
    """외부 일정 조회 날짜 범위의 ISO datetime에서 날짜 부분만 남깁니다."""

    normalized_date_from = str(date_from).split("T", 1)[0].strip() if date_from is not None else ""
    normalized_date_to = str(date_to).split("T", 1)[0].strip() if date_to is not None else ""
    return normalized_date_from, normalized_date_to


def external_schedule_summary(rows: list[dict[str, Any]]) -> str:
    """일정 row 목록을 LLM 답변 근거로 쓰기 쉬운 한글 요약 문자열로 바꿉니다."""

    if not rows:
        return "조회된 외부 일정이 없습니다."
    lines: list[str] = []
    for row in rows:
        clean_row = strip_external_row_parentheticals(row)
        member_name = clean_row.get("member_name") or "이름 미정"
        title = clean_row.get("title") or "제목 없음"
        date_text = row.get("date") or "날짜 미정"
        start_time = row.get("start_time") or "시간 미정"
        end_time = row.get("end_time") or "시간 미정"
        notes = clean_row.get("notes")
        line = f"- {member_name} | {title} | {date_text} {start_time}-{end_time}"
        if notes:
            line += f" | {notes}"
        lines.append(line)
    return "\n".join(lines)


class ExternalPeopleSQLiteStore(SQLiteFileStore):
    """외부 멤버 대화/일정 샘플 DB 저장소입니다.

    Week 5 MCP 도구와 Week 6 Kana agent가 여러 사람의 이전 대화와 바쁜 시간을
    조회할 때 사용합니다. 앱 내부 DB와 분리된 SQLite 파일을 씁니다.
    """

    def __init__(self, path: Path):
        """DB 파일을 준비하고 스키마 생성 및 데모 데이터 보정을 수행합니다."""

        super().__init__(path)
        self.initialize()
        self.seed()

    def initialize(self) -> None:
        """외부 대화/메시지/일정 테이블을 생성합니다."""

        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS external_conversations (
                    conversation_id TEXT PRIMARY KEY,
                    member_name TEXT NOT NULL,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS external_messages (
                    message_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    sender TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(conversation_id) REFERENCES external_conversations(conversation_id)
                );

                CREATE TABLE IF NOT EXISTS external_schedules (
                    schedule_id TEXT PRIMARY KEY,
                    member_name TEXT NOT NULL,
                    title TEXT NOT NULL,
                    date TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT NOT NULL,
                    source_conversation_id TEXT,
                    notes TEXT
                );
                """
            )

    def seed(self) -> None:
        """수업 fixture용 외부 멤버 대화와 기본 공유 일정을 채웁니다.

        seed source에 해당하는 row만 지우고 다시 넣습니다. 앱에서 동기화한 `app:`
        또는 `group:` 공유 일정은 학생 실습 결과일 수 있으므로 보존합니다.
        """

        with self.connect() as conn:
            placeholders = ",".join("?" for _ in JULY_PRACTICE_CONVERSATION_IDS)
            conn.execute(
                f"DELETE FROM external_schedules WHERE source_conversation_id IN ({placeholders})",
                JULY_PRACTICE_CONVERSATION_IDS,
            )
            conn.execute(
                f"DELETE FROM external_messages WHERE conversation_id IN ({placeholders})",
                JULY_PRACTICE_CONVERSATION_IDS,
            )
            conn.execute(
                f"DELETE FROM external_conversations WHERE conversation_id IN ({placeholders})",
                JULY_PRACTICE_CONVERSATION_IDS,
            )

            created_at = app_started_at_iso()
            for conversation_id, member_name, title, content in JULY_PRACTICE_CONVERSATIONS:
                conn.execute(
                    "INSERT INTO external_conversations VALUES (?, ?, ?, ?)",
                    (conversation_id, member_name, title, created_at),
                )
                conn.execute(
                    "INSERT INTO external_messages VALUES (?, ?, 'user', ?, ?, ?)",
                    (f"extmsg_{conversation_id}", conversation_id, member_name, content, created_at),
                )
            for schedule in JULY_PRACTICE_SCHEDULES:
                conn.execute("INSERT INTO external_schedules VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (*schedule, ""))

    def create_shared_schedule(
        self,
        member_name: str,
        title: str,
        date: str,
        start_time: str,
        end_time: str = "미정",
        notes: str | None = None,
        source_conversation_id: str | None = None,
        schedule_id: str | None = None,
    ) -> dict[str, Any]:
        """공유 일정 저장소에 일정 하나를 등록하거나 같은 ID의 일정을 갱신합니다."""

        normalized_member_name = strip_parenthetical_text(
            str(member_name or PERSONAL_SHARED_MEMBER_NAME).strip()
        ) or PERSONAL_SHARED_MEMBER_NAME
        normalized_title = strip_parenthetical_text(str(title or "제목 없음").strip()) or "제목 없음"
        normalized_date = str(date).split("T", 1)[0].strip() if date is not None else ""
        if not normalized_date:
            raise ValueError("date is required to create a shared schedule")
        normalized_start_time = str(start_time or "미정").strip() or "미정"
        normalized_end_time = str(end_time or "미정").strip() or "미정"
        normalized_notes = strip_parenthetical_text(notes or "공유 일정") or "공유 일정"
        selected_schedule_id = schedule_id or new_id("shared")
        sync_status = "created"

        with self.connect() as conn:
            existing = conn.execute(
                "SELECT schedule_id FROM external_schedules WHERE schedule_id = ?",
                (selected_schedule_id,),
            ).fetchone()
            if existing:
                sync_status = "updated"
            conn.execute(
                """
                INSERT INTO external_schedules
                    (schedule_id, member_name, title, date, start_time, end_time, source_conversation_id, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(schedule_id) DO UPDATE SET
                    member_name = excluded.member_name,
                    title = excluded.title,
                    date = excluded.date,
                    start_time = excluded.start_time,
                    end_time = excluded.end_time,
                    source_conversation_id = excluded.source_conversation_id,
                    notes = excluded.notes
                """,
                (
                    selected_schedule_id,
                    normalized_member_name,
                    normalized_title,
                    normalized_date,
                    normalized_start_time,
                    normalized_end_time,
                    source_conversation_id,
                    normalized_notes,
                ),
            )

        return {
            "schedule_id": selected_schedule_id,
            "member_name": normalized_member_name,
            "title": normalized_title,
            "date": normalized_date,
            "start_time": normalized_start_time,
            "end_time": normalized_end_time,
            "source_conversation_id": source_conversation_id,
            "notes": normalized_notes,
            "sync_status": sync_status,
        }

    def delete_shared_schedules(
        self,
        schedule_id: str | None = None,
        source_conversation_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """공유 일정 저장소에서 ID 또는 원본 요청 ID에 해당하는 일정을 삭제합니다."""

        if not schedule_id and not source_conversation_id:
            return []

        where: list[str] = []
        params: list[Any] = []
        if schedule_id:
            where.append("schedule_id = ?")
            params.append(schedule_id)
        if source_conversation_id:
            where.append("source_conversation_id = ?")
            params.append(source_conversation_id)

        with self.connect() as conn:
            cur = conn.execute(
                f"""
                SELECT schedule_id, member_name, title, date, start_time, end_time, notes, source_conversation_id
                FROM external_schedules
                WHERE {" OR ".join(where)}
                """,
                params,
            )
            rows = [dict(row) for row in cur.fetchall()]
            conn.execute(f"DELETE FROM external_schedules WHERE {' OR '.join(where)}", params)
        return [strip_external_row_parentheticals(row) for row in rows]

    def list_shared_schedules(
        self,
        member_names: list[str] | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        source_conversation_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """공유 일정 저장소의 row를 필터링해 조회합니다."""

        where: list[str] = []
        params: list[Any] = []
        has_explicit_filter = any([member_names is not None, date_from, date_to, source_conversation_id])
        if not has_explicit_filter:
            member_names = JULY_PRACTICE_MEMBER_NAMES
            date_from = JULY_PRACTICE_DATE_FROM
            date_to = JULY_PRACTICE_DATE_TO

        normalized_members = [
            EXTERNAL_MEMBER_ALIAS.get(str(name).strip(), str(name).strip())
            for name in (member_names or [])
            if str(name).strip()
        ]
        if member_names is not None and not normalized_members:
            return []
        if normalized_members:
            placeholders = ",".join("?" for _ in normalized_members)
            where.append(f"member_name IN ({placeholders})")
            params.extend(normalized_members)

        normalized_date_from = str(date_from).split("T", 1)[0].strip() if date_from is not None else ""
        normalized_date_to = str(date_to).split("T", 1)[0].strip() if date_to is not None else ""
        if normalized_date_from:
            where.append("date >= ?")
            params.append(normalized_date_from)
        if normalized_date_to:
            where.append("date <= ?")
            params.append(normalized_date_to)
        if source_conversation_id:
            where.append("source_conversation_id = ?")
            params.append(source_conversation_id)

        sql = """
            SELECT schedule_id, member_name, title, date, start_time, end_time, notes, source_conversation_id
            FROM external_schedules
        """
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY date, start_time, member_name LIMIT ?"
        params.append(max(1, min(int(limit or 50), 200)))
        with self.connect() as conn:
            return [strip_external_row_parentheticals(dict(row)) for row in conn.execute(sql, params).fetchall()]

    def search_previous_conversations(
        self,
        query: str,
        member_names: list[str] | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """LLM이 넘긴 query와 멤버 필터로 외부 멤버의 과거 메시지를 검색합니다."""

        query_text = str(query or "").strip()
        clauses: list[str] = []
        params: list[Any] = []
        if member_names is not None:
            normalized_members = normalize_external_member_names(member_names)
            if not normalized_members:
                return []
            placeholders = ",".join("?" for _ in normalized_members)
            clauses.append(f"c.member_name IN ({placeholders})")
            params.extend(normalized_members)
        if query_text:
            clauses.append("(m.content LIKE ? OR c.title LIKE ?)")
            token = f"%{query_text}%"
            params.extend([token, token])
        sql = """
            SELECT c.conversation_id, c.member_name, c.title, m.content, m.created_at
            FROM external_conversations c
            JOIN external_messages m ON m.conversation_id = c.conversation_id
        """
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY m.created_at DESC LIMIT ?"
        params.append(limit)
        with self.connect() as conn:
            return [strip_external_row_parentheticals(dict(row)) for row in conn.execute(sql, params).fetchall()]

    def load_conversation_messages(self, conversation_id: str) -> list[dict[str, Any]]:
        """conversation_id 하나에 속한 외부 메시지를 작성 순서대로 반환합니다."""

        with self.connect() as conn:
            cur = conn.execute(
                """
                SELECT role, sender, content, created_at
                FROM external_messages
                WHERE conversation_id = ?
                ORDER BY created_at ASC
                """,
                (conversation_id,),
            )
            return [strip_external_row_parentheticals(dict(row)) for row in cur.fetchall()]

    def extract_schedules_from_history(
        self,
        member_names: list[str] | None,
        date_from: str,
        date_to: str,
    ) -> list[dict[str, Any]]:
        """멤버와 날짜 범위에 맞는 외부 busy-time row를 반환합니다.

        이 프로젝트에서는 "대화에서 일정을 추출"하는 기능을 실제 LLM 추출 대신
        seed된 `external_schedules` 테이블 조회로 재현합니다.
        """

        normalized_members = normalize_external_member_names(member_names)
        date_from, date_to = normalize_external_schedule_date_bounds(member_names, date_from, date_to)
        member_filter = ""
        params: list[Any] = [date_from, date_to]
        if member_names is not None:
            if not normalized_members:
                return []
            placeholders = ",".join("?" for _ in normalized_members)
            member_filter = f"AND member_name IN ({placeholders})"
            params.extend(normalized_members)
        with self.connect() as conn:
            cur = conn.execute(
                f"""
                SELECT member_name, title, date, start_time, end_time, notes, source_conversation_id
                FROM external_schedules
                WHERE date >= ?
                  AND date <= ?
                  {member_filter}
                ORDER BY date, start_time
                """,
                params,
            )
            return [strip_external_row_parentheticals(dict(row)) for row in cur.fetchall()]
