from __future__ import annotations

"""앱 내부 SQLite 데이터베이스를 다루는 저장소입니다.

이 저장소는 두 종류의 데이터를 함께 관리합니다.
1. Gradio 채팅 UI가 보여 줄 대화와 메시지
2. Week 3 이후 agent가 저장한 structured request, 일정, 할 일, 알림

개인 일정은 외부 공유 일정 저장소에도 복사되어 Week 5/6의 여러 사람 일정 조율에서
"나"의 busy time으로 보이므로, 저장/수정/삭제 시 외부 MCP 동기화도 함께 수행합니다.
"""

import json
from pathlib import Path
from typing import Any

from fixed.external_mcp import (
    delete_group_schedule_from_shared,
    delete_personal_schedule_from_shared,
    sync_group_schedule_to_shared,
    sync_personal_schedule_to_shared,
)
from fixed.store_base import (
    SCHEDULE_COLUMNS,
    SCHEDULE_COLUMNS_WITH_KIND,
    SQLiteFileStore,
    decode_schedule_row,
    new_id,
    now_iso,
)


class AppSQLiteStore(SQLiteFileStore):
    """앱 내부 DB 저장소입니다.

    대화 로그, Week 3 structured output, 정규화된 개인 일정/할 일/알림을 같은
    SQLite 파일에 보관합니다. Week 4+ 도구는 이 저장소의 schedules 테이블을
    RAG 후보 데이터로 사용합니다.
    """

    def __init__(self, path: Path):
        """DB 파일 경로를 준비하고 필요한 테이블을 생성합니다."""

        super().__init__(path)
        self.initialize()

    def initialize(self) -> None:
        """앱 DB 스키마를 생성합니다.

        이미 테이블이 있으면 그대로 두므로 앱 시작 때마다 안전하게 호출할 수 있습니다.
        `structured_requests`는 LLM이 뽑은 원본 payload를 보관하고, `schedules`,
        `todos`, `reminders`는 화면/검색/삭제에 쓰기 쉬운 정규화 row입니다.
        """

        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    message_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(conversation_id) REFERENCES conversations(conversation_id)
                );

                CREATE TABLE IF NOT EXISTS structured_requests (
                    request_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    title TEXT,
                    date TEXT,
                    start_time TEXT,
                    end_time TEXT,
                    members_json TEXT NOT NULL DEFAULT '[]',
                    priority TEXT,
                    reason TEXT,
                    raw_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS schedules (
                    schedule_id TEXT PRIMARY KEY,
                    request_id TEXT,
                    owner TEXT NOT NULL DEFAULT 'me',
                    title TEXT NOT NULL,
                    date TEXT,
                    start_time TEXT,
                    end_time TEXT,
                    attendees_json TEXT NOT NULL DEFAULT '[]',
                    source TEXT NOT NULL DEFAULT 'structured_output',
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS todos (
                    todo_id TEXT PRIMARY KEY,
                    request_id TEXT,
                    title TEXT NOT NULL,
                    due_date TEXT,
                    priority TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS reminders (
                    reminder_id TEXT PRIMARY KEY,
                    request_id TEXT,
                    title TEXT NOT NULL,
                    date TEXT,
                    start_time TEXT,
                    reason TEXT,
                    created_at TEXT NOT NULL
                );
                """
            )

    # Conversation history

    def create_conversation(self, title: str = "새 대화") -> dict[str, Any]:
        """새 채팅 대화를 만들고 UI가 사용할 `conversation_id`와 제목을 반환합니다."""

        conversation_id = new_id("conv")
        created_at = now_iso()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO conversations (conversation_id, title, status, created_at, updated_at)
                VALUES (?, ?, 'active', ?, ?)
                """,
                (conversation_id, title[:80] or "새 대화", created_at, created_at),
            )
        return {"conversation_id": conversation_id, "title": title[:80] or "새 대화"}

    def append_message(self, conversation_id: str, role: str, content: str) -> dict[str, Any]:
        """대화에 user/assistant 메시지를 추가하고 대화의 최근 갱신 시각을 갱신합니다.

        새 대화의 기본 제목이 아직 "새 대화"라면 첫 메시지 앞부분으로 제목도 자동 설정합니다.
        """

        message_id = new_id("msg")
        created_at = now_iso()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO messages (message_id, conversation_id, role, content, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (message_id, conversation_id, role, content, created_at),
            )
            conn.execute(
                "UPDATE conversations SET updated_at = ?, title = CASE WHEN title = '새 대화' THEN ? ELSE title END WHERE conversation_id = ?",
                (created_at, content[:40] or "새 대화", conversation_id),
            )
        return {"message_id": message_id, "conversation_id": conversation_id}

    def list_conversations(self) -> list[dict[str, Any]]:
        """사이드바에 보여 줄 최근 활성 대화 목록을 반환합니다."""

        with self.connect() as conn:
            cur = conn.execute(
                """
                SELECT c.conversation_id, c.title, c.status, c.updated_at,
                       COUNT(m.message_id) AS message_count,
                       COALESCE((SELECT content FROM messages WHERE conversation_id = c.conversation_id ORDER BY created_at DESC, rowid DESC LIMIT 1), '') AS last_message
                FROM conversations c
                LEFT JOIN messages m ON m.conversation_id = c.conversation_id
                WHERE c.status = 'active'
                GROUP BY c.conversation_id
                ORDER BY c.updated_at DESC, c.rowid DESC
                LIMIT 30
                """
            )
            return [dict(row) for row in cur.fetchall()]

    def load_conversation(self, conversation_id: str) -> list[dict[str, str]]:
        """한 대화의 메시지를 시간순으로 불러옵니다."""

        with self.connect() as conn:
            cur = conn.execute(
                """
                SELECT role, content FROM messages
                WHERE conversation_id = ?
                ORDER BY created_at ASC, rowid ASC
                """,
                (conversation_id,),
            )
            return [dict(row) for row in cur.fetchall()]

    def search_conversation_messages(
        self,
        query: str,
        conversation_id: str | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """앱 DB에 저장된 일반 채팅 메시지를 검색합니다.

        `search_saved_requests`가 구조화된 일정/할 일/알림만 보는 것과 달리,
        이 조회는 Gradio 채팅 UI의 `messages` 테이블을 대상으로 합니다.
        """

        query_text = str(query or "").strip()
        if not query_text:
            return []

        try:
            normalized_limit = int(limit or 5)
        except (TypeError, ValueError):
            normalized_limit = 5
        normalized_limit = max(1, min(normalized_limit, 50))
        fetch_limit = max(normalized_limit * 10, 50)
        clauses: list[str] = []
        params: list[Any] = []
        if conversation_id:
            clauses.append("m.conversation_id = ?")
            params.append(conversation_id)
        clauses.append("(m.content LIKE ? OR c.title LIKE ?)")
        token = f"%{query_text}%"
        params.extend([token, token])

        sql = """
            SELECT m.message_id, m.conversation_id, c.title AS conversation_title,
                   m.role, m.content, m.created_at
            FROM messages m
            JOIN conversations c ON c.conversation_id = m.conversation_id
        """
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY m.created_at DESC, m.rowid DESC LIMIT ?"
        params.append(fetch_limit)

        with self.connect() as conn:
            rows = [dict(row) for row in conn.execute(sql, params).fetchall()]

        lowered_query = query_text.lower()

        def score(row: dict[str, Any]) -> tuple[int, str]:
            content_text = str(row.get("content") or "").lower()
            title_text = str(row.get("conversation_title") or "").lower()
            value = 0
            if lowered_query in content_text:
                value += min(len(query_text), 8)
            if lowered_query in title_text:
                value += 1
            if row.get("role") == "user":
                value += 3
            if str(row.get("content") or "").strip().endswith(("?", "？")):
                value -= 5
            return value, str(row.get("created_at") or "")

        rows.sort(key=score, reverse=True)
        return rows[:normalized_limit]

    def archive_conversation(self, conversation_id: str) -> dict[str, Any]:
        """대화를 목록에서 숨기기 위해 상태를 archived로 바꿉니다."""

        with self.connect() as conn:
            conn.execute(
                "UPDATE conversations SET status = 'archived', updated_at = ? WHERE conversation_id = ?",
                (now_iso(), conversation_id),
            )
        return {"conversation_id": conversation_id, "status": "archived"}

    def delete_conversation(self, conversation_id: str | None) -> dict[str, Any]:
        """대화와 그 메시지를 물리 삭제합니다. ID가 없으면 아무 작업도 하지 않습니다."""

        if not conversation_id:
            return {"conversation_id": "", "deleted": False}
        with self.connect() as conn:
            conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
            cur = conn.execute("DELETE FROM conversations WHERE conversation_id = ?", (conversation_id,))
        return {"conversation_id": conversation_id, "deleted": cur.rowcount > 0}

    # Week 3 structured output persistence

    def save_structured_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """LLM structured output을 원본 row와 목적별 정규화 row로 저장합니다.

        `payload["kind"]`에 따라 저장 위치가 달라집니다.
        - personal_schedule/group_schedule: `structured_requests`와 `schedules`에 저장
        - todo: `structured_requests`와 `todos`에 저장
        - reminder: `structured_requests`와 `reminders`에 저장

        개인 일정은 "나" busy-time으로, 그룹 일정은 참석자별 busy-time으로 외부 공유 저장소에도 복사합니다.
        """

        request_id = new_id("req")
        kind = payload.get("kind", "unknown")
        title = payload.get("title") or "제목 없음"
        date = payload.get("date")
        start_time = payload.get("start_time")
        end_time = payload.get("end_time")
        members = payload.get("members") or []
        priority = payload.get("priority")
        reason = payload.get("reason")
        created_at = now_iso()
        saved_rows: list[dict[str, Any]] = []
        shared_sync: dict[str, Any] | None = None
        schedule_for_shared: dict[str, Any] | None = None
        source_schedule_id = str(payload.get("source_schedule_id") or "").strip()

        with self.connect() as conn:
            if kind in {"personal_schedule", "group_schedule"} and source_schedule_id:
                existing_schedule = conn.execute(
                    """
                    SELECT schedule_id, request_id
                    FROM schedules
                    WHERE schedule_id = ?
                    """,
                    (source_schedule_id,),
                ).fetchone()
                if existing_schedule is not None:
                    return {
                        "request_id": existing_schedule["request_id"],
                        "kind": kind,
                        "saved_rows": [
                            {"table": "structured_requests", "id": existing_schedule["request_id"], "existing": True},
                            {"table": "schedules", "id": existing_schedule["schedule_id"], "existing": True},
                        ],
                        "shared_sync": None,
                        "already_exists": True,
                    }
            conn.execute(
                """
                INSERT INTO structured_requests
                    (request_id, kind, title, date, start_time, end_time, members_json, priority, reason, raw_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request_id,
                    kind,
                    title,
                    date,
                    start_time,
                    end_time,
                    json.dumps(members, ensure_ascii=False),
                    priority,
                    reason,
                    json.dumps(payload, ensure_ascii=False),
                    created_at,
                ),
            )
            saved_rows.append({"table": "structured_requests", "id": request_id})

            # structured_requests는 원본 감사 로그이고, 아래 분기들은 agent/tool이
            # 빠르게 조회하기 쉬운 테이블로 payload를 한 번 더 풀어 저장합니다.
            if kind in {"personal_schedule", "group_schedule"}:
                schedule_id = source_schedule_id or new_id("sch")
                conn.execute(
                    """
                    INSERT INTO schedules
                        (schedule_id, request_id, owner, title, date, start_time, end_time, attendees_json, source, created_at)
                    VALUES (?, ?, 'me', ?, ?, ?, ?, ?, 'structured_output', ?)
                    """,
                    (
                        schedule_id,
                        request_id,
                        title,
                        date,
                        start_time,
                        end_time,
                        json.dumps(members, ensure_ascii=False),
                        created_at,
                    ),
                )
                saved_rows.append({"table": "schedules", "id": schedule_id})
                if kind in {"personal_schedule", "group_schedule"}:
                    schedule_for_shared = {
                        "schedule_id": schedule_id,
                        "request_id": request_id,
                        "owner": "me",
                        "title": title,
                        "date": date,
                        "start_time": start_time,
                        "end_time": end_time,
                        "attendees": members,
                        "source": "structured_output",
                        "created_at": created_at,
                    }
            elif kind == "todo":
                todo_id = new_id("todo")
                conn.execute(
                    """
                    INSERT INTO todos (todo_id, request_id, title, due_date, priority, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (todo_id, request_id, title, date, priority, created_at),
                )
                saved_rows.append({"table": "todos", "id": todo_id})
            elif kind == "reminder":
                reminder_id = new_id("rem")
                conn.execute(
                    """
                    INSERT INTO reminders (reminder_id, request_id, title, date, start_time, reason, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (reminder_id, request_id, title, date, start_time, reason, created_at),
                )
                saved_rows.append({"table": "reminders", "id": reminder_id})

        if schedule_for_shared is not None:
            # 외부 공유 저장소 동기화는 앱 DB transaction 바깥에서 수행합니다.
            # 외부 MCP 실패가 앱 DB 저장 자체를 되돌리지 않게 하기 위함입니다.
            if kind == "group_schedule":
                shared_sync = sync_group_schedule_to_shared(schedule_for_shared)
            else:
                shared_sync = sync_personal_schedule_to_shared(schedule_for_shared)

        return {"request_id": request_id, "kind": kind, "saved_rows": saved_rows, "shared_sync": shared_sync}

    # Structured request lookup

    def list_saved_requests(
        self,
        kind: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """저장된 structured request를 종류와 날짜 범위로 필터링해 조회합니다."""

        where: list[str] = []
        params: list[Any] = []
        if kind:
            where.append("kind = ?")
            params.append(kind)
        if date_from:
            where.append("date >= ?")
            params.append(date_from)
        if date_to:
            where.append("date <= ?")
            params.append(date_to)
        query = "SELECT * FROM structured_requests"
        if where:
            query += " WHERE " + " AND ".join(where)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self.connect() as conn:
            return [dict(row) for row in conn.execute(query, params).fetchall()]

    def get_saved_request(self, request_id: str) -> dict[str, Any] | None:
        """request_id로 저장된 structured request 하나를 조회합니다."""

        with self.connect() as conn:
            cur = conn.execute("SELECT * FROM structured_requests WHERE request_id = ?", (request_id,))
            row = cur.fetchone()
            return dict(row) if row else None

    def search_saved_requests(self, query: str, kind: str | None = None, limit: int = 5) -> list[dict[str, Any]]:
        """저장된 요청 원문/제목/근거를 단순 LIKE 검색으로 찾습니다.

        Week 4의 RAG 보조 도구가 SQLite 기록에서 일정/할 일/알림 근거를 찾을 때 사용합니다.
        """

        query_text = str(query or "").strip()
        clauses = []
        params: list[Any] = []
        if kind:
            clauses.append("kind = ?")
            params.append(kind)
        if query_text:
            clauses.append("(raw_json LIKE ? OR title LIKE ? OR reason LIKE ?)")
            token = f"%{query_text}%"
            params.extend([token, token, token])
        sql = "SELECT * FROM structured_requests"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self.connect() as conn:
            return [dict(row) for row in conn.execute(sql, params).fetchall()]

    # Schedule lookup and deletion

    def list_schedules(
        self,
        limit: int = 12,
        kind: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[dict[str, Any]]:
        """저장된 일정 후보를 날짜/시간순으로 반환합니다."""

        where: list[str] = []
        params: list[Any] = []
        if kind:
            where.append("(SELECT kind FROM structured_requests WHERE request_id = schedules.request_id) = ?")
            params.append(kind)
        if date_from:
            where.append("date >= ?")
            params.append(date_from)
        if date_to:
            where.append("date <= ?")
            params.append(date_to)

        query = f"""
                SELECT {SCHEDULE_COLUMNS_WITH_KIND}
                FROM schedules
                """
        if where:
            query += " WHERE " + " AND ".join(where)
        query += """
                ORDER BY (date IS NULL), date ASC, (start_time IS NULL), start_time ASC, created_at DESC
                LIMIT ?
                """
        params.append(limit)
        with self.connect() as conn:
            cur = conn.execute(query, params)
            rows = [dict(row) for row in cur.fetchall()]
        return [decode_schedule_row(row) for row in rows]

    def update_schedule(
        self,
        schedule_id: str,
        title: str | None = None,
        date: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        attendees: list[str] | None = None,
    ) -> dict[str, Any] | None:
        """앱 DB의 일정 원본을 수정하고 필요한 경우 공유 일정 복사본도 갱신합니다.

        schedules 테이블뿐 아니라 연결된 `structured_requests.raw_json`도 같은 값으로 갱신합니다.
        이렇게 해야 이후 조회/RAG/삭제 도구가 서로 다른 값을 보지 않습니다.
        """

        with self.connect() as conn:
            row = conn.execute(
                f"""
                SELECT {SCHEDULE_COLUMNS_WITH_KIND}
                FROM schedules
                WHERE schedule_id = ?
                """,
                (schedule_id,),
            ).fetchone()
            if row is None:
                return None

            current = decode_schedule_row(dict(row))
            next_attendees = attendees if attendees is not None else current.get("attendees", [])
            updated = {
                **current,
                "title": title if title is not None else current.get("title"),
                "date": date if date is not None else current.get("date"),
                "start_time": start_time if start_time is not None else current.get("start_time"),
                "end_time": end_time if end_time is not None else current.get("end_time"),
                "attendees": next_attendees,
            }
            attendees_json = json.dumps(next_attendees, ensure_ascii=False)
            conn.execute(
                """
                UPDATE schedules
                SET title = ?,
                    date = ?,
                    start_time = ?,
                    end_time = ?,
                    attendees_json = ?
                WHERE schedule_id = ?
                """,
                (
                    updated["title"],
                    updated["date"],
                    updated["start_time"],
                    updated["end_time"],
                    attendees_json,
                    schedule_id,
                ),
            )
            raw_row = conn.execute(
                "SELECT raw_json FROM structured_requests WHERE request_id = ?",
                (current.get("request_id"),),
            ).fetchone()
            raw_payload: dict[str, Any] = {}
            if raw_row:
                try:
                    raw_payload = json.loads(raw_row["raw_json"] or "{}")
                except Exception:
                    raw_payload = {}
            raw_payload.update(
                {
                    "title": updated["title"],
                    "date": updated["date"],
                    "start_time": updated["start_time"],
                    "end_time": updated["end_time"],
                    "members": next_attendees,
                }
            )
            # structured_requests는 "사용자 요청의 원본 기록" 역할도 하지만,
            # 저장 일정 수정 이후에는 화면과 검색 결과가 최신 값을 보여야 하므로 같이 갱신합니다.
            conn.execute(
                """
                UPDATE structured_requests
                SET title = ?,
                    date = ?,
                    start_time = ?,
                    end_time = ?,
                    members_json = ?,
                    raw_json = ?
                WHERE request_id = ?
                  AND kind IN ('personal_schedule', 'group_schedule')
                """,
                (
                    updated["title"],
                    updated["date"],
                    updated["start_time"],
                    updated["end_time"],
                    attendees_json,
                    json.dumps(raw_payload, ensure_ascii=False),
                    current.get("request_id"),
                ),
            )

        shared_sync = None
        if updated.get("request_kind") == "personal_schedule":
            shared_sync = sync_personal_schedule_to_shared(updated)
        elif updated.get("request_kind") == "group_schedule":
            delete_group_schedule_from_shared(current)
            shared_sync = sync_group_schedule_to_shared(updated)
        return {"schedule": updated, "shared_sync": shared_sync}

    def find_schedules(
        self,
        schedule_ids: list[str] | None = None,
        date: str | None = None,
        title: str | None = None,
        start_time: str | None = None,
        time_unspecified: bool = False,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """일정 ID나 날짜/제목/시간 필터에 맞는 저장 일정을 찾습니다.

        삭제/수정 전 agent가 후보를 좁히는 용도입니다. `time_unspecified=True`는
        "시간 미정 일정"처럼 start_time이 비어 있는 row를 찾을 때 사용합니다.
        """

        if schedule_ids is not None and not schedule_ids:
            return []

        where: list[str] = []
        params: list[Any] = []
        if schedule_ids is not None:
            placeholders = ", ".join("?" for _ in schedule_ids)
            where.append(f"schedule_id IN ({placeholders})")
            params.extend(schedule_ids)
        if date:
            where.append("date = ?")
            params.append(date)
        if title:
            where.append("title LIKE ?")
            params.append(f"%{title}%")
        if start_time:
            where.append("start_time = ?")
            params.append(start_time)
        if time_unspecified:
            where.append("(start_time IS NULL OR start_time = '' OR start_time = '미정')")

        sql = f"""
            SELECT {SCHEDULE_COLUMNS}
            FROM schedules
        """
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        with self.connect() as conn:
            rows = [dict(row) for row in conn.execute(sql, params).fetchall()]
        return [decode_schedule_row(row) for row in rows]

    def delete_schedule(self, schedule_id: str) -> dict[str, Any] | None:
        """schedule_id 하나를 삭제하고 연결된 structured request도 함께 정리합니다."""

        with self.connect() as conn:
            row = conn.execute(
                f"""
                SELECT {SCHEDULE_COLUMNS_WITH_KIND}
                FROM schedules
                WHERE schedule_id = ?
                """,
                (schedule_id,),
            ).fetchone()
            if row is None:
                return None
            decoded = decode_schedule_row(dict(row))
            conn.execute("DELETE FROM schedules WHERE schedule_id = ?", (schedule_id,))
            conn.execute(
                """
                DELETE FROM structured_requests
                WHERE request_id = ?
                  AND kind IN ('personal_schedule', 'group_schedule')
                """,
                (row["request_id"],),
            )

        if decoded.get("request_kind") == "personal_schedule" and decoded.get("request_id"):
            # 개인 일정은 외부 공유 저장소에 복사본이 있으므로 앱 DB 삭제와 함께 제거합니다.
            delete_personal_schedule_from_shared(decoded["request_id"])
        elif decoded.get("request_kind") == "group_schedule" and decoded.get("request_id"):
            delete_group_schedule_from_shared(decoded)

        return decoded

    def delete_schedules_by_filter(
        self,
        schedule_ids: list[str] | None = None,
        date: str | None = None,
        title: str | None = None,
        start_time: str | None = None,
        time_unspecified: bool = False,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """필터로 찾은 일정들을 모두 삭제하고 삭제된 row 목록을 반환합니다."""

        if schedule_ids is None and not any([date, title, start_time, time_unspecified]):
            return []

        rows = self.find_schedules(
            schedule_ids=schedule_ids,
            date=date,
            title=title,
            start_time=start_time,
            time_unspecified=time_unspecified,
            limit=limit,
        )
        deleted: list[dict[str, Any]] = []
        for row in rows:
            deleted_row = self.delete_schedule(row["schedule_id"])
            if deleted_row:
                deleted.append(deleted_row)
        return deleted

    def delete_all_schedules(self) -> list[dict[str, Any]]:
        """앱 DB에 저장된 모든 일정과 일정 구조화 요청을 삭제합니다.

        개인 일정의 공유 저장소 복사본도 request_id 기준으로 함께 삭제합니다.
        """

        with self.connect() as conn:
            cur = conn.execute(
                f"""
                SELECT {SCHEDULE_COLUMNS_WITH_KIND}
                FROM schedules
                ORDER BY created_at DESC
                """
            )
            deleted_rows = [dict(row) for row in cur.fetchall()]
            conn.execute("DELETE FROM schedules")
            conn.execute(
                """
                DELETE FROM structured_requests
                WHERE kind IN ('personal_schedule', 'group_schedule')
                """
            )

        decoded_rows = [decode_schedule_row(row) for row in deleted_rows]
        for row in decoded_rows:
            if row.get("request_kind") == "personal_schedule" and row.get("request_id"):
                delete_personal_schedule_from_shared(row["request_id"])
            elif row.get("request_kind") == "group_schedule" and row.get("request_id"):
                delete_group_schedule_from_shared(row)
        return decoded_rows
