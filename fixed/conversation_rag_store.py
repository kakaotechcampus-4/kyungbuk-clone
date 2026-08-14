from __future__ import annotations

"""SQLite 대화 목록을 ChromaDB 메시지 window 청크로 동기화하고 검색하는 저장소입니다.

대화 1건을 통째로 청크 하나에 담으면 메시지가 한 건 붙을 때마다 대화 전문을 다시
embedding해야 해서 누적 비용이 대화 길이의 제곱으로 늘어납니다. 그래서 메시지를
`WINDOW_SIZE`개씩 끊어 청크로 만들고, 이미 가득 찬 window는 내용이 더 바뀌지 않으므로
다시 embedding하지 않습니다.

대화 제목은 사용자가 대화를 부르는 이름이라 검색어로 자주 쓰이므로 embedding 본문에
포함합니다. 제목은 메시지를 붙일 때 바뀌지 않으므로 append 비용은 여전히 window 1개입니다.
반대로 상태와 수정 시각은 메시지를 붙일 때마다 바뀌지만 검색 대상 본문은 아니므로
`source_hash`에서 제외하고 metadata만 갱신합니다.
"""

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from fixed.app_store import AppSQLiteStore
from fixed.config import CONFIG
from fixed.reference_store import OpenAIEmbeddingFunction


class ConversationRAGStore:
    """앱 SQLite 대화 전사를 메시지 window 단위 ChromaDB 청크로 관리합니다."""

    COLLECTION_NAME = "kanana_conversation_chunks_openai"
    WINDOW_SIZE = 8
    # 제목은 embedding 본문에 들어가므로 metadata-only 갱신 대상이 아닙니다.
    MUTABLE_METADATA_FIELDS = ("status", "updated_at")
    # 한 대화가 검색 결과를 독점하지 않도록 대화당 기본 window 수를 제한합니다.
    MAX_WINDOWS_PER_CONVERSATION = 2

    def __init__(
        self,
        chroma_dir: Path,
        *,
        embedding_function: Any | None = None,
        collection_name: str | None = None,
    ) -> None:
        """ChromaDB collection을 열고 테스트용 embedding/collection 주입을 허용합니다."""

        import chromadb

        self.chroma_dir = chroma_dir
        self.collection_name = collection_name or self.COLLECTION_NAME
        self.embedding_function = embedding_function or OpenAIEmbeddingFunction(
            api_key=CONFIG.proxy_token,
            base_url=CONFIG.embedding_proxy_url,
            model=CONFIG.openai_embedding_model,
        )
        chroma_dir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(chroma_dir))
        self.collection = client.get_or_create_collection(
            self.collection_name,
            embedding_function=self.embedding_function,
            metadata={
                "description": "Kanana SQLite conversation chunks",
                "embedding_provider": "openai",
                "embedding_model": CONFIG.openai_embedding_model,
            },
        )
        self._sync_cache_key: tuple[str, tuple[str, ...]] | None = None
        self._sync_total = 0

    def backend_info(self) -> dict[str, Any]:
        """Tool payload에 포함할 vector backend 정보를 반환합니다."""

        return {
            "vector_store": "chromadb",
            "embedding_provider": "openai",
            "embedding_model": CONFIG.openai_embedding_model,
            "embedding_base_url": CONFIG.embedding_proxy_url,
            "collection_name": self.collection_name,
            "chroma_dir": str(self.chroma_dir),
        }

    def sync_from_sqlite(
        self,
        sqlite_store: AppSQLiteStore,
        *,
        skip_conversation_ids: str | Iterable[str] | None = None,
    ) -> dict[str, int]:
        """SQLite 대화 목록을 읽어 신규/변경/삭제분만 ChromaDB에 반영합니다.

        `skip_conversation_ids`에 있는 대화는 어차피 검색에서 제외되므로 embedding도
        하지 않습니다. 기존 청크는 지우지 않고 그대로 두어, 나중에 그 대화가 검색
        대상이 될 때 한 번만 다시 embedding되게 합니다.
        """

        skip_ids = self._normalized_skip_ids(skip_conversation_ids)
        cache_key = (self._sqlite_fingerprint(sqlite_store), tuple(sorted(skip_ids)))
        if cache_key == self._sync_cache_key:
            return {"upserted": 0, "skipped": self._sync_total, "deleted": 0, "total": self._sync_total}

        chunks = self._conversation_chunks(sqlite_store)
        chunk_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
        existing = self._existing_metadata_by_id()

        stale_ids = [chunk_id for chunk_id in existing if chunk_id not in chunk_by_id]
        if stale_ids:
            self.collection.delete(ids=stale_ids)

        upsert_chunks: list[dict[str, Any]] = []
        refresh_chunks: list[dict[str, Any]] = []
        skipped = 0
        for chunk_id, chunk in chunk_by_id.items():
            metadata = chunk["metadata"]
            existing_metadata = existing.get(chunk_id) or {}
            if existing_metadata.get("source_hash") == metadata["source_hash"]:
                skipped += 1
                if self._metadata_needs_refresh(existing_metadata, metadata):
                    refresh_chunks.append(chunk)
                continue
            if metadata["conversation_id"] in skip_ids:
                skipped += 1
                continue
            upsert_chunks.append(chunk)

        if upsert_chunks:
            self.collection.upsert(
                ids=[chunk["chunk_id"] for chunk in upsert_chunks],
                documents=[chunk["content"] for chunk in upsert_chunks],
                metadatas=[chunk["metadata"] for chunk in upsert_chunks],
            )
        if refresh_chunks:
            # document를 넘기지 않으므로 embedding 재계산 없이 metadata만 갱신됩니다.
            self.collection.update(
                ids=[chunk["chunk_id"] for chunk in refresh_chunks],
                metadatas=[chunk["metadata"] for chunk in refresh_chunks],
            )

        self._sync_cache_key = cache_key
        self._sync_total = len(chunks)
        return {
            "upserted": len(upsert_chunks),
            "skipped": skipped,
            "deleted": len(stale_ids),
            "total": len(chunks),
        }

    def search(
        self,
        *,
        query: str,
        top_k: int = 5,
        exclude_conversation_id: str | None = None,
        conversation_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """query와 가까운 대화 청크를 검색합니다."""

        query_text = str(query or "").strip()
        if not query_text:
            return []

        collection_count = self.collection.count()
        if collection_count <= 0:
            return []

        try:
            normalized_limit = int(top_k or 5)
        except (TypeError, ValueError):
            normalized_limit = 5
        normalized_limit = max(1, min(normalized_limit, 50))
        fetch_limit = min(collection_count, max(normalized_limit * 5, normalized_limit))
        where = {"conversation_id": conversation_id} if conversation_id else None
        if where:
            result = self.collection.query(query_texts=[query_text], n_results=fetch_limit, where=where)
        else:
            result = self.collection.query(query_texts=[query_text], n_results=fetch_limit)

        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        ids = result.get("ids", [[]])[0]
        hits: list[dict[str, Any]] = []
        # 한 대화의 window들이 결과를 독점하지 않도록 대화당 개수를 제한하고,
        # 넘친 것은 자리가 남을 때만 뒤에 채웁니다. 특정 대화를 지목한 검색은 제한하지 않습니다.
        overflow: list[dict[str, Any]] = []
        windows_per_conversation: dict[str, int] = {}

        for index, document in enumerate(documents):
            metadata = metadatas[index] or {}
            hit_conversation_id = str(metadata.get("conversation_id") or "")
            if not conversation_id and exclude_conversation_id and hit_conversation_id == exclude_conversation_id:
                continue
            hit = {
                "chunk_id": ids[index],
                "conversation_id": hit_conversation_id,
                "title": metadata.get("title", ""),
                "status": metadata.get("status", ""),
                "content": document,
                "distance": distances[index] if index < len(distances) else None,
                "metadata": {
                    "conversation_id": hit_conversation_id,
                    "title": metadata.get("title", ""),
                    "status": metadata.get("status", ""),
                    "created_at": metadata.get("created_at", ""),
                    "updated_at": metadata.get("updated_at", ""),
                    "window_index": metadata.get("window_index", 0),
                    "message_count": metadata.get("message_count", 0),
                    "last_message_at": metadata.get("last_message_at", ""),
                    "source_hash": metadata.get("source_hash", ""),
                },
            }
            seen_windows = windows_per_conversation.get(hit_conversation_id, 0)
            if conversation_id or seen_windows < self.MAX_WINDOWS_PER_CONVERSATION:
                windows_per_conversation[hit_conversation_id] = seen_windows + 1
                hits.append(hit)
                if len(hits) >= normalized_limit:
                    return hits
            else:
                overflow.append(hit)

        for hit in overflow:
            if len(hits) >= normalized_limit:
                break
            hits.append(hit)
        return hits

    def context_from_hits(self, hits: list[dict[str, Any]]) -> str:
        """검색 결과를 agent가 바로 근거로 쓰기 쉬운 문자열로 만듭니다."""

        lines = ["[SQLite 대화 RAG 검색 결과]"]
        if not hits:
            lines.append("- 검색된 이전 대화가 없습니다.")
            return "\n".join(lines)

        for index, hit in enumerate(hits, start=1):
            metadata = hit.get("metadata") or {}
            title = hit.get("title") or "새 대화"
            conversation_id = hit.get("conversation_id") or "unknown"
            updated_at = metadata.get("updated_at") or "시간 미정"
            lines.append(f"[{index}] {title} | conversation_id={conversation_id} | updated_at={updated_at}")
            lines.append(str(hit.get("content") or "").strip())
        return "\n\n".join(lines)

    def _normalized_skip_ids(self, skip_conversation_ids: str | Iterable[str] | None) -> frozenset[str]:
        """문자열 하나 또는 여러 개를 받아 공백을 제거한 conversation_id 집합으로 만듭니다."""

        if not skip_conversation_ids:
            return frozenset()
        candidates = [skip_conversation_ids] if isinstance(skip_conversation_ids, str) else list(skip_conversation_ids)
        return frozenset(str(candidate).strip() for candidate in candidates if str(candidate).strip())

    def _sqlite_fingerprint(self, sqlite_store: AppSQLiteStore) -> str:
        """대화/메시지 테이블의 가벼운 지문을 만들어 불필요한 재청킹을 막습니다.

        메시지 본문까지 읽지 않고 대화 row와 메시지 개수/마지막 rowid만 봅니다.
        앱에서 메시지는 추가만 되고 수정되지 않으므로 이 조합으로 충분합니다.
        """

        with sqlite_store.connect() as conn:
            conversation_rows = [
                [
                    str(row["conversation_id"]),
                    str(row["title"] or ""),
                    str(row["status"] or ""),
                    str(row["updated_at"] or ""),
                ]
                for row in conn.execute(
                    """
                    SELECT conversation_id, title, status, updated_at
                    FROM conversations
                    WHERE status IN ('active', 'archived')
                    ORDER BY conversation_id ASC
                    """
                ).fetchall()
            ]
            message_row = conn.execute(
                """
                SELECT COUNT(*) AS message_count,
                       COALESCE(MAX(rowid), 0) AS max_rowid,
                       COALESCE(MAX(created_at), '') AS max_created_at
                FROM messages
                """
            ).fetchone()

        source = {
            "conversations": conversation_rows,
            "messages": [
                int(message_row["message_count"]),
                int(message_row["max_rowid"]),
                str(message_row["max_created_at"]),
            ],
        }
        return self._digest(source)

    def _metadata_needs_refresh(self, existing_metadata: dict[str, Any], metadata: dict[str, Any]) -> bool:
        """embedding은 그대로 두고 metadata만 갱신하면 되는지 판단합니다."""

        return any(
            str(existing_metadata.get(field) or "") != str(metadata.get(field) or "")
            for field in self.MUTABLE_METADATA_FIELDS
        )

    def _existing_metadata_by_id(self) -> dict[str, dict[str, Any]]:
        result = self.collection.get(include=["metadatas"])
        ids = result.get("ids", [])
        metadatas = result.get("metadatas", [])
        return {chunk_id: (metadatas[index] or {}) for index, chunk_id in enumerate(ids)}

    def _conversation_chunks(self, sqlite_store: AppSQLiteStore) -> list[dict[str, Any]]:
        conversations: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        with sqlite_store.connect() as conn:
            rows = [
                dict(row)
                for row in conn.execute(
                    """
                    SELECT c.conversation_id,
                           c.title,
                           c.status,
                           c.created_at AS conversation_created_at,
                           c.updated_at AS conversation_updated_at,
                           m.message_id,
                           m.role,
                           m.content,
                           m.created_at AS message_created_at
                    FROM conversations c
                    LEFT JOIN messages m ON m.conversation_id = c.conversation_id
                    WHERE c.status IN ('active', 'archived')
                    ORDER BY c.created_at ASC, c.rowid ASC, m.created_at ASC, m.rowid ASC
                    """
                ).fetchall()
            ]

        for row in rows:
            conversation_id = str(row["conversation_id"])
            if conversation_id not in conversations:
                conversations[conversation_id] = {
                    "conversation_id": conversation_id,
                    "title": row.get("title") or "새 대화",
                    "status": row.get("status") or "active",
                    "created_at": row.get("conversation_created_at") or "",
                    "updated_at": row.get("conversation_updated_at") or "",
                    "messages": [],
                }
                order.append(conversation_id)
            if row.get("message_id"):
                conversations[conversation_id]["messages"].append(
                    {
                        "message_id": row.get("message_id") or "",
                        "role": row.get("role") or "",
                        "content": row.get("content") or "",
                        "created_at": row.get("message_created_at") or "",
                    }
                )

        chunks: list[dict[str, Any]] = []
        for conversation_id in order:
            chunks.extend(self._chunks_from_conversation(conversations[conversation_id]))
        return chunks

    def _chunks_from_conversation(self, conversation: dict[str, Any]) -> list[dict[str, Any]]:
        """대화 하나를 `WINDOW_SIZE`개 메시지씩 끊어 청크 목록으로 만듭니다."""

        messages = conversation["messages"]
        if not messages:
            return []

        conversation_id = str(conversation["conversation_id"])
        title = str(conversation.get("title") or "새 대화")
        shared_metadata = {
            "conversation_id": conversation_id,
            "title": title,
            "status": str(conversation.get("status") or "active"),
            "created_at": str(conversation.get("created_at") or ""),
            "updated_at": str(conversation.get("updated_at") or ""),
        }
        windows = [messages[start : start + self.WINDOW_SIZE] for start in range(0, len(messages), self.WINDOW_SIZE)]
        return [
            {
                "chunk_id": f"conversation:{conversation_id}#w{window_index}",
                "content": "\n".join(
                    [f"대화 제목: {title}", *(self._message_line(message) for message in window)]
                ),
                "metadata": {
                    **shared_metadata,
                    "window_index": window_index,
                    "message_count": len(window),
                    "last_message_at": str(window[-1].get("created_at") or ""),
                    "source_hash": self._source_hash(conversation_id, window_index, title, window),
                },
            }
            for window_index, window in enumerate(windows)
        ]

    def _message_line(self, message: dict[str, Any]) -> str:
        """메시지 한 건을 embedding 대상 한 줄로 만듭니다."""

        role = str(message.get("role") or "")
        content = " ".join(str(message.get("content") or "").split())
        created_at = str(message.get("created_at") or "")
        return f"{created_at} | {role} | {content}"

    def _source_hash(
        self,
        conversation_id: str,
        window_index: int,
        title: str,
        messages: list[dict[str, Any]],
    ) -> str:
        """embedding 본문에 들어가는 값만 해싱합니다.

        제목은 본문에 포함되므로 해시에도 넣습니다. 제목이 바뀌면 그 대화의 window만
        다시 embedding되는데, 제목은 메시지를 붙일 때 바뀌지 않으므로 append 비용에는
        영향이 없습니다.

        반대로 상태와 대화 수정 시각은 본문이 아니므로 제외합니다. 이 값들이 해시에
        들어가 있으면 메시지 한 건이 추가될 때마다 과거 window까지 전부 다시
        embedding됩니다.
        """

        source = {
            "conversation_id": conversation_id,
            "window_index": window_index,
            "title": title,
            "messages": messages,
        }
        return self._digest(source)

    def _digest(self, source: Any) -> str:
        encoded = json.dumps(source, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()
