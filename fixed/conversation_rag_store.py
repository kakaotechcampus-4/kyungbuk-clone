from __future__ import annotations

"""SQLite 대화 목록을 ChromaDB 대화 청크로 동기화하고 검색하는 저장소입니다."""

import hashlib
import json
from pathlib import Path
from typing import Any

from fixed.app_store import AppSQLiteStore
from fixed.config import CONFIG
from fixed.reference_store import OpenAIEmbeddingFunction


class ConversationRAGStore:
    """앱 SQLite 대화 전사를 대화 단위 ChromaDB 청크로 관리합니다."""

    COLLECTION_NAME = "kanana_conversation_chunks_openai"

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

    def sync_from_sqlite(self, sqlite_store: AppSQLiteStore) -> dict[str, int]:
        """SQLite 대화 목록을 읽어 신규/변경/삭제분만 ChromaDB에 반영합니다."""

        chunks = self._conversation_chunks(sqlite_store)
        chunk_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
        existing = self._existing_metadata_by_id()

        stale_ids = [chunk_id for chunk_id in existing if chunk_id not in chunk_by_id]
        if stale_ids:
            self.collection.delete(ids=stale_ids)

        upsert_chunks: list[dict[str, Any]] = []
        skipped = 0
        for chunk_id, chunk in chunk_by_id.items():
            existing_metadata = existing.get(chunk_id) or {}
            if existing_metadata.get("source_hash") == chunk["metadata"]["source_hash"]:
                skipped += 1
                continue
            upsert_chunks.append(chunk)

        if upsert_chunks:
            self.collection.upsert(
                ids=[chunk["chunk_id"] for chunk in upsert_chunks],
                documents=[chunk["content"] for chunk in upsert_chunks],
                metadatas=[chunk["metadata"] for chunk in upsert_chunks],
            )

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

        for index, document in enumerate(documents):
            metadata = metadatas[index] or {}
            hit_conversation_id = str(metadata.get("conversation_id") or "")
            if not conversation_id and exclude_conversation_id and hit_conversation_id == exclude_conversation_id:
                continue
            hits.append(
                {
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
                        "message_count": metadata.get("message_count", 0),
                        "last_message_at": metadata.get("last_message_at", ""),
                        "source_hash": metadata.get("source_hash", ""),
                    },
                }
            )
            if len(hits) >= normalized_limit:
                break
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
            conversation = conversations[conversation_id]
            messages = conversation["messages"]
            if not messages:
                continue
            chunks.append(self._chunk_from_conversation(conversation))
        return chunks

    def _chunk_from_conversation(self, conversation: dict[str, Any]) -> dict[str, Any]:
        messages = conversation["messages"]
        title = str(conversation.get("title") or "새 대화")
        status = str(conversation.get("status") or "active")
        created_at = str(conversation.get("created_at") or "")
        updated_at = str(conversation.get("updated_at") or "")
        last_message_at = str(messages[-1].get("created_at") or "")
        source_hash = self._source_hash(conversation)
        lines = [
            f"대화 제목: {title}",
            f"대화 상태: {status}",
            f"대화 생성: {created_at}",
            f"대화 수정: {updated_at}",
            "[메시지]",
        ]
        for message in messages:
            role = str(message.get("role") or "")
            content = " ".join(str(message.get("content") or "").split())
            message_created_at = str(message.get("created_at") or "")
            lines.append(f"{message_created_at} | {role} | {content}")

        conversation_id = str(conversation["conversation_id"])
        return {
            "chunk_id": f"conversation:{conversation_id}",
            "content": "\n".join(lines),
            "metadata": {
                "conversation_id": conversation_id,
                "title": title,
                "status": status,
                "created_at": created_at,
                "updated_at": updated_at,
                "message_count": len(messages),
                "last_message_at": last_message_at,
                "source_hash": source_hash,
            },
        }

    def _source_hash(self, conversation: dict[str, Any]) -> str:
        source = {
            "conversation_id": conversation.get("conversation_id"),
            "title": conversation.get("title"),
            "status": conversation.get("status"),
            "created_at": conversation.get("created_at"),
            "updated_at": conversation.get("updated_at"),
            "messages": conversation.get("messages") or [],
        }
        encoded = json.dumps(source, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()
