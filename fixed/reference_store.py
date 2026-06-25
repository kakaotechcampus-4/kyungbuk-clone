from __future__ import annotations

"""Week 4 개인 참고자료 RAG를 위한 ChromaDB 저장소입니다."""

from pathlib import Path
from typing import Any

from fixed.config import CONFIG, PROXY_TOKEN_PLACEHOLDER
from fixed.store_base import new_id


class OpenAIEmbeddingFunction:
    """ChromaDB가 호출할 수 있는 OpenAI embeddings adapter입니다."""

    def __init__(self, api_key: str | None, base_url: str, model: str):
        """OpenAI 호환 embeddings API 호출에 필요한 설정을 보관합니다."""

        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self._client: Any | None = None

    def name(self) -> str:
        """ChromaDB가 embedding function을 식별할 때 사용할 이름을 반환합니다."""

        return f"openai_{self.model}".replace("/", "_")

    def is_legacy(self) -> bool:
        """ChromaDB의 custom embedding function 호환 경로 사용 여부를 알립니다."""

        return True

    def _openai_client(self) -> Any:
        """OpenAI client를 지연 생성하고 PROXY_TOKEN 누락을 명확한 오류로 바꿉니다."""

        if not self.api_key or self.api_key.strip() == PROXY_TOKEN_PLACEHOLDER:
            raise RuntimeError("PROXY_TOKEN이 필요합니다. .env에 키를 추가한 뒤 다시 실행하세요.")
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def __call__(self, input: list[str]) -> list[list[float]]:
        """ChromaDB가 문서/쿼리 embedding을 요청할 때 호출하는 진입점입니다."""

        response = self._openai_client().embeddings.create(model=self.model, input=input)
        return [item.embedding for item in response.data]

    def embed_query(self, input: list[str]) -> list[list[float]]:
        """일부 Chroma/LangChain 경로가 기대하는 query embedding 메서드입니다."""

        return self(input)

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        """일부 Chroma/LangChain 경로가 기대하는 document embedding 메서드입니다."""

        return self(input)


class PersonalReferenceStore:
    """Week 4 개인 참고자료 RAG 저장소입니다.

    참고자료는 ChromaDB에 저장하고, 벡터는 `.env`의 embedding proxy 설정으로
    생성합니다. PROXY_TOKEN이 없으면 앱 import는 가능하지만 실제 add/query
    시점에 명확한 오류를 냅니다.
    """

    COLLECTION_NAME = "kanana_personal_references_openai"
    DEFAULT_REFERENCES = [
        {
            "id": "ref_focus",
            "title": "집중 회의 선호",
            "content": "나는 오전 10시에서 12시 사이에 집중도가 높아서 중요한 회의는 오전 중반을 선호한다.",
            "tags": ["preference", "meeting"],
        },
        {
            "id": "ref_lunch",
            "title": "점심 시간 보호",
            "content": "점심 시간 12:00-13:00은 되도록 회의 없이 비워둔다.",
            "tags": ["preference", "lunch"],
        },
        {
            "id": "ref_sync",
            "title": "팀 싱크 방식",
            "content": "팀 싱크는 60분 이하로 잡고 회의 전날 아젠다를 공유하면 좋다.",
            "tags": ["team", "meeting"],
        },
    ]

    def __init__(self, chroma_dir: Path):
        """ChromaDB persistent collection을 열고, 키가 있으면 기본 참고자료를 seed합니다."""

        import chromadb

        self.chroma_dir = chroma_dir
        chroma_dir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(chroma_dir))
        self.collection = client.get_or_create_collection(
            self.COLLECTION_NAME,
            embedding_function=OpenAIEmbeddingFunction(
                api_key=CONFIG.proxy_token,
                base_url=CONFIG.embedding_proxy_url,
                model=CONFIG.openai_embedding_model,
            ),
            metadata={
                "description": "Kanana course personal references",
                "embedding_provider": "openai",
                "embedding_model": CONFIG.openai_embedding_model,
            },
        )
        if CONFIG.has_openai_key:
            self.seed()

    def backend_info(self) -> dict[str, Any]:
        """Week 4가 사용하는 vector store와 embedding backend를 설명합니다."""

        return {
            "vector_store": "chromadb",
            "embedding_provider": "openai",
            "embedding_model": CONFIG.openai_embedding_model,
            "embedding_base_url": CONFIG.embedding_proxy_url,
            "collection_name": self.COLLECTION_NAME,
            "chroma_dir": str(self.chroma_dir),
        }

    def seed(self) -> None:
        """기본 개인 참고자료가 비어 있을 때만 추가합니다."""

        if self.collection.count():
            return
        self.collection.add(
            ids=[item["id"] for item in self.DEFAULT_REFERENCES],
            documents=[item["content"] for item in self.DEFAULT_REFERENCES],
            metadatas=[{"title": item["title"], "tags": ",".join(item["tags"])} for item in self.DEFAULT_REFERENCES],
        )

    def add_personal_reference(self, title: str, content: str, tags: list[str] | None = None) -> dict[str, Any]:
        """개인 참고자료 하나를 ChromaDB에 저장하고 저장된 메타데이터를 반환합니다."""

        reference_id = new_id("ref")
        self.collection.add(
            ids=[reference_id],
            documents=[content],
            metadatas=[{"title": title, "tags": ",".join(tags or [])}],
        )
        return {
            "reference_id": reference_id,
            "title": title,
            "content": content,
            "tags": tags or [],
            "backend": self.backend_info(),
        }

    def search_personal_references(self, query: str, limit: int = 3) -> list[dict[str, Any]]:
        """query와 가까운 개인 참고자료를 ChromaDB에서 검색합니다."""

        result = self.collection.query(query_texts=[query], n_results=limit)
        hits: list[dict[str, Any]] = []
        for index, document in enumerate(result.get("documents", [[]])[0]):
            metadata = result.get("metadatas", [[]])[0][index] or {}
            distance = result.get("distances", [[]])[0][index]
            hits.append(
                {
                    "id": result.get("ids", [[]])[0][index],
                    "title": metadata.get("title", ""),
                    "content": document,
                    "tags": metadata.get("tags", ""),
                    "distance": distance,
                }
            )
        return hits
