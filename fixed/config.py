from __future__ import annotations

"""앱 전체에서 공유하는 환경 설정을 한 번 읽어 고정하는 모듈입니다.

이 프로젝트는 수업용 로컬 앱이라 실행 중 설정이 바뀌는 것을 전제로 하지 않습니다.
따라서 import 시점에 `.env`를 읽고 `CONFIG` 객체 하나로 경로, 모델명, 프록시 URL을
공유합니다.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT
DATA_DIR = PACKAGE_ROOT / "data"
STATIC_DIR = PACKAGE_ROOT / "static"
BRAND_DIR = STATIC_DIR / "brand"
PROXY_TOKEN_PLACEHOLDER = "여기에 api key 입력"
DEFAULT_CHAT_PROXY_URL = "https://mlapi.run/4bbd0c4d-bf02-4e59-a635-457b1c30c56a/v1"
DEFAULT_EMBEDDING_PROXY_URL = "https://mlapi.run/b54ff33e-6d14-42df-93f9-0f1132160ee8/v1"


@dataclass(frozen=True)
class AppConfig:
    """저장소의 .env 파일에서 읽어 온 실행 설정입니다."""

    proxy_token: str | None
    chat_proxy_url: str
    embedding_proxy_url: str
    openai_model: str
    openai_embedding_model: str
    use_llm: bool
    llm_assist: bool
    active_week: int
    app_db_path: Path
    external_db_path: Path
    chroma_dir: Path

    @property
    def has_openai_key(self) -> bool:
        """실제 API 키가 설정되어 있는지 확인합니다.

        학생용 `.env.example`에는 안내 문구가 들어갈 수 있으므로 값이 있더라도
        placeholder와 같으면 키가 없는 상태로 취급합니다.
        """

        if not self.proxy_token:
            return False
        return self.proxy_token.strip() != PROXY_TOKEN_PLACEHOLDER


def load_config() -> AppConfig:
    """비밀 값을 출력하지 않고 `.env`와 기본값을 합쳐 `AppConfig`를 만듭니다.

    데이터베이스와 ChromaDB가 들어갈 `data/` 디렉터리도 여기서 보장합니다.
    앱의 다른 모듈은 파일 경로나 모델명을 직접 계산하지 않고 `CONFIG`만 참조합니다.
    """

    load_dotenv(REPO_ROOT / ".env")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    proxy_token = os.getenv("PROXY_TOKEN")
    use_llm = os.getenv("KANANA_USE_LLM", "0").lower() in {"1", "true", "yes", "on"}
    llm_assist = os.getenv("KANANA_LLM_ASSIST", os.getenv("KANANA_USE_LLM", "0")).lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    active_week_raw = os.getenv("KANANA_ACTIVE_WEEK", "1").strip()
    try:
        active_week = int(active_week_raw)
    except ValueError:
        active_week = 1

    return AppConfig(
        proxy_token=proxy_token,
        chat_proxy_url=os.getenv("CHAT_PROXY_URL", DEFAULT_CHAT_PROXY_URL),
        embedding_proxy_url=os.getenv("EMBEDDING_PROXY_URL", DEFAULT_EMBEDDING_PROXY_URL),
        openai_model=os.getenv("OPENAI_MODEL", "openai/gpt-4.1-mini"),
        openai_embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "openai/text-embedding-3-small"),
        use_llm=use_llm,
        llm_assist=llm_assist,
        active_week=active_week,
        app_db_path=DATA_DIR / "kanana_app.sqlite3",
        external_db_path=DATA_DIR / "kanana_external_people.sqlite3",
        chroma_dir=DATA_DIR / "chroma",
    )


CONFIG = load_config()
