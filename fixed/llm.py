from __future__ import annotations

"""LangChain agent들이 공통으로 쓰는 채팅 모델 생성 모듈입니다."""

from langchain_openai import ChatOpenAI

from fixed.config import CONFIG


def chat_model(*, temperature: float = 0) -> ChatOpenAI:
    """프록시 서버를 사용하는 공통 채팅 모델 생성 helper입니다."""

    if not CONFIG.has_openai_key:
        raise RuntimeError("PROXY_TOKEN이 .env에 필요합니다.")
    return ChatOpenAI(
        model=CONFIG.openai_model,
        api_key=CONFIG.proxy_token,
        base_url=CONFIG.chat_proxy_url,
        temperature=temperature,
    )
