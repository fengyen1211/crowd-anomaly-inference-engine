"""
llm/factory.py
=================
LLM Provider 的工廠，跟 embedding/factory.py、vlm/factory.py 完全對稱的設計。

依照傳入的 provider_name，動態建立對應的 BaseLLMClient 實例，
在每個分支裡才 import 對應的 Provider 類別，避免不需要的相依套件。

目前支援 "mock"（預設，不需要任何外部依賴）、"openai"（可選，
需要 OPENAI_API_KEY）、"ollama"（本地執行 Qwen3 / Gemma3 / Llama3 等，
需要先啟動 Ollama 服務並 `ollama pull` 對應模型，見 llm/ollama_llm.py）。
"""

from typing import Optional

from llm.base import BaseLLMClient

SUPPORTED_PROVIDERS = ("mock", "openai", "ollama")


class LLMFactory:
    """依名稱建立 LLM Provider 的工廠類別。"""

    @classmethod
    def create(
        cls,
        provider_name: str,
        model: Optional[str] = None,
        **kwargs,
    ) -> BaseLLMClient:
        """
        Args:
            provider_name: "mock" / "openai"
            model: 模型名稱，不同 provider 有各自的預設值，可不傳。
            **kwargs: 傳給對應 Provider 建構子的其他參數。

        Raises:
            ValueError: provider_name 不在支援清單內。
        """
        if provider_name == "mock":
            from llm.mock import MockLLMClient

            return MockLLMClient(**kwargs)

        if provider_name == "openai":
            from llm.openai_llm import OpenAILLMClient

            return OpenAILLMClient(**kwargs)

        if provider_name == "ollama":
            from llm.ollama_llm import OllamaLLMClient

            return OllamaLLMClient(model=model or "qwen3:4b", **kwargs)

        raise ValueError(
            f"不支援的 LLM provider：'{provider_name}'，"
            f"支援的選項：{SUPPORTED_PROVIDERS}"
        )
