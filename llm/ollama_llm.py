"""
llm/ollama_llm.py
====================
BaseLLMClient 的 Ollama 實作。

Ollama 是本地執行開源 LLM 的服務，跑在背景（預設
http://localhost:11434），透過 HTTP API 呼叫，模型量化、載入、
記憶體管理都交給 Ollama 處理。這也是為什麼 Qwen3 / Gemma3 / Llama3
這幾個「本地 LLM」候選共用同一個 Provider 類別就夠了：差別只在
config.yaml 的 model 名稱（例如 "qwen3:4b" / "gemma3:4b"），
不需要為每個候選各寫一份 transformers 版本的載入邏輯。

TODO：
- [ ] 支援 streaming 回應（目前固定 stream=False，一次拿到完整回應）
"""

from typing import Dict, List, Optional

import requests

from config.settings import Settings, get_settings
from llm.base import BaseLLMClient
from utils.logger import get_logger

logger = get_logger(__name__)


class OllamaLLMClient(BaseLLMClient):
    """透過本地 Ollama 服務呼叫開源 LLM（Qwen3 / Gemma3 / Llama3 等）。"""

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        settings: Optional[Settings] = None,
        timeout: float = 300.0,
    ) -> None:
        """
        Args:
            model: Ollama 已下載的模型標籤（例如 "qwen3:4b"，對應 `ollama list` 的 NAME）。
            base_url: Ollama 服務位址，預設讀取 Settings.ollama_base_url。
            timeout: 單次請求逾時秒數（本地小模型在消費級硬體上生成
                     需要一段時間，預設放寬到 300 秒）。
        """
        settings = settings or get_settings()
        self._model = model
        self._base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self._timeout = timeout

    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> str:
        payload = {"model": self._model, "messages": messages, "stream": False}
        if kwargs:
            payload["options"] = kwargs

        logger.info("OllamaLLMClient 呼叫 model=%s", self._model)
        response = requests.post(f"{self._base_url}/api/chat", json=payload, timeout=self._timeout)
        response.raise_for_status()
        return response.json()["message"]["content"]
