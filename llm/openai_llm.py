"""
llm/openai_llm.py
====================
BaseLLMClient 的 OpenAI 實作（預設供應商）。

底層預計透過 LangChain 的 ChatOpenAI 呼叫（而不是直接用 openai SDK），
方便之後利用 LangChain 生態系的其他功能
（例如 output parser、chain、memory）。

目前只是骨架，尚未實作真正的呼叫邏輯。

TODO：
- [ ] 初始化 langchain_openai.ChatOpenAI(
          model=settings.openai_llm_model,
          api_key=settings.openai_api_key,
      )
- [ ] 將 messages 轉換成 LangChain 的訊息格式
      （HumanMessage / SystemMessage / AIMessage）
- [ ] 呼叫 self._chat_model.invoke(...) 並回傳生成的文字內容
"""

from typing import Dict, List

from config.settings import Settings, get_settings
from llm.base import BaseLLMClient
from utils.logger import get_logger

logger = get_logger(__name__)


class OpenAILLMClient(BaseLLMClient):
    """
    使用 LangChain 的 ChatOpenAI 實作 LLM 對話生成功能。
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """
        Args:
            settings: 系統設定物件。
        """
        self._settings = settings or get_settings()
        self._model = self._settings.openai_llm_model
        self._chat_model = None  # TODO: 實際初始化 ChatOpenAI 實例

    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        TODO: 見檔案頂端的 TODO 說明。
        """
        logger.info("OpenAILLMClient.chat_completion() 尚未實作")
        raise NotImplementedError("TODO: 尚未實作 OpenAI LLM 呼叫邏輯")
