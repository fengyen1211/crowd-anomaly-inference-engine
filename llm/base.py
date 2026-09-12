"""
llm/base.py
=============
定義 LLM 客戶端的抽象介面。

設計理念與 vlm/base.py 相同：上層模組只依賴這個抽象介面，
不直接依賴特定供應商（OpenAI / Anthropic / 其他），
方便未來更換或同時支援多個 LLM 供應商。

這裡刻意設計成「通用的對話式介面」（chat_completion），
而不是分別設計「產生摘要的介面」「產生建議的介面」...等多個方法，
原因是：
- 摘要 / 原因推論 / 建議生成 / 最終判定，這些不同任務的差異
  應該體現在 prompts/ 模組裡的不同 prompt 內容，
  而不是體現在 LLM client 要實作很多不同方法。
- 這樣之後要新增新的推理任務時，只需要新增新的 prompt，
  不需要修改 BaseLLMClient 介面。

TODO：
- [ ] 確認是否需要支援 streaming 回應（給 Web 即時顯示用）
- [ ] 確認是否需要支援 function calling / structured output
"""

from abc import ABC, abstractmethod
from typing import Dict, List


class BaseLLMClient(ABC):
    """
    所有 LLM 客戶端實作都必須繼承這個抽象類別。
    """

    @abstractmethod
    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        執行一次對話式生成。

        Args:
            messages: OpenAI 風格的訊息列表，
                      例如 [{"role": "system", "content": "..."},
                            {"role": "user", "content": "..."}]
            **kwargs: 供應商特定的額外參數（例如 temperature、max_tokens）。

        Returns:
            str: 模型生成的文字內容。

        TODO: 各供應商子類別實作實際 API 呼叫邏輯。
        """
        raise NotImplementedError
