"""
llm/mock.py
==============
BaseLLMClient 的假資料實作（Mock）。

★★★ 目前預設的 LLM Provider（跟 embedding/mock.py、vlm/mock.py 呼應）★★★

用途：
- 在還沒接上真正的本地 LLM（Qwen3 / Gemma3 / Llama3）或決定要不要用
  OpenAI 之前，先讓整個推論端 pipeline（尤其是 llm/reasoning_pipeline.py）
  可以真的端到端跑起來、可以測試。
- 不呼叫任何外部 API、不需要任何 API Key。

設計方式：依 prompts/ 組出來的訊息內容，判斷目前是哪一個任務
（summary / reasoning / alert），回傳格式正確、內容誠實標示為
「[MOCK]」的假回應，而不是隨機亂數字串——這樣下游的 JSON 解析邏輯
（llm/reasoning_pipeline.py）才能被真的測試到，而不是被假資料繞過去。
"""

import json
from typing import Dict, List

from llm.base import BaseLLMClient
from utils.logger import get_logger

logger = get_logger(__name__)

# 這些標記字串來自 prompts/reasoning_prompt.py 與 prompts/alert_prompt.py
# 的任務指示文字，用來判斷目前是哪一個任務。
# 注意：這是刻意的耦合——Mock 本來就需要知道 prompt 的內容長什麼樣子，
# 才能回傳格式正確的假回應；換成真正的 LLM 後，這個判斷邏輯就不需要了。
_REASONING_MARKER = "原因推論與驗證"
_ALERT_MARKER = "警報內容生成"


class MockLLMClient(BaseLLMClient):
    """依 prompt 內容判斷任務類型，回傳格式正確的假回應。"""

    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> str:
        user_content = self._latest_user_content(messages)

        if _REASONING_MARKER in user_content:
            return self._mock_reasoning_response(user_content)
        if _ALERT_MARKER in user_content:
            return self._mock_alert_response()
        return self._mock_summary_response()

    @staticmethod
    def _latest_user_content(messages: List[Dict[str, str]]) -> str:
        for message in reversed(messages):
            if message.get("role") == "user":
                return message.get("content", "")
        return ""

    @staticmethod
    def _mock_reasoning_response(user_content: str) -> str:
        """
        依「檢索結果區塊裡有沒有查到東西」決定假回應的內容，
        藉此也測試到 reasoning_prompt.py 要求的「查無資料要誠實說」規則。
        """
        has_similar_cases = "查無相似的歷史事件紀錄" not in user_content
        has_knowledge = "查無相關的知識庫文件片段" not in user_content

        causes = (
            ["[MOCK] 依據檢索到的相似歷史事件推測的可能原因"]
            if has_similar_cases
            else ["證據不足，無法判斷可能原因"]
        )
        actions = (
            ["[MOCK] 依據檢索到的知識庫規範產生的應變建議"]
            if has_knowledge
            else ["證據不足，無法提供具體應變建議"]
        )

        payload = {
            "final_verdict": "confirmed" if has_similar_cases else "uncertain",
            "final_confidence": 0.75 if has_similar_cases else 0.4,
            "possible_causes": causes,
            "recommended_actions": actions,
        }
        logger.info("MockLLMClient 回傳假的 reasoning 結果")
        return json.dumps(payload, ensure_ascii=False)

    @staticmethod
    def _mock_summary_response() -> str:
        logger.info("MockLLMClient 回傳假的 summary 結果")
        return "[MOCK 摘要] 系統偵測到一起群體異常行為事件，詳細內容請參考原始事件資料。"

    @staticmethod
    def _mock_alert_response() -> str:
        payload = {
            "title": "[MOCK] 異常事件警報",
            "short_message": "[MOCK] 偵測到異常行為，請留意現場狀況。",
            "full_report": "[MOCK] 這是測試用的完整報告內容，尚未接上真實 LLM。",
        }
        logger.info("MockLLMClient 回傳假的 alert 結果")
        return json.dumps(payload, ensure_ascii=False)
