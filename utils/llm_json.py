"""
utils/llm_json.py
====================
解析 LLM 回傳的 JSON 字串。

從 llm/reasoning_pipeline.py 抽出，讓其他也需要「要求 LLM 輸出 JSON」的
呼叫端（例如新聞摘要腳本）可以共用同一套容錯邏輯，不用各自重寫一份。
"""

import json
from typing import Any, Dict

from utils.logger import get_logger

logger = get_logger(__name__)


def parse_json_response(raw_response: str) -> Dict[str, Any]:
    """
    解析 LLM 回傳的 JSON 字串。

    LLM 有時會不小心用 Markdown 程式碼區塊（```json ... ```）包住 JSON，
    儘管 prompt 已經要求不要這樣做，這裡做基本的清理，提高容錯度。

    另外用 strict=False 容許字串值內出現未跳脫的控制字元（例如摘要文字
    裡直接換行，而不是寫成 \\n）——實測發現本地小型模型（Gemma3 4b）
    產生長篇摘要時常會這樣做，字串本身其實是合法內容，只是不符合嚴格
    JSON 語法，用 strict=False 放寬這點比要求模型「絕對不要換行」更可靠。

    TODO: 之後若真實 LLM 常常輸出不合法的 JSON，考慮改用
          LangChain 的 PydanticOutputParser 或加上重試機制。
    """
    text = raw_response.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        return json.loads(text, strict=False)
    except json.JSONDecodeError as exc:
        logger.error("LLM 回應無法解析成 JSON：%s", raw_response)
        raise ValueError(f"LLM 回應不是合法的 JSON：{exc}") from exc
