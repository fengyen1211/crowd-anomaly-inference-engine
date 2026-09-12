"""
vlm/schemas.py
================
定義 VLM（視覺語言模型）模組的輸出資料結構。

這個 schema 是 VLM Observation Module 的「輸出介面」：
不論底層實際呼叫哪一個 VLM 供應商（OpenAI GPT-4o、Gemini、
自架 LLaVA...），最終都必須轉換成這個統一格式，
下游的 RAG / LLM 模組才不會受供應商更換影響。

TODO：
- [ ] 之後可加入 bounding box 層級的細部觀察
- [ ] 之後可加入多幀（前後幾幀）比對的結果欄位
"""

from enum import Enum
from typing import List

from pydantic import BaseModel, Field


class VerificationVerdict(str, Enum):
    """
    VLM 對「規則系統判斷的事件類型」的驗證結論。

    - SUPPORT：畫面內容支持規則系統的判斷
    - CONTRADICT：畫面內容與規則系統的判斷矛盾
    - UNCERTAIN：畫面內容無法明確支持或反駁（例如畫面模糊、遮擋）
    """

    SUPPORT = "support"
    CONTRADICT = "contradict"
    UNCERTAIN = "uncertain"


class VLMObservation(BaseModel):
    """
    VLM Observation Module 的輸出結果。

    對應架構設計中的 [3] VLM Observation Module：
    - 輸入：畫面（裁切圖 + 原圖）、predicted_event_type、
            collective_motion、模板 caption
    - 輸出：這個 class
    """

    record_id: str

    # VLM 產生的、獨立於規則系統的畫面描述（不是模板組字，是真的看圖生成）
    visual_description: str

    # VLM 對規則系統標籤的驗證結論
    verification_verdict: VerificationVerdict

    # VLM 自評的信心程度（0.0 ~ 1.0），用來取代影像端寫死的假 confidence
    visual_confidence: float = Field(..., ge=0.0, le=1.0)

    # 規則系統看不到、但畫面中存在的額外線索
    # 例如：["煙霧", "出口方向", "障礙物"]
    extra_context: List[str] = Field(default_factory=list)
