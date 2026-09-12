"""
llm/schemas.py
================
定義 Reasoning Module（LLM）的輸出資料結構。

對應架構設計中的 [5] Reasoning Module：
輸入原始事件資料 + VLMObservation + RetrievedContext，
輸出這裡定義的 AnalysisResult。

TODO：
- [ ] 補上「引用來源」欄位，記錄每個 cause / action 是依據哪一筆
      RetrievedCase 或 KnowledgeSnippet 生成的，方便追溯與除錯
"""

from enum import Enum
from typing import List

from pydantic import BaseModel


class FinalVerdict(str, Enum):
    """
    LLM 綜合規則系統標籤、VLM 驗證結果、歷史相似案例後的最終判定。

    注意：這不是「重新分類事件類型」，而是對規則系統標籤的信任程度判斷。

    - CONFIRMED：維持原本規則系統的判斷
    - UNCERTAIN：證據不足，無法確定
    - OVERTURNED：證據顯示規則系統的判斷可能是錯的
    """

    CONFIRMED = "confirmed"
    UNCERTAIN = "uncertain"
    OVERTURNED = "overturned"


class AnalysisResult(BaseModel):
    """
    Reasoning Module 的完整輸出，會被送進 Alert Composer 產生警報。
    """

    record_id: str

    # 事件摘要（自然語言，給人看的完整敘述）
    summary: str

    # 最終判定
    final_verdict: FinalVerdict

    # 最終信心分數，取代影像端寫死的假 confidence
    final_confidence: float

    # 可能原因（需要 grounding 在 RetrievedContext 上，避免憑空生成）
    possible_causes: List[str]

    # 應變建議（需要 grounding 在知識庫 SOP 上）
    recommended_actions: List[str]
