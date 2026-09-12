"""
services/schemas.py
======================
定義 OrchestratorService 對外輸出的最終結果格式。

這是整個推論端 pipeline（JSON -> VLM -> Embedding -> Retriever ->
Knowledge -> LLM）的最終輸出，是給呼叫端（Web / API）看的精簡結果，
不是內部各模組的完整資料結構——那些細節分別在 vlm.schemas.VLMObservation、
rag.schemas.RetrievedContext、llm.schemas.AnalysisResult、
utils.schemas.Alert 裡，這裡只保留最外部呼叫端在意的欄位。
"""

from typing import List

from pydantic import BaseModel

from utils.schemas import AlertSeverity


class PipelineOutput(BaseModel):
    """OrchestratorService.process_event() 的最終回傳結果。"""

    summary: str
    event_type: str
    possible_causes: List[str]
    recommendation: List[str]
    alert_level: AlertSeverity
