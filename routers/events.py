"""
routers/events.py
====================
事件相關的 API 路由。

對應架構設計中的 API Flow：
- POST /events/ingest                 影像端呼叫，送入新事件 JSON
- GET  /events/{record_id}/analysis   Web 呼叫，取得完整推理結果
- POST /events/{record_id}/reanalyze  手動觸發重新分析

TODO：
- [ ] 目前 /ingest 是同步等待完整 pipeline 跑完才回應，事件量大時
      應該改成非同步排入佇列（例如 Celery / 背景任務），先回 202
      再非同步處理
- [ ] 補上適當的錯誤處理（例如 record_id 不存在時回傳 404，
      目前 ingest 已經有基本的驗證/檔案讀取失敗處理）
- [ ] /analysis、/reanalyze 需要資料庫持久化就緒後才能實作
      （目前 OrchestratorService.process_event() 是無狀態的單次執行，
      沒有把中間結果存下來，無法之後再查詢）
"""

from fastapi import APIRouter, Body, Depends, HTTPException
from pydantic import ValidationError

from app.dependencies import get_orchestrator_service
from services.orchestrator_service import OrchestratorService
from services.schemas import PipelineOutput

router = APIRouter(prefix="/events", tags=["events"])


@router.post("/ingest", response_model=PipelineOutput)
def ingest_event(
    raw_json: dict = Body(..., description="影像端輸出的單筆事件 JSON"),
    orchestrator: OrchestratorService = Depends(get_orchestrator_service),
) -> PipelineOutput:
    """
    接收影像端輸出的事件 JSON，同步執行完整推論端 pipeline
    （JSON -> VLM -> Embedding -> Retriever -> Knowledge -> LLM -> Output）。
    """
    try:
        return orchestrator.process_event(raw_json)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{record_id}/analysis")
def get_event_analysis(record_id: str):
    """
    取得指定事件的完整推理結果（VLM 觀察 + RAG 檢索 + LLM 分析）。

    TODO: 從資料庫查詢並組成回應。
    """
    raise NotImplementedError("TODO: 尚未實作 /events/{record_id}/analysis")


@router.post("/{record_id}/reanalyze")
def reanalyze_event(record_id: str):
    """
    手動觸發重新跑一次推論流程（例如更新知識庫後，想重新產生建議）。

    TODO: 從資料庫取回原始事件資料，重新呼叫 OrchestratorService。
    """
    raise NotImplementedError("TODO: 尚未實作 /events/{record_id}/reanalyze")
