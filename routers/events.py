"""
routers/events.py
====================
事件相關的 API 路由。

對應架構設計中的 API Flow：
- POST /events/ingest                 影像端呼叫，送入新事件 JSON
- GET  /events/{record_id}/analysis   Web 呼叫，取得完整推理結果
- POST /events/{record_id}/reanalyze  手動觸發重新分析

/ingest 採「先發警報，報告晚點補」：VLM 看完畫面、初步警報一產生就回
202，RAG + LLM 在背景執行，結果寫進 services/analysis_store.py，
再由 /analysis 查詢。

TODO：
- [ ] 背景任務目前用 FastAPI BackgroundTasks，跟 API 同一個行程；事件量
      大時應該改成獨立的工作佇列（例如 Celery），也才能跨 worker 共用結果
- [ ] analysis_store 是記憶體版本，服務重啟就消失，需要資料庫持久化
- [ ] /reanalyze 需要資料庫保存原始事件資料後才能實作
"""

from fastapi import APIRouter, BackgroundTasks, Body, Depends, HTTPException
from pydantic import ValidationError

from app.dependencies import get_analysis_store, get_orchestrator_service
from app.security import verify_api_key
from services.analysis_store import AnalysisRecord, InMemoryAnalysisStore
from services.orchestrator_service import ObservationStage, OrchestratorService
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/events", tags=["events"], dependencies=[Depends(verify_api_key)])


@router.post("/ingest", response_model=AnalysisRecord, status_code=202)
def ingest_event(
    background_tasks: BackgroundTasks,
    raw_json: dict = Body(..., description="影像端輸出的單筆事件 JSON"),
    orchestrator: OrchestratorService = Depends(get_orchestrator_service),
    store: InMemoryAnalysisStore = Depends(get_analysis_store),
) -> AnalysisRecord:
    """
    接收影像端輸出的事件 JSON，VLM 看完畫面就回傳初步警報（202，
    status=analyzing），RAG + LLM 在背景補上完整分析，完成後用
    GET /events/{record_id}/analysis 查詢。
    """
    try:
        stage = orchestrator.observe(raw_json)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    record = store.start(stage.preliminary_alert)
    background_tasks.add_task(_analyze_in_background, orchestrator, store, stage)
    return record


def _analyze_in_background(
    orchestrator: OrchestratorService, store: InMemoryAnalysisStore, stage: ObservationStage
) -> None:
    record_id = stage.record.record_id
    try:
        store.complete(record_id, orchestrator.analyze(stage))
    except Exception as exc:  # noqa: BLE001 - 背景任務沒有人接例外，必須記下來讓 /analysis 查得到
        logger.exception("record_id=%s 背景分析失敗", record_id)
        store.fail(record_id, f"{type(exc).__name__}: {exc}")


@router.get("/{record_id}/analysis", response_model=AnalysisRecord)
def get_event_analysis(
    record_id: str,
    store: InMemoryAnalysisStore = Depends(get_analysis_store),
) -> AnalysisRecord:
    """
    取得指定事件目前的處理狀態：analyzing（只有初步警報）、done（附完整
    結果）、failed（附錯誤訊息）。
    """
    record = store.get(record_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"找不到 record_id={record_id}（服務重啟後記錄會消失）")
    return record


@router.post("/{record_id}/reanalyze")
def reanalyze_event(record_id: str):
    """
    手動觸發重新跑一次推論流程（例如更新知識庫後，想重新產生建議）。

    TODO: 從資料庫取回原始事件資料，重新呼叫 OrchestratorService。
    """
    raise NotImplementedError("TODO: 尚未實作 /events/{record_id}/reanalyze")
