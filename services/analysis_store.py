"""
services/analysis_store.py
=============================
保存「初步警報已送出、完整分析在背景補上」的事件狀態，供
GET /events/{record_id}/analysis 查詢（見 routers/events.py）。

★ 暫時的記憶體版本 ★
還沒有資料庫，結果只存在這個行程的記憶體裡：服務重啟就消失，
多個 worker 行程之間也不共用。之後接上資料庫時，把這個類別換成
資料庫實作即可，router 只依賴這裡的四個方法。
"""

import threading
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Optional

from pydantic import BaseModel

from services.schemas import PipelineOutput
from utils.schemas import Alert


class AnalysisStatus(str, Enum):
    ANALYZING = "analyzing"
    DONE = "done"
    FAILED = "failed"


class AnalysisRecord(BaseModel):
    """單筆事件目前的處理狀態。"""

    record_id: str
    status: AnalysisStatus
    preliminary_alert: Alert
    # status=done 時才有值
    result: Optional[PipelineOutput] = None
    # status=failed 時才有值
    error: Optional[str] = None
    updated_at: datetime


class InMemoryAnalysisStore:
    """執行緒安全的記憶體存放區：背景任務寫入、API 請求讀取。"""

    def __init__(self) -> None:
        self._records: Dict[str, AnalysisRecord] = {}
        self._lock = threading.Lock()

    def start(self, preliminary_alert: Alert) -> AnalysisRecord:
        """登記初步警報、狀態設為 analyzing。同一 record_id 重送時覆寫舊結果。"""
        record = AnalysisRecord(
            record_id=preliminary_alert.record_id,
            status=AnalysisStatus.ANALYZING,
            preliminary_alert=preliminary_alert,
            updated_at=_now(),
        )
        with self._lock:
            self._records[record.record_id] = record
        return record

    def complete(self, record_id: str, result: PipelineOutput) -> None:
        self._update(record_id, status=AnalysisStatus.DONE, result=result)

    def fail(self, record_id: str, error: str) -> None:
        self._update(record_id, status=AnalysisStatus.FAILED, error=error)

    def get(self, record_id: str) -> Optional[AnalysisRecord]:
        with self._lock:
            return self._records.get(record_id)

    def _update(self, record_id: str, **changes) -> None:
        with self._lock:
            current = self._records[record_id]
            self._records[record_id] = current.model_copy(update={**changes, "updated_at": _now()})


def _now() -> datetime:
    return datetime.now(timezone.utc)
