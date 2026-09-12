"""
services/ingestion_service.py
================================
對應架構設計中的 [1] Ingestion Layer。

負責：
- 接收影像端輸出的 JSON（單筆或整批）
- 驗證資料格式（透過 utils.schemas.RawEventRecord 的 Pydantic 驗證）

目前範圍：只做「驗證」，不做「儲存」——資料庫連線
（database/db.py）還沒有實作，持久化留給之後接上 DB 時再處理，
這裡先確保 pipeline 的其餘部分（VLM/Embedding/RAG/LLM）可以拿到
一份「保證欄位齊全、型別正確」的事件資料。

TODO：
- [ ] database/db.py 的 DatabaseManager 就緒後，把驗證通過的資料寫入
      EventRecordORM（見 database/models.py），狀態標記為 pending
- [ ] 支援去重（同一個 record_id 重複匯入時的處理策略）
"""

from typing import List

from utils.logger import get_logger
from utils.schemas import RawEventRecord

logger = get_logger(__name__)


class IngestionService:
    """
    負責驗證影像端輸出的事件資料。
    """

    def ingest_single(self, raw_json: dict) -> RawEventRecord:
        """
        Args:
            raw_json: 單筆事件的原始 JSON dict。

        Returns:
            RawEventRecord: 驗證通過的內部資料物件。

        Raises:
            pydantic.ValidationError: 欄位缺漏或型別不符時。
        """
        record = RawEventRecord(**raw_json)
        logger.info("IngestionService 驗證通過：record_id=%s", record.record_id)
        return record

    def ingest_batch(self, raw_json_list: List[dict]) -> List[RawEventRecord]:
        """
        批次匯入多筆事件資料，單筆驗證失敗不影響其他筆。

        Returns:
            List[RawEventRecord]: 驗證通過的記錄列表（失敗的筆數只記錄
            log，不中斷整批處理）。
        """
        records: List[RawEventRecord] = []
        for index, raw_json in enumerate(raw_json_list):
            try:
                records.append(self.ingest_single(raw_json))
            except Exception as exc:  # noqa: BLE001 - 記錄後繼續處理下一筆
                logger.error("第 %d 筆事件資料驗證失敗：%s", index, exc)

        logger.info("IngestionService.ingest_batch() 完成：%d/%d 筆驗證通過", len(records), len(raw_json_list))
        return records
