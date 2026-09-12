"""
services/feedback_service.py
===============================
對應架構設計中的 [10] Feedback Module（非必要，但建議實作，
可作為未來 VLM/LLM fine-tune 的資料來源）。

負責：
- 接收人工對 VLM / LLM 判斷結果的標註（對 / 錯 / 修正後的內容）
- 存入資料庫，累積成未來 fine-tune 用的標籤資料集

TODO：
- [ ] 驗證 record_id 是否存在
- [ ] 寫入 database.models.FeedbackRecordORM
- [ ] 決定 feedback 是針對整個 AnalysisResult，還是可以細到單一欄位
      （例如只針對 possible_causes 提出修正）
"""

from utils.logger import get_logger

logger = get_logger(__name__)


class FeedbackService:
    """
    負責記錄人工標註的回饋資料。
    """

    def submit_feedback(
        self,
        record_id: str,
        is_correct: bool,
        corrected_content: dict | None = None,
    ) -> None:
        """
        TODO: 見檔案頂端的 TODO 說明。
        """
        logger.info(
            "FeedbackService.submit_feedback() 尚未實作，record_id=%s",
            record_id,
        )
        raise NotImplementedError
