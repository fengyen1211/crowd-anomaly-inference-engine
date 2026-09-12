"""
routers/feedback.py
======================
人工回饋相關的 API 路由。

- POST /events/{record_id}/feedback   人工標註 VLM/LLM 判斷結果的對錯

這些資料會被 FeedbackService 儲存，作為未來 fine-tune 的資料來源。

TODO：
- [ ] 注入 FeedbackService
- [ ] 定義 request body 的 Pydantic schema（is_correct, corrected_content）
"""

from fastapi import APIRouter

router = APIRouter(prefix="/events", tags=["feedback"])


@router.post("/{record_id}/feedback")
def submit_feedback(record_id: str):
    """
    提交人工回饋。

    TODO:
        1. 解析 request body（是否正確、修正後內容）
        2. 呼叫 FeedbackService.submit_feedback(...)
    """
    raise NotImplementedError("TODO: 尚未實作 /events/{record_id}/feedback")
