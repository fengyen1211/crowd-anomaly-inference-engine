"""
rag/loaders/schemas.py
=========================
定義文件載入器（Document Loader）的輸出資料結構。
"""

from typing import Any, Dict

from pydantic import BaseModel, Field


class LoadedDocument(BaseModel):
    """
    文件載入器讀取單一文件（或單一 JSON 記錄）後的統一輸出格式。

    不論來源是 txt / markdown / json / pdf，載入後都會轉換成這個格式，
    後續的 Chunker 只需要處理這一種資料結構，不需要知道原始檔案格式。
    """

    content: str = Field(..., description="這份文件的純文字內容")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="來源資訊，例如檔名、格式、原始欄位（record_id、caption...等）",
    )
