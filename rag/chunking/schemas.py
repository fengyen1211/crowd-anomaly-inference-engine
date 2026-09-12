"""
rag/chunking/schemas.py
==========================
定義文件切塊（Chunk）後的資料結構。
"""

from typing import Any, Dict

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """
    一份文件被切塊後的單一片段，是實際會被 embedding 並寫入
    向量資料庫的最小單位。
    """

    chunk_id: str = Field(..., description="這個片段的唯一識別碼，寫入向量資料庫時作為 id")
    content: str = Field(..., description="片段的純文字內容")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="繼承自來源文件的 metadata，並加上 chunk_index",
    )
