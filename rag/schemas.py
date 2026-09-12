"""
rag/schemas.py
================
定義 RAG（檢索增強生成）模組的輸出資料結構。

TODO：
- [ ] RetrievedCase 補上該歷史事件當時的後續處理結果（如果有記錄的話），
      這樣才能真正幫助 LLM 判斷「這種情況通常怎麼發展」
- [ ] KnowledgeSnippet 補上文件版本 / 發布單位等 metadata，方便 LLM 引用來源
"""

from typing import List

from pydantic import BaseModel


class RetrievedCase(BaseModel):
    """
    向量相似度檢索出來的「歷史相似事件」。
    對應架構設計中的 [4a] 視覺相似度檢索輸出。
    """

    record_id: str
    similarity_score: float
    caption: str
    predicted_event_type: str


class KnowledgeSnippet(BaseModel):
    """
    從知識庫（SOP、法規、歷史事件報告）中檢索出來的文字片段。
    對應架構設計中的 [4b] 知識庫文字檢索輸出。
    """

    source_document: str
    content: str
    relevance_score: float


class RetrievedContext(BaseModel):
    """
    RAG 模組的完整輸出，同時包含視覺相似案例與知識庫片段，
    會一起送進 Reasoning Module（LLM）作為 grounding 依據，
    避免 LLM 生成原因/建議時憑空捏造。
    """

    similar_cases: List[RetrievedCase] = []
    knowledge_snippets: List[KnowledgeSnippet] = []
    news_precedents: List[KnowledgeSnippet] = []
