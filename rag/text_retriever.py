"""
rag/text_retriever.py
========================
負責「知識庫文字檢索」：
輸入事件的文字描述（caption / predicted_event_type / 使用者問題），
去知識庫（SOP、法規、歷史事件報告）中找出相關的文字片段，
提供給 LLM 做 grounding，避免生成建議時憑空捏造。

對應架構設計中的 [4b] 知識庫文字檢索。

注意：跟 VisualSimilarityRetriever 不同，這裡輸入的是「文字」而不是
已經算好的向量，所以需要額外注入一個 BaseEmbeddingProvider，
在檢索前先把查詢文字轉成向量。
"""

from typing import List

from embedding.base import BaseEmbeddingProvider
from rag.interfaces import BaseRetriever, BaseVectorStore
from rag.schemas import KnowledgeSnippet
from utils.logger import get_logger

logger = get_logger(__name__)


class KnowledgeTextRetriever(BaseRetriever):
    """
    使用文字 embedding，從知識庫文件中找出最相關的段落。
    """

    def __init__(self, vector_store: BaseVectorStore, embedding: BaseEmbeddingProvider) -> None:
        """
        Args:
            vector_store: 知識庫專用的向量資料庫實作
                           （對應 config.chroma_collection_knowledge）。
            embedding: 用來把查詢文字轉成向量的 Embedding Provider
                       （目前預設 SentenceTransformerEmbeddingProvider / BAAI/bge-m3）。
        """
        self._vector_store = vector_store
        self._embedding = embedding

    def retrieve(self, query: str, top_k: int = 5) -> List[KnowledgeSnippet]:
        """
        Args:
            query: 查詢用的文字（例如 caption 或使用者問題）。
            top_k: 回傳最相關的筆數。
        """
        # embed_text() 是批次介面，查詢單一字串時傳長度為 1 的 list，取 [0]
        query_embedding = self._embedding.embed_text([query])[0]
        raw_results = self._vector_store.similarity_search(query_embedding, top_k)
        snippets = [self._to_snippet(raw) for raw in raw_results]
        logger.info("KnowledgeTextRetriever 找到 %d 筆相關知識片段", len(snippets))
        return snippets

    @staticmethod
    def _to_snippet(raw: dict) -> KnowledgeSnippet:
        """
        把 ChromaVectorStore.similarity_search() 回傳的原始 dict，
        轉換成對外統一的 KnowledgeSnippet。
        """
        metadata = raw.get("metadata") or {}
        return KnowledgeSnippet(
            source_document=str(metadata.get("source", "unknown")),
            content=raw.get("content", ""),
            relevance_score=raw["score"],
        )
