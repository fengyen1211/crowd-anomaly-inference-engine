"""
rag/visual_retriever.py
==========================
負責「視覺相似度檢索」：
輸入某筆事件的 embedding（目前是假向量，未來是真實 CLIP/SigLIP 向量），
去向量資料庫中找出最相似的歷史事件。

對應架構設計中的 [4a] 視覺相似度檢索。

注意：這裡輸入的 query 已經是「算好的向量」，
不像 KnowledgeTextRetriever 需要額外的 embedding 模型把文字轉成向量。
"""

from typing import List

from rag.interfaces import BaseRetriever, BaseVectorStore
from rag.schemas import RetrievedCase
from utils.logger import get_logger

logger = get_logger(__name__)


class VisualSimilarityRetriever(BaseRetriever):
    """
    使用向量相似度，從歷史事件中找出視覺特徵最相近的案例。
    """

    def __init__(self, vector_store: BaseVectorStore) -> None:
        """
        Args:
            vector_store: 已經注入好的向量資料庫實作
                           （目前預設是 ChromaVectorStore，但這裡只依賴抽象介面）。
        """
        self._vector_store = vector_store

    def retrieve(self, query: List[float], top_k: int = 5) -> List[RetrievedCase]:
        """
        Args:
            query: 查詢用的 embedding 向量（512 維，見 utils/schemas.py 的 embedding_dim）。
            top_k: 回傳最相似的筆數。
        """
        raw_results = self._vector_store.similarity_search(query, top_k)
        cases = [self._to_retrieved_case(raw) for raw in raw_results]
        logger.info("VisualSimilarityRetriever 找到 %d 筆相似案例", len(cases))
        return cases

    @staticmethod
    def _to_retrieved_case(raw: dict) -> RetrievedCase:
        """
        把 ChromaVectorStore.similarity_search() 回傳的原始 dict，
        轉換成對外統一的 RetrievedCase。
        """
        metadata = raw.get("metadata") or {}
        return RetrievedCase(
            record_id=str(metadata.get("record_id", raw.get("id", ""))),
            similarity_score=raw["score"],
            caption=str(metadata.get("caption", raw.get("content", ""))),
            predicted_event_type=str(metadata.get("predicted_event_type", "")),
        )
