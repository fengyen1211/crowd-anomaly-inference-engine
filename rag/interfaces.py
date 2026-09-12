"""
rag/interfaces.py
====================
定義 RAG 模組內部使用的抽象介面。

分成兩層抽象：
1. BaseVectorStore：對向量資料庫的抽象（目前預設用 ChromaDB，
   但介面隔離讓之後可以換成 FAISS、Milvus 等其他向量庫）。
2. BaseRetriever：對「檢索邏輯」的抽象（視覺相似度檢索 / 知識庫文字檢索
   是兩種不同的檢索邏輯，但對外提供一致的 retrieve() 介面）。

"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional


class BaseVectorStore(ABC):
    """
    向量資料庫的抽象介面。
    """

    @abstractmethod
    def add_texts(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[dict],
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        將文字、向量、metadata 寫入向量資料庫。

        採用 upsert 語意：若 ids 中的某個 id 已存在，會覆寫舊資料，
        而不是報錯或產生重複資料。這是刻意設計，搭配
        rag/chunking/recursive_chunker.py 用內容雜湊產生穩定 chunk_id，
        讓「重新對同一份文件建立索引」的行為等同於更新，而不是疊加。

        Args:
            ids: 可選的自訂 id 列表；未提供時由實作自行產生（例如 uuid4）。

        Returns:
            List[str]: 實際寫入時使用的 id 列表。
        """
        raise NotImplementedError

    @abstractmethod
    def similarity_search(self, query_embedding: List[float], top_k: int) -> List[dict]:
        """依向量相似度取回最相近的 top_k 筆資料。"""
        raise NotImplementedError


class BaseRetriever(ABC):
    """
    檢索邏輯的抽象介面。
    VisualSimilarityRetriever 與 KnowledgeTextRetriever 都需要實作這個介面，
    讓 Reasoning Module 可以用一致的方式呼叫兩種不同的檢索器。
    """

    @abstractmethod
    def retrieve(self, query: Any, top_k: int = 5) -> List[Any]:
        """
        執行檢索。

        Args:
            query: 查詢內容，視實作而定可能是 embedding 向量或文字字串。
            top_k: 回傳筆數上限。

        Returns:
            檢索結果列表（RetrievedCase 或 KnowledgeSnippet）。
        """
        raise NotImplementedError
