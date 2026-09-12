"""
rag/vector_store.py
======================
BaseVectorStore 的 ChromaDB 實作。

使用 chromadb 的 PersistentClient，將向量資料持久化到本機磁碟
（路徑由 config.settings.chroma_persist_dir 決定），重啟服務後索引不會遺失。
"""

import uuid
from typing import List, Optional

import chromadb

from config.settings import Settings, get_settings
from rag.interfaces import BaseVectorStore
from utils.logger import get_logger

logger = get_logger(__name__)


class ChromaVectorStore(BaseVectorStore):
    """
    使用 ChromaDB 作為向量資料庫的實作。

    這是預設的向量資料庫實作，透過 BaseVectorStore 介面隔離，
    未來若要換成 FAISS / Milvus / Pinecone，只需新增另一個類別。
    """

    def __init__(self, collection_name: str, settings: Optional[Settings] = None) -> None:
        """
        Args:
            collection_name: 要操作的 Chroma collection 名稱
                              （例如視覺向量用 chroma_collection_visual，
                              知識庫文字用 chroma_collection_knowledge）。
            settings: 系統設定物件。
        """
        self._settings = settings or get_settings()
        self._client = chromadb.PersistentClient(path=self._settings.chroma_persist_dir)

        # 明確指定用 cosine 距離：Chroma 定義 cosine 距離 = 1 - cosine 相似度，
        # 這樣 similarity_search() 裡「1 - distance」轉換出來的分數，
        # 才會是數學上正確的 cosine 相似度（範圍 -1 ~ 1），
        # 而不是預設 l2（歐氏距離平方）算出來沒有直覺意義的數字。
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaVectorStore 連線至 collection='%s'（path=%s）",
            collection_name,
            self._settings.chroma_persist_dir,
        )

    def add_texts(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[dict],
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        if not texts:
            return []

        ids = ids or [str(uuid.uuid4()) for _ in texts]
        cleaned_metadatas = [self._clean_metadata(m) for m in metadatas]

        # 用 upsert 而不是 add：add 遇到已存在的 id 會直接報錯，
        # upsert 則是「存在就覆寫、不存在就新增」，符合 interfaces.py
        # 文件所描述的語意（搭配穩定 chunk_id 設計，重新索引=更新）。
        self._collection.upsert(
            documents=texts,
            embeddings=embeddings,
            metadatas=cleaned_metadatas,
            ids=ids,
        )
        logger.info("ChromaVectorStore 寫入（upsert）%d 筆資料到 collection", len(texts))
        return ids

    def similarity_search(self, query_embedding: List[float], top_k: int) -> List[dict]:
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        # Chroma 的回傳格式是「每個欄位一個 list of list」（因為支援一次查多筆），
        # 這裡只查詢一筆，所以統一取索引 [0]。
        documents = results.get("documents") or [[]]
        metadatas = results.get("metadatas") or [[]]
        distances = results.get("distances") or [[]]
        ids = results.get("ids") or [[]]

        documents = documents[0] if documents else []
        metadatas = metadatas[0] if metadatas else []
        distances = distances[0] if distances else []
        ids = ids[0] if ids else []

        return [
            {
                "id": ids[i],
                "content": documents[i],
                "metadata": metadatas[i] or {},
                # cosine 距離轉成「分數越高越相關」，見建構子註解
                "score": 1.0 - distances[i],
            }
            for i in range(len(ids))
        ]

    @staticmethod
    def _clean_metadata(metadata: dict) -> dict:
        """
        Chroma 的 metadata 值只接受 str / int / float / bool，
        這裡把其餘型別（例如 list、dict、None）轉成字串或直接濾掉，
        避免寫入時報錯。

        TODO: 若之後發現有欄位需要保留原始型別做過濾條件（例如
              數字範圍查詢），需要更精細的清洗策略，而不是一律轉字串。
        """
        cleaned = {}
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                cleaned[key] = value
            else:
                cleaned[key] = str(value)
        return cleaned
