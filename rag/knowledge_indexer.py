"""
rag/knowledge_indexer.py
===========================
負責建立與維護「知識庫」的向量索引。

知識庫來源預期包含：
- 人群安全 SOP 文件、相關法規、歷史事件報告（txt / markdown）
- 也可以直接餵影像端輸出的事件 JSON（例如 rag_vlm_mock_dataset.json），
  讓歷史事件的 caption 也能被檢索到

這個模組與事件處理主流程（ingestion -> vlm -> rag -> llm -> alert）
完全解耦，可以獨立開發、獨立測試，也可以最早開始實作。

處理流程：
document_paths -> [Loader]  -> LoadedDocument
                -> [Chunker] -> Chunk
                -> [Embedding] -> 向量
                -> [VectorStore] -> 寫入 ChromaDB
"""

from pathlib import Path
from typing import List, Optional

from embedding.base import BaseEmbeddingProvider
from rag.chunking.base import BaseChunker
from rag.chunking.recursive_chunker import RecursiveChunker
from rag.interfaces import BaseVectorStore
from rag.loaders.factory import get_loader
from utils.logger import get_logger

logger = get_logger(__name__)


class KnowledgeBaseIndexer:
    """
    負責把知識庫原始文件切塊、embedding、寫入向量資料庫。
    """

    def __init__(
        self,
        vector_store: BaseVectorStore,
        embedding: BaseEmbeddingProvider,
        chunker: Optional[BaseChunker] = None,
    ) -> None:
        """
        Args:
            vector_store: 知識庫專用的向量資料庫實作。
            embedding: 把文字轉成向量的 embedding 實作。
            chunker: 文件切塊策略，預設使用 RecursiveChunker。
        """
        self._vector_store = vector_store
        self._embedding = embedding
        self._chunker = chunker or RecursiveChunker()

    def build_index_from_documents(self, document_paths: List[Path]) -> int:
        """
        從指定的文件路徑列表建立（或更新）知識庫索引。

        Returns:
            int: 實際寫入向量資料庫的 chunk 筆數。
        """
        chunks = []
        for path in document_paths:
            loader = get_loader(path)
            documents = loader.load(path)
            for document in documents:
                chunks.extend(self._chunker.split(document))

        if not chunks:
            logger.warning(
                "build_index_from_documents() 沒有產生任何 chunk，document_paths=%s",
                document_paths,
            )
            return 0

        texts = [chunk.content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        ids = [chunk.chunk_id for chunk in chunks]

        embeddings = self._embedding.embed_text(texts)
        self._vector_store.add_texts(texts, embeddings, metadatas, ids=ids)

        logger.info("KnowledgeBaseIndexer 完成索引，共寫入 %d 個 chunk", len(chunks))
        return len(chunks)

    def rebuild_index(self, document_paths: List[Path]) -> int:
        """
        重新建立索引（文件更新時使用）。

        因為 RecursiveChunker 是用「內容雜湊」產生穩定的 chunk_id
        （見 rag/chunking/recursive_chunker.py），且 ChromaVectorStore.add_texts()
        採用 upsert 語意，所以對「內容沒變」的片段重新索引等於原地更新，
        對「內容有改」的片段會產生新的 chunk_id（視為新增）。

        TODO: 目前 BaseVectorStore 介面沒有「清空 / 刪除」的方法，
              所以「文件中被移除的舊內容」不會自動從向量資料庫清掉。
              若需要嚴格的重建語意，之後要在 BaseVectorStore 補上
              clear() 或 delete_by_source() 之類的管理方法。
        """
        return self.build_index_from_documents(document_paths)
