"""
rag/visual_indexer.py
========================
負責建立與維護「視覺相似度檢索」的向量索引。

跟 rag/knowledge_indexer.py 是同一種角色（把原始素材轉成向量、寫入
ChromaDB），差別在來源素材與 embedding 模型不同：

    knowledge_indexer.py：SOP 文件（文字）    -> 文字 embedding (bge-m3)
    visual_indexer.py   ：事件畫面／裁切區域（圖片）-> 視覺 embedding (CLIP/SigLIP)

在這之前，chroma_collection_visual 這個 collection 完全沒有任何模組
會寫入資料——VisualSimilarityRetriever.retrieve() 實際上永遠查詢一個
空的 collection，只是因為 ChromaVectorStore.similarity_search() 對空
collection 回傳空 list、不會噴錯，才沒有被發現這個缺口。這個模組把
「歷史事件（或任何有畫面的素材）該怎麼被索引進視覺向量庫」補上。

處理流程：
image_paths -> [embedding.embed_image()] -> 向量 -> [VectorStore] -> 寫入 ChromaDB
"""

from pathlib import Path
from typing import List, Optional

from embedding.base import BaseEmbeddingProvider
from rag.interfaces import BaseVectorStore
from utils.logger import get_logger

logger = get_logger(__name__)


class VisualEventIndexer:
    """
    負責把畫面（歷史事件的完整幀或裁切區域）轉成視覺向量，寫入向量資料庫。
    """

    def __init__(self, vector_store: BaseVectorStore, embedding: BaseEmbeddingProvider) -> None:
        """
        Args:
            vector_store: 視覺事件專用的向量資料庫實作
                          （預期是 chroma_collection_visual）。
            embedding: 支援 embed_image() 的 embedding 實作（CLIP / SigLIP）。
                       傳入純文字 provider（例如 bge-m3）會在呼叫
                       embed_image() 時得到 NotImplementedError。
        """
        self._vector_store = vector_store
        self._embedding = embedding

    def index_images(
        self,
        image_paths: List[Path],
        captions: List[str],
        metadatas: List[dict],
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        讀取一批圖片檔案，算出視覺向量，寫入向量資料庫。

        Args:
            image_paths: 圖片檔案路徑列表。
            captions: 對應每張圖片的文字說明，存成 Chroma 的 document
                      內容，方便之後直接讀出人類可讀的描述
                      （不影響向量相似度比對，比對只看 embedding）。
            metadatas: 對應每張圖片的 metadata（例如 record_id、
                       predicted_event_type、frame_idx）。
            ids: 可選的自訂 id，未提供時由 vector_store 自行產生。

        Returns:
            List[str]: 實際寫入時使用的 id 列表。
        """
        if not image_paths:
            logger.warning("index_images() 收到空的 image_paths，沒有東西可以索引")
            return []

        if not (len(image_paths) == len(captions) == len(metadatas)):
            raise ValueError(
                "image_paths / captions / metadatas 長度必須一致："
                f"{len(image_paths)} / {len(captions)} / {len(metadatas)}"
            )

        images_bytes = [path.read_bytes() for path in image_paths]
        embeddings = self._embedding.embed_image(images_bytes)

        written_ids = self._vector_store.add_texts(captions, embeddings, metadatas, ids=ids)
        logger.info("VisualEventIndexer 完成索引，共寫入 %d 張畫面", len(written_ids))
        return written_ids
