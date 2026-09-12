"""
embedding/sentence_transformer.py
====================================
BaseEmbeddingProvider 的 Sentence-Transformers 實作。

★★★ 目前的預設 Embedding Provider ★★★

使用本地端執行的 sentence-transformers 模型，完全不需要呼叫任何
外部 API（不需要網路、不需要 API Key），符合「完全採用 Local AI」的目標。

預設模型：BAAI/bge-m3
- 多語言（含中文）文字 embedding 模型，輸出 1024 維向量
- 支援長文本（最長 8192 tokens），適合本專案的 caption / SOP 文件檢索

注意：這是「純文字」embedding 模型，不支援圖片輸入，
embed_image() 會丟出 NotImplementedError（圖片 embedding 之後由
CLIP / SigLIP Provider 負責，見 clip.py / siglip.py）。
"""

from typing import List, Optional

from embedding.base import BaseEmbeddingProvider
from utils.logger import get_logger

logger = get_logger(__name__)


class SentenceTransformerEmbeddingProvider(BaseEmbeddingProvider):
    """使用本地 sentence-transformers 模型產生文字 embedding。"""

    def __init__(self, model_name: str = "BAAI/bge-m3", device: Optional[str] = None) -> None:
        # 延遲載入（lazy import）：sentence-transformers 底層依賴 torch，
        # 安裝體積不小，只有真的建立這個 Provider 時才 import，
        # 避免其他 Provider（例如 mock）也被迫承擔這個相依性。
        from sentence_transformers import SentenceTransformer

        self._model_name = model_name
        logger.info("正在載入本地 Embedding 模型：%s（第一次執行需要下載，會較慢）", model_name)
        self._model = SentenceTransformer(model_name, device=device)

        # sentence-transformers 新版把 get_sentence_embedding_dimension()
        # 改名為 get_embedding_dimension()，這裡兩個都試一次以相容新舊版本。
        if hasattr(self._model, "get_embedding_dimension"):
            self._dimension = self._model.get_embedding_dimension()
        else:
            self._dimension = self._model.get_sentence_embedding_dimension()
        logger.info("模型載入完成：%s（維度=%d）", model_name, self._dimension)

    def embed_text(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        # normalize_embeddings=True：輸出向量做 L2 正規化，
        # 讓向量的內積等於 cosine 相似度，跟 rag/vector_store.py
        # 裡 ChromaVectorStore 設定的 hnsw:space="cosine" 一致。
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vectors.tolist()

    def embed_image(self, images: List[bytes]) -> List[List[float]]:
        raise NotImplementedError(
            f"{self._model_name} 是純文字 embedding 模型，不支援圖片輸入。"
            "圖片 embedding 請改用 CLIP 或 SigLIP Provider（見 clip.py / siglip.py）。"
        )

    def get_dimension(self) -> int:
        return self._dimension
