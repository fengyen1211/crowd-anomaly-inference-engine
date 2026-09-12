"""
embedding/siglip.py
======================
BaseEmbeddingProvider 的 SigLIP 實作（骨架，之後才實作）。

SigLIP 是 CLIP 的替代方案之一（Google 提出），
之後若評估 SigLIP 的視覺特徵品質優於 CLIP，可以切換到這個實作，
上層程式碼（KnowledgeBaseIndexer / Retriever）不需要修改
（透過 BaseEmbeddingProvider 介面隔離）。

TODO：
- [ ] 加入 SigLIP 相依套件（例如 transformers 的 SiglipModel）並更新 requirements.txt
- [ ] 實作 embed_text() / embed_image()
- [ ] 確認輸出維度與影像端 JSON 的 embedding_dim 一致
      （SigLIP 不同版本維度不同，需明確指定使用哪個版本）
"""

from typing import List

from embedding.base import BaseEmbeddingProvider
from utils.logger import get_logger

logger = get_logger(__name__)


class SigLIPEmbeddingProvider(BaseEmbeddingProvider):
    """使用 SigLIP 模型產生視覺／文字 embedding（骨架，尚未實作）。"""

    def __init__(self, model_name: str = "google/siglip-base-patch16-224", dimension: int = 768) -> None:
        self._model_name = model_name
        self._dimension = dimension

    def embed_text(self, texts: List[str]) -> List[List[float]]:
        logger.info("SigLIPEmbeddingProvider.embed_text() 尚未實作")
        raise NotImplementedError("TODO: SigLIP embedding 尚未實作")

    def embed_image(self, images: List[bytes]) -> List[List[float]]:
        logger.info("SigLIPEmbeddingProvider.embed_image() 尚未實作")
        raise NotImplementedError("TODO: SigLIP embedding 尚未實作")

    def get_dimension(self) -> int:
        return self._dimension
