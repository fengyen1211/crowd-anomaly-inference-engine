"""
embedding/clip.py
====================
BaseEmbeddingProvider 的 CLIP 實作。

背景（見 rag_vlm_mock_dataset_README.md）：
影像端當初的 embedding 欄位是假的隨機向量，因為機器硬碟空間不足以
下載 CLIP 權重。現在權重可以正常下載，這裡改用 Hugging Face
transformers 內建的 CLIPModel/CLIPProcessor 在本地端執行——
不需要額外安裝 open_clip_torch，本專案已經因為 sentence-transformers
而依賴 torch + transformers，這裡只是直接用 transformers 提供的
CLIP 權重載入介面。

CLIP 是多模態模型：文字與圖片分別經過各自的 encoder，
但輸出落在同一個向量空間，因此 embed_text() 與 embed_image()
的結果可以直接互相比較（以文找圖／以圖找文／以圖找圖）。
"""

import io
from typing import List, Optional

from embedding.base import BaseEmbeddingProvider
from utils.logger import get_logger

logger = get_logger(__name__)


class CLIPEmbeddingProvider(BaseEmbeddingProvider):
    """使用本地 CLIP 模型產生視覺／文字 embedding（共用同一個向量空間）。"""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: Optional[str] = None) -> None:
        # 延遲載入（lazy import）：CLIP 底層依賴 torch + transformers，
        # 安裝體積不小，只有真的建立這個 Provider 時才 import，
        # 避免其他 Provider（例如 mock）也被迫承擔這個相依性
        # （跟 embedding/sentence_transformer.py 的做法一致）。
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self._model_name = model_name
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._torch = torch

        logger.info("正在載入本地 CLIP 模型：%s（第一次執行需要下載權重，會較慢）", model_name)
        self._model = CLIPModel.from_pretrained(model_name).to(self._device).eval()
        self._processor = CLIPProcessor.from_pretrained(model_name)

        # 直接讀模型設定裡的 projection_dim，不寫死 512——
        # 換成其他 CLIP 變體（例如 ViT-L/14）時維度會不同，這裡自動對齊。
        self._dimension = self._model.config.projection_dim
        logger.info("CLIP 模型載入完成：%s（device=%s，維度=%d）", model_name, self._device, self._dimension)

    def embed_text(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        inputs = self._processor(text=texts, padding=True, truncation=True, return_tensors="pt")
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        with self._torch.no_grad():
            features = self._model.get_text_features(**inputs)
        return self._normalize(features).tolist()

    def embed_image(self, images: List[bytes]) -> List[List[float]]:
        if not images:
            return []

        from PIL import Image

        pil_images = [Image.open(io.BytesIO(data)).convert("RGB") for data in images]
        inputs = self._processor(images=pil_images, return_tensors="pt")
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        with self._torch.no_grad():
            features = self._model.get_image_features(**inputs)
        return self._normalize(features).tolist()

    def _normalize(self, features):
        # L2 正規化：讓向量的內積等於 cosine 相似度，跟 rag/vector_store.py
        # 裡 ChromaVectorStore 設定的 hnsw:space="cosine" 一致
        # （跟 sentence_transformer.py 的做法相同）。
        return features / features.norm(p=2, dim=-1, keepdim=True)

    def get_dimension(self) -> int:
        return self._dimension
