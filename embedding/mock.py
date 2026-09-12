"""
embedding/mock.py
====================
BaseEmbeddingProvider 的假資料實作（Mock）。

用途：
- 單元測試 / pipeline 串接測試，不需要載入任何真實模型
  （不需要下載模型權重、不需要 GPU/CPU 運算時間、不需要網路）。
- 呼應目前影像端 JSON 裡「假 embedding」的角色——
  格式、維度都對，但內容是隨機數字，純粹讓上下游可以先接起來測試。

注意：這個 Provider 產生的向量之間沒有真實的語意相似度關係，
不能拿來評估檢索品質，只能用來測試「流程有沒有跑通」。
"""

import hashlib
import random
from typing import List, Union

from embedding.base import BaseEmbeddingProvider


class MockEmbeddingProvider(BaseEmbeddingProvider):
    """回傳固定維度的假向量，不做任何真實運算。"""

    def __init__(self, dimension: int = 1024) -> None:
        self._dimension = dimension

    def embed_text(self, texts: List[str]) -> List[List[float]]:
        return [self._fake_vector(text) for text in texts]

    def embed_image(self, images: List[bytes]) -> List[List[float]]:
        return [self._fake_vector(image) for image in images]

    def get_dimension(self) -> int:
        return self._dimension

    def _fake_vector(self, seed_data: Union[str, bytes]) -> List[float]:
        """
        用輸入內容的雜湊值當作亂數種子，讓「相同輸入」每次都得到相同的假向量
        （方便測試斷言／除錯），而不是每次呼叫都完全隨機。
        """
        if isinstance(seed_data, bytes):
            digest = hashlib.sha256(seed_data).hexdigest()
        else:
            digest = hashlib.sha256(str(seed_data).encode("utf-8")).hexdigest()

        rng = random.Random(digest)
        return [rng.uniform(-1.0, 1.0) for _ in range(self._dimension)]
