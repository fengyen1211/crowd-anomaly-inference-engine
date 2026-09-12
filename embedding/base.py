"""
embedding/base.py
====================
定義所有 Embedding Provider 的共同抽象介面。

★★★ 這是「Embedding 模型可自由替換」需求的核心設計 ★★★

所有上層模組（rag/text_retriever.py、rag/knowledge_indexer.py，
以及未來 vlm 模組如果需要視覺 embedding）都只依賴這個抽象介面，
不會直接依賴任何特定的模型或套件
（sentence-transformers、openai、transformers...）。

Provider 分兩大類：
- 純文字模型（例如 BGE-M3、OpenAI text-embedding）：
  embed_image() 應該 raise NotImplementedError。
- 多模態模型（例如 CLIP、SigLIP）：
  embed_text() 和 embed_image() 都要能算，
  且兩者輸出必須落在同一個向量空間，才能做「以文找圖／以圖找文」。
"""

from abc import ABC, abstractmethod
from typing import List


class BaseEmbeddingProvider(ABC):
    """所有 Embedding Provider 都必須繼承這個抽象類別。"""

    @abstractmethod
    def embed_text(self, texts: List[str]) -> List[List[float]]:
        """
        將一批文字轉成向量。

        不論是要 embedding「多筆文件」還是「單一查詢字串」，
        都呼叫這個方法（查詢時傳入長度為 1 的 list，取結果 [0] 即可）——
        刻意不再區分 embed_documents() / embed_query() 兩個方法，
        減少每個 Provider 需要實作的介面數量。
        """
        raise NotImplementedError

    @abstractmethod
    def embed_image(self, images: List[bytes]) -> List[List[float]]:
        """
        將一批圖片（二進位資料）轉成向量。

        純文字模型（例如 BGE-M3）應該 raise NotImplementedError，
        由呼叫端自行決定要不要捕捉這個例外並提示「請換成 CLIP/SigLIP」。
        """
        raise NotImplementedError

    @abstractmethod
    def get_dimension(self) -> int:
        """回傳這個模型輸出的向量維度。"""
        raise NotImplementedError
