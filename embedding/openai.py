"""
embedding/openai.py
======================
BaseEmbeddingProvider 的 OpenAI 實作（保留，可選，非預設）。

★ 這個 Provider 不是預設選項 ★
本專案的設計目標是完全採用 Local AI，不依賴外部 API
（見 embedding/sentence_transformer.py 才是預設 Provider）。
保留這個實作是為了：
- 需要快速比較「本地模型 vs OpenAI」的檢索品質時可以切換
- 沒有 GPU / 本地運算資源時的備援方案

要啟用時，把 config.yaml 的 embedding.provider 改成 "openai"，
並在 .env 設定 OPENAI_API_KEY。
"""

from typing import Dict, List, Optional

from config.settings import Settings, get_settings
from embedding.base import BaseEmbeddingProvider
from utils.logger import get_logger

logger = get_logger(__name__)

# 常見 OpenAI embedding 模型的輸出維度（用於在第一次呼叫 API 前就能得知 dimension）。
_MODEL_DIMENSIONS: Dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """使用 OpenAI embeddings API 產生文字向量。"""

    def __init__(self, settings: Optional[Settings] = None, model: Optional[str] = None) -> None:
        # 延遲載入：不安裝 openai 套件，也不影響其他（本地）Provider 的使用。
        from openai import OpenAI

        self._settings = settings or get_settings()
        self._model = model or self._settings.openai_embedding_model
        self._client = OpenAI(api_key=self._settings.openai_api_key)
        self._dimension = _MODEL_DIMENSIONS.get(self._model)

        if self._dimension is None:
            logger.warning(
                "未知的 embedding 模型 '%s'，無法預先得知向量維度，"
                "將於第一次呼叫 API 後動態判斷。",
                self._model,
            )

    def embed_text(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        response = self._client.embeddings.create(model=self._model, input=texts)
        embeddings = [item.embedding for item in response.data]

        if self._dimension is None and embeddings:
            self._dimension = len(embeddings[0])

        logger.info("OpenAIEmbeddingProvider 產生 %d 筆向量（model=%s）", len(embeddings), self._model)
        return embeddings

    def embed_image(self, images: List[bytes]) -> List[List[float]]:
        raise NotImplementedError("OpenAI text-embedding 模型不支援圖片輸入")

    def get_dimension(self) -> int:
        if self._dimension is None:
            raise RuntimeError(
                "尚未呼叫過 embed_text()，無法得知向量維度。"
                "請先呼叫一次，或在建構子明確指定已知的模型名稱。"
            )
        return self._dimension
