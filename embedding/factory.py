"""
embedding/factory.py
=======================
Embedding Provider 的工廠。

依照傳入的 provider_name，動態建立對應的 BaseEmbeddingProvider 實例。

刻意在每個分支裡才 import 對應的 Provider 類別（而不是在檔案頂端一次性
import 全部），是因為每個 Provider 底層的套件都不小
（sentence-transformers 需要 torch，openai 需要 openai SDK，
CLIP/SigLIP 之後需要 transformers），使用者只想用其中一個 Provider 時，
不應該被迫把其他 Provider 的相依套件也裝起來。
"""

from typing import Optional

from embedding.base import BaseEmbeddingProvider

# 支援的 provider 名稱，對應 config.yaml 的 embedding.provider 欄位
SUPPORTED_PROVIDERS = ("mock", "sentence_transformer", "openai", "clip", "siglip")


class EmbeddingFactory:
    """依名稱建立 Embedding Provider 的工廠類別。"""

    @classmethod
    def create(
        cls,
        provider_name: str,
        model: Optional[str] = None,
        **kwargs,
    ) -> BaseEmbeddingProvider:
        """
        Args:
            provider_name: "mock" / "sentence_transformer" / "openai" / "clip" / "siglip"
            model: 模型名稱，不同 provider 有各自的預設值，可不傳。
            **kwargs: 傳給對應 Provider 建構子的其他參數（例如 device、dimension）。

        Raises:
            ValueError: provider_name 不在支援清單內。
        """
        if provider_name == "mock":
            from embedding.mock import MockEmbeddingProvider

            return MockEmbeddingProvider(**kwargs)

        if provider_name == "sentence_transformer":
            from embedding.sentence_transformer import SentenceTransformerEmbeddingProvider

            return SentenceTransformerEmbeddingProvider(model_name=model or "BAAI/bge-m3", **kwargs)

        if provider_name == "openai":
            from embedding.openai import OpenAIEmbeddingProvider

            return OpenAIEmbeddingProvider(model=model, **kwargs)

        if provider_name == "clip":
            from embedding.clip import CLIPEmbeddingProvider

            return CLIPEmbeddingProvider(**({"model_name": model} if model else {}), **kwargs)

        if provider_name == "siglip":
            from embedding.siglip import SigLIPEmbeddingProvider

            return SigLIPEmbeddingProvider(**({"model_name": model} if model else {}), **kwargs)

        raise ValueError(
            f"不支援的 embedding provider：'{provider_name}'，"
            f"支援的選項：{SUPPORTED_PROVIDERS}"
        )
