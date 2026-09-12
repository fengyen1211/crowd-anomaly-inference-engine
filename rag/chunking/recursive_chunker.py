"""
rag/chunking/recursive_chunker.py
====================================
使用 LangChain 的 RecursiveCharacterTextSplitter 實作文件切塊。

這個切塊策略會依序嘗試用「段落 -> 換行 -> 句子 -> 字元」等分隔符切割，
盡量讓每個 chunk 在語意上完整，同時保持在 chunk_size 限制內。
"""

import hashlib
from typing import List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import Settings, get_settings
from rag.chunking.base import BaseChunker
from rag.chunking.schemas import Chunk
from rag.loaders.schemas import LoadedDocument
from utils.logger import get_logger

logger = get_logger(__name__)


class RecursiveChunker(BaseChunker):
    """
    依 chunk_size / chunk_overlap 設定，把文件切成重疊的片段。
    """

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        settings: Settings | None = None,
    ) -> None:
        settings = settings or get_settings()
        self._chunk_size = chunk_size or settings.chunk_size
        self._chunk_overlap = chunk_overlap or settings.chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )

    def split(self, document: LoadedDocument) -> List[Chunk]:
        texts = self._splitter.split_text(document.content)
        source = document.metadata.get("source", "unknown")

        chunks: List[Chunk] = []
        for index, text in enumerate(texts):
            chunk_metadata = {**document.metadata, "chunk_index": index}
            chunk_id = self._make_chunk_id(source, index, text)
            chunks.append(Chunk(chunk_id=chunk_id, content=text, metadata=chunk_metadata))

        logger.info("RecursiveChunker 將文件 %s 切成 %d 個片段", source, len(chunks))
        return chunks

    @staticmethod
    def _make_chunk_id(source: str, index: int, text: str) -> str:
        """
        用「來源路徑 + chunk 序號 + 內容雜湊」組成穩定的 chunk_id，
        確保同一份文件重新索引時，內容沒變的片段會得到相同的 id
        （對應 ChromaVectorStore 的 upsert 語意：同 id 會覆寫而不是重複新增）。
        """
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
        return f"{source}:{index}:{digest}"
