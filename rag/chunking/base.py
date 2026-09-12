"""
rag/chunking/base.py
=======================
定義文件切塊器（Chunker）的抽象介面。

設計理念與其他 base.py 一致：上層（KnowledgeBaseIndexer）只依賴
BaseChunker 介面，之後若要換切塊策略（例如依語意切、依標題切），
只需要新增子類別，不需要修改呼叫端邏輯。
"""

from abc import ABC, abstractmethod
from typing import List

from rag.chunking.schemas import Chunk
from rag.loaders.schemas import LoadedDocument


class BaseChunker(ABC):
    """所有切塊器都必須實作 split()。"""

    @abstractmethod
    def split(self, document: LoadedDocument) -> List[Chunk]:
        """把一份 LoadedDocument 切成多個 Chunk。"""
        raise NotImplementedError
