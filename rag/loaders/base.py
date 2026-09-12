"""
rag/loaders/base.py
======================
定義文件載入器的抽象介面。

設計理念與 vlm/base.py、llm/base.py 一致：
上層（KnowledgeBaseIndexer）只依賴 BaseDocumentLoader 介面，
新增支援的檔案格式時，只需要新增一個子類別並註冊到 factory.py，
不需要修改呼叫端邏輯。
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from rag.loaders.schemas import LoadedDocument


class BaseDocumentLoader(ABC):
    """所有文件載入器都必須實作 load()。"""

    @abstractmethod
    def load(self, file_path: Path) -> List[LoadedDocument]:
        """
        讀取指定檔案，回傳一個或多個 LoadedDocument。

        回傳 List 而非單一物件，是因為像 JSON 這種格式，
        一個檔案可能包含多筆獨立的記錄（例如 records 陣列），
        每一筆都應該被視為獨立的文件分別處理，
        檢索時才能定位到「單一事件」而不是整個檔案。
        """
        raise NotImplementedError
