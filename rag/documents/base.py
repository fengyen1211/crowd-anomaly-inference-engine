"""
rag/documents/base.py
========================
定義 Document Builder 的抽象介面。

Document Builder 負責把「結構化資料」（例如影像端輸出的事件 JSON）
轉換成「適合拿去 embedding 的自然語言文字」，而不是把原始 JSON
（或任意巢狀資料結構）直接字串化後丟去 embedding。

為什麼不能直接把 JSON 存進向量資料庫？
- Embedding 模型（不論是本地的 BGE-M3 還是 OpenAI）都是針對自然語言
  訓練的，JSON 的大括號、引號、巢狀結構對語意相似度來說是雜訊，
  會讓向量沒辦法準確反映「這個事件實際上在描述什麼」。
- 把關鍵欄位（事件類型、群體移動特徵、人數、關鍵字...）組成一段
  人看得懂的描述，才能讓 embedding 抓到真正重要的語意特徵，
  檢索出來的相似案例才會準確。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

from rag.loaders.schemas import LoadedDocument


class BaseDocumentBuilder(ABC):
    """所有 Document Builder 都必須實作 build()。"""

    @abstractmethod
    def build(self, item: Dict[str, Any]) -> LoadedDocument:
        """把單一筆結構化資料組成 LoadedDocument。"""
        raise NotImplementedError
