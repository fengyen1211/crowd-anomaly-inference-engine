"""
rag/loaders/json_loader.py
=============================
JSON 檔案載入器。

知識庫的來源之一是「影像端輸出的事件 JSON」
（例如 rag_vlm_mock_dataset.json），這個載入器設計成可以把 JSON 裡的
「每一筆記錄」拆成獨立的 LoadedDocument，而不是把整個檔案當成一份文件
（那樣檢索出來的顆粒度太粗，沒辦法定位到單一事件）。

支援三種 JSON 結構：
1. {"records": [ {...}, {...} ]}   -> 每個 record 一份文件（優先比對）
2. [ {...}, {...} ]                -> 每個 list item 一份文件
3. { ... 其他任意結構 ... }         -> 整份 JSON 當成一份文件

★ 重構重點 ★
不再把「事件記錄」整筆用 json.dumps 直接當作 embedding 內容
（JSON 語法對 embedding 模型來說是雜訊，語意品質差）。
偵測到一個 dict 含有 predicted_event_type 欄位時，改交給
rag.documents.event_document_builder.EventDocumentBuilder 組成
自然語言描述；其餘一般 JSON（例如非事件類的知識庫結構化資料）
維持原本的 json.dumps 行為。

TODO：
- [ ] 讓 record_list_key（目前寫死找 "records"）可透過建構子參數自訂
"""

import json
from pathlib import Path
from typing import Any, List, Optional

from rag.documents.base import BaseDocumentBuilder
from rag.documents.event_document_builder import EventDocumentBuilder
from rag.documents.news_document_builder import NewsDocumentBuilder
from rag.loaders.base import BaseDocumentLoader
from rag.loaders.schemas import LoadedDocument
from utils.logger import get_logger

logger = get_logger(__name__)


class JSONLoader(BaseDocumentLoader):
    """
    讀取 .json 檔案，依結構拆分成一或多份 LoadedDocument。
    """

    # 若 JSON 是 dict 且含有這個 key 對應到一個 list，優先把該 list 展開
    _RECORD_LIST_KEY = "records"

    # 一般（非事件記錄）JSON 額外攤平進 metadata 的欄位
    _METADATA_FIELDS = ("record_id", "group_id", "predicted_event_type", "caption")

    # 判斷一個 dict 是不是「事件記錄」的依據欄位
    _EVENT_MARKER_FIELD = "predicted_event_type"

    # 判斷一個 dict 是不是「新聞事件記錄」的依據欄位
    # （見 scripts/fetch_and_index_news.py 寫出的 JSON 結構）
    _NEWS_MARKER_FIELD = "incident_type"

    def __init__(
        self,
        event_document_builder: Optional[BaseDocumentBuilder] = None,
        news_document_builder: Optional[BaseDocumentBuilder] = None,
    ) -> None:
        """
        Args:
            event_document_builder: 用來把事件記錄組成描述文字的 Document Builder，
                                     預設使用 EventDocumentBuilder。
            news_document_builder: 用來把新聞事件記錄組成描述文字的 Document Builder，
                                    預設使用 NewsDocumentBuilder。
        """
        self._event_document_builder = event_document_builder or EventDocumentBuilder()
        self._news_document_builder = news_document_builder or NewsDocumentBuilder()

    def load(self, file_path: Path) -> List[LoadedDocument]:
        raw = json.loads(file_path.read_text(encoding="utf-8"))

        if isinstance(raw, dict) and isinstance(raw.get(self._RECORD_LIST_KEY), list):
            items = raw[self._RECORD_LIST_KEY]
            logger.info(
                "JSONLoader 偵測到 '%s' 陣列，%s 展開成 %d 份文件",
                self._RECORD_LIST_KEY, file_path, len(items),
            )
            return [self._item_to_document(item, file_path, index) for index, item in enumerate(items)]

        if isinstance(raw, list):
            logger.info("JSONLoader 偵測到頂層陣列，%s 展開成 %d 份文件", file_path, len(raw))
            return [self._item_to_document(item, file_path, index) for index, item in enumerate(raw)]

        logger.info("JSONLoader 將 %s 視為單一文件", file_path)
        return [self._item_to_document(raw, file_path, 0)]

    def _item_to_document(self, item: Any, file_path: Path, index: int) -> LoadedDocument:
        """
        把單一 JSON 項目轉成 LoadedDocument。

        - 若是事件記錄（含 predicted_event_type 欄位）：交給 Document Builder
          組成自然語言描述，語意品質較好，適合拿去 embedding。
        - 若是新聞事件記錄（含 incident_type 欄位）：交給 NewsDocumentBuilder
          組成自然語言描述，原理跟事件記錄相同。
        - 其餘 JSON：維持原本用 json.dumps 保留完整資訊的行為。
        """
        if isinstance(item, dict) and self._EVENT_MARKER_FIELD in item:
            document = self._event_document_builder.build(item)
            document.metadata["source"] = str(file_path)
            document.metadata["item_index"] = index
            return document

        if isinstance(item, dict) and self._NEWS_MARKER_FIELD in item:
            document = self._news_document_builder.build(item)
            document.metadata["source"] = str(file_path)
            document.metadata["item_index"] = index
            return document

        content = json.dumps(item, ensure_ascii=False, indent=2) if isinstance(item, (dict, list)) else str(item)

        metadata: dict = {
            "source": str(file_path),
            "file_type": "json",
            "item_index": index,
        }
        if isinstance(item, dict):
            for key in self._METADATA_FIELDS:
                if key in item:
                    metadata[key] = item[key]

        return LoadedDocument(content=content, metadata=metadata)
