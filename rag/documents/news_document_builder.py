"""
rag/documents/news_document_builder.py
==========================================
把新聞爬蟲/摘要腳本（scripts/fetch_and_index_news.py）產生的「單筆新聞事件記錄」
組成適合 embedding 的自然語言文件。

組合的欄位（對應 scripts/fetch_and_index_news.py 寫出的 JSON 結構）：
- incident_type：事件分類（stampede / crowd_crush / panic / riot / other）
- headline：新聞標題
- summary：本地 LLM 產生的中文摘要
- event_location：事件發生地點
- publish_date：新聞發布日期
- source_url / source_domain：新聞來源

內容開頭明確標註「【真實新聞案例】」，讓 chunk 內容本身就能讓 LLM 分辨
這是真實案例（precedent），而不是法規條文——跟 EventDocumentBuilder
組「規則系統偵測到的事件」描述文字是不同性質的資料，用不同標記區分，
方便之後在 prompts/rag_prompt.py 組 prompt 時也能沿用同樣的判斷方式。
"""

from typing import Any, Dict

from rag.documents.base import BaseDocumentBuilder
from rag.loaders.schemas import LoadedDocument

# 事件分類 -> 中文標籤，跟 event_document_builder.py 的 _EVENT_TYPE_INFO 同樣性質，
# 差別是這裡的分類是新聞摘要 LLM 自己判斷輸出的，不是規則系統的固定列舉，
# 所以用 .get(..., incident_type) 容錯，不強制列舉完整。
_INCIDENT_TYPE_LABELS: Dict[str, str] = {
    "stampede": "踩踏事故",
    "crowd_crush": "群眾推擠",
    "panic": "恐慌逃散",
    "riot": "騷亂",
    "other": "其他群眾異常事件",
}


class NewsDocumentBuilder(BaseDocumentBuilder):
    """把單筆新聞事件記錄組成自然語言描述的 LoadedDocument。"""

    def build(self, item: Dict[str, Any]) -> LoadedDocument:
        incident_type = item.get("incident_type", "other")
        label = _INCIDENT_TYPE_LABELS.get(incident_type, incident_type)

        lines = [f"【真實新聞案例】事件類型：{label}（{incident_type}）"]

        event_location = item.get("event_location")
        if event_location:
            lines.append(f"發生地點：{event_location}")

        publish_date = item.get("publish_date")
        if publish_date:
            lines.append(f"發生時間：{publish_date}")

        headline = item.get("headline")
        if headline:
            lines.append(f"新聞標題：{headline}")

        summary = item.get("summary")
        if summary:
            lines.append(f"事件摘要：{summary}")

        source_domain = item.get("source_domain")
        source_url = item.get("source_url")
        if source_url:
            lines.append(f"資料來源：{source_domain or '未知'}（{source_url}）")

        content = "\n".join(lines)

        metadata: Dict[str, Any] = {
            "file_type": "news_article",
            "incident_type": incident_type,
            "headline": headline,
            "event_location": event_location,
            "publish_date": publish_date,
            "source_url": source_url,
            "source_domain": source_domain,
        }

        return LoadedDocument(content=content, metadata=metadata)
