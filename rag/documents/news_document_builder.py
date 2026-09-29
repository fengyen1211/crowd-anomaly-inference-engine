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

# 影像端行為類型 -> 中文標籤。
#
# 跟 incident_type 是兩個不同層次的分類：incident_type 講的是「這是哪一類
# 新聞事件」（踩踏／騷亂／火災恐慌），由爬蟲階段的 LLM 自行判斷；
# behavior_type 講的是「這起事故的人群動力學機制，對應影像端偵測的哪一種
# 行為」，目前只有 data/news_curated_cases.json 裡人工策展的案例才有這個
# 欄位——爬蟲摘要多半只寫死傷數字，沒寫機制，硬要 LLM 事後回推會編造。
#
# 標上這個欄位的目的是讓檢索能對上事件類型：影像端判定 counter_flow 時，
# 查詢文字裡就會帶到「逆向流動」，而不是所有案例都只匹配到「踩踏」。
_BEHAVIOR_TYPE_LABELS: Dict[str, str] = {
    "sprint_spike": "突然加速（個體或群體移動速度異常尖峰）",
    "crowd_locked": "人群卡死（局部密度過高，無法自主移動）",
    "panic_scatter": "恐慌四散（受外部刺激後朝多方向逃散）",
    "counter_flow": "逆向流動（與主流方向相反，或兩股人流對撞）",
}


class NewsDocumentBuilder(BaseDocumentBuilder):
    """把單筆新聞事件記錄組成自然語言描述的 LoadedDocument。"""

    def build(self, item: Dict[str, Any]) -> LoadedDocument:
        incident_type = item.get("incident_type", "other")
        label = _INCIDENT_TYPE_LABELS.get(incident_type, incident_type)

        lines = [f"【真實新聞案例】事件類型：{label}（{incident_type}）"]

        # 人工策展的案例才有 behavior_type（見 _BEHAVIOR_TYPE_LABELS 的說明）。
        # 容許字串或字串陣列兩種寫法，一起事故常常同時展現多種機制
        # （例如貝斯納爾格林是 sprint_spike 觸發、crowd_locked 致死）。
        behavior_types = item.get("behavior_type") or []
        if isinstance(behavior_types, str):
            behavior_types = [behavior_types]
        if behavior_types:
            described = "；".join(
                f"{_BEHAVIOR_TYPE_LABELS.get(bt, bt)}（{bt}）" for bt in behavior_types
            )
            lines.append(f"對應行為類型：{described}")

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
            # Chroma 的 metadata 只吃純量，list 會被 ChromaVectorStore._clean_metadata()
            # 轉成字串；這裡先自己 join 成可讀字串，避免存進去變成 "['a', 'b']"
            "behavior_type": "、".join(behavior_types) if behavior_types else None,
            "headline": headline,
            "event_location": event_location,
            "publish_date": publish_date,
            "source_url": source_url,
            "source_domain": source_domain,
        }

        return LoadedDocument(content=content, metadata=metadata)
