"""
rag/documents/event_document_builder.py
==========================================
把影像端輸出的「單筆事件記錄」組成適合 embedding 的自然語言文件。

組合的欄位（對應 rag_vlm_mock_dataset_README.md 描述的資料結構）：
- predicted_event_type：事件類型（轉成中文描述 + 關鍵字）
- caption：畫面描述（模板組字，但內容依據真實數字）
- collective_motion：群體移動特徵
- member_count：群體人數
- frame_file / frame_idx：畫面來源

注意：predicted_event_type 目前是規則系統的猜測，尚未經過驗證
（label_source = "rule_based_unvalidated"），這裡只是忠實地把它組進
描述文字方便檢索，不代表推論端已經認定這個標籤是對的——
「驗證」是 vlm/llm 模組的責任，不是 Document Builder 的責任。
"""

from typing import Any, Dict, List

from rag.documents.base import BaseDocumentBuilder
from rag.loaders.schemas import LoadedDocument

# 事件類型 -> (中文描述, 關鍵字列表)
# TODO: 之後若規則系統新增事件類型，記得同步補上對應項目
_EVENT_TYPE_INFO: Dict[str, Dict[str, Any]] = {
    "panic_dispersal": {
        "label": "恐慌性移動",
        "keywords": ["恐慌", "逃散", "加速", "四散", "緊急疏散"],
    },
    "reverse_flow": {
        "label": "逆向流動",
        "keywords": ["逆向", "逆流", "反方向", "對撞風險"],
    },
    "abnormal_gathering": {
        "label": "異常聚集",
        "keywords": ["聚集", "群聚", "停滯", "圍觀", "密度上升"],
    },
    "sprint_spike": {
        "label": "個體速度異常尖峰",
        "keywords": ["奔跑", "衝刺", "加速", "逃離", "速度異常"],
    },
    "crowd_locked": {
        "label": "群體鎖定聚集",
        "keywords": ["群聚", "鎖定", "密度上升", "停滯", "擁擠"],
    },
}


class EventDocumentBuilder(BaseDocumentBuilder):
    """把單筆事件記錄組成自然語言描述的 LoadedDocument。"""

    def build(self, item: Dict[str, Any]) -> LoadedDocument:
        event_type = item.get("predicted_event_type", "unknown")
        info = _EVENT_TYPE_INFO.get(event_type, {"label": event_type, "keywords": []})

        lines: List[str] = [
            f"事件類型：{info['label']}（{event_type}）",
            f"群體人數：{item.get('member_count', '未知')} 人",
        ]

        collective_motion = item.get("collective_motion") or {}
        if collective_motion:
            # 欄位名稱已對照 rag_vlm_mock_dataset.json 的實際輸出核對過
            # （跟 utils/schemas.py 的 CollectiveMotion 保持一致）。
            lines.append(
                "群體移動特徵：平均方向 {avg_direction_deg} 度，"
                "場景主流方向 {majority_flow_direction_deg} 度，"
                "偏差 {direction_deviation_deg} 度，"
                "密度變化趨勢：{density_trend}".format(
                    avg_direction_deg=self._round(collective_motion.get("avg_direction_deg")),
                    majority_flow_direction_deg=self._round(collective_motion.get("majority_flow_direction_deg")),
                    direction_deviation_deg=self._round(collective_motion.get("direction_deviation_deg")),
                    density_trend=collective_motion.get("density_trend", "未知"),
                )
            )

        caption = item.get("caption")
        if caption:
            lines.append(f"畫面描述：{caption}")

        frame_file = item.get("frame_file")
        frame_idx = item.get("frame_idx")
        if frame_file is not None:
            lines.append(f"畫面來源：{frame_file}（第 {frame_idx} 幀）")

        if info["keywords"]:
            lines.append(f"關鍵字：{'、'.join(info['keywords'])}")

        content = "\n".join(lines)

        metadata: Dict[str, Any] = {
            "file_type": "event_record",
            "record_id": item.get("record_id"),
            "group_id": item.get("group_id"),
            "predicted_event_type": event_type,
            "caption": caption,
            "member_count": item.get("member_count"),
            "frame_file": frame_file,
            "frame_idx": frame_idx,
        }

        return LoadedDocument(content=content, metadata=metadata)

    @staticmethod
    def _round(value: Any) -> Any:
        """把角度數值四捨五入到小數點後一位，避免組出的描述文字出現一長串浮點數雜訊。"""
        if isinstance(value, (int, float)):
            return round(value, 1)
        return "未知"
