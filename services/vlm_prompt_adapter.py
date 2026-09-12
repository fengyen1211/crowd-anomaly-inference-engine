"""
services/vlm_prompt_adapter.py
=================================
把影像端「逐幀 VLM 提示」格式（例如 visual_tracks_events_13.json 的
vlm_visual_prompts 陣列）轉接成推論端 pipeline 認得的 RawEventRecord
相容 dict。

背景：這是比 rag_vlm_mock_dataset.json / timeline_events_7.json 更底層
的影像端輸出——沒有經過 DBSCAN 分群聚合，是逐幀、逐個體/小群體的統計
異常偵測（sprint_spike：個體速度 z-score 異常；crowd_locked：3-5 人小
群體密度鎖定），同一個人連續數十幀都會各產生一筆記錄。

如果一筆原始記錄對應一次 LLM+VLM 呼叫，一支影片動輒上千筆、要跑上千次
本地模型，不切實際。所以這裡先做「聚合」：把同一人/同一群、時間相鄰的
逐幀偵測合併成一筆「事件片段」，才送進現有 pipeline（一個片段 = 一次
VLM 觀察 + 一次 LLM 推理，跟其餘來源的處理粒度一致）。
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils.logger import get_logger

logger = get_logger(__name__)

_METRIC_PATTERN = re.compile(r"\(([a-zA-Z_]+)=([\d.]+)\)")


def load_vlm_prompts(path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """讀取 vlm_visual_prompts 格式的 JSON，回傳 (metadata, prompts)。"""
    raw = json.loads(path.read_text(encoding="utf-8"))
    metadata = raw.get("metadata", {})
    prompts = raw.get("vlm_visual_prompts", [])
    logger.info("load_vlm_prompts 讀取 %s：%d 筆原始偵測", path, len(prompts))
    return metadata, prompts


def _base_event_type(event_type: str) -> str:
    """"sprint_spike(z=3.1)" -> "sprint_spike"。"""
    return event_type.split("(")[0]


def _extract_metric(event_type: str) -> Optional[float]:
    """從 "sprint_spike(z=3.1)" 這種字串取出括號內的數值（3.1），查無則回傳 None。"""
    match = _METRIC_PATTERN.search(event_type)
    if not match:
        return None
    return float(match.group(2))


def segment_prompts(prompts: List[Dict[str, Any]], max_frame_gap: int = 15) -> List[List[Dict[str, Any]]]:
    """
    把逐幀偵測依「事件基礎類型 + 時間相鄰 + member_ids 有交集」聚合成事件片段。

    先依基礎事件類型分組，組內依 frame_id 排序後逐筆嘗試併入某個仍在
    「開放中」的片段：跟該片段最後一筆的 frame_id 差距在 max_frame_gap
    以內、且 member_ids 有交集，就併入；找不到符合的就另開一個新片段。

    這裡刻意維護「多個同時開放中的片段」，而不是只追蹤單一個「目前片段」
    ——實測資料裡，不同人（不同 member_id）的偵測會依 frame_id 交錯出現
    （例如 frame 411 先出現 member 631，接著 member 657，frame 412 又
    出現 member 631...），如果只追蹤單一個「目前片段」，member 657 的
    記錄插進來就會讓 member 631 的片段追蹤斷掉、下次 631 出現時被誤判成
    新片段，導致 1022 筆幾乎沒被聚合（實測踩過這個坑：只追蹤單一片段時
    1022 筆只聚合成 816 個片段，改成多片段並行後才真正聚合到位）。

    用「有交集」而不是「完全相同」比對 member_ids，是因為 crowd_locked
    的 member 組成會隨幀數緩慢變動（例如 [224,438,577] 幾幀後變成
    [224,186,863]，只有 224 是共同的）——完全比對會把同一個持續中的
    群聚事件切得太碎。
    """
    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for prompt in prompts:
        base_type = _base_event_type(prompt["event_type"])
        by_type.setdefault(base_type, []).append(prompt)

    all_segments: List[List[Dict[str, Any]]] = []
    for base_type, group in by_type.items():
        group_sorted = sorted(group, key=lambda p: p["frame_id"])

        # 每個開放中的片段是 {"prompts": [...], "members": set, "last_frame": int}
        open_segments: List[Dict[str, Any]] = []

        for prompt in group_sorted:
            member_ids = set(prompt["member_ids"])
            frame_id = prompt["frame_id"]

            match = next(
                (
                    seg
                    for seg in open_segments
                    if (frame_id - seg["last_frame"]) <= max_frame_gap and (member_ids & seg["members"])
                ),
                None,
            )

            if match is not None:
                match["prompts"].append(prompt)
                match["members"] |= member_ids
                match["last_frame"] = frame_id
            else:
                open_segments.append({"prompts": [prompt], "members": set(member_ids), "last_frame": frame_id})

        all_segments.extend(seg["prompts"] for seg in open_segments)

    logger.info(
        "segment_prompts 把 %d 筆原始偵測聚合成 %d 個事件片段（max_frame_gap=%d）",
        len(prompts), len(all_segments), max_frame_gap,
    )
    return all_segments


def build_record(segment: List[Dict[str, Any]], video: str, segment_index: int) -> Dict[str, Any]:
    """把一個事件片段組成 RawEventRecord 相容的 dict。"""
    base_type = _base_event_type(segment[0]["event_type"])

    representative = max(segment, key=lambda p: _extract_metric(p["event_type"]) or 0.0)
    max_metric = _extract_metric(representative["event_type"]) or 0.0

    member_ids: List[int] = sorted({m for p in segment for m in p["member_ids"]})

    start_frame = segment[0]["frame_id"]
    end_frame = segment[-1]["frame_id"]
    start_sec = segment[0]["timestamp_sec"]
    end_sec = segment[-1]["timestamp_sec"]

    if base_type == "sprint_spike":
        caption = (
            f"追蹤目標 {member_ids} 在第 {start_frame}-{end_frame} 幀"
            f"（約 {start_sec:.1f}-{end_sec:.1f} 秒）期間偵測到速度異常尖峰，"
            f"z-score 最高達 {max_metric:.1f}"
        )
    else:
        caption = (
            f"偵測到 {len(member_ids)} 人的群體在第 {start_frame}-{end_frame} 幀"
            f"（約 {start_sec:.1f}-{end_sec:.1f} 秒）期間呈現群聚鎖定狀態，"
            f"密度指標最高達 {max_metric:.2f}"
        )

    record_id = f"{video}_{base_type}_{segment_index:03d}"

    return {
        "record_id": record_id,
        "group_id": record_id,
        "member_ids": member_ids,
        "member_count": len(member_ids),
        "frame_file": f"{representative['state']}/frame_{representative['frame_id']:05d}.jpg",
        "frame_idx": representative["frame_id"],
        "spatial_bbox": representative["visual_prompt_bbox"],
        "predicted_event_type": base_type,
        "label_source": "rule_based_unvalidated",
        "caption": caption,
    }
