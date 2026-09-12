"""
scripts/build_tinyformer_events.py
======================================
把 0901/ 資料夾（影像端新模型 TinyFormer+SAM2 的輸出）轉換成推論端可以吃的
事件 JSON（符合 utils.schemas.RawEventRecord 格式）。

來源資料長什麼樣子、為什麼不能直接 1:1 轉換
--------------------------------------------
0901/tinyformer_events_7.json 的 vlm_visual_prompts 不是「事件列表」，
是逐幀、逐人的原始異常訊號：同一支影片（video 7，DURING 階段，
frame 378-1086，共 23.6 秒）裡，總共有 2,997 筆紀錄，涵蓋 315 個不同的人。
如果每一筆都當成一個獨立事件塞進 RawEventRecord，會產生數百到近三千筆
「事件」，既不符合 RawEventRecord 原本「一筆＝一個群體層級事件」的設計
（一個 bbox、一個 caption、一組 member_ids），也不可能在這台機器的硬體
（VLM/LLM 各要 20-40 秒）上一一跑完整條 pipeline。

實際作法：依「行為類型」（sprint_spike / crowd_locked / panic_scatter /
counter_flow）聚合成 4 筆事件，每一筆代表「這支影片裡這種行為模式的整體
狀況」：
- member_ids／member_count：該類型底下涉及到的所有不重複的人物 ID
- 代表幀（frame_idx／frame_file／spatial_bbox）：該類型裡「異常程度最高」
  的單一實例（sprint_spike 用 z-score 最大值；crowd_locked 用密度最大值；
  panic_scatter／counter_flow 沒有強度數值，取第一個實例）——VLM 只能看
  一張畫面，選最極端的例子最有代表性
- caption：誠實列出聚合統計數字（實例數、涉及人數、時間跨度、
  z-score/密度的範圍），並明確註明「畫面顯示的只是其中一個代表性瞬間，
  不是整段時間的畫面」，避免 LLM 誤以為代表幀 = 全部異常都發生在那一幀

哪些欄位是真的、哪些是誠實的佔位值
------------------------------------
- frame_file／frame_idx／畫面內容：真的（複製自 0901/tinyformer_dataset_7
  的實際幀）
- member_ids／spatial_bbox／predicted_event_type：真的，來自 TinyFormer+SAM2
  的實際輸出（聚合方式見上）
- embedding：真的，用 embedding/clip.py 現場算出代表幀的 CLIP 向量
- collective_motion：None——來源資料沒有方向性欄位（TinyFormer 這個逐幀
  偵測格式只給 z-score／密度，不像舊版規則系統會算 avg_direction_deg 等
  群體移動角度），沒有數字就不編造
- confidence：0.0（沿用專案慣例，這個欄位目前系統邏輯不讀取，真正的強度
  資訊已經寫進 caption 讓 LLM 看得到）

使用方式（在 project/ 目錄下執行）：
    python scripts/build_tinyformer_events.py

輸出：data/tinyformer7_events.json（{"records": [...]} 格式，
可直接被 scripts/run_pipeline.py --input ... --index 0~3 使用）
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

from app.dependencies import get_visual_embedding_client  # noqa: E402
from config.settings import get_settings  # noqa: E402

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SOURCE_DIR = _PROJECT_ROOT / "0901"
_SOURCE_JSON = _SOURCE_DIR / "tinyformer_events_7.json"
_FRAMES_SOURCE_DIR = _SOURCE_DIR / "tinyformer_dataset_7" / "DURING"
_OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "tinyformer7_events.json"

_EVENT_TYPE_RE = re.compile(r"^([a-z_]+)(?:\(([a-z]+)=([-0-9.]+)\))?$")

_CAPTION_TEMPLATE = (
    "【TinyFormer+SAM2 影像端偵測，聚合自逐幀原始訊號，非人工標註】"
    "來源影片：video 7，DURING 異常階段（frame 378-1086，約 12.6s-36.2s）。"
    "行為類型「{base_type}」在這段期間內，系統總共標記了 {instance_count} 次"
    "逐幀觀測，涉及 {member_count} 個不重複的追蹤人物 ID"
    "（時間跨度 frame {frame_min}-{frame_max}，約 {ts_min:.2f}s-{ts_max:.2f}s）。"
    "{metric_desc}"
    "本筆畫面（frame_idx={peak_frame}）取的是這個類型裡「異常程度最高」的"
    "單一代表性瞬間，不代表整段時間畫面都長這樣，也不代表所有 {member_count} "
    "個相關人物都同時出現在這張畫面裡——請 VLM 只描述這張畫面實際看到的內容。"
    "TinyFormer+SAM2 目前只輸出 bbox 與強度數值（z-score／密度），沒有輸出"
    "群體移動方向角度，因此 collective_motion 欄位誠實留空，不是遺漏。"
)

_METRIC_LABEL = {"z": "z-score（相對正常速度分布的離群程度）", "den": "局部人群密度"}


def _parse_event_type(event_type: str) -> Tuple[str, Optional[str], Optional[float]]:
    match = _EVENT_TYPE_RE.match(event_type)
    if not match:
        return event_type, None, None
    base, metric, value = match.groups()
    return base, metric, (float(value) if value is not None else None)


def _load_source() -> List[Dict[str, Any]]:
    raw = json.loads(_SOURCE_JSON.read_text(encoding="utf-8"))
    prompts = raw["vlm_visual_prompts"]
    for p in prompts:
        base, metric, value = _parse_event_type(p["event_type"])
        p["_base"], p["_metric"], p["_value"] = base, metric, value
    return prompts


def _group_by_category(prompts: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for p in prompts:
        grouped[p["_base"]].append(p)
    return grouped


def _pick_peak_instance(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    has_metric = any(p["_value"] is not None for p in items)
    if has_metric:
        return max(items, key=lambda p: p["_value"] if p["_value"] is not None else float("-inf"))
    return items[0]


def _build_caption(base_type: str, items: List[Dict[str, Any]], peak: Dict[str, Any]) -> str:
    members: set = set()
    for p in items:
        members.update(p["member_ids"])
    frames = sorted(p["frame_id"] for p in items)
    timestamps = sorted(p["timestamp_sec"] for p in items)

    values = [p["_value"] for p in items if p["_value"] is not None]
    if values:
        metric_name = _METRIC_LABEL.get(peak["_metric"], peak["_metric"])
        metric_desc = (
            f"強度指標為{metric_name}，範圍 {min(values):.2f} 至 {max(values):.2f}"
            f"（平均 {sum(values) / len(values):.2f}，本筆取樣的是最高值 {peak['_value']:.2f}）。"
        )
    else:
        metric_desc = "此類型來源資料未提供強度數值，僅有位置與涉及人物紀錄。"

    return _CAPTION_TEMPLATE.format(
        base_type=base_type,
        instance_count=len(items),
        member_count=len(members),
        frame_min=frames[0],
        frame_max=frames[-1],
        ts_min=timestamps[0],
        ts_max=timestamps[-1],
        metric_desc=metric_desc,
        peak_frame=peak["frame_id"],
    )


def _copy_peak_frame(base_type: str, peak_frame_id: int, frames_dir: Path) -> str:
    source_path = _FRAMES_SOURCE_DIR / f"frame_{peak_frame_id:05d}.jpg"
    if not source_path.exists():
        raise SystemExit(f"找不到代表幀畫面：{source_path}")

    frames_dir.mkdir(parents=True, exist_ok=True)
    dest_filename = f"tinyformer7_{base_type}_frame_{peak_frame_id:05d}.jpg"
    dest_path = frames_dir / dest_filename
    dest_path.write_bytes(source_path.read_bytes())
    return dest_filename


def main() -> None:
    prompts = _load_source()
    grouped = _group_by_category(prompts)

    frames_dir = Path(get_settings().frames_dir)
    clip = get_visual_embedding_client()

    records = []
    print(f"來源：{_SOURCE_JSON}（共 {len(prompts)} 筆逐幀原始訊號，聚合成 {len(grouped)} 筆事件）\n")

    for base_type, items in sorted(grouped.items(), key=lambda kv: -len(kv[1])):
        peak = _pick_peak_instance(items)
        peak_frame_id = peak["frame_id"]

        dest_filename = _copy_peak_frame(base_type, peak_frame_id, frames_dir)
        source_path = _FRAMES_SOURCE_DIR / f"frame_{peak_frame_id:05d}.jpg"
        width, height = Image.open(source_path).size
        embedding = clip.embed_image([source_path.read_bytes()])[0]

        members = sorted({m for p in items for m in p["member_ids"]})
        bbox = [max(0.0, min(float(c), dim)) for c, dim in zip(peak["visual_prompt_bbox"], [width, height, width, height])]

        record = {
            "record_id": f"tinyformer7_{base_type}",
            "group_id": "tinyformer7_video7_during",
            "member_ids": members,
            "member_count": len(members),
            "frame_file": dest_filename,
            "frame_idx": peak_frame_id,
            "spatial_bbox": bbox,
            "predicted_event_type": base_type,
            "label_source": "tinyformer_sam2_zscore_unvalidated",
            "confidence": 0.0,
            "collective_motion": None,
            "caption": _build_caption(base_type, items, peak),
            "embedding": embedding,
            "embedding_model": "openai/clip-vit-base-patch32",
            "embedding_is_simulated": False,
            "embedding_dim": len(embedding),
        }
        records.append(record)

        print(f"[{base_type}] {len(items)} 筆原始訊號 / {len(members)} 人 -> 代表幀 frame_{peak_frame_id:05d}.jpg")

    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _OUTPUT_PATH.write_text(json.dumps({"records": records}, ensure_ascii=False, indent=2), encoding="utf-8")

    relative_output = _OUTPUT_PATH.relative_to(Path(__file__).resolve().parent.parent)
    print(f"\n已寫入 {len(records)} 筆事件：{_OUTPUT_PATH}")
    print(f"\n接著可用：\n  python scripts/run_pipeline.py --input {relative_output} --index 0")


if __name__ == "__main__":
    main()
