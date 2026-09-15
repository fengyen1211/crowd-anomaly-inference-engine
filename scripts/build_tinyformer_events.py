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
- frame_file／frame_idx／畫面內容：真的，但**不是**從 0901/tinyformer_dataset_7
  複製（那批是影像端已疊加 bbox／軌跡線／文字標籤的渲染輸出，VLM 會把標註
  誤認成畫面內容，見下方「畫面來源修正」）。實際是從乾淨的
  7_clean/tinyformer_clean_dataset_7/DURING 讀取
- member_ids／spatial_bbox／predicted_event_type：真的，來自 TinyFormer+SAM2
  的實際輸出（聚合方式見上）
- embedding：真的，用 embedding/clip.py 現場算出代表幀的 CLIP 向量
- collective_motion：None——來源資料沒有方向性欄位（TinyFormer 這個逐幀
  偵測格式只給 z-score／密度，不像舊版規則系統會算 avg_direction_deg 等
  群體移動角度），沒有數字就不編造
- confidence：0.0（沿用專案慣例，這個欄位目前系統邏輯不讀取，真正的強度
  資訊已經寫進 caption 讓 LLM 看得到）

畫面來源修正：乾淨幀 + padding 裁切預覽
--------------------------------------------
影像端後來提供了 7_clean/tinyformer_clean_dataset_7/DURING：跟
0901/tinyformer_dataset_7/DURING（疊加標註版）同一份 30fps 輸出、逐幀一一
對應的 frame_id，只是沒有畫 bbox／軌跡線／文字標籤，所以可以直接用
tinyformer_events_7.json 裡的 frame_id 去這個資料夾找對應畫面，不需要
（也不再需要之前用 0901/7.mp4 時做的）24fps/30fps 時間戳換算。

另外，實際送進 VLM 的畫面不是這裡存的完整乾淨幀，而是 pipeline 執行時由
services/frame_service.py::crop_bbox() 依 spatial_bbox 外擴 padding
（3 倍寬高、最小 100px）裁出來的範圍——這一步在系統裡是全域生效的，不需要
在這支腳本裡重做。但為了能像除錯時那樣人工檢查「VLM 實際看到什麼」，
這支腳本另外呼叫同一個 crop_bbox() 存一份 `*_cropped_preview.jpg`
（不會被 pipeline 讀取，純供人工檢視用）。

panic_scatter／counter_flow 的事件前後脈絡畫面
--------------------------------------------
`panic_scatter`（人群四散）、`counter_flow`（逆向流動）本質上是「隨時間
變化」的行為，但 TinyFormer+SAM2 只給 z-score／密度這種瞬時強度數值，
沒有方向角度，VLM 又只看 DURING 單一靜態幀，看不出「四散」「反向」這種
需要前後對照才能判斷的動態。因此這兩類事件額外帶上 `before_frame_file`／
`after_frame_file`：取 `7_clean/.../BEFORE` 裡最接近 DURING 開始的最後一幀、
`AFTER` 裡最接近 DURING 結束的第一幀（同機位、同 `spatial_bbox` 座標，只是
時間點不同），讓 VLM 能比較事件前中後的畫面差異（見
vlm/observation_builder.py 的 prompt 組裝邏輯）。其餘兩類事件
（`sprint_spike`／`crowd_locked`）沒有這個時間比對的需求，維持原本單張
畫面的流程，這兩個欄位留 `None`。

使用方式（在 project/ 目錄下執行）：
    python scripts/build_tinyformer_events.py

輸出：data/tinyformer7_events.json（{"records": [...]} 格式，
可直接被 scripts/run_pipeline.py --input ... --index 0~3 使用）
"""

import io
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
from services.frame_service import FrameLoaderService  # noqa: E402

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SOURCE_DIR = _PROJECT_ROOT / "0901"
_SOURCE_JSON = _SOURCE_DIR / "tinyformer_events_7.json"
_CLEAN_DATASET_DIR = _PROJECT_ROOT / "7_clean" / "tinyformer_clean_dataset_7"
_CLEAN_FRAMES_DIR = _CLEAN_DATASET_DIR / "DURING"
_BEFORE_FRAMES_DIR = _CLEAN_DATASET_DIR / "BEFORE"
_AFTER_FRAMES_DIR = _CLEAN_DATASET_DIR / "AFTER"
_OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "tinyformer7_events.json"

# 需要「事件前後對照畫面」才能判斷的行為類型，見上方docstring。
_CONTEXT_FRAME_TYPES = {"panic_scatter", "counter_flow"}

_EVENT_TYPE_RE = re.compile(r"^([a-z_]+)(?:\(([a-z]+)=([-0-9.]+)\))?$")
_FRAME_ID_RE = re.compile(r"frame_(\d+)\.jpg$")

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


def _load_clean_frame(peak_frame_id: int) -> bytes:
    """從乾淨（無疊加標註）的 DURING 畫面資料夾讀取跟標註版一一對應的 frame_id。"""
    source_path = _CLEAN_FRAMES_DIR / f"frame_{peak_frame_id:05d}.jpg"
    if not source_path.exists():
        raise SystemExit(f"找不到乾淨代表幀畫面：{source_path}")
    return source_path.read_bytes()


def _find_boundary_frame(directory: Path, pick_max: bool) -> Path:
    """
    BEFORE 資料夾取最接近 DURING 開始的最後一幀（pick_max=True）；
    AFTER 資料夾取最接近 DURING 結束的第一幀（pick_max=False）。
    """
    candidates = [
        (int(match.group(1)), path)
        for path in directory.glob("frame_*.jpg")
        if (match := _FRAME_ID_RE.search(path.name))
    ]
    if not candidates:
        raise SystemExit(f"找不到任何畫面檔案：{directory}")
    picker = max if pick_max else min
    return picker(candidates, key=lambda c: c[0])[1]


def _save_context_frame(kind: str, source_path: Path, frames_dir: Path) -> str:
    """存 panic_scatter／counter_flow 共用的事件前／後脈絡畫面（只存一份，
    兩個事件類型的 record 都指向同一個檔案）。"""
    frames_dir.mkdir(parents=True, exist_ok=True)
    dest_filename = f"tinyformer7_context_{kind}.jpg"
    (frames_dir / dest_filename).write_bytes(source_path.read_bytes())
    return dest_filename


def _save_peak_frame(base_type: str, peak_frame_id: int, frame_bytes: bytes, frames_dir: Path) -> str:
    frames_dir.mkdir(parents=True, exist_ok=True)
    dest_filename = f"tinyformer7_{base_type}_frame_{peak_frame_id:05d}.jpg"
    (frames_dir / dest_filename).write_bytes(frame_bytes)
    return dest_filename


def _save_cropped_preview(
    base_type: str,
    peak_frame_id: int,
    frame_bytes: bytes,
    bbox: List[float],
    frames_dir: Path,
    frame_loader: FrameLoaderService,
) -> Tuple[str, Tuple[int, int]]:
    """存一份跟 pipeline 執行時同樣邏輯（含 padding）裁切出來的預覽圖，供人工檢查
    「VLM 實際看到的畫面」是否可辨識——不會被 pipeline 讀取。"""
    cropped = frame_loader.crop_bbox(frame_bytes, bbox)
    preview_filename = f"tinyformer7_{base_type}_frame_{peak_frame_id:05d}_cropped_preview.jpg"
    (frames_dir / preview_filename).write_bytes(cropped)
    return preview_filename, Image.open(io.BytesIO(cropped)).size


def main() -> None:
    prompts = _load_source()
    grouped = _group_by_category(prompts)

    frames_dir = Path(get_settings().frames_dir)
    clip = get_visual_embedding_client()
    frame_loader = FrameLoaderService(frames_dir=str(frames_dir))

    if not _CLEAN_FRAMES_DIR.exists():
        raise SystemExit(f"找不到乾淨畫面資料夾：{_CLEAN_FRAMES_DIR}")

    before_source_path = _find_boundary_frame(_BEFORE_FRAMES_DIR, pick_max=True)
    after_source_path = _find_boundary_frame(_AFTER_FRAMES_DIR, pick_max=False)
    before_frame_file = _save_context_frame("before", before_source_path, frames_dir)
    after_frame_file = _save_context_frame("after", after_source_path, frames_dir)

    records = []
    print(f"來源：{_SOURCE_JSON}（共 {len(prompts)} 筆逐幀原始訊號，聚合成 {len(grouped)} 筆事件）")
    print(f"乾淨畫面：{_CLEAN_FRAMES_DIR}")
    print(
        f"事件前後脈絡畫面（僅 {sorted(_CONTEXT_FRAME_TYPES)} 使用）："
        f"before={before_source_path.name}->{before_frame_file}，"
        f"after={after_source_path.name}->{after_frame_file}\n"
    )

    for base_type, items in sorted(grouped.items(), key=lambda kv: -len(kv[1])):
        peak = _pick_peak_instance(items)
        peak_frame_id = peak["frame_id"]

        frame_bytes = _load_clean_frame(peak_frame_id)
        dest_filename = _save_peak_frame(base_type, peak_frame_id, frame_bytes, frames_dir)
        width, height = Image.open(io.BytesIO(frame_bytes)).size
        embedding = clip.embed_image([frame_bytes])[0]

        members = sorted({m for p in items for m in p["member_ids"]})
        bbox = [max(0.0, min(float(c), dim)) for c, dim in zip(peak["visual_prompt_bbox"], [width, height, width, height])]

        preview_filename, preview_size = _save_cropped_preview(
            base_type, peak_frame_id, frame_bytes, bbox, frames_dir, frame_loader
        )

        has_context = base_type in _CONTEXT_FRAME_TYPES
        record = {
            "record_id": f"tinyformer7_{base_type}",
            "group_id": "tinyformer7_video7_during",
            "member_ids": members,
            "member_count": len(members),
            "frame_file": dest_filename,
            "frame_idx": peak_frame_id,
            "before_frame_file": before_frame_file if has_context else None,
            "after_frame_file": after_frame_file if has_context else None,
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

        context_note = "，含事件前後脈絡畫面" if has_context else ""
        print(
            f"[{base_type}] {len(items)} 筆原始訊號 / {len(members)} 人 -> "
            f"代表幀 frame_{peak_frame_id:05d} -> {dest_filename}"
            f"（裁切預覽 {preview_filename}，{preview_size[0]}x{preview_size[1]}）{context_note}"
        )

    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _OUTPUT_PATH.write_text(json.dumps({"records": records}, ensure_ascii=False, indent=2), encoding="utf-8")

    relative_output = _OUTPUT_PATH.relative_to(Path(__file__).resolve().parent.parent)
    print(f"\n已寫入 {len(records)} 筆事件：{_OUTPUT_PATH}")
    print(f"\n接著可用：\n  python scripts/run_pipeline.py --input {relative_output} --index 0")


if __name__ == "__main__":
    main()
