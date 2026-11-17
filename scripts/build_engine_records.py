"""
scripts/build_engine_records.py
================================
把影像端直接給的 `engine_records`（非群聚資料第二批，2026-11-16 起）轉成
推論端的事件 JSON（{"records": [...]}），可以直接給 scripts/run_pipeline.py
或 scripts/nano4/run_model_sweep.sbatch 使用。

和 build_tinyformer_events.py 的差別
------------------------------------
第一批只有逐幀訊號（vlm_visual_prompts），切段、選代表幀、寫 caption 都是
推論端自己做（build_tinyformer_events.py --ucf-root）。第二批的 events JSON
多了 `engine_records`：影像端已經把每段 DURING 整理成一筆 RawEventRecord，
這支腳本**不重新切段**，只補上推論端執行時需要、影像端沒附的東西：

- 代表幀圖片：engine_records 的 frame_file 只是檔名，圖片要用 frame_idx 去
  同一支影片的 DURING/frame_<5 位數>.jpg 找，複製到 frames/
- 前後脈絡畫面：before/after_frame_file 同樣只是檔名。沿用第一批的規則：
  sprint_spike 取代表幀前後 15 幀，panic_scatter／counter_flow 取該段開始前、
  結束後 30 幀（段落範圍從 caption 的「異常區間 frame A-B」讀），其他類型留 None
- embedding：用 embedding/clip.py 算代表幀的 CLIP 向量
- spatial_bbox：夾在畫面範圍內（同 build_tinyformer_events.py）

其餘欄位（member_ids、predicted_event_type、collective_motion、caption、
label_source）照影像端原樣保留。confidence 影像端固定 0.0，也照抄。

命名
----
record_id 加上 `ucf2_` 前綴（例如 Arrest001_616 -> ucf2_Arrest001_616），
group_id 改成 `ucf2_<影片>_during`，畫面檔名也用同樣前綴，避免跟第一批
（ucf_ 前綴）在 frames/ 裡撞名。

資料夾結構
----------
第二批解壓後多一層：<root>/<影片>_clean/<影片>_clean/...，兩種結構都支援。

使用方式（在 project/ 目錄下執行）：
    python scripts/build_engine_records.py --root ../非群聚資料_給推論端_第二批_1116/非群聚資料_給AI端_第二批_1116
    python scripts/build_engine_records.py --root <同上> --videos Arrest001 Explosion007

輸出：預設 data/ucf_batch2_events.json；接著用
    python scripts/build_ucf_labels.py --events data/ucf_batch2_events.json \
        --ranking <root>/ranking.md --output data/labels_ucf2_gt.json
產生標準答案。
"""

import argparse
import io
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from PIL import Image  # noqa: E402

from app.dependencies import get_visual_embedding_client  # noqa: E402
from build_tinyformer_events import (  # noqa: E402
    _CONTEXT_FRAME_TYPES,
    _CONTEXT_OFFSET_FRAMES,
    _PEAK_CONTEXT_OFFSET_FRAMES,
    _PEAK_CONTEXT_TYPES,
    _find_context_frame,
    _index_clean_frames,
)
from config.settings import get_settings  # noqa: E402
from services.frame_service import FrameLoaderService  # noqa: E402
from utils.schemas import RawEventRecord  # noqa: E402

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_OUTPUT_PATH = _DATA_DIR / "ucf_batch2_events.json"
_PREFIX = "ucf2"
_SEGMENT_RE = re.compile(r"異常區間 frame (\d+)-(\d+)")


def _video_dirs(root: Path, videos: Optional[List[str]]) -> List[Tuple[str, Path]]:
    """回傳 (影片名, 含 events JSON 的資料夾)，支援多包一層的解壓結構。"""
    found = []
    for d in sorted(p for p in root.iterdir() if p.is_dir() and p.name.endswith("_clean")):
        name = d.name.removesuffix("_clean")
        inner = d / d.name
        found.append((name, inner if inner.is_dir() else d))
    if videos:
        wanted = set(videos)
        missing = wanted - {n for n, _ in found}
        if missing:
            raise SystemExit(f"找不到影片：{sorted(missing)}（在 {root}）")
        found = [(n, d) for n, d in found if n in wanted]
    if not found:
        raise SystemExit(f"{root} 底下沒有 *_clean 資料夾")
    return found


def _context_range(record: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    """依事件類型決定前後脈絡畫面要找的時間點（規則同 build_tinyformer_events.py）。"""
    event_type = record["predicted_event_type"]
    peak = record["frame_idx"]
    if event_type in _PEAK_CONTEXT_TYPES:
        return peak - _PEAK_CONTEXT_OFFSET_FRAMES, peak + _PEAK_CONTEXT_OFFSET_FRAMES
    if event_type in _CONTEXT_FRAME_TYPES:
        m = _SEGMENT_RE.search(record.get("caption", ""))
        start, end = (int(m.group(1)), int(m.group(2))) if m else (peak, peak)
        return start - _CONTEXT_OFFSET_FRAMES, end + _CONTEXT_OFFSET_FRAMES
    return None


def _build_video(
    name: str,
    video_dir: Path,
    frames_dir: Path,
    clip: Any,
    frame_loader: FrameLoaderService,
) -> List[Dict[str, Any]]:
    source_json = video_dir / f"tinyformer_clean_events_{name}.json"
    dataset_dir = video_dir / f"tinyformer_clean_dataset_{name}"
    data = json.loads(source_json.read_text(encoding="utf-8"))
    engine_records = data.get("engine_records") or []
    if not engine_records:
        print(f"{name}：沒有 engine_records（不產生事件）")
        return []
    clean_index = _index_clean_frames(dataset_dir)

    records = []
    for src in engine_records:
        frame_idx = int(src["frame_idx"])
        frame_path = dataset_dir / "DURING" / f"frame_{frame_idx:05d}.jpg"
        if not frame_path.exists():
            raise SystemExit(f"找不到代表幀：{frame_path}（record {src['record_id']}）")
        frame_bytes = frame_path.read_bytes()
        width, height = Image.open(io.BytesIO(frame_bytes)).size

        record_id = f"{_PREFIX}_{src['record_id']}"
        frame_file = f"{record_id}_frame_{frame_idx:05d}.jpg"
        (frames_dir / frame_file).write_bytes(frame_bytes)

        bbox = [
            max(0.0, min(float(c), dim))
            for c, dim in zip(src["spatial_bbox"], [width, height, width, height])
        ]
        # 供人工檢查用的裁切預覽（pipeline 不讀），同 build_tinyformer_events.py
        (frames_dir / f"{record_id}_frame_{frame_idx:05d}_cropped_preview.jpg").write_bytes(
            frame_loader.crop_bbox(frame_bytes, bbox)
        )

        before_file = after_file = None
        context = _context_range(src)
        if context:
            ids = (
                _find_context_frame(clean_index, context[0], before=True),
                _find_context_frame(clean_index, context[1], before=False),
            )
            names = []
            for fid in ids:
                fname = f"{_PREFIX}_{name}_context_frame_{fid:05d}.jpg"
                (frames_dir / fname).write_bytes(clean_index[fid].read_bytes())
                names.append(fname)
            before_file, after_file = names

        embedding = clip.embed_image([frame_bytes])[0]
        record = {
            "record_id": record_id,
            "group_id": f"{_PREFIX}_{name}_during",
            "member_ids": src.get("member_ids", []),
            "member_count": src.get("member_count", len(src.get("member_ids", []))),
            "frame_file": frame_file,
            "frame_idx": frame_idx,
            "before_frame_file": before_file,
            "after_frame_file": after_file,
            "spatial_bbox": bbox,
            "predicted_event_type": src["predicted_event_type"],
            "label_source": src.get("label_source", "unvalidated"),
            "confidence": src.get("confidence", 0.0),
            "collective_motion": src.get("collective_motion"),
            "caption": src["caption"],
            "embedding": embedding,
            "embedding_model": "openai/clip-vit-base-patch32",
            "embedding_is_simulated": False,
            "embedding_dim": len(embedding),
        }
        RawEventRecord.model_validate(record)  # 欄位不符就在這裡停下，不要等到 Nano4 上才發現
        records.append(record)

        context_note = f"，前後 {before_file} / {after_file}" if context else ""
        print(f"  {record_id:<28}{src['predicted_event_type']:<22}frame {frame_idx}{context_note}")
    print(f"{name}：{len(records)} 筆")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="影像端 engine_records -> 推論端事件 JSON")
    parser.add_argument("--root", type=Path, required=True, help="非群聚資料第二批的資料夾（含 *_clean 與 ranking.md）")
    parser.add_argument("--videos", nargs="+", help="只處理這幾支")
    parser.add_argument("--output", type=Path, default=_OUTPUT_PATH)
    args = parser.parse_args()

    frames_dir = Path(get_settings().frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    clip = get_visual_embedding_client()
    frame_loader = FrameLoaderService(frames_dir=str(frames_dir))

    records: List[Dict[str, Any]] = []
    for name, video_dir in _video_dirs(args.root, args.videos):
        records.extend(_build_video(name, video_dir, frames_dir, clip, frame_loader))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已寫入 {len(records)} 筆事件：{args.output}（畫面在 {frames_dir}）")


if __name__ == "__main__":
    main()
