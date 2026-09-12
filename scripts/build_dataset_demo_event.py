"""
scripts/build_dataset_demo_event.py
======================================
把 dataset/dataset_export_7 的真實劇院畫面，組成一筆符合
utils.schemas.RawEventRecord 格式的「demo 事件 JSON」，讓你可以用
scripts/run_pipeline.py 跑完整的 9 步驟 pipeline——跟 rag_vlm_mock_dataset.json
不同的是，這裡 VLM 看到的是真實畫面內容，不是純色占位圖。

老實講清楚哪些欄位是真的、哪些是為了湊齊格式而放的佔位值：
- frame_file / frame_idx／畫面內容：真的（複製自 dataset_export_7 的實際幀）
- embedding：真的，用 embedding/clip.py 現場算出這張幀的 CLIP 向量
- spatial_bbox / group_id / member_ids / member_count / collective_motion /
  predicted_event_type：全部是佔位值，因為這支影片沒有跑過 YOLOv8+SAM2
  的人物追蹤與軌跡分析、也沒有跑過規則系統分類，這些數字不存在。
  caption 裡會誠實註明這一點，不會假裝是真的偵測結果。

使用方式（在 project/ 目錄下執行）：
    python scripts/build_dataset_demo_event.py --phase DURING --frame-file frame_00670.jpg
    python scripts/run_pipeline.py --input data/dataset7_demo_event.json

不帶 --frame-file 時，預設用該階段的第一張幀；--phase 可選 BEFORE/DURING/AFTER。
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

from app.dependencies import get_visual_embedding_client  # noqa: E402
from config.settings import get_settings  # noqa: E402

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DATASET_DIR = _PROJECT_ROOT / "dataset" / "dataset_export_7"
_OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "dataset7_demo_event.json"

_CAPTION_TEMPLATE = (
    "【Demo 資料，非影像端正式偵測結果】此幀取自 dataset/dataset_export_7"
    "（劇院場景測試影片），屬於 {phase} 階段（frame_idx={frame_idx}）。"
    "這支影片沒有跑過 YOLOv8+SAM2 軌跡分析與規則系統分類，因此 "
    "predicted_event_type、collective_motion、spatial_bbox、member_count "
    "皆為佔位值，僅用來讓 VLM 能看到真實畫面內容、示範完整 pipeline 如何運作。"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="把 dataset_export_7 的真實幀組成 demo 用的事件 JSON")
    parser.add_argument("--phase", choices=["BEFORE", "DURING", "AFTER"], default="DURING")
    parser.add_argument("--frame-file", default=None, help="指定幀檔名（例如 frame_00670.jpg），不指定則用該階段第一張")
    args = parser.parse_args()

    phase_dir = _DATASET_DIR / args.phase
    if args.frame_file:
        source_path = phase_dir / args.frame_file
    else:
        candidates = sorted(phase_dir.glob("*.jpg"))
        if not candidates:
            raise SystemExit(f"{phase_dir} 底下沒有任何幀")
        source_path = candidates[0]

    if not source_path.exists():
        raise SystemExit(f"找不到幀檔案：{source_path}")

    frame_idx = int("".join(ch for ch in source_path.stem if ch.isdigit()) or -1)

    frames_dir = Path(get_settings().frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    dest_filename = f"dataset7_{args.phase}_{source_path.name}"
    dest_path = frames_dir / dest_filename
    dest_path.write_bytes(source_path.read_bytes())

    width, height = Image.open(source_path).size

    clip = get_visual_embedding_client()
    embedding = clip.embed_image([source_path.read_bytes()])[0]

    record = {
        "record_id": f"dataset7_{args.phase}_{frame_idx}",
        "group_id": "dataset7_demo_group",
        "member_ids": [],
        "member_count": 0,
        "frame_file": dest_filename,
        "frame_idx": frame_idx,
        "spatial_bbox": [0, 0, float(width), float(height)],
        "predicted_event_type": "abnormal_gathering",
        "label_source": "manual_demo_no_rule_classification",
        "confidence": 0.0,
        "collective_motion": {
            "avg_direction_deg": 0.0,
            "majority_flow_direction_deg": 0.0,
            "direction_deviation_deg": 0.0,
            "density_trend": "unknown",
        },
        "caption": _CAPTION_TEMPLATE.format(phase=args.phase, frame_idx=frame_idx),
        "embedding": embedding,
        "embedding_model": "openai/clip-vit-base-patch32",
        "embedding_is_simulated": False,
        "embedding_dim": len(embedding),
    }

    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _OUTPUT_PATH.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    relative_output = _OUTPUT_PATH.relative_to(Path(__file__).resolve().parent.parent)
    print(f"已複製畫面到：{dest_path}")
    print(f"已寫入 demo 事件：{_OUTPUT_PATH}")
    print(f"\n接著執行：\n  python scripts/run_pipeline.py --input {relative_output}")


if __name__ == "__main__":
    main()
