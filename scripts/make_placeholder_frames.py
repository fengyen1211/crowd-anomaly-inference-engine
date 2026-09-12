"""
scripts/make_placeholder_frames.py
=====================================
產生測試用的占位畫面檔案。

背景：這個開發環境目前沒有真實的影片/畫面檔案，但
FrameLoaderService 需要 frames_dir 底下真的存在
record.frame_file 對應的圖片才能運作（見 services/frame_service.py）。

這個腳本讀取事件 JSON，找出所有用到的 frame_file，在 frames_dir
底下產生對應的空白圖片——純粹讓 pipeline 的「讀圖、裁切、丟給 VLM」
這幾個步驟可以真的執行，VLM（目前是 MockVLM）看到的內容當然沒有
真實意義，不能拿來評估 VLM 的表現。

等你有真實的影片 keyframe 圖片後，把它們放進 frames_dir
（預設 ./frames，可在 .env 用 FRAMES_DIR 覆寫），
就不需要再跑這個腳本了。

使用方式（在 project/ 目錄下執行）：
    python scripts/make_placeholder_frames.py --input ../rag_vlm_mock_dataset.json
"""

import argparse
import json
import sys
from pathlib import Path

# 讓這個腳本不論從哪個工作目錄執行，都能 import 到 project 底下的套件
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

from config.settings import get_settings  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="為測試用途產生占位畫面檔案（非真實內容）")
    parser.add_argument("--input", required=True, help="事件 JSON 檔案路徑（單筆事件或整份資料集皆可）")
    parser.add_argument("--size", default="1920x1080", help="畫面尺寸，預設 1920x1080")
    args = parser.parse_args()

    raw = json.loads(Path(args.input).read_text(encoding="utf-8"))
    records = raw["records"] if isinstance(raw, dict) and "records" in raw else raw
    if isinstance(records, dict):
        records = [records]

    frame_files = sorted({record["frame_file"] for record in records if "frame_file" in record})
    if not frame_files:
        print("在輸入的 JSON 裡找不到任何 frame_file 欄位，沒有東西可以產生。")
        return

    width, height = (int(x) for x in args.size.split("x"))
    frames_dir = Path(get_settings().frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    created = 0
    for frame_file in frame_files:
        path = frames_dir / frame_file
        if path.exists():
            continue
        Image.new("RGB", (width, height), color=(120, 140, 160)).save(path, format="JPEG")
        created += 1

    print(f"完成：{frames_dir} 底下共有 {len(frame_files)} 個需要的畫面檔案，本次新建立 {created} 個。")
    print("提醒：這些都是純色占位圖片，只能用來測試流程是否跑得通，不是真實畫面內容。")


if __name__ == "__main__":
    main()
