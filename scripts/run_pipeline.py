"""
scripts/run_pipeline.py
==========================
命令列工具：讀取一份事件 JSON，跑過完整的推論端 pipeline，印出結果。

這是最直接測試「JSON -> VLM -> Embedding -> Retriever -> Knowledge
-> LLM -> Output」整條管線的方式，不需要另外啟動 API 伺服器。

使用方式（在 project/ 目錄下執行）：
    python scripts/run_pipeline.py --input path/to/event.json
    python scripts/run_pipeline.py --input path/to/dataset.json --index 3

支援兩種 JSON 結構：
- 單筆事件：{"record_id": ..., "predicted_event_type": ..., ...}
- 整份資料集：{"records": [{...}, {...}]}（用 --index 選擇要跑哪一筆，預設 0）

如果遇到「找不到畫面檔案」的錯誤，先執行：
    python scripts/make_placeholder_frames.py --input <同一份 JSON>
"""

import argparse
import json
import sys
from pathlib import Path

# 讓這個腳本不論從哪個工作目錄執行，都能 import 到 project 底下的套件
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import ValidationError  # noqa: E402

from app.dependencies import get_orchestrator_service  # noqa: E402


def load_record(input_path: Path, index: int) -> dict:
    raw = json.loads(input_path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and isinstance(raw.get("records"), list):
        return raw["records"][index]
    if isinstance(raw, list):
        return raw[index]
    return raw


def main() -> None:
    parser = argparse.ArgumentParser(description="跑一次推論端 pipeline（JSON -> ... -> Output）")
    parser.add_argument("--input", required=True, help="事件 JSON 檔案路徑")
    parser.add_argument("--index", type=int, default=0, help="若輸入是整份資料集，指定要跑第幾筆（預設 0）")
    args = parser.parse_args()

    record = load_record(Path(args.input), args.index)
    print(f"處理事件：record_id={record.get('record_id')}, frame_file={record.get('frame_file')}")

    orchestrator = get_orchestrator_service()

    try:
        output = orchestrator.process_event(record)
    except ValidationError as exc:
        print("\n事件 JSON 驗證失敗（欄位缺漏或型別不符）：")
        print(exc)
        sys.exit(1)
    except FileNotFoundError as exc:
        print(f"\n找不到畫面檔案：{exc}")
        print("提示：先執行以下指令產生測試用占位畫面：")
        print(f"  python scripts/make_placeholder_frames.py --input {args.input}")
        sys.exit(1)

    print("\n=== 推論結果（PipelineOutput）===")
    print(output.model_dump_json(indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
