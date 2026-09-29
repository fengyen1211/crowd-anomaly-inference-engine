"""
scripts/run_tinyformer_batch.py
===================================
把 data/tinyformer7_events.json（scripts/build_tinyformer_events.py 產生）
裡的 4 筆事件，逐一跑過完整推論端 pipeline，並把結果彙整成一份報告。

跟直接呼叫 4 次 scripts/run_pipeline.py 的差別：這裡在同一個 process
裡重複呼叫 get_orchestrator_service()，VLM/LLM/Embedding client 都是
lru_cache 過的單例，只會在第一次呼叫時真的載入模型，後面 3 筆事件
不會重複付出模型載入的時間成本。

使用方式（在 project/ 目錄下執行）：
    python scripts/run_tinyformer_batch.py
    python scripts/run_tinyformer_batch.py --output tinyformer7_pipeline_report_v2.json

輸出：
    - 主控台：每一筆事件的處理進度與耗時
    - tinyformer7_pipeline_report.json（預設）：完整結果彙整

注意：--output 預設會覆寫既有報告。這支腳本跑一次要 5-7 分鐘、且報告檔
在 .gitignore 裡（不進版控），覆寫掉就沒有了，因此要做前後對照實驗時
請明確指定不同的 --output，不要靠預設值。
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import ValidationError  # noqa: E402

from app.dependencies import get_orchestrator_service  # noqa: E402

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_INPUT_PATH = _PROJECT_ROOT / "data" / "tinyformer7_events.json"
_OUTPUT_PATH = _PROJECT_ROOT / "tinyformer7_pipeline_report.json"


def main() -> None:
    # 先前這支腳本把輸入/輸出路徑寫死、也沒有 argparse，導致傳入
    # --input / --output 會被「靜默忽略」——不會報錯，但報告仍然寫到預設
    # 路徑，覆蓋掉上一次實驗的結果。加上 argparse 讓這兩個參數真的生效。
    parser = argparse.ArgumentParser(description="批次跑 TinyFormer 事件的推論端 pipeline")
    parser.add_argument("--input", default=str(_INPUT_PATH), help="事件 JSON 路徑")
    parser.add_argument("--output", default=str(_OUTPUT_PATH), help="報告輸出路徑（會覆寫既有檔案）")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    raw = json.loads(input_path.read_text(encoding="utf-8"))
    records = raw["records"]

    orchestrator = get_orchestrator_service()

    results = []
    overall_start = time.time()

    for i, record in enumerate(records):
        record_id = record.get("record_id")
        print(f"\n[{i + 1}/{len(records)}] 處理事件：{record_id}（predicted_event_type={record.get('predicted_event_type')}）")
        start = time.time()
        try:
            output = orchestrator.process_event(record)
            elapsed = time.time() - start
            print(f"  -> 完成，耗時 {elapsed:.1f}s，alert_level={output.alert_level.value}")
            results.append({
                "record_id": record_id,
                "status": "ok",
                "elapsed_sec": round(elapsed, 1),
                "output": json.loads(output.model_dump_json()),
            })
        except (ValidationError, FileNotFoundError, Exception) as exc:  # noqa: BLE001
            elapsed = time.time() - start
            print(f"  -> 失敗（{type(exc).__name__}）：{exc}")
            results.append({
                "record_id": record_id,
                "status": "error",
                "elapsed_sec": round(elapsed, 1),
                "error_type": type(exc).__name__,
                "error": str(exc),
            })

    total_elapsed = time.time() - overall_start

    report = {
        "source": f"0901/tinyformer_events_7.json -> {input_path.name}",
        "total_records": len(records),
        "total_elapsed_sec": round(total_elapsed, 1),
        "results": results,
    }
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== 全部完成，共耗時 {total_elapsed:.1f}s ===")
    print(f"報告已寫入：{output_path}")


if __name__ == "__main__":
    main()
