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

輸出：
    - 主控台：每一筆事件的處理進度與耗時
    - tinyformer7_pipeline_report.json：完整結果彙整
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import ValidationError  # noqa: E402

from app.dependencies import get_orchestrator_service  # noqa: E402

_INPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "tinyformer7_events.json"
_OUTPUT_PATH = Path(__file__).resolve().parent.parent / "tinyformer7_pipeline_report.json"


def main() -> None:
    raw = json.loads(_INPUT_PATH.read_text(encoding="utf-8"))
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
        "source": "0901/tinyformer_events_7.json -> data/tinyformer7_events.json",
        "total_records": len(records),
        "total_elapsed_sec": round(total_elapsed, 1),
        "results": results,
    }
    _OUTPUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== 全部完成，共耗時 {total_elapsed:.1f}s ===")
    print(f"報告已寫入：{_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
