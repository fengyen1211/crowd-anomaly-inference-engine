"""
scripts/run_pipeline_vlm_prompts.py
======================================
命令列工具：讀取影像端「逐幀 VLM 提示」格式的事件 JSON
（例如 13_8-9/visual_tracks_events_13.json 的 vlm_visual_prompts 陣列），
聚合成事件片段後，逐一跑過完整的推論端 pipeline，印出結果並存檔。

跟 scripts/run_pipeline.py 的差異：run_pipeline.py 吃的是已經聚合好的
單筆／整份 RawEventRecord 資料集；這支腳本吃的是更底層、逐幀輸出的原始
偵測資料，需要先用 services/vlm_prompt_adapter.py 聚合，且整份檔案通常
對應「一支影片的多個事件」，所以是跑一整批、不是單筆。

一個事件片段要跑完 VLM + LLM 全部步驟大約 70-90 秒，一支影片動輒上百個
片段，單次執行往往要跑上數小時，很容易被系統中途中斷（單次執行時間
上限、網路問題等）。所以這裡採用跟 scripts/fetch_and_index_news.py
相同的「可續跑」設計：每處理完一筆就立即把目前累積的結果覆寫存檔，且
啟動時會先讀取輸出檔裡已經有的 record_id 直接跳過——重複執行同一條
指令，就能接續處理剩下的事件片段，不用擔心中途中斷會遺失進度或重工。

使用方式（在 project/ 目錄下執行）：
    python scripts/run_pipeline_vlm_prompts.py --events ../13_8-9/visual_tracks_events_13.json
    python scripts/run_pipeline_vlm_prompts.py --events ... --frames-dir ../13_8-9/visual_tracks_dataset_13
    python scripts/run_pipeline_vlm_prompts.py --events ... --limit 6   # 本次最多處理 6 個「新」事件片段
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import ValidationError  # noqa: E402

from app.dependencies import (  # noqa: E402
    get_alert_service,
    get_ingestion_service,
    get_news_retriever,
    get_reasoning_pipeline,
    get_text_retriever,
    get_visual_embedding_client,
    get_visual_retriever,
    get_vlm_client,
)
from services.frame_service import FrameLoaderService  # noqa: E402
from services.orchestrator_service import OrchestratorService  # noqa: E402
from services.vlm_prompt_adapter import build_record, load_vlm_prompts, segment_prompts  # noqa: E402
from utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


def load_existing_outputs(output_path: Path) -> list:
    """讀取既有輸出檔（若存在），供接續執行時跳過已處理過的 record_id。"""
    if not output_path.exists():
        return []
    try:
        return json.loads(output_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("既有輸出檔 %s 無法解析，視為沒有先前進度", output_path)
        return []


def save_outputs(outputs: list, output_path: Path) -> None:
    """把目前累積的結果整批覆寫進輸出檔，讓大批次執行中途被中斷也不會遺失進度。"""
    output_path.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")


def build_orchestrator(frames_dir: Path) -> OrchestratorService:
    """
    比照 app.dependencies.get_orchestrator_service() 的組裝方式手動組一份，
    只有 frame_service 換成指向這批資料專屬的 frames_dir——
    get_orchestrator_service() 刻意設計成無參數（給 FastAPI Depends() 用），
    沒辦法直接拿來覆寫 frames_dir，其餘元件仍然重用同一批 get_xxx() 工廠。
    """
    return OrchestratorService(
        ingestion_service=get_ingestion_service(),
        frame_service=FrameLoaderService(frames_dir=str(frames_dir)),
        vlm_client=get_vlm_client(),
        visual_embedding_provider=get_visual_embedding_client(),
        visual_retriever=get_visual_retriever(),
        text_retriever=get_text_retriever(),
        news_retriever=get_news_retriever(),
        reasoning_pipeline=get_reasoning_pipeline(),
        alert_service=get_alert_service(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="跑一批影像端逐幀 VLM 提示格式的事件，過完整推論端 pipeline")
    parser.add_argument("--events", required=True, help="vlm_visual_prompts 格式的事件 JSON 路徑")
    parser.add_argument(
        "--frames-dir",
        default=None,
        help="畫面檔案目錄（底下要有 BEFORE/DURING/AFTER 子資料夾）。"
        "未指定時，預設推導成事件 JSON 同層的 visual_tracks_dataset_<video>/",
    )
    parser.add_argument("--max-frame-gap", type=int, default=15, help="聚合時允許的最大幀數間隔（預設 15）")
    parser.add_argument(
        "--limit", type=int, default=None, help="本次最多處理幾個「尚未處理過」的事件片段，不含已跳過的部分"
    )
    args = parser.parse_args()

    events_path = Path(args.events)
    metadata, prompts = load_vlm_prompts(events_path)
    video = metadata.get("video", "unknown")

    frames_dir = Path(args.frames_dir) if args.frames_dir else events_path.parent / f"visual_tracks_dataset_{video}"
    print(f"影片：{video}，畫面目錄：{frames_dir}")

    segments = segment_prompts(prompts, max_frame_gap=args.max_frame_gap)
    print(f"共 {len(prompts)} 篇原始偵測 -> 聚合成 {len(segments)} 個事件片段")

    all_records = [build_record(seg, video, i) for i, seg in enumerate(segments)]

    output_path = Path(f"pipeline_output_{video}.json")
    outputs = load_existing_outputs(output_path)
    processed_ids = {item["record_id"] for item in outputs}
    pending_records = [r for r in all_records if r["record_id"] not in processed_ids]
    if processed_ids:
        print(f"{len(processed_ids)} 個事件片段先前已處理過，本次跳過。")

    if args.limit is not None:
        records = pending_records[: args.limit]
        remaining = len(pending_records) - len(records)
        if remaining > 0:
            print(f"本次限制最多處理 {args.limit} 個新事件片段，剩餘 {remaining} 個留待下次執行同一條指令時接續。")
    else:
        records = pending_records

    if not records:
        print("沒有需要新處理的事件片段。")
        return

    orchestrator = build_orchestrator(frames_dir)

    failed = 0
    for i, record in enumerate(records, start=1):
        print(f"\n[{i}/{len(records)}] 處理事件：record_id={record['record_id']}, frame_file={record['frame_file']}")
        try:
            output = orchestrator.process_event(record)
        except ValidationError as exc:
            logger.warning("事件 JSON 驗證失敗，跳過 record_id=%s：%s", record["record_id"], exc)
            failed += 1
            continue
        except FileNotFoundError as exc:
            logger.warning("找不到畫面檔案，跳過 record_id=%s：%s", record["record_id"], exc)
            failed += 1
            continue
        except Exception as exc:  # noqa: BLE001 - 單筆失敗不該中斷整批，錯誤原因記錄下來即可
            logger.warning("處理失敗，跳過 record_id=%s：%s", record["record_id"], exc)
            failed += 1
            continue

        print(f"  -> event_type={output.event_type}, alert_level={output.alert_level.value}")
        outputs.append({"record_id": record["record_id"], **output.model_dump()})
        save_outputs(outputs, output_path)

    print(f"\n本次處理 {len(records)} 個事件片段 -> 成功 {len(records) - failed} 筆，失敗 {failed} 筆")
    print(f"累計已寫入 {output_path}（共 {len(outputs)} 筆）")

    alert_levels: dict = {}
    for item in outputs:
        alert_levels[item["alert_level"]] = alert_levels.get(item["alert_level"], 0) + 1
    print(f"目前累計警報等級分布：{alert_levels}")


if __name__ == "__main__":
    main()
