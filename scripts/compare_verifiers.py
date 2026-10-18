"""
scripts/compare_verifiers.py
===============================
固定同一份 VLM 描述，只換驗證用的 LLM（llm/visual_verifier.py），比較各模型
的判斷與耗時。

為什麼不直接跑完整 sweep：驗證只吃描述文字，描述固定後不必重跑 VLM、RAG、
完整推論，每個模型幾十秒就能跑完；而且所有模型看的是同一份描述，差異只來自
驗證模型本身（完整 sweep 裡 3b／7b 的描述不同，判斷好壞混在一起分不開）。

分三步，每一步獨立一個 process（config 與 client 都是 lru_cache 單例，
同一個 process 裡換 VLM_MODEL／LLM_MODEL 不會生效）：

    # 1. 用 VLM 產生描述（VLM_MODEL 覆寫 config.yaml）
    VLM_MODEL=qwen2.5vl:7b python scripts/compare_verifiers.py describe \
        --input data/tinyformer7_events.json --output reports/verifiers/descriptions.json

    # 2. 每個驗證模型各跑一次（LLM_MODEL 覆寫 config.yaml）
    LLM_MODEL=gemma3:12b python scripts/compare_verifiers.py verify \
        --descriptions reports/verifiers/descriptions.json --output reports/verifiers/gemma3_12b.json

    # 3. 彙整成一張表
    python scripts/compare_verifiers.py summarize reports/verifiers --reference gemma3:27b

Nano4 上用 scripts/nano4/run_verifier_compare.sbatch 一次跑完三步。
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.dependencies import get_frame_service, get_ingestion_service, get_llm_client, get_vlm_client  # noqa: E402
from config.yaml_config import load_config  # noqa: E402
from llm.visual_verifier import VisualVerifier  # noqa: E402
from vlm.observation_builder import VLMObservationBuilder  # noqa: E402
from vlm.schemas import VerificationVerdict, VLMObservation  # noqa: E402


def describe(input_path: Path, output_path: Path) -> None:
    """跟 OrchestratorService.observe() 前半段相同：讀畫面、裁切、VLM 中立描述。"""
    records = json.loads(input_path.read_text(encoding="utf-8"))["records"]
    ingestion = get_ingestion_service()
    frames = get_frame_service()
    vlm = get_vlm_client()

    items = []
    for raw in records:
        record = ingestion.ingest_single(raw)
        cropped = frames.crop_bbox(frames.load_full_frame(record), record.spatial_bbox)
        context = [
            frames.crop_bbox(frames.load_context_frame(f, record.record_id), record.spatial_bbox)
            for f in (record.before_frame_file, record.after_frame_file)
            if f
        ] or None
        start = time.time()
        observation = VLMObservationBuilder.build(vlm, cropped, record, context_images=context)
        print(f"{record.record_id}（{time.time() - start:.1f}s）：{observation.visual_description}")
        items.append({
            "record": raw,
            "event_type": record.predicted_event_type,
            "description": observation.visual_description,
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"vlm_model": load_config().vlm.model, "items": items}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"描述已寫入：{output_path}")


def verify(descriptions_path: Path, output_path: Path) -> None:
    data = json.loads(descriptions_path.read_text(encoding="utf-8"))
    ingestion = get_ingestion_service()
    verifier = VisualVerifier(get_llm_client())
    model = load_config().llm.model

    def observe(item):
        record = ingestion.ingest_single(item["record"])
        observation = VLMObservation(
            record_id=record.record_id,
            visual_description=item["description"],
            verification_verdict=VerificationVerdict.UNCERTAIN,
            visual_confidence=0.5,
        )
        return record, observation

    # 第一次呼叫包含載入模型的時間，先暖機一次不計時，耗時才是穩態的判斷速度
    verifier.verify(*observe(data["items"][0]))

    results = []
    for item in data["items"]:
        record, observation = observe(item)
        start = time.time()
        verified = verifier.verify(record, observation)
        elapsed = time.time() - start
        # VisualVerifier 失敗時回傳原物件（reason 為空），在這裡記成失敗
        failed = verified.verification_reason == ""
        print(
            f"{record.record_id:<36} {verified.verification_verdict.value:<10} "
            f"{verified.visual_confidence:.2f} {elapsed:5.1f}s {'FAILED' if failed else verified.verification_reason}"
        )
        results.append({
            "record_id": record.record_id,
            "event_type": item["event_type"],
            "verdict": verified.verification_verdict.value,
            "confidence": verified.visual_confidence,
            "reason": verified.verification_reason,
            "failed": failed,
            "sec": round(elapsed, 2),
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {"llm_model": model, "vlm_model": data["vlm_model"], "results": results},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    print(f"結果已寫入：{output_path}")


def summarize(report_dir: Path, reference: str) -> None:
    reports = []
    for path in sorted(report_dir.glob("*.json")):
        if path.name == "descriptions.json":
            continue
        reports.append(json.loads(path.read_text(encoding="utf-8")))
    if not reports:
        print(f"{report_dir} 裡沒有驗證結果")
        return

    by_model = {r["llm_model"]: {x["record_id"]: x for x in r["results"]} for r in reports}
    models = list(by_model)
    record_ids = [x["record_id"] for x in reports[0]["results"]]
    ref = by_model.get(reference)

    print(f"描述來源 VLM：{reports[0]['vlm_model']}")
    print(f"\n{'模型':<26}{'中位耗時':>8}{'失敗':>6}  support/contradict/uncertain  與 {reference} 一致")
    for model in models:
        rows = by_model[model]
        verdicts = [rows[r]["verdict"] for r in record_ids if r in rows]
        agree = f"{sum(ref.get(r, {}).get('verdict') == rows[r]['verdict'] for r in rows)}/{len(rows)}" if ref else "-"
        print(
            f"{model:<26}{statistics.median(x['sec'] for x in rows.values()):>7.2f}s"
            f"{sum(x['failed'] for x in rows.values()):>6}  "
            f"{verdicts.count('support'):>7}/{verdicts.count('contradict')}/{verdicts.count('uncertain'):<18}{agree}"
        )

    short = {"support": "S", "contradict": "C", "uncertain": "U"}
    print("\n逐筆判斷（S=support C=contradict U=uncertain，數字為信心）")
    print(f"{'record_id':<36}" + "".join(f"{m[:14]:>16}" for m in models))
    for rid in record_ids:
        cells = []
        for model in models:
            x = by_model[model].get(rid)
            cells.append(f"{short[x['verdict']]} {x['confidence']:.2f}" if x else "-")
        print(f"{rid:<36}" + "".join(f"{c:>16}" for c in cells))


def main() -> None:
    parser = argparse.ArgumentParser(description="固定 VLM 描述，比較不同驗證模型")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("describe", help="用 VLM（VLM_MODEL）產生描述")
    p.add_argument("--input", required=True, help="事件 JSON 路徑")
    p.add_argument("--output", required=True, help="描述輸出路徑")

    p = sub.add_parser("verify", help="用 LLM（LLM_MODEL）驗證描述")
    p.add_argument("--descriptions", required=True, help="describe 產生的描述檔")
    p.add_argument("--output", required=True, help="結果輸出路徑")

    p = sub.add_parser("summarize", help="彙整同一目錄下各模型的結果")
    p.add_argument("report_dir", help="verify 輸出所在的目錄")
    p.add_argument("--reference", default="gemma3:27b", help="作為比較基準的模型")

    args = parser.parse_args()
    if args.command == "describe":
        describe(Path(args.input), Path(args.output))
    elif args.command == "verify":
        verify(Path(args.descriptions), Path(args.output))
    else:
        summarize(Path(args.report_dir), args.reference)


if __name__ == "__main__":
    main()
