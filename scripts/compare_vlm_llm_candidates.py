"""
scripts/compare_vlm_llm_candidates.py
========================================
拿專題裡真實的資料，實際跑過 4 個候選模型（2 個 VLM + 2 個 LLM），
比較輸出品質與延遲，作為選型依據。

前提：
- Ollama 服務已啟動，且已用 `ollama pull` 下載以下 4 個模型：
  qwen2.5vl:3b、minicpm-v（VLM 候選）、qwen3:4b、gemma3:4b（LLM 候選）
- 這裡直接透過 VLMFactory / LLMFactory 建立 provider="ollama" 的實例，
  跟接進正式 pipeline 時用的是同一套 Provider 程式碼
  （vlm/ollama_vlm.py、llm/ollama_llm.py），確保比較結果跟正式接上後
  的行為一致，選完之後只要改 config.yaml 就能直接上線，不需要重寫。

VLM 比較：用 dataset/dataset_export_7 的真實幀（BEFORE 一張、DURING
    一張），讓兩個候選各自描述畫面內容，人工比較誰看得比較準、比較快。

LLM 比較：用 rag_vlm_mock_dataset.json 裡一筆真實事件紀錄，直接呼叫
    ReasoningPipeline（跟正式 pipeline 完全相同的 prompt 組裝與 JSON
    解析邏輯），檢查兩個候選誰能穩定輸出合法 JSON、內容有沒有照規則
    誠實回報「證據不足」（這裡刻意用空的 RetrievedContext，對應知識庫
    /視覺歷史案例都還沒正式建置的真實現況）。

使用方式（在 project/ 目錄下執行）：
    python scripts/compare_vlm_llm_candidates.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm.factory import LLMFactory  # noqa: E402
from llm.reasoning_pipeline import ReasoningPipeline  # noqa: E402
from rag.schemas import RetrievedContext  # noqa: E402
from utils.schemas import RawEventRecord  # noqa: E402
from vlm.factory import VLMFactory  # noqa: E402

VLM_CANDIDATES = ["qwen2.5vl:3b", "minicpm-v"]
LLM_CANDIDATES = ["qwen3:4b", "gemma3:4b"]

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DATASET_DIR = _PROJECT_ROOT / "dataset" / "dataset_export_7"
_MOCK_DATASET_PATH = _PROJECT_ROOT / "rag_vlm_mock_dataset.json"

_VLM_PROMPT = "請客觀描述這張畫面中人群的行為與動態，包含是否有任何異常跡象（例如逆向移動、恐慌性移動、異常聚集）。"


def compare_vlm() -> None:
    test_images = [
        ("BEFORE", _DATASET_DIR / "BEFORE" / "frame_00000.jpg"),
        ("DURING", _DATASET_DIR / "DURING" / "frame_00670.jpg"),
    ]

    print("=" * 70)
    print("VLM 候選比較")
    print("=" * 70)

    for model_name in VLM_CANDIDATES:
        print(f"\n--- {model_name} ---")
        vlm = VLMFactory.create("ollama", model=model_name)
        vlm.load_model()

        for phase, path in test_images:
            image_bytes = path.read_bytes()
            start = time.time()
            try:
                description = vlm.describe_image(image_bytes, prompt=_VLM_PROMPT)
                elapsed = time.time() - start
                print(f"[{phase}] {path.name}（耗時 {elapsed:.1f} 秒）：\n{description}\n")
            except Exception as exc:  # noqa: BLE001 - 比較腳本，記錄失敗原因即可，不中斷其他候選
                elapsed = time.time() - start
                print(f"[{phase}] {path.name}（耗時 {elapsed:.1f} 秒）失敗：{exc}\n")


def _load_sample_record() -> RawEventRecord:
    raw = json.loads(_MOCK_DATASET_PATH.read_text(encoding="utf-8"))
    return RawEventRecord(**raw["records"][0])


def compare_llm() -> None:
    record = _load_sample_record()
    empty_context = RetrievedContext()  # 對應知識庫/視覺歷史案例都還沒正式建置的真實現況

    print("=" * 70)
    print(f"LLM 候選比較（事件：{record.record_id}，規則系統標籤：{record.predicted_event_type}）")
    print("=" * 70)

    for model_name in LLM_CANDIDATES:
        print(f"\n--- {model_name} ---")
        llm_client = LLMFactory.create("ollama", model=model_name)
        pipeline = ReasoningPipeline(llm_client)

        start = time.time()
        try:
            result = pipeline.run(record, vlm_observation=None, retrieved_context=empty_context)
            elapsed = time.time() - start
            print(f"耗時 {elapsed:.1f} 秒，成功解析為合法 JSON：")
            print(result.model_dump_json(indent=2, ensure_ascii=False))
        except Exception as exc:  # noqa: BLE001 - 比較腳本，記錄失敗原因即可，不中斷其他候選
            elapsed = time.time() - start
            print(f"耗時 {elapsed:.1f} 秒，失敗：{exc}")


def main() -> None:
    compare_vlm()
    print()
    compare_llm()


if __name__ == "__main__":
    main()
