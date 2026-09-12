"""
scripts/index_dataset_frames.py
==================================
把 dataset/dataset_export_7（BEFORE/DURING/AFTER 三個資料夾）裡的真實
影片幀，跑過真正的 CLIP 模型算出視覺 embedding，寫進視覺相似度向量庫
（chroma_collection_visual），並且做一次示範查詢，證明整條「視覺
embedding pipeline」是真的通的（不是假資料、也不是空的向量庫）。

背景：
- embedding/clip.py 原本只是骨架（NotImplementedError），rag_vlm_mock_dataset.json
  裡 42 筆事件的 embedding 欄位也是隨機假向量（見 rag_vlm_mock_dataset_README.md）。
  project/frames/ 底下對應那 42 筆事件的圖片檔案，本身也只是純色占位圖
  （見 scripts/make_placeholder_frames.py），就算補上真的 CLIP 也算不出
  有意義的視覺特徵。
- dataset/dataset_export_7 底下是另一支測試影片（劇院場景）真正渲染出來的
  畫面幀，並且已經用 timeline_segments_7.json 的異常區間切成 BEFORE/
  DURING/AFTER，畫面內容是真的，適合拿來驗證 CLIP embedding 是否真的能
  跑、算出來的向量是否真的有區分度，而不是像之前一樣只能空跑流程。

使用方式（在 project/ 目錄下執行）：
    python scripts/index_dataset_frames.py
    python scripts/index_dataset_frames.py --dataset-dir ../dataset/dataset_export_7 --batch-size 16
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.dependencies import get_visual_embedding_client, get_visual_indexer, get_visual_vector_store  # noqa: E402

PHASES = ("BEFORE", "DURING", "AFTER")
_FRAME_IDX_PATTERN = re.compile(r"(\d+)")


def collect_frames(dataset_dir: Path) -> List[Tuple[Path, str, int]]:
    """回傳 (檔案路徑, phase, frame_idx) 的列表，依 phase 再依 frame_idx 排序。"""
    frames: List[Tuple[Path, str, int]] = []
    for phase in PHASES:
        phase_dir = dataset_dir / phase
        if not phase_dir.is_dir():
            continue
        for path in sorted(phase_dir.glob("*.jpg")):
            match = _FRAME_IDX_PATTERN.search(path.stem)
            frame_idx = int(match.group(1)) if match else -1
            frames.append((path, phase, frame_idx))
    frames.sort(key=lambda item: (item[1], item[2]))
    return frames


def index_frames(dataset_dir: Path, batch_size: int) -> List[Tuple[Path, str, int]]:
    frames = collect_frames(dataset_dir)
    if not frames:
        print(f"在 {dataset_dir} 底下找不到任何 BEFORE/DURING/AFTER 幀，請確認路徑。")
        return frames

    indexer = get_visual_indexer()
    print(f"共找到 {len(frames)} 張幀，開始用 CLIP 計算視覺 embedding（batch_size={batch_size}）...")
    print("（第一次執行會先下載 CLIP 權重，約數百 MB，請耐心等待）")

    for start in range(0, len(frames), batch_size):
        batch = frames[start : start + batch_size]
        image_paths = [path for path, _, _ in batch]
        captions = [
            f"劇院場景示範幀（dataset_export_7，{phase} 階段，frame_idx={frame_idx}）"
            for _, phase, frame_idx in batch
        ]
        metadatas = [
            {
                "record_id": f"dataset7_{phase}_{frame_idx}",
                "caption": caption,
                "predicted_event_type": phase,  # 借用這個欄位標示所屬階段，方便示範查詢時比對
                "frame_file": path.name,
                "frame_idx": frame_idx,
                "source": "dataset_export_7",
            }
            for (path, phase, frame_idx), caption in zip(batch, captions)
        ]
        ids = [metadata["record_id"] for metadata in metadatas]

        indexer.index_images(image_paths, captions, metadatas, ids=ids)
        print(f"  已索引 {min(start + batch_size, len(frames))}/{len(frames)} 張")

    print("索引完成。")
    return frames


def demo_query(frames: List[Tuple[Path, str, int]], top_k: int) -> None:
    """挑一張 DURING 幀當查詢圖片，看看向量資料庫找回來的最相似幀分佈在哪些階段。"""
    during_frames = [item for item in frames if item[1] == "DURING"]
    if not during_frames:
        print("沒有 DURING 幀可以用來示範查詢。")
        return

    query_path, query_phase, query_idx = during_frames[len(during_frames) // 2]
    print(f"\n=== 示範查詢：以 {query_path.name}（{query_phase} 階段）為查詢圖片，找最相似的 {top_k} 張 ===")

    embedding_client = get_visual_embedding_client()
    query_embedding = embedding_client.embed_image([query_path.read_bytes()])[0]

    vector_store = get_visual_vector_store()
    results = vector_store.similarity_search(query_embedding, top_k)

    for rank, result in enumerate(results, start=1):
        metadata = result["metadata"]
        print(
            f"  {rank}. {metadata.get('frame_file')}"
            f"（phase={metadata.get('predicted_event_type')}，"
            f"score={result['score']:.4f}）"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="把 dataset/dataset_export_7 的真實幀索引進視覺向量庫並示範查詢")
    parser.add_argument(
        "--dataset-dir",
        default=str(Path(__file__).resolve().parent.parent.parent / "dataset" / "dataset_export_7"),
        help="dataset_export_7 資料夾路徑（預設：專題根目錄下的 dataset/dataset_export_7）",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="CLIP 一次處理幾張圖片（預設 16）")
    parser.add_argument("--top-k", type=int, default=6, help="示範查詢回傳的相似筆數（預設 6）")
    args = parser.parse_args()

    frames = index_frames(Path(args.dataset_dir), args.batch_size)
    if frames:
        demo_query(frames, args.top_k)


if __name__ == "__main__":
    main()
