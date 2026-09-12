"""
scripts/index_knowledge_base.py
==================================
把 data/knowledge_base_sources/ 底下的真實 SOP／法規文件（消防法、
各類場所消防安全設備設置標準、災害防救法、大型群聚活動安全管理要點）
切塊、算 embedding、寫進知識庫向量庫（chroma_collection_knowledge），
並做一次示範查詢，證明「有資料時 LLM 真的查得到、引用得到」。

使用方式（在 project/ 目錄下執行）：
    python scripts/index_knowledge_base.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.dependencies import get_knowledge_indexer, get_text_retriever  # noqa: E402

_SOURCE_DIR = Path(__file__).resolve().parent.parent / "data" / "knowledge_base_sources"

_DEMO_QUERIES = [
    "大型活動的緊急疏散通道與應變計畫應該包含哪些項目？",
    "公共場所發現有發生火災爆炸之虞時，消防機關可以怎麼處置？",
]


def main() -> None:
    document_paths = sorted(_SOURCE_DIR.glob("*.txt")) + sorted(_SOURCE_DIR.glob("*.pdf"))
    if not document_paths:
        print(f"在 {_SOURCE_DIR} 底下找不到任何文件，請先確認來源檔案存在。")
        return

    print(f"共找到 {len(document_paths)} 份來源文件：")
    for path in document_paths:
        print(f"  - {path.name}")

    indexer = get_knowledge_indexer()
    start = time.time()
    chunk_count = indexer.build_index_from_documents(document_paths)
    elapsed = time.time() - start
    print(f"\n索引完成，共寫入 {chunk_count} 個 chunk（耗時 {elapsed:.1f} 秒）。")

    retriever = get_text_retriever()
    for query in _DEMO_QUERIES:
        print(f"\n=== 示範查詢：{query} ===")
        snippets = retriever.retrieve(query, top_k=3)
        if not snippets:
            print("（查無相關片段）")
            continue
        for rank, snippet in enumerate(snippets, start=1):
            print(f"{rank}. （來源：{snippet.source_document}，相關度 {snippet.relevance_score:.2f}）")
            print(f"   {snippet.content[:150]}...")


if __name__ == "__main__":
    main()
