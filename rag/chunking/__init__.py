"""
rag/chunking package
=======================
負責把 LoadedDocument 切成較小的 Chunk，才適合送進 embedding 模型
與向量資料庫（原始文件通常太長，直接整篇 embedding 會讓檢索精準度下降）。

- schemas.py             Chunk 資料結構
- base.py                BaseChunker 抽象介面
- recursive_chunker.py   RecursiveChunker：目前預設的切塊策略
"""
