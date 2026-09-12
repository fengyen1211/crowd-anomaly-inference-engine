"""
rag/loaders package
======================
負責把不同格式的原始文件（txt / markdown / json / pdf）
統一載入成 LoadedDocument 格式，供後續 chunking 使用。

- schemas.py          LoadedDocument 資料結構
- base.py             BaseDocumentLoader 抽象介面
- text_loader.py      .txt
- markdown_loader.py  .md / .markdown
- json_loader.py      .json（支援展開 records 陣列成多份文件）
- pdf_loader.py       .pdf（骨架，之後實作）
- factory.py          get_loader()：依副檔名分派對應載入器
"""
