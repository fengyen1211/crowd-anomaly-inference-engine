"""
rag/loaders/markdown_loader.py
=================================
Markdown 檔（.md）載入器。

目前先原樣讀取檔案內容當作純文字（保留 markdown 語法），
因為 LLM 通常可以直接理解 markdown 格式，不強制剝除。

TODO：
- [ ] 視需求依標題（# / ##）切分成多個 LoadedDocument，
      而不是整份檔案當作一份文件（目前交給後續的 Chunker 處理切塊）
"""

from pathlib import Path
from typing import List

from rag.loaders.base import BaseDocumentLoader
from rag.loaders.schemas import LoadedDocument
from utils.logger import get_logger

logger = get_logger(__name__)


class MarkdownLoader(BaseDocumentLoader):
    """讀取 .md 檔案，整份檔案視為一份文件。"""

    def load(self, file_path: Path) -> List[LoadedDocument]:
        content = file_path.read_text(encoding="utf-8")
        logger.info("MarkdownLoader 讀取完成：%s（%d 字元）", file_path, len(content))
        return [
            LoadedDocument(
                content=content,
                metadata={
                    "source": str(file_path),
                    "file_type": "markdown",
                },
            )
        ]
