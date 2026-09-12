"""
rag/loaders/text_loader.py
=============================
純文字檔（.txt）載入器。
"""

from pathlib import Path
from typing import List

from rag.loaders.base import BaseDocumentLoader
from rag.loaders.schemas import LoadedDocument
from utils.logger import get_logger

logger = get_logger(__name__)


class TextLoader(BaseDocumentLoader):
    """讀取 .txt 檔案，整份檔案視為一份文件。"""

    def load(self, file_path: Path) -> List[LoadedDocument]:
        content = file_path.read_text(encoding="utf-8")
        logger.info("TextLoader 讀取完成：%s（%d 字元）", file_path, len(content))
        return [
            LoadedDocument(
                content=content,
                metadata={
                    "source": str(file_path),
                    "file_type": "txt",
                },
            )
        ]
