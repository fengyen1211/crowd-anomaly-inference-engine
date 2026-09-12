"""
rag/loaders/factory.py
=========================
依副檔名分派對應的文件載入器。

新增支援的檔案格式時，只需要：
1. 新增一個繼承 BaseDocumentLoader 的類別
2. 在下面的 _LOADERS 註冊對應副檔名
不需要修改呼叫端（knowledge_indexer.py）的程式碼。
"""

from pathlib import Path
from typing import Dict

from rag.loaders.base import BaseDocumentLoader
from rag.loaders.json_loader import JSONLoader
from rag.loaders.markdown_loader import MarkdownLoader
from rag.loaders.pdf_loader import PDFLoader
from rag.loaders.text_loader import TextLoader

# 副檔名 -> 載入器實例。載入器本身無狀態，共用同一個實例即可。
_LOADERS: Dict[str, BaseDocumentLoader] = {
    ".txt": TextLoader(),
    ".md": MarkdownLoader(),
    ".markdown": MarkdownLoader(),
    ".json": JSONLoader(),
    ".pdf": PDFLoader(),
}


def get_loader(file_path: Path) -> BaseDocumentLoader:
    """
    依檔案副檔名回傳對應的載入器實例。

    Raises:
        ValueError: 找不到對應副檔名的載入器時。
    """
    suffix = file_path.suffix.lower()
    loader = _LOADERS.get(suffix)
    if loader is None:
        raise ValueError(f"不支援的檔案格式：{suffix}（檔案：{file_path}）")
    return loader
