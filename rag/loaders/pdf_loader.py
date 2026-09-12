"""
rag/loaders/pdf_loader.py
============================
PDF 檔案載入器。

用 pypdf 逐頁擷取文字，每一頁回傳一份 LoadedDocument，metadata 記錄
頁碼（page_number），方便之後引用來源時標明頁數（例如「大型群聚活動
安全管理要點 第 3 頁」）。
"""

from pathlib import Path
from typing import List

from rag.loaders.base import BaseDocumentLoader
from rag.loaders.schemas import LoadedDocument
from utils.logger import get_logger

logger = get_logger(__name__)


class PDFLoader(BaseDocumentLoader):
    """讀取 .pdf 檔案，每一頁視為一份獨立文件。"""

    def load(self, file_path: Path) -> List[LoadedDocument]:
        # 延遲載入：pypdf 不是所有 Provider 都需要的相依套件，
        # 只有真的要讀 PDF 時才 import（跟 embedding/clip.py 等的做法一致）。
        from pypdf import PdfReader

        reader = PdfReader(str(file_path))
        documents: List[LoadedDocument] = []

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if not text.strip():
                continue
            documents.append(
                LoadedDocument(
                    content=text,
                    metadata={
                        "source": str(file_path),
                        "file_type": "pdf",
                        "page_number": page_number,
                    },
                )
            )

        logger.info("PDFLoader 讀取完成：%s（共 %d 頁，%d 頁有文字內容）", file_path, len(reader.pages), len(documents))
        return documents
