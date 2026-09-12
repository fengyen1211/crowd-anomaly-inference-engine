"""
vlm/internvl.py
==================
BaseVLM 的 InternVL 實作（骨架，之後才實作，不下載任何模型）。

InternVL 是上海 AI Lab 開源的視覺語言模型系列，以強大的視覺編碼器
（InternViT）著稱，多個 benchmark 表現優異，是候選 VLM 之一。

TODO：
- [ ] 加入相依套件並更新 requirements.txt（transformers + timm 等）
- [ ] 實作 load_model()：依官方文件用 AutoModel.from_pretrained() 載入
- [ ] 實作 describe_image()：組成對話格式 + 圖片輸入，取得文字輸出
- [ ] 實作 generate_embedding()：InternViT 是獨立的視覺編碼器，
      可以考慮直接拿它的輸出當作視覺 embedding
- [ ] 確認模型版本（InternVL2 / InternVL2.5 等）與硬體資源需求
"""

from typing import List, Optional

from vlm.base import BaseVLM
from utils.logger import get_logger

logger = get_logger(__name__)


class InternVLProvider(BaseVLM):
    """使用 InternVL 模型（骨架，尚未實作）。"""

    def __init__(self, model_name: str = "OpenGVLab/InternVL2-8B", device: Optional[str] = None) -> None:
        self._model_name = model_name
        self._device = device
        self._model = None

    def load_model(self) -> None:
        logger.info("InternVLProvider.load_model() 尚未實作：%s", self._model_name)
        raise NotImplementedError("TODO: InternVL 尚未實作，需先安裝相依套件並下載權重")

    def describe_image(self, image: bytes, prompt: Optional[str] = None) -> str:
        raise NotImplementedError("TODO: InternVL 尚未實作")

    def generate_embedding(self, image: bytes) -> List[float]:
        raise NotImplementedError("TODO: InternVL 尚未實作")
