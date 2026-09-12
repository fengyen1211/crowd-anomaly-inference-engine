"""
vlm/minicpm_v.py
===================
BaseVLM 的 MiniCPM-V 實作（骨架，之後才實作，不下載任何模型）。

MiniCPM-V 是面向端側部署設計的輕量級視覺語言模型，
相對前面幾個候選模型參數量小很多，適合硬體資源有限時的備援選項
（例如只有消費級顯卡，甚至只能用 CPU 推論）。

TODO：
- [ ] 加入相依套件並更新 requirements.txt
- [ ] 實作 load_model()：依官方文件用 AutoModel.from_pretrained() 載入
- [ ] 實作 describe_image()：組成對話格式 + 圖片輸入，取得文字輸出
- [ ] 實作 generate_embedding()：確認視覺編碼器輸出的取得方式
- [ ] 確認模型版本（MiniCPM-V 2.6 等）與硬體資源需求
      （這是這幾個候選裡對硬體資源要求最低的，資源不足時優先考慮）
"""

from typing import List, Optional

from vlm.base import BaseVLM
from utils.logger import get_logger

logger = get_logger(__name__)


class MiniCPMVProvider(BaseVLM):
    """使用 MiniCPM-V 模型（骨架，尚未實作）。"""

    def __init__(self, model_name: str = "openbmb/MiniCPM-V-2_6", device: Optional[str] = None) -> None:
        self._model_name = model_name
        self._device = device
        self._model = None

    def load_model(self) -> None:
        logger.info("MiniCPMVProvider.load_model() 尚未實作：%s", self._model_name)
        raise NotImplementedError("TODO: MiniCPM-V 尚未實作，需先安裝相依套件並下載權重")

    def describe_image(self, image: bytes, prompt: Optional[str] = None) -> str:
        raise NotImplementedError("TODO: MiniCPM-V 尚未實作")

    def generate_embedding(self, image: bytes) -> List[float]:
        raise NotImplementedError("TODO: MiniCPM-V 尚未實作")
