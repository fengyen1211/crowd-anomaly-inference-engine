"""
vlm/qwen25_vl.py
===================
BaseVLM 的 Qwen2.5-VL 實作（骨架，之後才實作，不下載任何模型）。

Qwen2.5-VL 是阿里巴巴開源的視覺語言模型，支援圖片理解、細粒度物件定位、
影片理解等能力，是本專案未來優先考慮接上的 VLM（適合群眾異常行為的
畫面理解任務）。

TODO：
- [ ] 加入相依套件並更新 requirements.txt
      （transformers 需要夠新的版本才支援 Qwen2.5-VL，
      可能還需要官方的 qwen-vl-utils）
- [ ] 實作 load_model()：用 transformers 的
      Qwen2_5_VLForConditionalGeneration.from_pretrained() 載入權重
- [ ] 實作 describe_image()：組成 chat template + 圖片輸入，
      呼叫 processor / generate() 取得文字輸出
- [ ] 實作 generate_embedding()：確認 Qwen2.5-VL 是否有可用的圖像
      embedding 輸出（例如 vision encoder 的 pooled hidden state），
      或改用其他方式取得視覺特徵向量
- [ ] 確認模型權重大小與所需硬體資源（7B 模型至少需要一定顯存/記憶體）
"""

from typing import List, Optional

from vlm.base import BaseVLM
from utils.logger import get_logger

logger = get_logger(__name__)


class Qwen25VLProvider(BaseVLM):
    """使用 Qwen2.5-VL 模型（骨架，尚未實作）。"""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct", device: Optional[str] = None) -> None:
        self._model_name = model_name
        self._device = device
        self._model = None

    def load_model(self) -> None:
        logger.info("Qwen25VLProvider.load_model() 尚未實作：%s", self._model_name)
        raise NotImplementedError("TODO: Qwen2.5-VL 尚未實作，需先安裝相依套件並下載權重")

    def describe_image(self, image: bytes, prompt: Optional[str] = None) -> str:
        raise NotImplementedError("TODO: Qwen2.5-VL 尚未實作")

    def generate_embedding(self, image: bytes) -> List[float]:
        raise NotImplementedError("TODO: Qwen2.5-VL 尚未實作")
