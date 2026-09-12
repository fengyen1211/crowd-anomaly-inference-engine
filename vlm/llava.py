"""
vlm/llava.py
===============
BaseVLM 的 LLaVA 實作（骨架，之後才實作，不下載任何模型）。

LLaVA 是最早、也是社群生態最成熟的開源 VLM 之一，
transformers 官方原生支援，部署與文件資源相對豐富，適合作為
「先求有、再求好」階段的備援選項。

TODO：
- [ ] 加入相依套件並更新 requirements.txt
- [ ] 實作 load_model()：用 transformers 的
      LlavaForConditionalGeneration.from_pretrained() 載入權重
- [ ] 實作 describe_image()：組成 chat template + 圖片輸入，取得文字輸出
- [ ] 實作 generate_embedding()：確認要用 vision tower 的哪一層輸出
      當作視覺特徵向量
- [ ] 確認模型版本（llava-1.5 / llava-next 等）與硬體資源需求
"""

from typing import List, Optional

from vlm.base import BaseVLM
from utils.logger import get_logger

logger = get_logger(__name__)


class LLaVAProvider(BaseVLM):
    """使用 LLaVA 模型（骨架，尚未實作）。"""

    def __init__(self, model_name: str = "llava-hf/llava-1.5-7b-hf", device: Optional[str] = None) -> None:
        self._model_name = model_name
        self._device = device
        self._model = None

    def load_model(self) -> None:
        logger.info("LLaVAProvider.load_model() 尚未實作：%s", self._model_name)
        raise NotImplementedError("TODO: LLaVA 尚未實作，需先安裝相依套件並下載權重")

    def describe_image(self, image: bytes, prompt: Optional[str] = None) -> str:
        raise NotImplementedError("TODO: LLaVA 尚未實作")

    def generate_embedding(self, image: bytes) -> List[float]:
        raise NotImplementedError("TODO: LLaVA 尚未實作")
