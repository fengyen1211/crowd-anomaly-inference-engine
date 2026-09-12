"""
vlm/mock.py
=============
BaseVLM 的假資料實作（Mock）。

★★★ 目前唯一可用的 VLM Provider ★★★

用途：
- 開發流程串接測試，不需要下載或載入任何真實模型權重
  （Qwen2.5-VL / InternVL / LLaVA / MiniCPM-V 權重都是好幾 GB 起跳，
  在還沒確定要用哪個模型、硬體資源足夠之前，先用這個把上下游接起來）。
- 呼應 embedding/mock.py 的角色，兩者是對稱設計。

注意：這裡回傳的描述文字與向量都是假的（用圖片內容的雜湊值當種子，
確保同一張圖片每次都得到相同結果，方便測試斷言），不能拿來評估
真實效果，只能用來測試「流程有沒有跑通」。
"""

import hashlib
import random
from typing import List, Optional

from vlm.base import BaseVLM
from utils.logger import get_logger

logger = get_logger(__name__)


class MockVLM(BaseVLM):
    """回傳固定格式的假描述與假向量，不載入任何真實模型。"""

    def __init__(self, dimension: int = 512) -> None:
        self._dimension = dimension
        self._is_loaded = False

    def load_model(self) -> None:
        # 不需要真的載入任何東西，純粹模擬「已完成載入」的狀態，
        # 讓呼叫端可以測試「必須先 load_model() 才能使用」這個標準流程，
        # 跟真實本地模型的使用方式一致。
        logger.info("MockVLM.load_model()：no-op，未載入任何真實模型")
        self._is_loaded = True

    def describe_image(self, image: bytes, prompt: Optional[str] = None) -> str:
        self._ensure_loaded()
        digest = hashlib.sha256(image).hexdigest()[:8]
        description = f"[MOCK 描述] 這是一張測試圖片（雜湊={digest}）"
        if prompt:
            description += f"，提示詞：{prompt}"
        return description

    def generate_embedding(self, image: bytes) -> List[float]:
        self._ensure_loaded()
        digest = hashlib.sha256(image).hexdigest()
        rng = random.Random(digest)
        return [rng.uniform(-1.0, 1.0) for _ in range(self._dimension)]

    def _ensure_loaded(self) -> None:
        """
        強制要求呼叫端先呼叫過 load_model()，即使 Mock 本身不需要真的載入，
        也要求同樣的使用順序，避免之後換成真實模型時才發現忘了呼叫。
        """
        if not self._is_loaded:
            raise RuntimeError("尚未呼叫 load_model()，請先載入模型再使用")
