"""
vlm/ollama_vlm.py
====================
BaseVLM 的 Ollama 實作。

跟 llm/ollama_llm.py 是同樣的角色：透過本地 Ollama 服務呼叫開源 VLM
（Qwen2.5-VL / MiniCPM-V 等），量化、模型載入都交給 Ollama 處理，
一個 Provider 類別就能涵蓋所有「Ollama 有支援的視覺模型」，
差別只在 config.yaml 的 model 名稱，不需要為 qwen25_vl.py /
minicpm_v.py / internvl.py / llava.py 這幾個候選各自實作一份
transformers 版本的載入邏輯。
"""

import base64
from typing import List, Optional

import requests

from config.settings import Settings, get_settings
from utils.logger import get_logger
from vlm.base import BaseVLM

logger = get_logger(__name__)


class OllamaVLM(BaseVLM):
    """透過本地 Ollama 服務呼叫開源 VLM（Qwen2.5-VL / MiniCPM-V 等）。"""

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        settings: Optional[Settings] = None,
        timeout: float = 300.0,
    ) -> None:
        """
        Args:
            model: Ollama 已下載的模型標籤（例如 "qwen2.5vl:3b"）。
            base_url: Ollama 服務位址，預設讀取 Settings.ollama_base_url。
            timeout: 單次請求逾時秒數（本地 VLM 在消費級硬體上生成
                     需要一段時間，預設放寬到 300 秒）。
        """
        settings = settings or get_settings()
        self._model = model
        self._base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self._timeout = timeout
        self._is_loaded = False

    def load_model(self) -> None:
        # Ollama 本身常駐管理模型的載入/卸載（依需求在收到請求時把權重讀進
        # 記憶體），呼叫端不需要自己管理；這裡只確認服務有回應、標記成已就緒，
        # 跟其他 Provider（Mock/CLIP）維持一致的「先 load 再用」使用方式。
        response = requests.get(f"{self._base_url}/api/tags", timeout=10)
        response.raise_for_status()
        self._is_loaded = True
        logger.info("OllamaVLM 就緒：model=%s（實際權重由 Ollama 服務於首次請求時載入）", self._model)

    def describe_image(self, image: bytes, prompt: Optional[str] = None) -> str:
        if not self._is_loaded:
            raise RuntimeError("尚未呼叫 load_model()，請先載入模型再使用")

        image_b64 = base64.b64encode(image).decode("ascii")
        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt or "請描述這張圖片的內容。",
                    "images": [image_b64],
                }
            ],
            "stream": False,
        }
        logger.info("OllamaVLM 呼叫 model=%s", self._model)
        response = requests.post(f"{self._base_url}/api/chat", json=payload, timeout=self._timeout)
        response.raise_for_status()
        return response.json()["message"]["content"]

    def generate_embedding(self, image: bytes) -> List[float]:
        # 本專案的視覺相似度檢索已經由 embedding/clip.py（CLIP）負責，
        # 這裡的 VLM Provider 只需要「看圖產生描述」的能力，不重複提供
        # 視覺 embedding，避免同一件事有兩套不一致的實作。
        raise NotImplementedError(
            "OllamaVLM 不提供視覺 embedding，視覺相似度檢索請改用 "
            "embedding/clip.py（CLIPEmbeddingProvider.embed_image()）"
        )
