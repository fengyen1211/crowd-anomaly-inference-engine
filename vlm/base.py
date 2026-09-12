"""
vlm/base.py
=============
定義所有 VLM Provider 的共同抽象介面。

★★★ 這是「VLM 模型可自由替換」需求的核心設計 ★★★

跟 embedding/base.py 的 BaseEmbeddingProvider 是完全對稱的設計：
上層模組只依賴這個抽象介面，不會直接依賴任何特定的 VLM 模型或套件
（transformers、vllm...）。之後要換成 Qwen2.5-VL / InternVL / LLaVA /
MiniCPM-V，只需要新增一個子類別並在 vlm/factory.py 註冊，
不需要修改呼叫端程式碼。

三個方法對應 VLM 的三種基本能力：
- load_model()：把模型權重載入到記憶體。
  刻意跟 __init__ 分開，讓呼叫端可以控制「什麼時候才真的載入這個
  很重的模型」（例如應用程式啟動時才載入一次，而不是每次建立
  Provider 物件就載入，本地 VLM 權重通常好幾 GB，載入也需要時間）。
- describe_image()：輸入圖片，輸出模型生成的文字描述。
- generate_embedding()：輸入圖片，輸出這張圖片的視覺特徵向量
  （之後可以取代影像端 JSON 裡的假 embedding，或用來做視覺相似度檢索）。

注意：這裡刻意不處理「跟規則系統標籤做交叉比對」這類業務邏輯
（對應 vlm/schemas.py 的 VLMObservation / verification_verdict）。
那是更上層、之後才會建立的「VLM 業務服務層」該做的事——
BaseVLM 只負責「模型本身能做什麼」這個最基本的能力層，
維持單一職責，才能讓 Provider 之間互相替換不受業務邏輯牽動。
"""

from abc import ABC, abstractmethod
from typing import List, Optional


class BaseVLM(ABC):
    """所有 VLM Provider 都必須繼承這個抽象類別。"""

    @abstractmethod
    def load_model(self) -> None:
        """
        載入模型權重到記憶體。

        本地模型（Qwen2.5-VL / InternVL / LLaVA / MiniCPM-V）應該在這裡
        真的執行模型載入；如果之後有純 API 型 Provider（呼叫遠端服務），
        這裡可以是 no-op。
        """
        raise NotImplementedError

    @abstractmethod
    def describe_image(self, image: bytes, prompt: Optional[str] = None) -> str:
        """
        輸入圖片，回傳模型生成的文字描述。

        Args:
            image: 圖片的二進位資料。
            prompt: 可選的引導提示詞（例如「請描述這群人在做什麼」），
                    不同模型對 prompt 的支援程度可能不同。
        """
        raise NotImplementedError

    @abstractmethod
    def generate_embedding(self, image: bytes) -> List[float]:
        """
        輸入圖片，回傳這張圖片的視覺特徵向量。

        用途：之後可以把這組向量存進向量資料庫做視覺相似度檢索，
        取代目前影像端 JSON 裡「格式對、內容是隨機數字」的假 embedding。
        """
        raise NotImplementedError
