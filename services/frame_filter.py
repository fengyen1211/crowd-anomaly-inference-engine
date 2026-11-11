"""
services/frame_filter.py
===========================
在 VLM 之前擋掉「不是監視器畫面」的事件（插圖、卡通、片尾卡），這類事件
不進入判斷流程（VLM、驗證、RAG + LLM 都不跑），直接回 info。

為什麼需要：UCF 的 Fighting018 片尾接了一張空手道插圖，影像端把它偵測成
fight（Fighting018_fight_p2）。VLM 正確描述了「踢向對方」，所以每一組
sweep（3b／7b／32b、crop／zoom、新舊驗證 prompt）都判 support。描述本身
沒有錯，只看文字擋不掉，要在看圖這一步就判斷「這不是監視器畫面」。

做法：用 CLIP 零樣本分類整張畫面（不是裁切圖），比較兩句描述：
「真實場景的照片」vs「圖畫、卡通或插圖」。直接沿用視覺相似度檢索已經
載入的 CLIP（get_visual_embedding_client()），不多載模型。

門檻依據（本機 frames/ 中 UCF 99 筆與 video 7 的 184 張畫面）：空手道插圖
0.81，其餘最高 0.17（Fighting003 的昏暗畫面），video 7 的模擬 CG 畫面
都在 0.1 以下。只有一張插圖樣本，門檻取中間的 0.5；之後遇到新的非監視器
畫面要再檢查。

只在 visual_embedding 是 CLIP 時啟用（見 app/dependencies.py）：mock 的
向量是隨機的，SigLIP 的相似度尺度不同，套同一個門檻沒有意義。
"""

import math
from typing import List, Optional

from embedding.base import BaseEmbeddingProvider
from utils.logger import get_logger

logger = get_logger(__name__)

# 第一句是「正常」，其餘是要擋掉的類型。試過「監視器畫面 vs 黑白線稿」：
# 插圖只有 0.25、其他是 0.00，分得開但門檻太難定；換成下面這組差距最大。
_LABELS = ("a photo of a real scene", "a drawing, cartoon, or illustration")
# CLIP 訓練時學到的 logit_scale（clip-vit-base-patch32 為 100），
# 讓 cosine 相似度的 softmax 跟模型本身的零樣本機率一致
_LOGIT_SCALE = 100.0


class NonCameraFrameFilter:
    """判斷整張畫面是不是插圖／卡通，而不是真實場景。"""

    def __init__(self, embedding_provider: BaseEmbeddingProvider, threshold: float = 0.5) -> None:
        self._provider = embedding_provider
        self._threshold = threshold
        self._label_vectors: Optional[List[List[float]]] = None

    def illustration_score(self, full_frame: bytes) -> float:
        """畫面是插圖的機率（0～1）。"""
        if self._label_vectors is None:
            self._label_vectors = self._provider.embed_text(list(_LABELS))
        image_vector = self._provider.embed_image([full_frame])[0]
        logits = [_LOGIT_SCALE * sum(a * b for a, b in zip(image_vector, v)) for v in self._label_vectors]
        peak = max(logits)
        exps = [math.exp(x - peak) for x in logits]
        return 1 - exps[0] / sum(exps)

    def check(self, full_frame: bytes, record_id: str) -> Optional[str]:
        """是插圖就回傳略過的原因，否則回傳 None。"""
        score = self.illustration_score(full_frame)
        if score < self._threshold:
            return None
        logger.info("NonCameraFrameFilter record_id=%s 略過：插圖機率 %.2f", record_id, score)
        return f"畫面不是監視器畫面（插圖機率 {score:.2f}），未進入判斷流程"
