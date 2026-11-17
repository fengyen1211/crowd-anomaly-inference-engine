"""
vlm/observation_builder.py
=============================
把 BaseVLM 的原始能力（describe_image）轉換成業務層需要的 VLMObservation。

這是 vlm/base.py docstring 提到的「業務服務層」：BaseVLM 只知道怎麼
描述一張圖片，不知道「這個描述支不支持規則系統的判斷」，這件事需要跟
predicted_event_type 做比對，屬於業務邏輯，所以獨立成這個模組，
不要塞進 BaseVLM 的 Provider 實作裡（Provider 只管模型能力，
不管業務判斷，才能保持 Provider 之間可以互相替換）。

這裡只做中立描述：verification_verdict 固定回傳 UNCERTAIN、
visual_confidence 固定回傳 0.5，真正的驗證結論由下一步
llm/visual_verifier.py 依描述文字填入。

曾經試過讓 VLM 看圖時直接判斷（prompt 要求輸出 support／contradict），
但 qwen2.5vl:3b 幾乎一律回答 support、信心 0.90，描述也被系統給的行為
說明帶偏，所以 prompt 刻意維持「只描述、不判斷」（見
prompts/verification_prompt.py 的說明）。

prompt 也不告訴 VLM 系統判斷的事件類型：sweep_565835 裡 qwen2.5vl:3b 在
描述中直接寫出「這類行為可以被歸類為 counter_flow」，驗證模型便據此判成
support。描述必須獨立於規則系統，驗證才有意義。
"""

import re
from typing import List, Optional

from utils.schemas import RawEventRecord
from vlm.base import BaseVLM
from utils.logger import get_logger
from vlm.schemas import VerificationVerdict, VLMObservation

logger = get_logger(__name__)

# zoom 的說明只請它描述人在做什麼、彼此怎麼互動，不提任何事件類型，也不
# 列動作清單：verifiers_688422 裡列了「站、走、跑、蹲、倒地、伸手」，
# qwen2.5vl:3b 就照清單每項寫一條（沒打架的畫面也寫出「倒地」「身體接觸」）。
_VIEW_NOTES = {
    "full": (
        "畫面中的紅框標出影像分析系統關注的位置。請描述整個畫面的人群狀況，"
        "並特別說明紅框內及附近的人在做什麼。"
    ),
    "zoom": (
        "這是監視器畫面中某個區域的放大圖，畫質可能比較模糊。"
        "請描述圖中的人各自在做什麼，以及他們彼此之間的互動。"
    ),
}

# VLM 拒絕描述的開頭用語。verifiers_688422 裡 qwen2.5vl:3b 看到放大後清楚
# 的扭打畫面，直接回答「很抱歉，我無法提供您所要求的內容」。
_REFUSAL_MARKERS = ("無法提供", "无法提供", "無法協助", "无法协助", "I can't", "I cannot", "I'm sorry")


class VLMObservationBuilder:
    """把 BaseVLM.describe_image() 的原始輸出組成 VLMObservation。"""

    @classmethod
    def build(
        cls,
        vlm_client: BaseVLM,
        image: bytes,
        record: RawEventRecord,
        context_images: Optional[List[bytes]] = None,
        view: str = "crop",
        fallback_image: Optional[bytes] = None,
    ) -> VLMObservation:
        """
        Args:
            vlm_client: 已注入好的 VLM 實作（依賴抽象介面 BaseVLM）。
            image: 事件對應的畫面（通常是裁切後的區域）。
            record: 影像端輸出的原始事件資料（只用 record_id，事件類型
                    刻意不放進 prompt，見檔案頂端說明）。
            context_images: 額外的脈絡畫面（依序為事件發生前／後的同機位
                    畫面），只有 panic_scatter／counter_flow／sprint_spike 這類需要跨時間
                    比對的行為類型才會有值（見 RawEventRecord.before_frame_file
                    ／after_frame_file）；None 時沿用單張畫面的原有 prompt。
            view: 畫面的種類（Settings.vlm_view）。full 是整張畫面加紅框，
                    prompt 會說明紅框的意思；zoom 是放大的局部，prompt 會請
                    它描述人在做什麼與彼此的互動；crop 不加說明。
            fallback_image: VLM 拒絕描述時改看的畫面（zoom 時傳一般裁切
                    圖），用不加說明的 prompt 重新描述一次；None 時保留拒答。

        Returns:
            VLMObservation
        """
        box_note = _VIEW_NOTES.get(view, "")
        if context_images:
            prompt = box_note + (
                f"你會依序看到 {1 + len(context_images)} 張同一位置的畫面："
                f"第一張是事件發生當下，接下來依序是事件發生前、事件發生後的畫面。"
                f"請客觀比較這幾張畫面中人群隨時間的變化，包括人群的分布與疏密、"
                f"人的動作、移動的方向與速度，只描述你實際看到的內容。"
            )
        else:
            prompt = box_note + (
                "請客觀描述這張畫面中群體的行為與動態，包括人群的分布與疏密、"
                "人的動作、移動的方向與速度，只描述你實際看到的內容。"
            )
        description = _truncate_repetition(
            vlm_client.describe_image(image, prompt=prompt, context_images=context_images)
        )
        if fallback_image is not None and _is_refusal(description):
            logger.warning("VLM 拒絕描述 record_id=%s，改用備用畫面重新描述：%s", record.record_id, description)
            plain_prompt = prompt[len(box_note):]
            description = _truncate_repetition(vlm_client.describe_image(fallback_image, prompt=plain_prompt))

        # 描述寫進 log（換行改成「 / 」保持一行），之後可以不重跑 VLM、只換驗證
        # prompt 或模型重算
        logger.info("VLMDescription record_id=%s description=%s", record.record_id, description.replace("\n", " / "))

        return VLMObservation(
            record_id=record.record_id,
            visual_description=description,
            # TODO: 見檔案頂端說明，目前先誠實回報「尚無法判斷」，
            # 不假裝有能力做交叉驗證。
            verification_verdict=VerificationVerdict.UNCERTAIN,
            visual_confidence=0.5,
            extra_context=[],
        )


def _is_refusal(text: str) -> bool:
    """描述很短、而且開頭就是拒絕用語，才視為拒答（長描述裡偶爾出現「無法」不算）。"""
    head = text.strip()[:40]
    return len(text.strip()) < 120 and any(m in head for m in _REFUSAL_MARKERS)


# 句子單位：以句號／驚嘆號／問號或換行結尾；num_predict 截斷時最後半句沒有句號，也算一個單位
_UNIT_RE = re.compile(r"[^。！？\n]*[。！？]|[^。！？\n]+")
_MAX_BLOCK_UNITS = 8


def _truncate_repetition(text: str) -> str:
    """
    VLM 陷入重複迴圈時，從開始重複的地方截斷，保留第一份。

    num_predict 上限只保證會停下來，不保證內容可用：verifiers_565780 裡
    qwen2.5vl:7b 對 counter_flow 把「穿白色上衣的人在階梯上向右側移動」等
    幾行重複列了幾十次直到上限，下游驗證模型全都因為「有人往左、有人往右」
    判成 support。

    以「句子」為單位比對（句號、驚嘆號、問號或換行切開，空行忽略），找
    連續重複的區塊：
    - 1～2 句的區塊連續出現 3 次以上
    - 3 句以上（最多 8 句）的區塊連續出現 2 次以上

    舊版以「行」為單位、要求連續 3 行跟前面一一相同，sweep_752396／752397
    的 118 筆描述裡有兩種迴圈抓不到：
    - 同一行內重複：ucf_Assault005_crowd_locked 把同一句寫了 46 次，全在一行
    - 段落之間隔空行：tinyformer7_crowd_locked_p1 的劇院段落重複 16 次，
      空行讓「連續相同」的計數中斷

    同一句在不同段落各出現一次（例如事件前／後兩段都寫「櫃檯前的地板上
    鋪有彩色的地毯」）中間隔著別的句子，不是連續重複，不會被截掉。
    """
    units = [(m.start(), m.group().strip()) for m in _UNIT_RE.finditer(text)]
    units = [u for u in units if u[1]]
    keys = [k for _, k in units]
    n = len(keys)
    for i in range(n):
        for size in range(1, _MAX_BLOCK_UNITS + 1):
            repeats = 3 if size < 3 else 2
            if i + size * repeats > n:
                break
            block = keys[i:i + size]
            if all(keys[i + r * size:i + (r + 1) * size] == block for r in range(1, repeats)):
                return text[:units[i + size][0]].rstrip()
    return text
