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

from typing import List, Optional

from utils.schemas import RawEventRecord
from vlm.base import BaseVLM
from vlm.schemas import VerificationVerdict, VLMObservation


class VLMObservationBuilder:
    """把 BaseVLM.describe_image() 的原始輸出組成 VLMObservation。"""

    @classmethod
    def build(
        cls,
        vlm_client: BaseVLM,
        image: bytes,
        record: RawEventRecord,
        context_images: Optional[List[bytes]] = None,
        box_drawn: bool = False,
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
            box_drawn: 畫面是整張畫面加紅框（Settings.vlm_view="full"）時為
                    True，prompt 會說明紅框的意思；裁切圖時為 False。

        Returns:
            VLMObservation
        """
        box_note = (
            "畫面中的紅框標出影像分析系統關注的位置。請描述整個畫面的人群狀況，"
            "並特別說明紅框內及附近的人在做什麼。"
            if box_drawn
            else ""
        )
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

        return VLMObservation(
            record_id=record.record_id,
            visual_description=description,
            # TODO: 見檔案頂端說明，目前先誠實回報「尚無法判斷」，
            # 不假裝有能力做交叉驗證。
            verification_verdict=VerificationVerdict.UNCERTAIN,
            visual_confidence=0.5,
            extra_context=[],
        )


def _truncate_repetition(text: str, min_run: int = 3) -> str:
    """
    VLM 陷入重複迴圈時，從開始重複的地方截斷。

    num_predict 上限只保證會停下來，不保證內容可用：verifiers_565780 裡
    qwen2.5vl:7b 對 counter_flow 把「穿白色上衣的人在階梯上向右側移動」等
    幾行重複列了幾十次直到上限，下游驗證模型全都因為「有人往左、有人往右」
    判成 support。

    判斷方式：某一行跟前面某一行相同，而且接下來連續 min_run 行都跟前面
    那段一一相同，就視為進入迴圈。單獨一行重複（例如前後兩段都寫「環境
    比較平靜」）不算，不會被截掉。
    """
    lines = text.split("\n")
    seen = {}
    for i, line in enumerate(lines):
        key = line.strip()
        if not key:
            continue
        for j in seen.get(key, []):
            run = 0
            while (
                i + run < len(lines)
                and j + run < i
                and lines[i + run].strip()
                and lines[i + run].strip() == lines[j + run].strip()
            ):
                run += 1
            if run >= min_run:
                return "\n".join(lines[:i]).rstrip()
        seen.setdefault(key, []).append(i)
    return text
