"""
vlm/observation_builder.py
=============================
把 BaseVLM 的原始能力（describe_image）轉換成業務層需要的 VLMObservation。

這是 vlm/base.py docstring 提到的「業務服務層」：BaseVLM 只知道怎麼
描述一張圖片，不知道「這個描述支不支持規則系統的判斷」，這件事需要跟
predicted_event_type 做比對，屬於業務邏輯，所以獨立成這個模組，
不要塞進 BaseVLM 的 Provider 實作裡（Provider 只管模型能力，
不管業務判斷，才能保持 Provider 之間可以互相替換）。

目前實作：★ 暫時的簡化版本 ★
還沒有真正的交叉驗證邏輯（那其實是 LLM 的工作，見
prompts/reasoning_prompt.py 的 final_verdict）。這裡的
verification_verdict 先固定回傳 UNCERTAIN、visual_confidence 固定
回傳 0.5，誠實反映「目前 VLM 這一步只有描述能力，還沒有驗證能力」，
而不是編造一個看起來很篤定、實際上沒有依據的假結果。

TODO：
- [ ] 之後可以讓 VLM 的 prompt 本身就要求輸出結構化的驗證結果
      （例如直接問模型「這個描述支持 xxx 事件類型嗎？」），
      到時候這裡就能填入真正算出來的 verdict/confidence，
      而不是寫死的預設值。
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
    ) -> VLMObservation:
        """
        Args:
            vlm_client: 已注入好的 VLM 實作（依賴抽象介面 BaseVLM）。
            image: 事件對應的畫面（通常是裁切後的區域）。
            record: 影像端輸出的原始事件資料，提供 predicted_event_type
                    作為描述時的參考問題。
            context_images: 額外的脈絡畫面（依序為事件發生前／後的同機位
                    畫面），只有 panic_scatter／counter_flow 這類需要跨時間
                    比對的行為類型才會有值（見 RawEventRecord.before_frame_file
                    ／after_frame_file）；None 時沿用單張畫面的原有 prompt。

        Returns:
            VLMObservation
        """
        if context_images:
            prompt = (
                f"你會依序看到 {1 + len(context_images)} 張同一位置的畫面："
                f"第一張是事件發生當下，接下來依序是事件發生前、事件發生後的畫面。"
                f"系統判斷這是「{record.predicted_event_type}」，這類行為需要"
                f"比對人群隨時間的變化（例如是否明顯聚集、四散、或反方向移動）"
                f"才能判斷。請客觀比較這幾張畫面，描述群體行為與動態上的差異，"
                f"只描述你實際看到的內容，不需要判斷是否同意這個標籤。"
            )
        else:
            prompt = (
                f"請客觀描述這張畫面中群體的行為與動態。"
                f"系統判斷這可能是「{record.predicted_event_type}」，"
                f"但請只描述你實際看到的內容，不需要判斷是否同意這個標籤。"
            )
        description = vlm_client.describe_image(image, prompt=prompt, context_images=context_images)

        return VLMObservation(
            record_id=record.record_id,
            visual_description=description,
            # TODO: 見檔案頂端說明，目前先誠實回報「尚無法判斷」，
            # 不假裝有能力做交叉驗證。
            verification_verdict=VerificationVerdict.UNCERTAIN,
            visual_confidence=0.5,
            extra_context=[],
        )
