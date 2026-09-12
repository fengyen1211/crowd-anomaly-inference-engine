"""
prompts/user_prompt.py
=========================
定義「事件資料」的 User Prompt 區塊。

這是所有任務共用的「事件描述」文字區塊：把 RawEventRecord（影像端
輸出的原始資料）跟 VLMObservation（VLM 對畫面的觀察，若有）組成一段
LLM 看得懂的事件描述。summary/reasoning/alert 三個任務都會先組出這段
文字，再各自加上任務專屬的問題或輸出格式要求（見對應的 xxx_prompt.py）。

為什麼獨立成一個檔案，而不是每個任務各自組一次？
- 三個任務描述的是同一筆事件，欄位對應邏輯只需要寫一次、維護一次；
  之後事件資料格式若有變動（例如 utils/schemas.py 新增欄位），
  只需要改這裡，三個任務會同步更新，不會有的任務改了、有的忘記改。
"""

from typing import Optional

from utils.schemas import RawEventRecord
from vlm.schemas import VLMObservation


class UserPromptBuilder:
    """組裝「事件資料」文字區塊。"""

    @classmethod
    def build(cls, record: RawEventRecord, vlm_observation: Optional[VLMObservation] = None) -> str:
        """
        Args:
            record: 影像端輸出的原始事件資料。
            vlm_observation: VLM 對這一幀畫面的觀察結果，若尚未取得可不傳。

        Returns:
            str: 給 LLM 看的事件描述文字。
        """
        lines = [
            "以下是系統偵測到的一筆事件資料：",
            f"- 事件編號：{record.record_id}",
            f"- 規則系統判斷的事件類型：{record.predicted_event_type}"
            f"（標籤來源：{record.label_source}，尚未經過驗證）",
            f"- 群體人數：{record.member_count} 人",
        ]

        # 個體層級事件（例如單人速度異常尖峰）沒有群體移動統計可言，
        # collective_motion 會是 None——查無資料就整行省略，不留空欄位
        # 或塞假數字，避免誤導 LLM 以為這是「量測出來但剛好是零」的資料。
        if record.collective_motion is not None:
            motion = record.collective_motion
            lines.append(
                f"- 群體移動特徵：平均方向 {motion.avg_direction_deg:.1f} 度，"
                f"場景主流方向 {motion.majority_flow_direction_deg:.1f} 度，"
                f"偏差 {motion.direction_deviation_deg:.1f} 度，"
                f"密度變化趨勢：{motion.density_trend}"
            )

        lines.append(f"- 系統自動生成的描述：{record.caption}")
        lines.append(f"- 畫面來源：{record.frame_file}（第 {record.frame_idx} 幀）")

        if vlm_observation is not None:
            lines.append("")
            lines.append("視覺模型（VLM）對這一幀畫面的觀察：")
            lines.append(f"- 畫面描述：{vlm_observation.visual_description}")
            lines.append(f"- 對規則判斷的驗證結論：{vlm_observation.verification_verdict.value}")
            lines.append(f"- VLM 自評信心：{vlm_observation.visual_confidence:.2f}")
            if vlm_observation.extra_context:
                lines.append(f"- 額外注意到的線索：{'、'.join(vlm_observation.extra_context)}")
        else:
            lines.append("")
            lines.append("（目前尚未取得 VLM 對這一幀畫面的觀察結果，請只依據上述數字資料判斷）")

        return "\n".join(lines)
