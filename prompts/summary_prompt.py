"""
prompts/summary_prompt.py
============================
定義「事件摘要生成」任務的完整 Prompt。

用途：整合事件資料（+ VLM 觀察，若有），產生一段給人看的
自然語言事件摘要。

這個任務不需要 RAG 檢索結果——摘要只是忠實描述「發生了什麼」，
不涉及原因推論或應變建議，不需要外部知識庫佐證
（跟 reasoning_prompt.py 的職責明確分開）。

輸出格式：純文字摘要（不要求 JSON 結構化輸出），因為摘要本身
就是要給人直接閱讀的文字，不需要程式再解析欄位。
"""

from typing import Dict, List, Optional

from prompts.system_prompt import SystemPromptBuilder
from prompts.user_prompt import UserPromptBuilder
from utils.schemas import RawEventRecord
from vlm.schemas import VLMObservation

_TASK_INSTRUCTION = """
【目前任務：事件摘要】
請根據以上事件資料，用 2-4 句話寫出一段事件摘要，內容須包含：
- 發生了什麼類型的行為、涉及幾人
- 關鍵的移動特徵（如果跟事件類型的判斷有明顯關聯）
- 若有 VLM 觀察結果且與規則判斷不一致，需在摘要中提及這個落差

只需要輸出摘要文字本身，不要加上「摘要：」這類前綴，也不要輸出其他說明。
"""


class SummaryPromptBuilder:
    """組裝「事件摘要生成」任務的完整 Prompt（system + user + 任務指示）。"""

    @classmethod
    def build(
        cls,
        record: RawEventRecord,
        vlm_observation: Optional[VLMObservation] = None,
    ) -> List[Dict[str, str]]:
        """
        Args:
            record: 影像端輸出的原始事件資料。
            vlm_observation: VLM 對這一幀畫面的觀察結果，若尚未取得可不傳。

        Returns:
            List[Dict[str, str]]: 可直接傳給 BaseLLMClient.chat_completion()
            的訊息列表（[{"role": "system"/"user", "content": ...}]）。
        """
        user_content = UserPromptBuilder.build(record, vlm_observation) + "\n" + _TASK_INSTRUCTION

        return [
            {"role": "system", "content": SystemPromptBuilder.build()},
            {"role": "user", "content": user_content},
        ]
