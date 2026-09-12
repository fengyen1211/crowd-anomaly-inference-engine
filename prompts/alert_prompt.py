"""
prompts/alert_prompt.py
==========================
定義「警報內容生成」任務的完整 Prompt。

用途：把 Reasoning 任務的分析結果（llm.schemas.AnalysisResult），
轉換成給監控人員看的警報文字（標題、簡短訊息、完整報告）。

★ 職責劃分（重要）★
- 警報「等級」（info / warning / critical）是 services/alert_service.py
  的 AlertComposerService 用純規則邏輯決定，不是這裡的 LLM 決定
  ——安全性考量：警報要不要升級為 critical，不應該受 LLM 生成內容
  不穩定性的影響，規則判斷比較可預測、可稽核。
- 這裡的 LLM 只負責把「已經分析好的結果」轉換成「人類監控人員
  一眼就能看懂該做什麼」的通知文字，是「表達」而不是「分析」的任務，
  所以不需要重新做 RAG 檢索，直接使用 Reasoning 任務已經產出的結果即可。

輸出格式：要求 JSON，欄位對應 title / short_message / full_report。
"""

from typing import Dict, List, Optional

from llm.schemas import AnalysisResult
from prompts.system_prompt import SystemPromptBuilder
from prompts.user_prompt import UserPromptBuilder
from utils.schemas import RawEventRecord
from vlm.schemas import VLMObservation

_TASK_INSTRUCTION_TEMPLATE = """
【目前任務：警報內容生成】
分析結果如下：
- 最終判定：{final_verdict}（信心分數：{final_confidence:.2f}）
- 事件摘要：{summary}
- 可能原因：{possible_causes}
- 應變建議：{recommended_actions}

請根據以上分析結果，以 JSON 格式輸出警報內容：

{{
  "title": "簡短標題，15 字以內，讓人一眼看出事件類型與嚴重性",
  "short_message": "1-2 句話的簡短通知，適合推播通知顯示",
  "full_report": "完整報告，包含事件摘要、可能原因、應變建議，適合監控人員仔細閱讀"
}}

規則：
- 語氣要專業、急迫但不誇大，避免恐慌性用詞。
- full_report 裡的可能原因與應變建議，必須忠實反映分析結果，不要額外添加分析結果沒有的內容。
- 只輸出 JSON 本身，不要加上任何其他文字說明，也不要用 Markdown 程式碼區塊包住。
"""


class AlertPromptBuilder:
    """組裝「警報內容生成」任務的完整 Prompt（system + user + 分析結果 + 任務指示）。"""

    @classmethod
    def build(
        cls,
        record: RawEventRecord,
        analysis_result: AnalysisResult,
        vlm_observation: Optional[VLMObservation] = None,
    ) -> List[Dict[str, str]]:
        """
        Args:
            record: 影像端輸出的原始事件資料。
            analysis_result: Reasoning 任務（reasoning_prompt.py）已經產出的分析結果。
            vlm_observation: VLM 對這一幀畫面的觀察結果，若尚未取得可不傳。

        Returns:
            List[Dict[str, str]]: 可直接傳給 BaseLLMClient.chat_completion()
            的訊息列表。
        """
        task_instruction = _TASK_INSTRUCTION_TEMPLATE.format(
            final_verdict=analysis_result.final_verdict.value,
            final_confidence=analysis_result.final_confidence,
            summary=analysis_result.summary,
            possible_causes="、".join(analysis_result.possible_causes),
            recommended_actions="、".join(analysis_result.recommended_actions),
        )

        user_content = UserPromptBuilder.build(record, vlm_observation) + "\n\n" + task_instruction

        return [
            {"role": "system", "content": SystemPromptBuilder.build()},
            {"role": "user", "content": user_content},
        ]
