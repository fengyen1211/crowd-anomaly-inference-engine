"""
prompts/reasoning_prompt.py
==============================
定義「原因推論與驗證」任務的完整 Prompt。

用途：綜合事件資料、VLM 觀察、RAG 檢索到的相似案例與知識庫片段，
一次產生：
- final_verdict：對規則系統標籤的信任程度判斷（confirmed / uncertain / overturned）
- final_confidence：最終信心分數（取代影像端寫死的假 confidence）
- possible_causes：可能原因（必須有 RAG 依據，不能憑空生成）
- recommended_actions：應變建議（必須有知識庫 SOP 依據）

為什麼這四件事放在同一個 Prompt，而不是拆成三個？
- 這四件事都需要「同一份檢索資料」做 grounding，判斷 verdict 的證據
  跟推論 causes 的證據往往是同一批相似案例，拆開反而要重複貼一次
  檢索結果、也可能讓 LLM 在不同任務給出互相矛盾的結論。
- 這是整個 pipeline 裡唯一需要 RAG 檢索結果的任務——
  summary（摘要）只需要事件資料，alert（警報）只是把這裡的產出轉換
  成通知文字，真正「需要講道理、有依據」的分析工作集中在這裡。

輸出格式：要求 LLM 輸出 JSON，欄位對應 llm.schemas.AnalysisResult
的部分欄位，方便程式解析（實際解析與呼叫邏輯在
llm/reasoning_pipeline.py，這裡只負責組 prompt）。
"""

from typing import Dict, List, Optional

from prompts.rag_prompt import RAGPromptBuilder
from prompts.system_prompt import SystemPromptBuilder
from prompts.user_prompt import UserPromptBuilder
from rag.schemas import RetrievedContext
from utils.schemas import RawEventRecord
from vlm.schemas import VLMObservation

_TASK_INSTRUCTION = """
【目前任務：原因推論與驗證】
請根據以上事件資料與檢索到的參考資料，判斷以下四件事，並以 JSON 格式輸出：

{
  "final_verdict": "confirmed | uncertain | overturned",
  "final_confidence": 0.0 到 1.0 之間的浮點數,
  "possible_causes": ["原因1", "原因2", "..."],
  "recommended_actions": ["建議1", "建議2", "..."]
}

規則：
- final_verdict：畫面與檢索資料是否支持規則系統的判斷？
  支持 -> confirmed；證據不足 -> uncertain；證據矛盾 -> overturned。
- possible_causes：每一條原因都必須能對應到「相似歷史事件」「畫面描述」或
  「新聞真實案例」，如果查無相關資料，請回傳 ["證據不足，無法判斷可能原因"]，
  不要臆測。
- recommended_actions：每一條建議都必須能對應到「知識庫參考片段」或
  「新聞真實案例」，如果查無相關規範，請回傳 ["證據不足，無法提供具體應變建議"]，
  不要臆測。
- possible_causes、recommended_actions 每一條都控制在一句話以內（約 50 字），
  簡潔扼要，最多各列 3 條，不要展開成長篇說明。
- 只輸出 JSON 本身，不要加上任何其他文字說明，也不要用 Markdown 程式碼區塊包住。
"""


class ReasoningPromptBuilder:
    """組裝「原因推論與驗證」任務的完整 Prompt（system + user + rag + 任務指示）。"""

    @classmethod
    def build(
        cls,
        record: RawEventRecord,
        retrieved_context: RetrievedContext,
        vlm_observation: Optional[VLMObservation] = None,
    ) -> List[Dict[str, str]]:
        """
        Args:
            record: 影像端輸出的原始事件資料。
            retrieved_context: RAG 模組檢索到的相似案例與知識庫片段。
            vlm_observation: VLM 對這一幀畫面的觀察結果，若尚未取得可不傳。

        Returns:
            List[Dict[str, str]]: 可直接傳給 BaseLLMClient.chat_completion()
            的訊息列表。
        """
        user_content = (
            UserPromptBuilder.build(record, vlm_observation)
            + "\n\n"
            + RAGPromptBuilder.build(retrieved_context)
            + "\n"
            + _TASK_INSTRUCTION
        )

        return [
            {"role": "system", "content": SystemPromptBuilder.build()},
            {"role": "user", "content": user_content},
        ]
