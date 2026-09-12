"""
llm/reasoning_pipeline.py
============================
Reasoning Module 的核心串接邏輯。

對應架構設計中的 [5] Reasoning Module：
整合「原始事件資料」「VLM 觀察結果」「RAG 檢索結果」，
產生：驗證判定、事件摘要、可能原因、應變建議。

分成兩次 LLM 呼叫（而不是一次，也不是拆成四次）：
- _reason()：一次產生 final_verdict / final_confidence / possible_causes /
  recommended_actions（見 prompts/reasoning_prompt.py，這四件事都需要
  同一份 RAG 檢索結果做 grounding，放在同一個 prompt 裡比較不會互相矛盾）
- _summarize()：產生 summary（見 prompts/summary_prompt.py，
  不需要 RAG 檢索結果，是獨立的任務）

這個類別只依賴 BaseLLMClient 這個抽象介面（依賴注入），
不知道也不在乎背後是 MockLLMClient 還是 OpenAILLMClient。
"""

from llm.base import BaseLLMClient
from llm.schemas import AnalysisResult, FinalVerdict
from prompts.reasoning_prompt import ReasoningPromptBuilder
from prompts.summary_prompt import SummaryPromptBuilder
from rag.schemas import RetrievedContext
from utils.llm_json import parse_json_response
from utils.logger import get_logger
from utils.schemas import RawEventRecord
from vlm.schemas import VLMObservation

logger = get_logger(__name__)


class ReasoningPipeline:
    """
    協調 LLM 呼叫，產生最終的 AnalysisResult。
    """

    def __init__(self, llm_client: BaseLLMClient) -> None:
        """
        Args:
            llm_client: 已注入好的 LLM 客戶端實作（依賴抽象介面 BaseLLMClient）。
        """
        self._llm_client = llm_client

    def run(
        self,
        event_record: RawEventRecord,
        vlm_observation: VLMObservation,
        retrieved_context: RetrievedContext,
    ) -> AnalysisResult:
        """
        執行完整的推理流程：先做原因推論與驗證，再做摘要，組成 AnalysisResult。
        """
        logger.info("ReasoningPipeline.run() 開始處理 record_id=%s", event_record.record_id)

        verdict, confidence, causes, actions = self._reason(event_record, vlm_observation, retrieved_context)
        summary = self._summarize(event_record, vlm_observation)

        return AnalysisResult(
            record_id=event_record.record_id,
            summary=summary,
            final_verdict=verdict,
            final_confidence=confidence,
            possible_causes=causes,
            recommended_actions=actions,
        )

    def _reason(
        self,
        event_record: RawEventRecord,
        vlm_observation: VLMObservation,
        retrieved_context: RetrievedContext,
    ):
        """呼叫 LLM 產生 final_verdict / final_confidence / possible_causes / recommended_actions。"""
        messages = ReasoningPromptBuilder.build(event_record, retrieved_context, vlm_observation)
        raw_response = self._llm_client.chat_completion(messages)
        data = parse_json_response(raw_response)

        return (
            FinalVerdict(data["final_verdict"]),
            float(data["final_confidence"]),
            list(data["possible_causes"]),
            list(data["recommended_actions"]),
        )

    def _summarize(self, event_record: RawEventRecord, vlm_observation: VLMObservation) -> str:
        """呼叫 LLM 產生 summary。"""
        messages = SummaryPromptBuilder.build(event_record, vlm_observation)
        return self._llm_client.chat_completion(messages).strip()
