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
  不需要 RAG 檢索結果，是獨立的任務，所以跟 _reason() 同時送出）

這個類別只依賴 BaseLLMClient 這個抽象介面（依賴注入），
不知道也不在乎背後是 MockLLMClient 還是 OpenAILLMClient。
"""

import re
from concurrent.futures import ThreadPoolExecutor

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

# 每條原因／建議的字數與條數上限，寫進 JSON schema 由 Ollama 的文法強制。
#
# 為什麼需要：sweep_752396 有 3 筆 gemma3:4b 輸出停不下來（5～8K 字）。兩筆
# 是用全形「」」收尾字串（…協助判斷事件發展。」]}），一筆寫出 C:\b\b\b…，
# JSON 字串都沒有關閉。文法只允許字串內容，模型之後寫的「系統已完成任務。
# 謝謝。」等全部變成字串內容，一路寫到 context 上限，每筆浪費約 20 秒再重試。
# 有 maxLength 時，文法寫滿上限就強制關閉字串，JSON 一定合法（本機
# Ollama 0.40.16 + gemma3:4b 實測：maxLength 20 的字串剛好 20 字後關閉）。
#
# 200 字：過去所有 sweep 報告的 2759 條原因／建議最長 179 字、99% 在 106 字內，
# prompt 要求約 50 字。3 條：prompt 要求最多 3 條，過去只有 1 筆列了 4 條。
_MAX_ITEM_CHARS = 200
_MAX_ITEMS = 3
_LIST_SCHEMA = {
    "type": "array",
    "maxItems": _MAX_ITEMS,
    "items": {"type": "string", "maxLength": _MAX_ITEM_CHARS},
}

# _reason() 要求的輸出格式，欄位對應 prompts/reasoning_prompt.py 的說明
_REASONING_SCHEMA = {
    "type": "object",
    "properties": {
        "final_verdict": {"type": "string", "enum": [v.value for v in FinalVerdict]},
        "final_confidence": {"type": "number"},
        "possible_causes": _LIST_SCHEMA,
        "recommended_actions": _LIST_SCHEMA,
    },
    "required": ["final_verdict", "final_confidence", "possible_causes", "recommended_actions"],
}

# 文法已經限制了長度，num_predict 只是保險（不支援 schema 的 client 或文法
# 失效時）：6 條 × 200 字加欄位名稱，約 1,500 token 以內
_REASONING_NUM_PREDICT = 2048

# 模型以為已經結束 JSON 的位置：引號（含全形）後面接 ] 或 }
_LEAKED_CLOSE_RE = re.compile(r'[」"”]\s*[\]}]')


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

        # 摘要只需要事件資料與 VLM 觀察，不依賴推論結果，兩次 LLM 呼叫同時送出，
        # 每筆事件省下摘要那一次的等待（H200 上約 1～3 秒）。Ollama 要設
        # OLLAMA_NUM_PARALLEL >= 2 才會真的平行處理，否則只是在服務端排隊，
        # 結果相同、只是不會變快。任一邊拋出的例外都會照原樣往上傳。
        with ThreadPoolExecutor(max_workers=1) as pool:
            summary_future = pool.submit(self._summarize, event_record, vlm_observation)
            verdict, confidence, causes, actions = self._reason(event_record, vlm_observation, retrieved_context)
            summary = summary_future.result()

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
        # 用 JSON schema 限制輸出（支援的 client 才會生效，例如 Ollama）：Nano4 sweep 中
        # gemma3:4b 約 10% 回應因字串內非法跳脫（Invalid \escape）等語法錯誤而無法
        # 解析，整筆事件報廢。解析或欄位不符時再重新請求一次，作為不支援 schema 的
        # client 的保險。
        try:
            return self._request_reasoning(messages)
        except (ValueError, KeyError, TypeError) as exc:
            logger.warning("record_id=%s 的推論回應格式不符（%s），重新請求一次", event_record.record_id, exc)
            return self._request_reasoning(messages)

    def _request_reasoning(self, messages):
        data = parse_json_response(
            self._llm_client.chat_completion(
                messages, json_schema=_REASONING_SCHEMA, num_predict=_REASONING_NUM_PREDICT
            )
        )
        return (
            FinalVerdict(data["final_verdict"]),
            float(data["final_confidence"]),
            _clean_items(data["possible_causes"]),
            _clean_items(data["recommended_actions"]),
        )

    def _summarize(self, event_record: RawEventRecord, vlm_observation: VLMObservation) -> str:
        """呼叫 LLM 產生 summary。"""
        messages = SummaryPromptBuilder.build(event_record, vlm_observation)
        return self._llm_client.chat_completion(messages).strip()


def _clean_items(items) -> list:
    """
    清掉字串被文法強制關閉時留下的尾巴（見 _MAX_ITEM_CHARS 的說明）：

    - 模型以為已經結束的位置（「」]}」這類引號接括號）之後全部切掉
    - 控制字元（例如 \\b 解碼出來的退格）刪掉
    - 清完是空字串的條目不保留；條數超過上限只取前幾條
    """
    cleaned = []
    for item in list(items)[:_MAX_ITEMS]:
        text = str(item)
        if match := _LEAKED_CLOSE_RE.search(text):
            text = text[:match.start()]
        text = "".join(ch for ch in text if ch.isprintable() or ch in "\n\t").strip()
        if text:
            cleaned.append(text)
    return cleaned
