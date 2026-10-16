"""
llm/visual_verifier.py
=========================
依 VLM 的中立畫面描述，判斷畫面是否支持規則系統的事件類型，填入
VLMObservation 的 verification_verdict／visual_confidence，作為初步警報的
依據（見 services/alert_service.py 的 compose_preliminary()）。

這一步不做 RAG、只看描述文字，所以很快（H200 上約 0.5 秒），初步警報
不必等完整的推論流程。為什麼不讓 VLM 看圖時直接判斷，見
prompts/verification_prompt.py。

失敗（LLM 錯誤、輸出無法解析）時退回 uncertain／0.5，也就是這一步加入前
的預設值：初步警報只是提早通知，不能因為它讓整筆事件失敗。
"""

from llm.base import BaseLLMClient
from prompts.verification_prompt import VerificationPromptBuilder
from utils.llm_json import parse_json_response
from utils.logger import get_logger
from utils.schemas import RawEventRecord
from vlm.schemas import VerificationVerdict, VLMObservation

logger = get_logger(__name__)

_VERIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "verification_verdict": {"type": "string", "enum": [v.value for v in VerificationVerdict]},
        "visual_confidence": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["verification_verdict", "visual_confidence", "reason"],
}


class VisualVerifier:
    """用 LLM 判斷 VLM 描述是否支持規則系統的判斷。"""

    def __init__(self, llm_client: BaseLLMClient) -> None:
        self._llm_client = llm_client

    def verify(self, record: RawEventRecord, observation: VLMObservation) -> VLMObservation:
        """回傳填入驗證結論的新 VLMObservation（不修改傳入的物件）。"""
        messages = VerificationPromptBuilder.build(record.predicted_event_type, observation.visual_description)
        try:
            data = parse_json_response(
                self._llm_client.chat_completion(messages, json_schema=_VERIFICATION_SCHEMA)
            )
            verdict = VerificationVerdict(data["verification_verdict"])
            # schema 文法不保證數值範圍，超出 0～1 就夾回來
            confidence = min(max(float(data["visual_confidence"]), 0.0), 1.0)
            reason = str(data["reason"])
        except Exception as exc:  # noqa: BLE001 - 初步判斷失敗不能讓整筆事件失敗
            logger.warning("record_id=%s 的視覺驗證失敗（%s），退回 uncertain", record.record_id, exc)
            return observation

        logger.info(
            "VisualVerifier record_id=%s verdict=%s confidence=%.2f reason=%s",
            record.record_id, verdict.value, confidence, reason,
        )
        return observation.model_copy(
            update={
                "verification_verdict": verdict,
                "visual_confidence": confidence,
                "verification_reason": reason,
            }
        )
