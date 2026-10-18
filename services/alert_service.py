"""
services/alert_service.py
============================
對應架構設計中的 [6] Alert Composer。

負責：
- 依 AnalysisResult（final_verdict / final_confidence / event_type）
  套用規則，決定警報等級（info / warning / critical）
- 組成給 Web 介面顯示的警報內容（標題/簡短訊息/完整報告）

★ 這個模組是「純規則邏輯」，不需要任何 LLM/VLM 呼叫 ★
警報等級不應該受 LLM 生成內容不穩定性影響，規則判斷比較可預測、
可稽核；文字內容目前用簡單的字串模板組成（不是 LLM 生成），
如果之後需要更自然的警報文字，可以另外接上
prompts/alert_prompt.py + LLM（那是「表達」而不是「決策」，
應該是可選的加強，不應該取代這裡的規則判斷）。
"""

from llm.schemas import AnalysisResult, FinalVerdict
from utils.logger import get_logger
from utils.schemas import Alert, AlertSeverity, PredictedEventType, RawEventRecord
from vlm.schemas import VerificationVerdict, VLMObservation

logger = get_logger(__name__)

# 「恐慌性移動」在不同影像端格式下的命名不同：舊版規則系統輸出
# PredictedEventType.PANIC_DISPERSAL（"panic_dispersal"），較新的
# TinyFormer+SAM2 逐幀偵測格式輸出 "panic_scatter"——語意相同（人群
# 突然加速、四散），但字串不同，比對時兩者都要算。
_PANIC_EVENT_TYPES = {PredictedEventType.PANIC_DISPERSAL.value, "panic_scatter"}


class AlertComposerService:
    """
    依規則將 AnalysisResult 轉換成 Alert。
    """

    def compose(self, event_record: RawEventRecord, analysis_result: AnalysisResult) -> Alert:
        """
        組成最終的警報內容。
        """
        severity = self._decide_severity(event_record, analysis_result)

        title = f"[{severity.value.upper()}] {event_record.predicted_event_type} 事件"
        short_message = analysis_result.summary
        full_report = "\n".join(
            [
                f"事件摘要：{analysis_result.summary}",
                f"最終判定：{analysis_result.final_verdict.value}"
                f"（信心分數：{analysis_result.final_confidence:.2f}）",
                "可能原因：" + "、".join(analysis_result.possible_causes),
                "應變建議：" + "、".join(analysis_result.recommended_actions),
            ]
        )

        alert = Alert(
            record_id=event_record.record_id,
            severity=severity,
            title=title,
            short_message=short_message,
            full_report=full_report,
            recommended_actions=analysis_result.recommended_actions,
            frame_file=event_record.frame_file,
        )
        logger.info(
            "AlertComposerService 組成警報：record_id=%s severity=%s",
            event_record.record_id, severity.value,
        )
        return alert

    def compose_preliminary(self, event_record: RawEventRecord, vlm_observation: VLMObservation) -> Alert:
        """
        只依規則系統標籤與 VLM 驗證結論組成「初步警報」，不等 RAG + LLM。

        等級規則跟最終警報相同（見 _severity()），只是把 LLM 綜合 RAG 後的
        final_verdict 換成 verification_verdict（llm/visual_verifier.py 依
        VLM 描述判斷，不做 RAG）：support 對應
        confirmed、contradict 對應 overturned。最終警報出來後以最終警報為準，
        等級可能升高也可能降低。
        """
        severity = _severity(
            event_record.predicted_event_type,
            rejected=vlm_observation.verification_verdict == VerificationVerdict.CONTRADICT,
            confirmed=vlm_observation.verification_verdict == VerificationVerdict.SUPPORT,
            confidence=vlm_observation.visual_confidence,
        )

        alert = Alert(
            record_id=event_record.record_id,
            severity=severity,
            title=f"[{severity.value.upper()}・初步] {event_record.predicted_event_type} 事件",
            short_message=vlm_observation.visual_description,
            full_report="\n".join(
                [
                    "初步警報：僅依規則系統標籤與視覺模型觀察，完整分析（原因、應變建議）產生中。",
                    f"畫面觀察：{vlm_observation.visual_description}",
                    f"視覺驗證：{vlm_observation.verification_verdict.value}"
                    f"（信心分數：{vlm_observation.visual_confidence:.2f}）"
                    + (f"，依據：{vlm_observation.verification_reason}" if vlm_observation.verification_reason else ""),
                ]
            ),
            frame_file=event_record.frame_file,
        )
        logger.info(
            "AlertComposerService 組成初步警報：record_id=%s severity=%s",
            event_record.record_id, severity.value,
        )
        return alert

    def _decide_severity(self, event_record: RawEventRecord, analysis_result: AnalysisResult) -> AlertSeverity:
        return _severity(
            event_record.predicted_event_type,
            rejected=analysis_result.final_verdict == FinalVerdict.OVERTURNED,
            confirmed=analysis_result.final_verdict == FinalVerdict.CONFIRMED,
            confidence=analysis_result.final_confidence,
        )


def _severity(event_type: str, rejected: bool, confirmed: bool, confidence: float) -> AlertSeverity:
    """
    純規則邏輯，依事件類型、判定與信心分數決定嚴重程度。初步警報（VLM 判定）
    與最終警報（LLM 判定）共用同一套規則，兩者的等級才能直接比較。

    規則（由上而下依序判斷，符合就回傳，不繼續往下比對）：
    1. 規則系統的判斷被推翻（rejected）-> 原本的異常判斷不成立 -> info
    2. 恐慌性移動（見 _PANIC_EVENT_TYPES）判斷成立且信心夠高 -> 影響人身安全風險最高 -> critical
    3. 判斷維持成立（confirmed）且信心夠高 -> warning
    4. 其餘情況（uncertain、信心不足...）-> info，避免誤報造成不必要的恐慌

    confidence 是「對這個判定本身」的信心，不是「事件為真」的機率：uncertain
    搭配 0.7 代表「有七成把握無法確定」。所以第 2 條也要求 confirmed，否則
    uncertain 0.7 會被升成 critical（sweep 565487 的 panic_scatter_p3 就是這樣）。
    """
    if rejected:
        return AlertSeverity.INFO

    if event_type in _PANIC_EVENT_TYPES and confirmed and confidence >= 0.7:
        return AlertSeverity.CRITICAL

    if confirmed and confidence >= 0.6:
        return AlertSeverity.WARNING

    return AlertSeverity.INFO
