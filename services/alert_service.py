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

    def _decide_severity(self, event_record: RawEventRecord, analysis_result: AnalysisResult) -> AlertSeverity:
        """
        純規則邏輯，依事件類型、最終判定與信心分數決定嚴重程度。

        規則（由上而下依序判斷，符合就回傳，不繼續往下比對）：
        1. 規則系統的判斷被 LLM 推翻（overturned）-> 原本的異常判斷不成立 -> info
        2. 恐慌性移動（見 _PANIC_EVENT_TYPES）且信心夠高 -> 影響人身安全風險最高 -> critical
        3. 判斷維持成立（confirmed）且信心夠高 -> warning
        4. 其餘情況（uncertain、信心不足...）-> info，避免誤報造成不必要的恐慌
        """
        if analysis_result.final_verdict == FinalVerdict.OVERTURNED:
            return AlertSeverity.INFO

        if (
            event_record.predicted_event_type in _PANIC_EVENT_TYPES
            and analysis_result.final_confidence >= 0.7
        ):
            return AlertSeverity.CRITICAL

        if (
            analysis_result.final_verdict == FinalVerdict.CONFIRMED
            and analysis_result.final_confidence >= 0.6
        ):
            return AlertSeverity.WARNING

        return AlertSeverity.INFO
