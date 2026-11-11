"""
services/alert_service.py
============================
對應架構設計中的 [6] Alert Composer。

負責：
- 依事件類型與驗證結論（llm/visual_verifier.py 的 verification_verdict /
  visual_confidence）套用規則，決定警報等級（info / warning / critical）
- 組成給 Web 介面顯示的警報內容（標題/簡短訊息/完整報告）

等級採「信任影像端」：影像端偵測到就發，驗證只在 contradict（描述明顯是
別的事）時把它降成 info，uncertain 照發。見 _severity() 的說明。

最終警報的等級跟初步警報一樣只看驗證結論，不看推論 LLM 的 final_verdict。
sweep_565835 裡 gemma3:4b 的推論把驗證判為 contradict（描述寫「觀眾隨音樂
擺動，沒有明顯四散」）的 panic_scatter 事件判成 confirmed、升成 critical；
固定描述比較（verifiers_565780）也顯示專職驗證的 qwen3:8b 判斷比推論 LLM
可靠。所以推論 LLM 只負責摘要、可能原因與應變建議，它的判定寫進報告供
參考，不改變等級——最終警報與初步警報的等級因此一致。

★ 這個模組是「純規則邏輯」，不需要任何 LLM/VLM 呼叫 ★
警報等級不應該受 LLM 生成內容不穩定性影響，規則判斷比較可預測、
可稽核；文字內容目前用簡單的字串模板組成（不是 LLM 生成），
如果之後需要更自然的警報文字，可以另外接上
prompts/alert_prompt.py + LLM（那是「表達」而不是「決策」，
應該是可選的加強，不應該取代這裡的規則判斷）。
"""

from llm.schemas import AnalysisResult
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

    def compose(
        self,
        event_record: RawEventRecord,
        analysis_result: AnalysisResult,
        vlm_observation: VLMObservation,
    ) -> Alert:
        """
        組成最終的警報內容：等級依驗證結論（見檔案頂端說明），文字來自推論 LLM。
        """
        severity = _severity_from_verification(event_record, vlm_observation)

        title = f"[{severity.value.upper()}] {event_record.predicted_event_type} 事件"
        short_message = analysis_result.summary
        full_report = "\n".join(
            [
                f"事件摘要：{analysis_result.summary}",
                f"視覺驗證：{vlm_observation.verification_verdict.value}"
                f"（信心分數：{vlm_observation.visual_confidence:.2f}，決定警報等級）",
                f"推論模型判定：{analysis_result.final_verdict.value}"
                f"（信心分數：{analysis_result.final_confidence:.2f}，僅供參考）",
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

        等級跟最終警報用同一套規則、同一個依據（見 _severity_from_verification()），
        所以兩者等級相同；最終警報只是補上摘要、原因與應變建議。
        """
        severity = _severity_from_verification(event_record, vlm_observation)

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

    def compose_skipped(self, event_record: RawEventRecord, reason: str) -> Alert:
        """畫面沒進入判斷流程（例如是插圖，見 services/frame_filter.py）時的初步警報，固定 info。"""
        logger.info("AlertComposerService 略過事件：record_id=%s（%s）", event_record.record_id, reason)
        return Alert(
            record_id=event_record.record_id,
            severity=AlertSeverity.INFO,
            title=f"[INFO・略過] {event_record.predicted_event_type} 事件",
            short_message=reason,
            full_report=reason,
            frame_file=event_record.frame_file,
        )


def _severity_from_verification(event_record: RawEventRecord, vlm_observation: VLMObservation) -> AlertSeverity:
    """依驗證結論決定等級：只有 contradict 會推翻影像端的判斷。"""
    return _severity(
        event_record.predicted_event_type,
        rejected=vlm_observation.verification_verdict == VerificationVerdict.CONTRADICT,
    )


def _severity(event_type: str, rejected: bool) -> AlertSeverity:
    """
    純規則邏輯（信任影像端）：
    1. 驗證判 contradict（描述明顯是別的事）-> info
    2. 恐慌性移動（見 _PANIC_EVENT_TYPES）-> critical，影響人身安全風險最高
    3. 其餘（support、uncertain）-> warning

    原本是「support 且信心 >= 0.6 才發」，uncertain 一律降成 info。1103 的
    離線重算（UCF 684501，99 筆）顯示這樣分不出真假：該通報的留下 18/47
    （38%）、不該通報的留下 19/52（37%），比例幾乎一樣；VLM 在 320x240 的
    監視器畫面上大多寫不出打架動作，verifier 只能判 uncertain，真事件就被
    擋掉。改成只有 contradict 降級後，同一批判斷重算為 45/47 有通報、
    48/52 誤報——漏報幾乎消失，誤報量改由影像端的準確度決定。

    contradict 不看信心：contradict >= 0.8 才降級的版本重算結果幾乎相同。
    """
    if rejected:
        return AlertSeverity.INFO

    if event_type in _PANIC_EVENT_TYPES:
        return AlertSeverity.CRITICAL

    return AlertSeverity.WARNING
