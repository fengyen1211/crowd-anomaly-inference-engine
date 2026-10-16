"""
prompts/verification_prompt.py
=================================
定義「依畫面描述驗證規則系統判斷」任務的 Prompt，產生初步警報用的
verification_verdict／visual_confidence（呼叫邏輯在 llm/visual_verifier.py）。

為什麼不直接讓 VLM 看圖時順便判斷？
實測 qwen2.5vl:3b 被要求判斷時，video 7 的 10 筆事件有 9 筆回答 support、
信心全是 0.90，而且描述本身被系統給的行為說明帶偏（例如照抄「有人突然
加速奔跑」）。所以 VLM 維持只做中立描述（見 vlm/observation_builder.py），
判斷另外交給 LLM，而且只給它描述文字：
- 不給事件的系統 caption（裡面的強度數字與「異常」字眼會引導它附和）
- 只看描述寫出的內容、不自行推測；但用詞不必與行為說明完全相同
  （第一版禁止得太強硬，10 筆全部判成 uncertain，連「人群開始四散」也是）
"""

from typing import Dict, List

# 行為說明寫「核心動作」，不寫「突然」這類程度詞：程度詞會讓 LLM 要求描述
# 逐字出現同樣的字眼，第一版因此把「人群開始四散」也判成 uncertain
_EVENT_TYPE_HINTS = {
    "sprint_spike": "有人在奔跑或明顯加速移動",
    "crowd_locked": "人群擠在一起、停滯不動或難以移動",
    "panic_scatter": "人群向四周散開、四散奔逃",
    "counter_flow": "有人朝著與其他人相反的方向移動",
    "panic_dispersal": "人群加速、向四周散開",
    "reverse_flow": "有人朝著與大部分人相反的方向移動",
    "abnormal_gathering": "一群人越靠越近、停下來聚集",
}

_INSTRUCTION = """影像分析系統判斷某個畫面中出現了「{event_type}」{hint_clause}。
另一個視覺模型被要求只客觀描述同一畫面、不判斷是否同意，寫了以下描述：

「{description}」

請只根據這段描述中寫出的內容，判斷描述是否支持系統的判斷：
- support：描述中出現這個行為的核心動作。用詞不必完全相同（例如描述寫「有人在奔跑」就支持「有人在奔跑或明顯加速移動」）
- contradict：描述的情況明顯是別的事（例如只是正常走動、排練、表演、觀看），與這個行為不符
- uncertain：描述完全沒有提到相關的動作，或只有「有人在移動」這類泛泛的說法

不要自行推測描述沒寫到的事。

以 JSON 回答：
- verification_verdict：support／contradict／uncertain
- visual_confidence：你對這個結論的信心，0 到 1
- reason：一句話說明依據，引用描述中的關鍵字句"""


class VerificationPromptBuilder:
    """組裝「依畫面描述驗證規則系統判斷」的 Prompt。"""

    @classmethod
    def build(cls, event_type: str, description: str) -> List[Dict[str, str]]:
        hint = _EVENT_TYPE_HINTS.get(event_type)
        content = _INSTRUCTION.format(
            event_type=event_type,
            hint_clause=f"（{hint}）" if hint else "",
            description=description,
        )
        return [{"role": "user", "content": content}]
