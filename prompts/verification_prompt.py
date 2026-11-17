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
- UCF fight 抽樣（verifiers_685625）裡 10 筆有 4 筆 support 的理由是
  「快速移動／相向而行／遮臉，可能與打架有關」，所以再補一條：理由要靠
  「可能、暗示」才連得上就判 uncertain；fight 的行為說明也寫明什麼不算

contradict 的條件有三個版本（VERIFIER_PROMPT，見 config/settings.py）。信任
影像端之後只有 contradict 會撤回警報，所以這裡的寬嚴直接決定漏報：
- default：描述明顯是別的事就判 contradict（正常走動、表演、插圖……）
- contact：同 default，但 fight 的描述裡有身體接觸（擁抱、親吻、握手）時
  不判 contradict。UCF 751767 的 2 筆漏報都是真的打架被 VLM 寫成「接吻」
  「友好的握手或擁抱」，verifier 據此判 contradict 撤回了警報
- obvious：只有明顯的偵測錯誤才判 contradict（畫面裡沒有人、插圖或文字
  畫面、人數明顯不夠），描述的行為跟事件不同只判 uncertain
"""

from typing import Dict, List, Literal

PromptVariant = Literal["default", "contact", "obvious"]

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
    # UCF-Crime 非群聚資料（YOLO+ByteTrack＋CLIP 打架偵測）
    # 寫明「什麼不算」：UCF fight 抽樣裡 gemma3:27b 把「快速移動」「相向而行」
    # 「用手遮臉」都推論成可能在打架而判 support
    "fight": "人與人之間有肢體衝突，例如出手打人、推人、拉扯、扭打、把人壓倒在地；"
             "只是快速移動、相向而行、靠近、交談或圍觀都不算",
}

# 場景異常模型（影像端第二批，CLIP L/14 + MIL）觸發的事件，類型是 scene_ 加上
# 零樣本推測的類別。影像端說明「有異常」可靠、「是哪一種」常常不準（例如
# Explosion007 被標成 scene_arson、scene_road_accident），所以說明寫成
# 「畫面中有異常狀況」，推測類別只當參考。
_SCENE_PREFIX = "scene_"
_SCENE_HINT = (
    "場景異常模型判斷畫面整體出現異常，推測類型是「{category}」，但推測類型常常不準；"
    "請看描述中有沒有寫出異常狀況，例如火光、煙霧、爆炸、物品被破壞、有人持武器、"
    "打鬥、搶奪或有人倒地，不限於推測的類型"
)
# 場景事件的 contradict 只留給明顯的偵測錯誤。sweep_792426（第二批，qwen2.5vl:3b）
# 時段內 26 筆場景事件只有 1 筆描述提到火、煙、破壞這類字眼，真的爆炸、縱火
# 大多被寫成「在購物」「在走動」，跟時段外的誤報（49 筆中 1 筆）分不開；
# 若「只寫到一般活動」就判 contradict，真事件會跟誤報一起被撤回。
_SCENE_CONTRADICT_RULE = (
    "- contradict：只用在描述指出畫面是插圖、動畫或文字畫面，而不是真實的監視器場景。"
    "描述只寫到一般活動（購物、辦公、走動、交談）時判 uncertain，不要判 contradict："
    "視覺模型在低解析度的監視器畫面上常常看不出火光、煙霧或打鬥"
)


def _hint_for(event_type: str):
    """事件類型的行為說明；場景事件依推測類別產生，未知類型回傳 None。"""
    if event_type.startswith(_SCENE_PREFIX):
        return _SCENE_HINT.format(category=event_type[len(_SCENE_PREFIX):])
    return _EVENT_TYPE_HINTS.get(event_type)


_INSTRUCTION = """影像分析系統判斷某個畫面中出現了「{event_type}」{hint_clause}。
另一個視覺模型被要求只客觀描述同一畫面、不判斷是否同意，寫了以下描述：

「{description}」

請只根據這段描述中寫出的內容，判斷描述是否支持系統的判斷：
- support：描述中直接寫出這個行為的核心動作。用詞不必完全相同（例如描述寫「有人在奔跑」就支持「有人在奔跑或明顯加速移動」）
{contradict_rule}
- uncertain：描述完全沒有提到相關的動作，或只有「有人在移動」這類泛泛的說法

不要自行推測描述沒寫到的事：
- support 的依據必須是描述裡直接寫出的那個動作，不能從別的動作推論（例如「快速移動」不能推論成「推擠」，「相向而行」不能推論成「衝突」）
- 如果你的理由需要用到「可能」「暗示」「或許」才能連到這個行為，就判 uncertain

以 JSON 回答：
- verification_verdict：support／contradict／uncertain
- visual_confidence：你對這個結論的信心，0 到 1
- reason：一句話說明依據，引用描述中的關鍵字句"""


_CONTRADICT_RULES = {
    "default": "- contradict：描述的情況明顯是別的事（例如只是正常走動、排練、表演、比賽、觀看，"
               "或畫面是插圖、動畫而不是真實場景），與這個行為不符",
    "obvious": "- contradict：只用在偵測明顯出錯的情況：描述指出畫面（或紅框）裡沒有人、只有物體或空地；"
               "畫面是插圖、動畫或文字而不是真實場景；人數明顯不足以構成這個行為（例如只有一個人）。"
               "描述的人在做別的事（例如聊天、走動、擁抱）不算 contradict，判 uncertain",
}
_CONTRADICT_RULES["contact"] = _CONTRADICT_RULES["default"]

# contact 版本的額外規則：VLM 常把衝突合理化成友善的身體接觸（1103 報告四之 3）
_CONTACT_EXCEPTIONS = {
    "fight": "\n  例外：描述中的人有身體接觸（例如擁抱、親吻、握手、拉住、抱住、倒在一起）時不要判 contradict，"
             "視覺模型常把扭打寫成友善的接觸，這種情況判 uncertain",
}


class VerificationPromptBuilder:
    """組裝「依畫面描述驗證規則系統判斷」的 Prompt。"""

    @classmethod
    def build(cls, event_type: str, description: str, variant: PromptVariant = "default") -> List[Dict[str, str]]:
        hint = _hint_for(event_type)
        contradict_rule = _CONTRADICT_RULES[variant]
        if event_type.startswith(_SCENE_PREFIX):
            contradict_rule = _SCENE_CONTRADICT_RULE
        elif variant == "contact":
            contradict_rule += _CONTACT_EXCEPTIONS.get(event_type, "")
        content = _INSTRUCTION.format(
            contradict_rule=contradict_rule,
            event_type=event_type,
            hint_clause=f"（{hint}）" if hint else "",
            description=description,
        )
        return [{"role": "user", "content": content}]
