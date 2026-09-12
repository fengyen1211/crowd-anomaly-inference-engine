"""
prompts/news_summary_prompt.py
=================================
定義「新聞事件摘要」任務的完整 Prompt。

用途：scripts/fetch_and_index_news.py 從候選來源（GDELT 或 Wikipedia 搜尋）撈到
候選文章全文後，用這個 Prompt 請本地 LLM 做三件事：
1. 判斷這篇文章是否真的跟「群眾異常行為」（踩踏、推擠、恐慌逃散、騷亂）相關
   ——關鍵字搜尋會有誤判（例如球賽踩踏事故 vs. 純球賽賽果報導），
   讓 LLM 讀完全文再判斷一次，比純關鍵字比對準確。
2. 從文章內文判斷「事件實際發生日期」——不能直接信任來源提供的日期提示：
   GDELT 的 seendate 是「該篇報導被 GDELT 索引到的時間」，Wikipedia 條目更是
   完全沒有這種欄位（Wikipedia 只有「條目最後編輯時間」，跟事件本身何時發生
   毫無關係）。所以日期提示只當作參考，實際事件日期要求 LLM 從內文正文判斷。
3. 若相關，摘要成適合寫入知識庫的結構化欄位。

跟 prompts/system_prompt.py（推論任務共用角色設定）刻意不共用——
這裡是「新聞摘要」任務，角色是新聞事實查核/摘要助理，跟「群眾異常行為
監控推論助理」是不同性質的工作，硬套同一個 System Prompt 反而會讓
LLM 混淆兩種任務的行為規範。

輸出格式：要求 LLM 輸出 JSON，欄位對應
rag/documents/news_document_builder.py::NewsDocumentBuilder 預期的輸入欄位，
實際解析邏輯在 scripts/fetch_and_index_news.py（用 utils/llm_json.py 解析）。
"""

from typing import Dict, List

_SYSTEM_PROMPT = """你是一位新聞摘要助理，任務是判讀單篇新聞文章，判斷它是否描述一起
真實發生的「群眾異常行為」事件（例如：踩踏事故、群眾推擠、恐慌逃散、騷亂衝突），
並把相關資訊整理成結構化摘要，供另一套群眾異常行為監控系統的知識庫引用為真實案例。

你必須遵守以下規則：
- 只有文章確實描述一起「已經發生」的群眾傷亡/推擠/恐慌事件時，才判定為相關；
  單純預告、賽事報導、政策討論、與群眾安全無直接關係的新聞，都判定為不相關。
- 摘要必須忠於原文內容，不要加入原文沒有提到的細節或臆測原因。
- event_date 必須從內文正文找出的實際事件發生日期判斷，不能直接照抄「來源日期提示」——
  來源日期提示可能是報導索引時間或條目編輯時間，不等於事件發生日期；
  內文找不到明確日期時才使用來源日期提示，兩者都沒有就填 "未知"。
- 使用繁體中文輸出，摘要控制在 2-3 句話，簡潔扼要。
- 只輸出 JSON 本身，不要加上任何其他文字說明，也不要用 Markdown 程式碼區塊包住。"""

_TASK_INSTRUCTION_TEMPLATE = """
【新聞標題】
{title}

【來源日期提示（僅供參考，不一定是事件發生日期）】
{date_hint}

【新聞來源網址】
{source_url}

【新聞全文】
{article_text}

【任務】
請判讀以上文章，並以 JSON 格式輸出：

{{
  "is_relevant": true 或 false,
  "incident_type": "stampede | crowd_crush | panic | riot | other",
  "headline": "文章標題（可直接沿用或稍微精簡）",
  "summary": "2-3 句話的中文摘要",
  "event_location": "事件發生地點（城市、國家），查無明確地點請填 \\"未知\\"",
  "event_date": "事件實際發生日期（YYYY-MM-DD），從內文正文判斷，查無明確日期請填 \\"未知\\""
}}

規則：
- 若 is_relevant 為 false，其餘欄位可以省略或留空字串，不需要勉強填寫。
- incident_type 請從列舉值中選最接近的一個，無法歸類時選 "other"。
"""


class NewsSummaryPromptBuilder:
    """組裝「新聞事件摘要」任務的完整 Prompt（system + user）。"""

    @classmethod
    def build(cls, title: str, article_text: str, source_url: str, date_hint: str) -> List[Dict[str, str]]:
        """
        Args:
            title: 候選來源（GDELT 或 Wikipedia）回傳的文章標題。
            article_text: 用 requests + BeautifulSoup 擷取到的文章全文。
            source_url: 文章網址。
            date_hint: 來源提供的日期線索（GDELT 的 seendate，或 Wikipedia 沒有
                       時傳入的空字串/"未知"）——僅供 LLM 參考，不強制採用，
                       實際事件日期由 LLM 從全文內容判斷。

        Returns:
            List[Dict[str, str]]: 可直接傳給 BaseLLMClient.chat_completion()
            的訊息列表。
        """
        user_content = _TASK_INSTRUCTION_TEMPLATE.format(
            title=title,
            date_hint=date_hint or "未知",
            source_url=source_url,
            article_text=article_text,
        )

        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
