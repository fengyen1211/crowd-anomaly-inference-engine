"""
prompts/rag_prompt.py
========================
定義「檢索結果」的 Prompt 區塊（RAG grounding）。

把 rag.schemas.RetrievedContext（視覺相似歷史案例 + 知識庫文字片段 + 新聞真實案例）
組成一段文字，附加在 user prompt 之後，讓 LLM 在生成原因推論、
應變建議時「有東西可以引用」，而不是憑空生成。

刻意在「查無資料」時明確告知 LLM，而不是留空——
明確寫「查無相關資料」比留空更能避免 LLM 自己腦補內容
（留空的話 LLM 可能會忽略這一段，或誤以為是格式錯誤而自行腦補）。

實測發現：這台機器的 Ollama 依 VRAM（4GB）自動把 context window 設得很小
（num_ctx=4096），知識庫／新聞案例內容豐富之後，檢索片段全文塞進 prompt
很容易把可用於「輸出」的 token 空間擠光，導致 LLM 的 JSON 回應被硬生生
截斷（parse 失敗）。這裡刻意把每筆片段內容截短、且只顯示檔名而不是完整
路徑，把 prompt 控制在合理長度，優先保留「有幾筆依據、依據內容大致是
什麼」，而不是塞進逐字全文——citation 的來源檔名還在，需要追根究柢時
可以自己去查那個檔案。
"""

from pathlib import Path

from rag.schemas import RetrievedContext

_MAX_SNIPPET_CHARS = 150


def _truncate(text: str, max_chars: int = _MAX_SNIPPET_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "…"


def _short_source(source_document: str) -> str:
    """只顯示檔名，不要把使用者本機的完整路徑塞進 prompt 浪費 token。"""
    return Path(source_document).name


class RAGPromptBuilder:
    """組裝「檢索結果」文字區塊。"""

    @classmethod
    def build(cls, context: RetrievedContext) -> str:
        """
        Args:
            context: RAG 模組檢索到的相似案例與知識庫片段。

        Returns:
            str: 給 LLM 看的檢索結果文字，明確標示每筆資料的來源，
                 方便 LLM 生成內容時附上引用依據。
        """
        lines = ["以下是檢索到的參考資料，生成回答時請優先引用這些資料，不要憑空杜撰："]

        if context.similar_cases:
            lines.append("")
            lines.append("【相似歷史事件】")
            for i, case in enumerate(context.similar_cases, start=1):
                lines.append(
                    f"{i}. （相似度 {case.similarity_score:.2f}，事件類型：{case.predicted_event_type}）"
                    f"{_truncate(case.caption)}"
                )
        else:
            lines.append("")
            lines.append("【相似歷史事件】查無相似的歷史事件紀錄。")

        if context.knowledge_snippets:
            lines.append("")
            lines.append("【知識庫參考片段】")
            for i, snippet in enumerate(context.knowledge_snippets, start=1):
                lines.append(
                    f"{i}. （來源：{_short_source(snippet.source_document)}，相關度 {snippet.relevance_score:.2f}）"
                    f"{_truncate(snippet.content)}"
                )
        else:
            lines.append("")
            lines.append("【知識庫參考片段】查無相關的知識庫文件片段。")

        if context.news_precedents:
            lines.append("")
            lines.append("【新聞真實案例】")
            for i, precedent in enumerate(context.news_precedents, start=1):
                lines.append(
                    f"{i}. （來源：{_short_source(precedent.source_document)}，相關度 {precedent.relevance_score:.2f}）"
                    f"{_truncate(precedent.content)}"
                )
        else:
            lines.append("")
            lines.append("【新聞真實案例】查無相關的新聞案例。")

        return "\n".join(lines)
