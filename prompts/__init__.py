"""
prompts package
==================
存放所有 LLM 任務的 Prompt 組裝邏輯，分成「共用區塊」與「任務模板」兩層。

共用區塊（被下面的任務模板組合使用，不會單獨拿去呼叫 LLM）：
- system_prompt.py   所有任務共用的角色設定與硬性規則
- user_prompt.py     事件資料（+ VLM 觀察）組成的文字區塊
- rag_prompt.py      RAG 檢索結果（相似案例 + 知識庫片段）組成的文字區塊

任務模板（組合上面的共用區塊 + 任務專屬指示，回傳可直接餵給
BaseLLMClient.chat_completion() 的訊息列表）：
- summary_prompt.py    事件摘要生成（不需要 RAG）
- reasoning_prompt.py  原因推論與驗證（需要 RAG，是唯一需要「講依據」的任務）
- alert_prompt.py      警報內容生成（把分析結果轉換成通知文字，不做分析）

每個檔案獨立存放：修改某個任務的措辭時，不會影響到其他任務或共用區塊；
修改共用區塊時，所有引用它的任務會同步更新。
"""
