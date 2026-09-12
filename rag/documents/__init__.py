"""
rag/documents package
========================
負責把結構化資料（例如影像端輸出的事件 JSON）組成適合 embedding 的
自然語言文件（LoadedDocument），而不是把原始 JSON 直接字串化存進向量資料庫。

- base.py                    BaseDocumentBuilder 抽象介面
- event_document_builder.py  EventDocumentBuilder：把單筆事件記錄組成描述文字
"""
