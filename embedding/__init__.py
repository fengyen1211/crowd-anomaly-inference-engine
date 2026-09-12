"""
embedding package
====================
負責把文字／圖片轉成向量，是 RAG 檢索與未來 VLM 視覺比對的共用基礎設施。

★ 設計目標：完全支援 Local AI，不綁定任何特定供應商 ★
所有上層模組（rag/text_retriever.py、rag/knowledge_indexer.py...）
只依賴 base.BaseEmbeddingProvider 這個抽象介面，實際使用哪個 Provider
由 factory.EmbeddingFactory 依設定動態建立，不需要改動呼叫端程式碼。

- base.py                 BaseEmbeddingProvider 抽象介面
- factory.py              EmbeddingFactory：依 provider 名稱建立實例
- mock.py                 MockEmbeddingProvider：測試用假向量，不需載入任何模型
- sentence_transformer.py SentenceTransformerEmbeddingProvider
                          ★ 目前預設 Provider ★，完全在本地端執行，
                          預設模型 BAAI/bge-m3，不依賴任何外部 API
- openai.py               OpenAIEmbeddingProvider：保留，可選，非預設
- clip.py                 CLIPEmbeddingProvider：骨架，之後接視覺 embedding
- siglip.py               SigLIPEmbeddingProvider：骨架，CLIP 的備選方案
"""
