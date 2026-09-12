"""
rag package
=============
負責 RAG（檢索增強生成）相關功能，內部再拆成幾個子模組：

- loaders/            文件載入器（txt / markdown / json / pdf）
- chunking/            文件切塊（RecursiveChunker）
- embeddings/           Embedding 抽象介面 + OpenAI 實作（未來可換 CLIP/SigLIP）
- interfaces.py         BaseVectorStore / BaseRetriever 抽象介面
- vector_store.py       ChromaVectorStore：向量資料庫實作（預設 ChromaDB）
- visual_retriever.py   視覺相似度檢索（用 embedding 找相似歷史事件）
- text_retriever.py     知識庫文字檢索（用文字找相關 SOP/法規片段）
- knowledge_indexer.py  串接 loaders -> chunking -> embeddings -> vector_store 的索引建立流程
- schemas.py            RetrievedCase / KnowledgeSnippet / RetrievedContext 資料結構
"""
