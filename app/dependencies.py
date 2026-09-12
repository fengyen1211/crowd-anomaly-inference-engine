"""
app/dependencies.py
======================
FastAPI 依賴注入（Dependency Injection）的組裝點。

★★★ 這裡是「支援未來更換 VLM / LLM / 向量資料庫」的關鍵組裝位置 ★★★

設計理念：
- routers/ 裡的 API 端點，不會自己 new 一個 Provider 出來，
  而是透過 FastAPI 的 Depends(get_vlm_client) 取得。
- 之後如果要把 VLM 從 Mock 換成 Qwen2.5-VL / InternVL / LLaVA /
  MiniCPM-V，或把 LLM 從 Mock 換成本地的 Qwen3/Gemma3/Llama3，
  只需要修改 config.yaml，不需要修改任何 router 或 service 的程式碼
  （VLM / Embedding / LLM 三者現在是同一套 factory pattern）。

TODO：
- [ ] 加上 get_db() 等其他依賴注入函式
"""

from functools import lru_cache
from typing import Optional

from config.settings import Settings, get_settings
from config.yaml_config import AppConfig, load_config
from embedding.base import BaseEmbeddingProvider
from embedding.factory import EmbeddingFactory
from llm.base import BaseLLMClient
from llm.factory import LLMFactory
from llm.reasoning_pipeline import ReasoningPipeline
from rag.knowledge_indexer import KnowledgeBaseIndexer
from rag.text_retriever import KnowledgeTextRetriever
from rag.vector_store import ChromaVectorStore
from rag.visual_indexer import VisualEventIndexer
from rag.visual_retriever import VisualSimilarityRetriever
from services.alert_service import AlertComposerService
from services.frame_service import FrameLoaderService
from services.ingestion_service import IngestionService
from services.orchestrator_service import OrchestratorService
from vlm.base import BaseVLM
from vlm.factory import VLMFactory


@lru_cache
def get_vlm_client(app_config: Optional[AppConfig] = None) -> BaseVLM:
    """
    依 config.yaml 的 vlm 區塊，透過 VLMFactory 建立對應實作。

    預設（config.yaml 的預設值）：mock，不載入任何真實模型。
    要切換模型（例如之後的 qwen2.5-vl / internvl / llava / minicpm_v），
    只需要修改 config.yaml，不需要改這個函式。

    用 lru_cache 確保整個應用程式生命週期只建立一次、只呼叫一次
    load_model()——本地 VLM 權重載入通常很花時間，不應該每次注入
    都重新載入一次。
    """
    app_config = app_config or load_config()
    vlm_config = app_config.vlm
    client = VLMFactory.create(vlm_config.provider, model=vlm_config.model)
    client.load_model()
    return client


@lru_cache
def get_llm_client(app_config: Optional[AppConfig] = None) -> BaseLLMClient:
    """
    依 config.yaml 的 llm 區塊，透過 LLMFactory 建立對應實作。

    預設（config.yaml 的預設值）：mock，不需要任何外部依賴。
    要切換成 openai（需要 .env 的 OPENAI_API_KEY）或之後的本地 LLM，
    只需要修改 config.yaml，不需要改這個函式。
    """
    app_config = app_config or load_config()
    llm_config = app_config.llm
    return LLMFactory.create(llm_config.provider, model=llm_config.model)


@lru_cache
def get_embedding_client(app_config: Optional[AppConfig] = None) -> BaseEmbeddingProvider:
    """
    依 config.yaml 的 embedding 區塊，透過 EmbeddingFactory 建立對應實作。

    預設（config.yaml 的預設值）：sentence_transformer / BAAI/bge-m3，
    完全在本地端執行，不需要呼叫任何外部 API。
    要切換模型（例如換成 openai、或之後的 clip/siglip），
    只需要修改 config.yaml，不需要改這個函式。

    用 lru_cache 確保整個應用程式生命週期只載入一次模型權重——
    這個函式會被 get_text_retriever() 跟 get_knowledge_indexer()
    分別呼叫，沒有快取的話會重複載入同一個模型兩次。
    """
    app_config = app_config or load_config()
    embedding_config = app_config.embedding
    return EmbeddingFactory.create(embedding_config.provider, model=embedding_config.model)


@lru_cache
def get_visual_embedding_client(app_config: Optional[AppConfig] = None) -> BaseEmbeddingProvider:
    """
    依 config.yaml 的 visual_embedding 區塊，透過 EmbeddingFactory 建立對應實作。

    刻意跟 get_embedding_client() 分開：get_embedding_client() 預設是
    sentence_transformer（純文字，bge-m3），不支援 embed_image()；
    視覺相似度檢索需要的是 CLIP/SigLIP 這種文字與圖片落在同一個向量
    空間的多模態模型，兩者的 provider、model、向量維度都不同，
    不能共用同一個 Provider 實例或同一個 lru_cache key。
    """
    app_config = app_config or load_config()
    visual_embedding_config = app_config.visual_embedding
    return EmbeddingFactory.create(visual_embedding_config.provider, model=visual_embedding_config.model)


def get_visual_vector_store(settings: Settings | None = None) -> ChromaVectorStore:
    """
    取得「視覺相似事件」專用的向量資料庫（對應 chroma_collection_visual）。
    """
    settings = settings or get_settings()
    return ChromaVectorStore(settings.chroma_collection_visual, settings)


def get_knowledge_vector_store(settings: Settings | None = None) -> ChromaVectorStore:
    """
    取得「知識庫文件」專用的向量資料庫（對應 chroma_collection_knowledge）。
    """
    settings = settings or get_settings()
    return ChromaVectorStore(settings.chroma_collection_knowledge, settings)


def get_visual_retriever(settings: Settings | None = None) -> VisualSimilarityRetriever:
    """組裝視覺相似度檢索器。"""
    return VisualSimilarityRetriever(get_visual_vector_store(settings))


def get_text_retriever(settings: Settings | None = None) -> KnowledgeTextRetriever:
    """組裝知識庫文字檢索器（需要額外注入 embedding 實作）。"""
    return KnowledgeTextRetriever(
        get_knowledge_vector_store(settings),
        get_embedding_client(),
    )


def get_knowledge_indexer(settings: Settings | None = None) -> KnowledgeBaseIndexer:
    """組裝知識庫索引建立器（loaders -> chunking -> embeddings -> vector_store）。"""
    return KnowledgeBaseIndexer(
        get_knowledge_vector_store(settings),
        get_embedding_client(),
    )


def get_news_vector_store(settings: Settings | None = None) -> ChromaVectorStore:
    """
    取得「新聞真實案例」專用的向量資料庫（對應 chroma_collection_news）。

    刻意跟 get_knowledge_vector_store() 分開成獨立 collection，而不是
    把新聞案例跟法規文件塞進同一個 collection——KnowledgeTextRetriever
    目前是純向量相似度檢索、沒有 metadata 過濾機制，混在一起會沒辦法
    讓 LLM 分開引用「法規依據」跟「真實案例」。
    """
    settings = settings or get_settings()
    return ChromaVectorStore(settings.chroma_collection_news, settings)


def get_news_retriever(settings: Settings | None = None) -> KnowledgeTextRetriever:
    """
    組裝新聞案例文字檢索器。

    直接重用 KnowledgeTextRetriever（跟資料來源無關，純粹是
    「向量相似度查詢某個 collection」），只是指向新聞 collection，
    不需要另外寫一個檢索器類別。
    """
    return KnowledgeTextRetriever(
        get_news_vector_store(settings),
        get_embedding_client(),
    )


def get_news_indexer(settings: Settings | None = None) -> KnowledgeBaseIndexer:
    """組裝新聞案例索引建立器，同樣重用 KnowledgeBaseIndexer，只是指向新聞 collection。"""
    return KnowledgeBaseIndexer(
        get_news_vector_store(settings),
        get_embedding_client(),
    )


def get_visual_indexer(settings: Settings | None = None) -> VisualEventIndexer:
    """組裝視覺事件索引建立器（image bytes -> CLIP embedding -> vector_store）。"""
    return VisualEventIndexer(
        get_visual_vector_store(settings),
        get_visual_embedding_client(),
    )


def get_ingestion_service() -> IngestionService:
    """組裝事件資料驗證服務。"""
    return IngestionService()


def get_frame_service(settings: Settings | None = None) -> FrameLoaderService:
    """組裝畫面讀取/裁切服務。"""
    return FrameLoaderService(settings=settings)


def get_alert_service() -> AlertComposerService:
    """組裝警報組成服務（純規則邏輯）。"""
    return AlertComposerService()


def get_reasoning_pipeline() -> ReasoningPipeline:
    """組裝 Reasoning Module（需要額外注入 LLM 實作）。"""
    return ReasoningPipeline(get_llm_client())


def get_orchestrator_service() -> OrchestratorService:
    """
    組裝整個推論端 pipeline 的 Orchestrator。

    這是唯一把所有模組串在一起的地方：每個模組都是透過上面各自的
    get_xxx() 函式建立、彼此互不依賴，只有這裡把它們組合起來。
    之後任何一個模組要換實作（例如 VLM 換成 Qwen2.5-VL、
    LLM 換成本地模型），都只需要改對應的 config.yaml 設定，
    這個函式跟 OrchestratorService 本身都不需要修改。

    注意：這個函式故意不接受任何參數（不像 get_frame_service() 等
    函式可以傳入自訂 settings）——因為這個函式會被 routers/ 用
    Depends(get_orchestrator_service) 直接注入，FastAPI 會遞迴檢查
    它的參數簽名；如果這裡也放一個沒有標記 Depends() 的 Pydantic
    參數（例如 settings: Settings），FastAPI 會誤判成請求 body 的
    一個欄位，導致 API 呼叫端必須多包一層 {"settings": ...} 才能用。
    """
    return OrchestratorService(
        ingestion_service=get_ingestion_service(),
        frame_service=get_frame_service(),
        vlm_client=get_vlm_client(),
        visual_embedding_provider=get_visual_embedding_client(),
        visual_retriever=get_visual_retriever(),
        text_retriever=get_text_retriever(),
        news_retriever=get_news_retriever(),
        reasoning_pipeline=get_reasoning_pipeline(),
        alert_service=get_alert_service(),
    )
