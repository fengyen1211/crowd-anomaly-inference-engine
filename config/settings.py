"""
config/settings.py
====================
本檔案負責定義整個推論端系統的「集中式設定」。

設計理念：
- 所有需要從環境變數 / .env 檔案讀取的設定值（API Key、模型名稱、
  資料庫路徑、向量資料庫路徑...等）都集中在這裡管理，
  避免設定值散落在各個模組中，難以維護。
- 使用 pydantic-settings 的 BaseSettings，可以自動從環境變數或 .env
  檔案讀取對應欄位，並且有型別驗證。
- Embedding / LLM / VLM 的 provider/model 選擇統一都在 config.yaml
  （見 config/yaml_config.py），因為那是「常態調整、需要被 git
  追蹤」的設定，跟這裡的機密/環境設定性質不同。這裡保留
  `openai_embedding_model` / `openai_llm_model` 純粹是 OpenAI Provider
  自己的內部預設模型名稱（僅 provider="openai" 時使用）。

TODO：
- [ ] 補上正式的環境變數驗證（例如 OPENAI_API_KEY 不可為空字串）
- [ ] 依部署環境（dev / staging / prod）切分不同的 .env 檔案
- [ ] 確認多 worker 環境下 lru_cache 單例是否符合預期行為
"""

from functools import lru_cache

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# pydantic-settings 的 env_file 只會把 .env 的值讀進 Settings 物件的欄位，
# 不會真的寫進 os.environ；但 huggingface_hub / sentence-transformers 這類
# 第三方套件是直接讀 os.environ 的 HF_HOME 等變數，不會理會 Settings。
# 這裡明確呼叫 load_dotenv()，讓 .env 裡任何變數（不只是 Settings 定義的
# 欄位）都真的進到行程環境變數，這樣使用者只要寫在 .env 就好，
# 不需要每次開新終端機都手動 export/$env: 設定一次。
load_dotenv()


class Settings(BaseSettings):
    """
    系統集中設定物件。

    每一個欄位都可以透過環境變數覆寫（大小寫不敏感），
    例如環境變數 OPENAI_API_KEY 會對應到 openai_api_key 欄位。
    """

    # ---------- 基本應用程式資訊 ----------
    app_name: str = "推論端 API（VLM + RAG + LLM）"
    app_version: str = "0.1.0"
    debug: bool = False

    # ---------- OpenAI 相關設定（可選 LLM / Embedding 供應商，非預設） ----------
    # 注意：這裡只放「設定值」，實際呼叫 OpenAI API 的程式碼
    # 分別在 llm/openai_llm.py、embedding/openai.py 中實作。
    # 本專案的 Embedding / LLM / VLM 預設都不依賴 OpenAI
    # （見 config.yaml），OpenAI 僅保留作為可選 provider。
    openai_api_key: str = ""
    openai_llm_model: str = "gpt-4o-mini"                   # 僅 provider="openai" 時使用
    openai_embedding_model: str = "text-embedding-3-small"  # 1536 維，僅 provider="openai" 時使用

    # ---------- Ollama（本地 LLM/VLM 執行服務）設定 ----------
    # llm/ollama_llm.py、vlm/ollama_vlm.py 共用，透過 HTTP API 呼叫本地跑的
    # Ollama 服務（例如 Qwen3 / Gemma3 / Qwen2.5-VL / MiniCPM-V），
    # 模型量化、載入、記憶體管理都交給 Ollama 處理。
    ollama_base_url: str = "http://localhost:11434"

    # ---------- 向量資料庫（ChromaDB）設定 ----------
    chroma_persist_dir: str = "./chroma_data"
    chroma_collection_visual: str = "event_visual_embeddings"
    chroma_collection_knowledge: str = "knowledge_base_documents"
    chroma_collection_news: str = "news_precedent_documents"

    # ---------- 畫面檔案（Frame）設定 ----------
    # FrameLoaderService 依 record.frame_file 到這個目錄底下找對應的圖片檔案
    frames_dir: str = "./frames"

    # ---------- 文件切塊（Chunking）設定 ----------
    # 對應 rag/chunking/recursive_chunker.py 的預設值
    chunk_size: int = 500      # 每個 chunk 的最大字元數
    chunk_overlap: int = 50    # 相鄰 chunk 之間的重疊字元數，避免語意在邊界被切斷

    # ---------- 關聯式資料庫設定 ----------
    database_url: str = "sqlite:///./inference_engine.db"  # TODO: 之後可換成 PostgreSQL

    # ---------- Pydantic Settings 讀取設定 ----------
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """
    取得全域唯一的 Settings 實例（單例模式）。

    使用 lru_cache 確保整個應用程式生命週期中只會建立一次 Settings，
    並且方便在 FastAPI 的 Depends() 中注入使用。
    """
    return Settings()
