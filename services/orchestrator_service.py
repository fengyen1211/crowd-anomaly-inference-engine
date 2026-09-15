"""
services/orchestrator_service.py
===================================
對應架構設計中的 [7] Orchestrator。

負責串接完整流程：

    JSON -> VLM -> Visual Description -> Embedding -> Retriever
         -> Knowledge -> LLM -> Output

也就是：
    Ingestion -> Frame Loader -> VLM(+ObservationBuilder) -> Embedding
    -> Visual Retriever（相似歷史事件） -> Knowledge Retriever（SOP 片段）
    -> Reasoning(LLM) -> Alert Composer -> PipelineOutput

這是唯一「同時依賴」其他所有 service / vlm / embedding / rag / llm
模組的地方，其他模組彼此之間不會互相依賴，全部由這裡透過建構子注入
（Dependency Injection）組合起來——每個模組只依賴抽象介面
（BaseVLM / BaseEmbeddingProvider / BaseRetriever / BaseLLMClient），
不知道也不在乎彼此背後實際是哪個實作，方便獨立替換、獨立測試。

TODO：
- [ ] 設計事件處理狀態機（pending -> analyzing -> done / failed），
      目前是同步、一次性執行完整流程，沒有中間狀態的持久化
- [ ] 加上錯誤處理與重試機制（例如 LLM 回應解析失敗時的重試）
- [ ] frame_service 讀不到畫面檔案時的降級策略（目前會直接讓例外往上拋）
"""

from typing import List, Optional

from embedding.base import BaseEmbeddingProvider
from llm.reasoning_pipeline import ReasoningPipeline
from rag.schemas import RetrievedContext
from rag.text_retriever import KnowledgeTextRetriever
from rag.visual_retriever import VisualSimilarityRetriever
from services.alert_service import AlertComposerService
from services.frame_service import FrameLoaderService
from services.ingestion_service import IngestionService
from services.schemas import PipelineOutput
from utils.logger import get_logger
from utils.schemas import RawEventRecord
from vlm.base import BaseVLM
from vlm.observation_builder import VLMObservationBuilder

logger = get_logger(__name__)


class OrchestratorService:
    """
    協調整個推論端 pipeline 的執行流程。

    所有相依元件都透過建構子注入（依賴注入），
    方便測試時替換成 mock 物件（例如用假的 VLM/LLM 測試流程串接，
    不需要真的載入模型或呼叫外部 API）。
    """

    def __init__(
        self,
        ingestion_service: IngestionService,
        frame_service: FrameLoaderService,
        vlm_client: BaseVLM,
        visual_embedding_provider: BaseEmbeddingProvider,
        visual_retriever: VisualSimilarityRetriever,
        text_retriever: KnowledgeTextRetriever,
        news_retriever: KnowledgeTextRetriever,
        reasoning_pipeline: ReasoningPipeline,
        alert_service: AlertComposerService,
    ) -> None:
        self._ingestion_service = ingestion_service
        self._frame_service = frame_service
        self._vlm_client = vlm_client
        self._visual_embedding_provider = visual_embedding_provider
        self._visual_retriever = visual_retriever
        self._text_retriever = text_retriever
        self._news_retriever = news_retriever
        self._reasoning_pipeline = reasoning_pipeline
        self._alert_service = alert_service

    def process_event(self, raw_json: dict) -> PipelineOutput:
        """
        執行完整的單筆事件處理流程，回傳給呼叫端看的精簡結果。

        Args:
            raw_json: 影像端輸出的單筆事件 JSON（dict）。

        Returns:
            PipelineOutput
        """
        # 1. Ingestion：驗證原始 JSON，取得型別安全的事件資料
        record = self._ingestion_service.ingest_single(raw_json)
        logger.info("Orchestrator 開始處理 record_id=%s", record.record_id)

        # 2. Frame Loader：讀取畫面並依 bbox 裁切出感興趣區域
        full_frame = self._frame_service.load_full_frame(record)
        cropped_frame = self._frame_service.crop_bbox(full_frame, record.spatial_bbox)
        context_frames = self._load_context_frames(record)

        # 3. VLM：描述畫面內容，組成業務層需要的 VLMObservation
        vlm_observation = VLMObservationBuilder.build(
            self._vlm_client, cropped_frame, record, context_images=context_frames
        )

        # 4. Embedding：
        #    - 視覺向量：直接對「裁切後的畫面」算 CLIP/SigLIP embedding，
        #      落在跟 rag/visual_indexer.py 寫入歷史事件時相同的向量空間，
        #      才能做視覺相似度比對（用文字 embedding 去查視覺向量庫，
        #      向量空間、維度都對不上，比對出來沒有意義）。
        #    - 文字向量：「系統描述 + VLM 觀察」組成查詢文字，供知識庫檢索使用。
        visual_query_embedding = self._visual_embedding_provider.embed_image([cropped_frame])[0]
        query_text = f"{record.caption}\n{vlm_observation.visual_description}"

        # 5. Retriever：用視覺向量查詢視覺相似的歷史事件
        similar_cases = self._visual_retriever.retrieve(visual_query_embedding)

        # 6. Knowledge：用文字查詢知識庫（SOP / 歷史事件描述）中的相關片段
        knowledge_snippets = self._text_retriever.retrieve(query_text)

        # 6b. News：用同一份查詢文字查詢新聞真實案例知識庫（獨立 collection，
        #     讓 LLM 可以把「法規依據」跟「真實案例」分開引用，見 rag_prompt.py）
        news_precedents = self._news_retriever.retrieve(query_text)

        retrieved_context = RetrievedContext(
            similar_cases=similar_cases,
            knowledge_snippets=knowledge_snippets,
            news_precedents=news_precedents,
        )

        # 7. LLM：綜合以上所有資訊，產生摘要、驗證判定、可能原因、應變建議
        analysis_result = self._reasoning_pipeline.run(record, vlm_observation, retrieved_context)

        # 8. Alert：純規則邏輯決定警報等級
        alert = self._alert_service.compose(record, analysis_result)

        logger.info("Orchestrator 完成處理 record_id=%s alert_level=%s", record.record_id, alert.severity.value)

        # 9. Output：組成呼叫端要的精簡結果
        return PipelineOutput(
            summary=analysis_result.summary,
            event_type=record.predicted_event_type,
            possible_causes=analysis_result.possible_causes,
            recommendation=analysis_result.recommended_actions,
            alert_level=alert.severity,
        )

    def _load_context_frames(self, record: RawEventRecord) -> Optional[List[bytes]]:
        """
        讀取＋裁切 before_frame_file／after_frame_file（若有）。

        兩者跟主要畫面共用同一個 spatial_bbox（同機位、同座標，只是時間點
        不同），所以套用一樣的 crop_bbox() 裁切邏輯。目前只有 panic_scatter
        ／counter_flow 這類需要跨時間比對的事件類型會帶這兩個欄位，其餘
        事件類型這裡會回傳 None，VLM 那端沿用單張畫面的原有流程。
        """
        frame_files = [f for f in (record.before_frame_file, record.after_frame_file) if f]
        if not frame_files:
            return None

        context_frames = []
        for frame_file in frame_files:
            full_frame = self._frame_service.load_context_frame(frame_file, record.record_id)
            context_frames.append(self._frame_service.crop_bbox(full_frame, record.spatial_bbox))
        return context_frames
