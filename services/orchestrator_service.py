"""
services/orchestrator_service.py
===================================
對應架構設計中的 [7] Orchestrator。

負責串接完整流程：

    JSON -> VLM -> Visual Description -> Embedding -> Retriever
         -> Knowledge -> LLM -> Output

也就是：
    Ingestion -> Frame Loader -> NonCameraFrameFilter -> VLM(+ObservationBuilder) -> VisualVerifier -> Embedding
    -> Visual Retriever（相似歷史事件） -> Knowledge Retriever（SOP 片段）
    -> Reasoning(LLM) -> Alert Composer -> PipelineOutput

分成三個階段：
- detect()：影像端偵測到就發初步警報（信任影像端，不等 VLM）
- verify()：VLM 描述 + 驗證，contradict 時撤回初步警報
- analyze()：RAG + LLM，補上摘要、可能原因與應變建議

畫面是插圖、卡通這類非監視器畫面時（services/frame_filter.py），detect()
就停下，警報是 info，不跑 VLM、驗證與 RAG + LLM。

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

from dataclasses import dataclass, replace
from typing import Callable, List, Optional

from embedding.base import BaseEmbeddingProvider
from llm.reasoning_pipeline import ReasoningPipeline
from llm.visual_verifier import VisualVerifier
from rag.schemas import RetrievedContext
from rag.text_retriever import KnowledgeTextRetriever
from rag.visual_retriever import VisualSimilarityRetriever
from services.alert_service import AlertComposerService
from services.frame_filter import NonCameraFrameFilter
from services.frame_service import FrameLoaderService
from services.ingestion_service import IngestionService
from services.schemas import PipelineOutput
from utils.logger import get_logger
from utils.schemas import Alert, AlertSeverity, RawEventRecord
from vlm.base import BaseVLM
from vlm.observation_builder import VLMObservationBuilder
from vlm.schemas import VLMObservation

logger = get_logger(__name__)


@dataclass
class ObservationStage:
    """detect()／verify() 的結果，原樣交給下一步接續處理。"""

    record: RawEventRecord
    cropped_frame: bytes
    full_frame: bytes
    # 影像端偵測到就發的初步警報（不等 VLM）
    preliminary_alert: Alert
    # verify() 之後才有值：VLM 觀察與依驗證結論修正後的警報
    vlm_observation: Optional[VLMObservation] = None
    verified_alert: Optional[Alert] = None
    # 被 NonCameraFrameFilter 略過時記原因，之後的步驟都不跑
    skipped_reason: Optional[str] = None


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
        visual_verifier: VisualVerifier,
        visual_embedding_provider: BaseEmbeddingProvider,
        visual_retriever: VisualSimilarityRetriever,
        text_retriever: KnowledgeTextRetriever,
        news_retriever: KnowledgeTextRetriever,
        reasoning_pipeline: ReasoningPipeline,
        alert_service: AlertComposerService,
        vlm_view: str = "crop",
        frame_filter: Optional[NonCameraFrameFilter] = None,
    ) -> None:
        self._ingestion_service = ingestion_service
        self._frame_service = frame_service
        self._vlm_client = vlm_client
        self._visual_verifier = visual_verifier
        self._visual_embedding_provider = visual_embedding_provider
        self._visual_retriever = visual_retriever
        self._text_retriever = text_retriever
        self._news_retriever = news_retriever
        self._reasoning_pipeline = reasoning_pipeline
        self._alert_service = alert_service
        # VLM 看裁切圖（crop）還是整張畫面加紅框（full），見 config/settings.py
        self._vlm_view = vlm_view
        # None 表示不過濾（visual_embedding 不是 CLIP 時，見 app/dependencies.py）
        self._frame_filter = frame_filter

    def process_event(
        self,
        raw_json: dict,
        on_preliminary_alert: Optional[Callable[[Alert], None]] = None,
        on_verified_alert: Optional[Callable[[Alert], None]] = None,
    ) -> PipelineOutput:
        """
        執行完整的單筆事件處理流程，回傳給呼叫端看的精簡結果。

        Args:
            raw_json: 影像端輸出的單筆事件 JSON（dict）。
            on_preliminary_alert: 初步警報一產生就呼叫（影像端偵測到就發，不等 VLM）。
            on_verified_alert: VLM 描述與驗證完成、警報等級確認（或撤回）時呼叫，
                    不等後面的 RAG + LLM。批次腳本用這兩個 callback 記錄時間點。

        Returns:
            PipelineOutput
        """
        stage = self.detect(raw_json)
        if on_preliminary_alert is not None:
            on_preliminary_alert(stage.preliminary_alert)
        stage = self.verify(stage)
        if on_verified_alert is not None:
            on_verified_alert(stage.verified_alert)
        return self.analyze(stage)

    def detect(self, raw_json: dict) -> ObservationStage:
        """
        第一階段：驗證事件、讀取畫面，影像端偵測到就組成初步警報，不等 VLM。

        等級採「信任影像端」（見 services/alert_service.py），預設就是要發，
        所以不必等 VLM 那 2～3 秒；VLM 判 contradict 時由 verify() 撤回。
        插圖等非監視器畫面在這裡就停下（只需要 CLIP，H200 上幾十毫秒）。
        """
        # 1. Ingestion：驗證原始 JSON，取得型別安全的事件資料
        record = self._ingestion_service.ingest_single(raw_json)
        logger.info("Orchestrator 開始處理 record_id=%s", record.record_id)

        # 2. Frame Loader：讀取畫面並依 bbox 裁切出感興趣區域
        full_frame = self._frame_service.load_full_frame(record)
        cropped_frame = self._frame_service.crop_bbox(full_frame, record.spatial_bbox)

        # 2b. 插圖、卡通等非監視器畫面不進入判斷流程（見 services/frame_filter.py）
        skipped_reason = self._frame_filter.check(full_frame, record.record_id) if self._frame_filter else None
        if skipped_reason:
            return ObservationStage(
                record=record,
                cropped_frame=cropped_frame,
                full_frame=full_frame,
                preliminary_alert=self._alert_service.compose_skipped(record, skipped_reason),
                skipped_reason=skipped_reason,
            )

        # 3. 初步警報：只看影像端的事件類型
        return ObservationStage(
            record=record,
            cropped_frame=cropped_frame,
            full_frame=full_frame,
            preliminary_alert=self._alert_service.compose_preliminary(record),
        )

    def verify(self, stage: ObservationStage) -> ObservationStage:
        """
        第二階段：VLM 中立描述 + 驗證，依驗證結論確認或撤回初步警報（不做 RAG）。
        """
        if stage.skipped_reason:
            return replace(stage, verified_alert=stage.preliminary_alert)

        record = stage.record
        full_frame = stage.full_frame
        context_frames = self._load_context_frames(record)

        # 4. VLM：中立地描述畫面內容（不判斷），組成 VLMObservation
        vlm_image = (
            stage.cropped_frame
            if self._vlm_view == "crop"
            else self._frame_service.vlm_view_image(full_frame, record.spatial_bbox, self._vlm_view)
        )
        vlm_observation = VLMObservationBuilder.build(
            self._vlm_client, vlm_image, record, context_images=context_frames, view=self._vlm_view,
            # zoom 被拒答時（畫面暴力程度高）退回一般裁切圖，context 畫面不帶
            fallback_image=stage.cropped_frame if self._vlm_view == "zoom" else None,
        )

        # 4b. 驗證：LLM 只看描述文字，判斷是否支持規則系統的判斷
        vlm_observation = self._visual_verifier.verify(record, vlm_observation)

        # 4c. 確認或撤回初步警報
        verified_alert = self._alert_service.compose_verified(record, vlm_observation, stage.preliminary_alert)
        return replace(stage, vlm_observation=vlm_observation, verified_alert=verified_alert)

    def observe(self, raw_json: dict) -> ObservationStage:
        """detect() + verify()，給不需要分開兩步的呼叫端。"""
        return self.verify(self.detect(raw_json))

    def analyze(self, stage: ObservationStage) -> PipelineOutput:
        """第三階段：RAG 檢索 + LLM 推論，組成最終警報與完整結果。"""
        record = stage.record
        if stage.skipped_reason:
            return PipelineOutput(
                summary=stage.skipped_reason,
                event_type=record.predicted_event_type,
                possible_causes=[],
                recommendation=[],
                alert_level=AlertSeverity.INFO,
                preliminary_alert_level=stage.preliminary_alert.severity,
                skipped_reason=stage.skipped_reason,
            )

        cropped_frame = stage.cropped_frame
        vlm_observation = stage.vlm_observation

        # 5. Embedding：
        #    - 視覺向量：直接對「裁切後的畫面」算 CLIP/SigLIP embedding，
        #      落在跟 rag/visual_indexer.py 寫入歷史事件時相同的向量空間，
        #      才能做視覺相似度比對（用文字 embedding 去查視覺向量庫，
        #      向量空間、維度都對不上，比對出來沒有意義）。
        #    - 文字向量：「系統描述 + VLM 觀察」組成查詢文字，供知識庫檢索使用。
        visual_query_embedding = self._visual_embedding_provider.embed_image([cropped_frame])[0]
        query_text = f"{record.caption}\n{vlm_observation.visual_description}"

        # 6. Retriever：用視覺向量查詢視覺相似的歷史事件
        similar_cases = self._visual_retriever.retrieve(visual_query_embedding)

        # 7. Knowledge：用文字查詢知識庫（SOP / 歷史事件描述）中的相關片段
        knowledge_snippets = self._text_retriever.retrieve(query_text)

        # 7b. News：用同一份查詢文字查詢新聞真實案例知識庫（獨立 collection，
        #     讓 LLM 可以把「法規依據」跟「真實案例」分開引用，見 rag_prompt.py）
        news_precedents = self._news_retriever.retrieve(query_text)

        retrieved_context = RetrievedContext(
            similar_cases=similar_cases,
            knowledge_snippets=knowledge_snippets,
            news_precedents=news_precedents,
        )

        # 8. LLM：綜合以上所有資訊，產生摘要、驗證判定、可能原因、應變建議
        analysis_result = self._reasoning_pipeline.run(record, vlm_observation, retrieved_context)

        # 9. Alert：純規則邏輯依驗證結論決定警報等級（推論 LLM 的判定只寫進報告）
        alert = self._alert_service.compose(record, analysis_result, vlm_observation)

        logger.info("Orchestrator 完成處理 record_id=%s alert_level=%s", record.record_id, alert.severity.value)

        # 10. Output：組成呼叫端要的精簡結果
        return PipelineOutput(
            summary=analysis_result.summary,
            event_type=record.predicted_event_type,
            possible_causes=analysis_result.possible_causes,
            recommendation=analysis_result.recommended_actions,
            alert_level=alert.severity,
            preliminary_alert_level=stage.preliminary_alert.severity,
            verification_verdict=vlm_observation.verification_verdict.value,
        )

    def _load_context_frames(self, record: RawEventRecord) -> Optional[List[bytes]]:
        """
        讀取 before_frame_file／after_frame_file（若有），依 vlm_view 裁切、
        畫紅框或放大，跟主要畫面一致。

        兩者跟主要畫面共用同一個 spatial_bbox（同機位、同座標，只是時間點
        不同），所以套用一樣的 crop_bbox() 裁切邏輯。目前只有 panic_scatter
        ／counter_flow／sprint_spike 這類需要跨時間比對的事件類型會帶這兩個欄位，其餘
        事件類型這裡會回傳 None，VLM 那端沿用單張畫面的原有流程。
        """
        frame_files = [f for f in (record.before_frame_file, record.after_frame_file) if f]
        if not frame_files:
            return None

        context_frames = []
        for frame_file in frame_files:
            full_frame = self._frame_service.load_context_frame(frame_file, record.record_id)
            context_frames.append(
                self._frame_service.vlm_view_image(full_frame, record.spatial_bbox, self._vlm_view)
            )
        return context_frames
