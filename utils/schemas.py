"""
utils/schemas.py
==================
定義「影像端輸出 JSON」對應的內部資料結構（Pydantic Models）。

為什麼放在 utils 而不是各自模組內？
- 這些是「跨模組共用」的基礎資料結構：Ingestion、VLM、RAG、LLM、
  Alert 等模組都需要讀取或參考同一份原始事件資料，
  集中定義在這裡，避免各模組各自定義造成不一致。

對應文件：rag_vlm_mock_dataset_README.md

TODO：
- [ ] 確認影像端 spatial_bbox 的實際座標格式定義
      （目前先假設為左上/右下座標 [x1, y1, x2, y2]）
- [ ] 依影像端實際輸出，補齊 CollectiveMotion 的其餘欄位
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class PredictedEventType(str, Enum):
    """
    影像端規則系統判斷出的「已知」異常事件類型，供規則判斷（例如
    services/alert_service.py 的嚴重程度分級）引用具名常數用。

    注意：RawEventRecord.predicted_event_type 欄位本身是開放字串，
    不強制檢查只能是這個列舉裡的值——影像端的分類法還在演進中
    （例如較新的逐幀偵測格式用的是 sprint_spike／crowd_locked，
    跟這裡列的三種完全不同），封裝成封閉列舉會導致每次影像端新增
    類型，推論端就要跟著改一次程式碼才能吃資料。這個列舉純粹是
    「已知常數」的集合，不是欄位型別的驗證依據。
    """

    PANIC_DISPERSAL = "panic_dispersal"        # 恐慌性移動：突然加速四散
    REVERSE_FLOW = "reverse_flow"              # 逆向流動：跟大部分人反方向走
    ABNORMAL_GATHERING = "abnormal_gathering"  # 異常聚集：一群人越靠越近、停下來


class CollectiveMotion(BaseModel):
    """
    群體移動的統計特徵，對應影像端 JSON 的 collective_motion 欄位。
    這些數字是從真實座標計算出來的，是可信的資料
    （跟 predicted_event_type 不同，這裡不是規則猜測，是數學計算結果）。

    欄位名稱已對照 rag_vlm_mock_dataset.json 的實際輸出核對過。
    """

    avg_direction_deg: float = Field(..., description="群體平均移動方向（角度）")
    majority_flow_direction_deg: float = Field(..., description="場景主流移動方向（角度）")
    direction_deviation_deg: float = Field(..., description="群體方向與主流方向的角度差")
    density_trend: str = Field(..., description="密度變化趨勢，例如 increasing/decreasing/stable")

    # TODO: 依實際影像端輸出格式，補上平均速度、加速度等欄位


class RawEventRecord(BaseModel):
    """
    影像端輸出 JSON 中，單一筆 record 的完整結構。

    這是整個推論端 pipeline 的「起點資料」，
    後續 VLM / RAG / LLM 模組的輸出都會參考或引用這裡的欄位。
    """

    record_id: str
    group_id: str
    member_ids: List[int] = Field(default_factory=list)
    member_count: int
    frame_file: str
    frame_idx: int

    # 注意：影像端實際輸出的是扁平陣列 [x1, y1, x2, y2]，不是具名物件，
    # 已對照 rag_vlm_mock_dataset.json 核對過。
    spatial_bbox: List[float] = Field(..., min_length=4, max_length=4)

    # 開放字串，不是封閉列舉——理由見 PredictedEventType 的 docstring。
    predicted_event_type: str
    label_source: str = "rule_based_unvalidated"

    # 注意：目前 confidence 是假值（影像端寫死 1.0），
    # 推論端不應該直接信任這個數字，而是要透過 VLM 驗證 + LLM 推理
    # 重新產生 final_confidence（見 llm/schemas.py 的 AnalysisResult）。
    # 部分來源（例如逐幀偵測格式）根本不提供這個欄位，且目前系統邏輯
    # 沒有任何地方讀取它，給預設值即可。
    confidence: float = 0.0

    # 個體層級事件（例如單人速度異常尖峰）沒有「群體」可言，
    # 自然也沒有群體移動統計，因此允許 None。
    collective_motion: Optional[CollectiveMotion] = None
    caption: str  # 目前是模板組字，不是 AI 生成

    # 視覺特徵向量，目前是假的隨機向量，未來會替換成真實 CLIP 向量。
    # 推論端的程式邏輯不應該因為向量從假換成真而需要修改。
    # 目前系統邏輯沒有任何地方讀取這幾個欄位，部分來源也不提供，給預設值即可。
    embedding: List[float] = Field(default_factory=list)
    embedding_model: str = ""
    # 明確標記這筆 embedding 是否為模擬值（見 rag_vlm_mock_dataset_README.md）
    embedding_is_simulated: bool = True
    embedding_dim: int = 0


class AlertSeverity(str, Enum):
    """警報等級，由 Alert Composer 模組（純規則邏輯）決定。"""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class Alert(BaseModel):
    """
    最終要提供給 Web 介面顯示的警報內容。
    由 services/alert_service.py 的 AlertComposerService 產生。
    """

    record_id: str
    severity: AlertSeverity
    title: str
    short_message: str
    full_report: str
    recommended_actions: List[str] = Field(default_factory=list)
    frame_file: str

    # TODO: 補上 alert_id、created_at、是否已讀/已處理 等欄位
