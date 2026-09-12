"""
database/models.py
=====================
定義資料庫的 ORM Model（使用 SQLAlchemy Declarative Base）。

這裡的欄位設計對應：
- utils.schemas.RawEventRecord（事件原始資料）  -> EventRecordORM
- llm.schemas.AnalysisResult（推理結果）        -> AnalysisResultORM
- utils.schemas.Alert（警報內容）               -> AlertORM
- 人工回饋資料                                  -> FeedbackRecordORM

注意：這裡只定義資料表結構（欄位），不包含任何 CRUD 邏輯，
實際的新增/查詢/更新由各個 service 模組實作。

TODO：
- [ ] 依實際需求調整欄位型態（例如 embedding 資料量大時，
      考慮改用專門的向量資料庫欄位型態，而不是存在關聯式資料庫裡）
- [ ] 加上索引（Index）設計，例如針對 group_id / predicted_event_type 建立索引
- [ ] 改用 Alembic 管理 schema migration，而不是仰賴 create_all()
"""

from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import declarative_base

# 所有 ORM model 的共同基底類別
Base = declarative_base()


class EventRecordORM(Base):
    """對應影像端輸出的原始事件資料表。"""

    __tablename__ = "event_records"

    record_id = Column(String, primary_key=True)
    group_id = Column(String, nullable=False)
    member_ids = Column(JSON, nullable=False, default=list)
    member_count = Column(Integer, nullable=False)
    frame_file = Column(String, nullable=False)
    frame_idx = Column(Integer, nullable=False)
    spatial_bbox = Column(JSON, nullable=False)  # {x1, y1, x2, y2}
    predicted_event_type = Column(String, nullable=False)
    label_source = Column(String, default="rule_based_unvalidated")

    # 注意：這是影像端給的假信心值（目前固定 1.0），
    # 推論端不應直接信任，真正的信心值在 AnalysisResultORM.final_confidence
    confidence = Column(Float, nullable=False)

    collective_motion = Column(JSON, nullable=False)
    caption = Column(Text, nullable=False)

    # 目前是假的隨機向量，未來會替換成真實 CLIP 向量，欄位結構不需改變
    embedding = Column(JSON, nullable=False)
    embedding_model = Column(String, nullable=False)
    embedding_dim = Column(Integer, nullable=False)

    # pipeline 處理狀態：pending / analyzing / done / failed
    status = Column(String, nullable=False, default="pending")

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class AnalysisResultORM(Base):
    """對應 Reasoning Module 產生的推理結果。"""

    __tablename__ = "analysis_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    record_id = Column(String, ForeignKey("event_records.record_id"), nullable=False)

    summary = Column(Text, nullable=False)
    final_verdict = Column(String, nullable=False)  # confirmed / uncertain / overturned
    final_confidence = Column(Float, nullable=False)
    possible_causes = Column(JSON, nullable=False, default=list)
    recommended_actions = Column(JSON, nullable=False, default=list)

    created_at = Column(DateTime, default=datetime.utcnow)


class AlertORM(Base):
    """對應最終要提供給 Web 介面的警報資料。"""

    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    record_id = Column(String, ForeignKey("event_records.record_id"), nullable=False)

    severity = Column(String, nullable=False)  # info / warning / critical
    title = Column(String, nullable=False)
    short_message = Column(Text, nullable=False)
    full_report = Column(Text, nullable=False)
    recommended_actions = Column(JSON, nullable=False, default=list)

    created_at = Column(DateTime, default=datetime.utcnow)


class FeedbackRecordORM(Base):
    """
    人工標註的回饋資料，未來作為 VLM/LLM fine-tune 的資料集來源。
    """

    __tablename__ = "feedback_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    record_id = Column(String, ForeignKey("event_records.record_id"), nullable=False)

    is_correct = Column(Boolean, nullable=False)
    corrected_content = Column(JSON, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
