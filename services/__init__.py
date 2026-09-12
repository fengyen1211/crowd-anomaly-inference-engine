"""
services package
===================
存放各個業務邏輯服務類別：

- ingestion_service.py    IngestionService：驗證影像端事件資料
- frame_service.py        FrameLoaderService：讀取/裁切畫面
- alert_service.py        AlertComposerService：依規則產生警報（純規則邏輯）
- orchestrator_service.py OrchestratorService：串接完整 pipeline
- schemas.py              PipelineOutput：OrchestratorService 的最終輸出格式
- feedback_service.py     FeedbackService：儲存人工回饋（未來 fine-tune 用）
"""
