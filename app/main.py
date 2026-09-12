"""
app/main.py
=============
FastAPI 應用程式進入點。

負責：
- 建立 FastAPI app 實例
- 掛載所有 routers（events / alerts / feedback）
- 註冊 startup / shutdown 事件（例如初始化資料庫、向量資料庫連線）

執行方式（開發模式，於 project/ 目錄下執行）：
    uvicorn app.main:app --reload

TODO：
- [ ] 加上 CORS middleware（Web 前端跨網域呼叫需要）
- [ ] 加上全域例外處理（統一錯誤回應格式）
- [ ] 在 startup 事件中呼叫 DatabaseManager.init_db()
- [ ] 加上健康檢查端點 /health
"""

from fastapi import FastAPI

from config.settings import get_settings
from routers import alerts, events, feedback

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
)

# 掛載各個功能模組的路由
app.include_router(events.router)
app.include_router(alerts.router)
app.include_router(feedback.router)


@app.on_event("startup")
def on_startup() -> None:
    """
    應用程式啟動時執行的初始化流程。

    TODO:
        1. 初始化資料庫連線（DatabaseManager().init_db()）
        2. 初始化向量資料庫連線（ChromaVectorStore）
        3. 視需要預先載入知識庫索引
    """
    pass


@app.on_event("shutdown")
def on_shutdown() -> None:
    """
    應用程式關閉時執行的清理流程。

    TODO: 釋放資料庫連線、向量資料庫連線等資源。
    """
    pass
