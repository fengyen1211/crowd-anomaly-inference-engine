"""
routers/alerts.py
====================
警報相關的 API 路由。

- GET /alerts             取得目前警報列表（給 Web 儀表板顯示）
- GET /alerts/{alert_id}  取得單一警報詳細內容

TODO：
- [ ] 支援分頁與依 severity / event_type 篩選
- [ ] 之後可考慮加上 WebSocket 端點，主動推播新警報給 Web，
      避免 Web 需要一直輪詢
"""

from fastapi import APIRouter

router = APIRouter(prefix="/alerts", tags=["alerts"])


@router.get("")
def list_alerts():
    """
    取得警報列表。

    TODO: 從資料庫查詢警報列表，支援分頁/篩選。
    """
    raise NotImplementedError("TODO: 尚未實作 /alerts")


@router.get("/{alert_id}")
def get_alert(alert_id: int):
    """
    取得單一警報詳細內容。

    TODO: 從資料庫查詢單一警報詳細內容。
    """
    raise NotImplementedError("TODO: 尚未實作 /alerts/{alert_id}")
