"""
app/security.py
==================
對外 API 的身份驗證。

背景：/events/ingest 會觸發真的 VLM/LLM 運算（單次 70-90 秒），且此前
完全沒有任何身份驗證機制——任何人都能無限次呼叫，既可能被拿來耗盡運算
資源（資源耗盡型 DoS），也讓沒有驗證身份的呼叫者可以透過 caption 等
自由文字欄位嘗試 prompt injection（見 README「安全性注意事項」）。

這裡先做最簡單、足夠擋掉「隨便亂打」的機制：單一共用 API Key，
從 Header `X-API-Key` 帶入。不是完整的多租戶／權限系統，只是把
「完全開放」變成「至少要知道一組 Key 才能呼叫」。

使用方式：在 router 上掛 `dependencies=[Depends(verify_api_key)]`，
FastAPI 會在進入實際的 endpoint 邏輯之前先執行這裡的檢查。
"""

import secrets

from fastapi import Header, HTTPException, status

from config.settings import get_settings


def verify_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    """
    驗證呼叫端帶的 API Key 是否正確。

    Raises:
        HTTPException 500: 伺服器沒有設定 INFERENCE_API_KEY——拒絕在
            「形同虛設的驗證機制」下對外服務，而不是安靜地放行所有請求。
        HTTPException 401: 呼叫端沒帶 Key，或帶的 Key 不正確。
    """
    settings = get_settings()
    if not settings.inference_api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="伺服器未設定 INFERENCE_API_KEY，拒絕在沒有驗證機制的情況下對外服務。",
        )

    # 用 secrets.compare_digest 做固定時間比對，避免字元一個一個比對時
    # 因為提早跳出造成的執行時間差異，被拿來推測 Key 內容（timing attack）。
    if not x_api_key or not secrets.compare_digest(x_api_key, settings.inference_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="缺少或不正確的 API Key（請在 X-API-Key header 帶入）。",
        )
