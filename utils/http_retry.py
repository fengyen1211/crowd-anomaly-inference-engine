"""
utils/http_retry.py
======================
帶重試的 HTTP POST，供 llm/ollama_llm.py 與 vlm/ollama_vlm.py 共用。

為什麼需要這個
--------------
實測案例（2026-09-29，8 筆事件的批次 pipeline）：`crowd_locked_p1` 這一筆
連續兩次在 `OllamaVLM.describe_image()` 逾時失敗（`ReadTimeout`，
read timeout=300.0），單筆事件整個報廢。但直接對同一張裁切圖、同一個
prompt 呼叫 Ollama，只花 14.3 秒就回來，`done_reason` 是 `stop`（自然結束，
不是被截斷），輸出 111 個 token、內容完全正常；第三次再跑完整 pipeline
也順利通過。

也就是說這不是資料問題、不是模型陷入無限生成，而是**狀態相依的暫時性
卡頓**：這台機器只有 4GB VRAM，pipeline 每處理一筆事件都要在
`qwen2.5vl:3b`（VLM，一次）與 `gemma3:4b`（LLM，兩次）之間來回切換，
Ollama 必須反覆把模型換出換入，某次重載撞上了寫死的 300 秒逾時。

原本的實作對這種情況完全沒有防護：單次 `requests.post()`，逾時就往上拋，
呼叫端（OrchestratorService）沒有任何重試，於是一次暫時性卡頓就等於
「這筆事件跑不出結果」。在批次跑實驗時，這會讓整份報告缺一筆。

設計取捨
--------
- **只重試「暫時性」錯誤**：連線失敗、逾時、以及 5xx（伺服器端錯誤）。
  4xx 是呼叫端自己的問題（例如模型名稱打錯、payload 格式不對），重試一百次
  也不會變成功，只會把失敗時間拉長 N 倍、讓錯誤訊息更晚出現。
- **退避時間刻意短**（預設 5s、10s）：這裡要等的不是「對方限流冷卻」，
  而是「Ollama 把模型載進 VRAM」。我們逾時放棄請求之後，Ollama 那邊的
  載入其實還在繼續，所以隔幾秒重試時模型通常已經就緒。指數退避只是
  避免在真的掛掉時狂打。
- **不吃掉最後一次的例外**：重試用完仍然失敗就照原樣往上拋，讓呼叫端
  看到真正的錯誤型別（`ReadTimeout` 等），而不是被包成一個模糊的自訂例外。
"""

import time
from typing import Any, Dict, Optional

import requests

from utils.logger import get_logger

logger = get_logger(__name__)

# 這些是「等一下再試可能就好」的錯誤
_RETRYABLE_EXCEPTIONS = (
    requests.exceptions.Timeout,
    requests.exceptions.ConnectionError,
)


def post_json_with_retry(
    url: str,
    payload: Dict[str, Any],
    timeout: float,
    max_attempts: int = 3,
    backoff_base: float = 5.0,
    context: Optional[str] = None,
) -> requests.Response:
    """
    對 url 送出 JSON POST，遇到暫時性錯誤時自動重試。

    Args:
        url: 目標網址。
        payload: 要送出的 JSON body。
        timeout: 單次請求的逾時秒數（每一次嘗試各自計時，不是總預算）。
        max_attempts: 總嘗試次數（含第一次），預設 3 次＝最多重試 2 次。
        backoff_base: 第一次重試前等待的秒數，之後每次加倍（5s -> 10s）。
        context: 寫進 log 的識別字串（例如 "OllamaVLM model=qwen2.5vl:3b"），
                 方便在批次執行的 log 裡看出是哪個 Provider 在重試。

    Returns:
        requests.Response（已通過 raise_for_status()）。

    Raises:
        最後一次嘗試的原始例外（requests.exceptions.*）。
    """
    label = context or url
    last_exception: Optional[BaseException] = None

    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.post(url, json=payload, timeout=timeout)

            # 5xx 視為暫時性：伺服器端出問題，重試有機會成功。
            # 4xx 不重試，直接讓 raise_for_status() 拋出去。
            if response.status_code >= 500:
                response.raise_for_status()

            response.raise_for_status()
            if attempt > 1:
                logger.info("%s 第 %d 次嘗試成功", label, attempt)
            return response

        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status is None or status < 500:
                raise  # 4xx：呼叫端自己的問題，重試沒有意義
            last_exception = exc

        except _RETRYABLE_EXCEPTIONS as exc:
            last_exception = exc

        if attempt < max_attempts:
            wait = backoff_base * (2 ** (attempt - 1))
            logger.warning(
                "%s 第 %d/%d 次嘗試失敗（%s: %s），%.0f 秒後重試",
                label, attempt, max_attempts,
                type(last_exception).__name__, last_exception, wait,
            )
            time.sleep(wait)
        else:
            logger.error(
                "%s 已重試 %d 次仍然失敗，放棄（%s）",
                label, max_attempts - 1, type(last_exception).__name__,
            )

    assert last_exception is not None  # 迴圈必定會設定它才走到這裡
    raise last_exception
