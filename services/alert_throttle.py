"""
services/alert_throttle.py
=============================
同一支攝影機短時間內的警報合併。

影像端常在同一段異常裡連續觸發好幾筆事件（第二批 UCF 的 Explosion013 有
7 筆、Shooting043 有 6 筆落在標準答案時段外），每筆都發一則警報會讓值班
人員被重複的通知淹沒。這裡記住每支攝影機上次「發出」的警報，之後
ALERT_MERGE_SEC 秒內、等級沒有更高的事件就併進那則警報（前端更新既有
警報，不另外跳通知）；等級升高（warning -> critical）照發。

依據：reports/alert_merge_1117/simulate_merge.py 用 sweep_792767（第二批 99
筆）離線模擬，20 秒時發出的警報 96 -> 51 則（時段外 64 -> 36）；標準答案
時段有警報涵蓋的仍是 20/26 段，被合併的時段都是 4～17 秒前已經從同一支
攝影機發出警報（影像端 DURING 常比標準答案早幾秒開始）。

被合併的事件照樣跑完 VLM 驗證與完整分析，結果寫進 Alert.merged_into 指向
的那則警報，資訊不會遺失；這裡只決定要不要另外通知。

時間用影像端的 frame_idx / VIDEO_FPS（不是處理當下的時鐘時間）：批次跑
歷史影片時事件是連續送進來的，時鐘時間跟影片裡的時間對不上。

攝影機以 group_id 區分。第二批 UCF 的 group_id 是 ucf2_<影片>_during，一支
影片一個；之後影像端若提供攝影機 ID，改 _camera_key() 即可。
"""

import threading
from dataclasses import dataclass
from typing import Dict, Optional

from utils.logger import get_logger
from utils.schemas import AlertSeverity, RawEventRecord

logger = get_logger(__name__)

_RANK = {AlertSeverity.INFO: 0, AlertSeverity.WARNING: 1, AlertSeverity.CRITICAL: 2}


@dataclass
class _OpenAlert:
    record_id: str
    time_sec: float
    severity: AlertSeverity


def _camera_key(record: RawEventRecord) -> str:
    return record.group_id


class AlertThrottle:
    """記住每支攝影機上次發出的警報，決定新事件要另發還是合併。"""

    def __init__(self, window_sec: float, fps: float) -> None:
        self._window_sec = window_sec
        self._fps = fps
        self._open: Dict[str, _OpenAlert] = {}
        # FastAPI 背景任務可能同時處理多筆事件
        self._lock = threading.Lock()

    def decide(self, record: RawEventRecord, severity: AlertSeverity) -> Optional[str]:
        """
        回傳要併入的警報 record_id；None 表示另外發出（並成為這支攝影機目前的警報）。

        info（略過）不通報，也不影響之後的判斷。
        """
        if severity == AlertSeverity.INFO:
            return None
        key = _camera_key(record)
        time_sec = record.frame_idx / self._fps
        with self._lock:
            prev = self._open.get(key)
            if (
                prev is not None
                and 0 <= time_sec - prev.time_sec <= self._window_sec
                and _RANK[severity] <= _RANK[prev.severity]
            ):
                logger.info(
                    "AlertThrottle 合併：record_id=%s 併入 %s（相隔 %.1f 秒）",
                    record.record_id, prev.record_id, time_sec - prev.time_sec,
                )
                return prev.record_id
            self._open[key] = _OpenAlert(record.record_id, time_sec, severity)
            return None

    def release(self, record: RawEventRecord) -> None:
        """這則警報被撤回（驗證判 contradict）時呼叫：之後的事件不再併進一則已撤回的警報。"""
        key = _camera_key(record)
        with self._lock:
            prev = self._open.get(key)
            if prev is not None and prev.record_id == record.record_id:
                del self._open[key]
