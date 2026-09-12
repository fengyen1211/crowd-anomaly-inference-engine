"""
utils/logger.py
=================
提供全系統統一的 logger 取得方式。

設計理念：
- 所有模組都應該用 get_logger(__name__) 取得 logger，
  而不是各自 import logging 之後各寫各的設定，
  確保 log 格式、等級在整個系統中一致。

TODO：
- [ ] 依 config/settings.py 的 debug 旗標調整 log level
- [ ] 加上寫入檔案 / 集中式 log 服務（例如 ELK、CloudWatch）的 handler
- [ ] 加上 request_id / trace_id 等結構化欄位，
      方便追蹤單一事件從 ingestion 到 alert 的完整處理流程
"""

import logging


def get_logger(name: str) -> logging.Logger:
    """
    取得指定名稱的 logger（通常傳入 __name__）。

    目前只做最基本的 StreamHandler 設定，
    之後應該擴充為結構化 logging（例如 JSON 格式，方便集中收集）。
    """
    logger = logging.getLogger(name)

    # 避免模組被重複 import 時，重複加入 handler 導致 log 重複輸出
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)  # TODO: 改成讀取 config 的 debug 設定

    return logger
