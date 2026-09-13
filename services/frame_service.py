"""
services/frame_service.py
============================
對應架構設計中的 [2] Frame/Evidence Loader。

負責：
- 依 frame_file，從 frames_dir（見 config.settings.frames_dir）讀取
  對應的畫面檔案
- 依 spatial_bbox 裁切出感興趣區域（給 VLM 使用）

TODO：
- [ ] 確認影像端實際交付畫面的方式：目前假設 frame_file 是已經存在於
      frames_dir 底下的圖片檔案；如果之後是直接從影片檔案依 frame_idx
      擷取，這裡的 load_full_frame() 需要改用 OpenCV/ffmpeg 之類的工具
- [ ] 決定圖片快取策略（避免重複讀取同一個檔案）
"""

import io
from pathlib import Path
from typing import List, Optional

from PIL import Image

from config.settings import Settings, get_settings
from utils.logger import get_logger
from utils.schemas import RawEventRecord

logger = get_logger(__name__)


class FrameLoaderService:
    """
    負責讀取與裁切事件對應的畫面。
    """

    def __init__(self, frames_dir: Optional[str] = None, settings: Optional[Settings] = None) -> None:
        """
        Args:
            frames_dir: 畫面檔案所在目錄，預設讀取 Settings.frames_dir。
            settings: 系統設定物件。
        """
        settings = settings or get_settings()
        self._frames_dir = Path(frames_dir or settings.frames_dir)

    def load_full_frame(self, event_record: RawEventRecord) -> bytes:
        """
        依 frame_file 讀取完整原始畫面。

        Raises:
            FileNotFoundError: frames_dir 底下找不到對應檔案時，
                               明確報錯而不是靜默回傳空資料。
        """
        frames_dir = self._frames_dir.resolve()
        frame_path = (frames_dir / event_record.frame_file).resolve()

        # frame_file 是從外部 JSON（/events/ingest 是公開端點，未驗證身份）
        # 直接吃進來的字串，未經檢查。pathlib 的 "/" 對絕對路徑會直接蓋掉
        # frames_dir（例如 frame_file="/etc/passwd" 會讓 frame_path 變成
        # "/etc/passwd"），".." 相對路徑穿越同樣有效。用 resolve() 正規化後
        # 檢查是否仍落在 frames_dir 底下，避免任意檔案讀取。
        if not frame_path.is_relative_to(frames_dir):
            raise FileNotFoundError(
                f"畫面檔案路徑不合法（超出 frames_dir 範圍）：{event_record.frame_file}"
                f"（record_id={event_record.record_id}）"
            )

        if not frame_path.exists():
            raise FileNotFoundError(
                f"找不到畫面檔案：{frame_path}"
                f"（record_id={event_record.record_id}，frames_dir={frames_dir}）"
            )

        logger.info("FrameLoaderService 讀取畫面：%s", frame_path)
        return frame_path.read_bytes()

    def crop_bbox(
        self,
        full_frame: bytes,
        bbox: List[float],
        padding_ratio: float = 3.0,
        min_padding_px: int = 100,
    ) -> bytes:
        """
        依 bbox 座標裁切出感興趣區域，並向外加上 padding。

        bbox 是影像端給的「單一人物追蹤框」，緊貼身形通常只有幾十像素見方
        （實測 35x46px），裁出來幾乎只剩一團模糊人影，VLM 完全看不出群體
        脈絡（走道、周圍人群等）。系統要驗證的是「群體異常」而非單一個人
        姿態，因此外擴一圈，讓 VLM 能看到人物所在的周邊場景。

        Args:
            full_frame: 完整畫面的圖片二進位資料。
            bbox: 扁平陣列 [x1, y1, x2, y2]（對應影像端實際輸出格式，
                  見 utils/schemas.py 的 RawEventRecord.spatial_bbox）。
            padding_ratio: 向外擴張的比例，乘上 bbox 寬/高（各邊分別計算）。
            min_padding_px: 最小擴張像素數，避免 bbox 本身就很小時
                             （如本例）padding 仍然太小。

        Returns:
            bytes: 裁切後的圖片二進位資料（維持原圖格式）。
        """
        image = Image.open(io.BytesIO(full_frame))
        img_w, img_h = image.size
        x1, y1, x2, y2 = bbox

        pad_x = max(min_padding_px, (x2 - x1) * padding_ratio)
        pad_y = max(min_padding_px, (y2 - y1) * padding_ratio)

        px1 = max(0, int(x1 - pad_x))
        py1 = max(0, int(y1 - pad_y))
        px2 = min(img_w, int(x2 + pad_x))
        py2 = min(img_h, int(y2 + pad_y))

        cropped = image.crop((px1, py1, px2, py2))

        buffer = io.BytesIO()
        cropped.save(buffer, format=image.format or "JPEG")
        logger.info(
            "FrameLoaderService 裁切完成：bbox=%s -> padded_box=%s（size=%s）",
            bbox, [px1, py1, px2, py2], cropped.size,
        )
        return buffer.getvalue()
