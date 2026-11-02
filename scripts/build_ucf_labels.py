"""
scripts/build_ucf_labels.py
=============================
把 UCF-Crime 非群聚資料的「標準答案時段」（每支影片一段秒數）轉成
scripts/evaluate_against_labels.py 吃的「每筆事件一個標註」。

為什麼需要轉換
--------------
video 7 的標註是人工一筆一筆看 record 標的（data/labels_video7*.json）。
UCF 資料只有影片層級的答案：例如 Explosion017 的爆炸發生在 33–48 秒。
scripts/build_tinyformer_events.py --ucf-root 把影像端的逐幀訊號切成一波
一波的事件（record），每筆有一張代表幀（frame_idx）——那張就是 VLM 實際
看到的畫面。所以每筆事件的答案取決於代表幀落在哪：

    代表幀在標準答案時段內 -> occurred_scene = yes，danger 依影片類別
    代表幀在時段外         -> occurred_scene = no， danger = none（影像端誤報）

只有時間、沒有位置
------------------
標準答案只有時段，沒有框，判斷不了「框內的人」對不對，所以
occurred_box 一律是 unknown，評估時 --target box 不會計入驗證判斷，
請用 --target scene（預設）。

通報程度是假設，不是標準答案
----------------------------
UCF 只有類別，沒有危險程度。_DANGER_BY_CATEGORY 是推論端自己訂的對應，
改這張表就能換評分標準。注意目前 services/alert_service.py 只有 panic 類
能升到 critical，所以標成 danger 的 fight／sprint_spike 最多拿到 warning，
在評估裡會算漏報。

標準答案來源（同 非群聚資料_給推論端/README.md）：Fighting006、Assault005、
Assault008 是影像端手標，其餘是 UCF-Crime 官方時段標註。

使用方式（在 project/ 目錄下執行）：
    python scripts/build_ucf_labels.py
    python scripts/evaluate_against_labels.py data/labels_ucf_gt.json \
        reports/sweep_<jobid> logs/model-sweep_<jobid>.out
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple

_PROJECT_DIR = Path(__file__).resolve().parent.parent
_DEFAULT_EVENTS = _PROJECT_DIR / "data" / "ucf_noncluster_events.json"
_DEFAULT_OUTPUT = _PROJECT_DIR / "data" / "labels_ucf_gt.json"
_FPS = 30  # README：UCF 影片都是 30 fps，frame_XXXXX 編號 = 秒數 × 30

# 標準答案時段（秒），抄自 非群聚資料_給推論端/README.md
_GROUND_TRUTH: Dict[str, Tuple[float, float]] = {
    "Assault008": (3.0, 21.0),
    "Arrest001": (39.5, 49.5),
    "Fighting006": (15.0, 25.5),
    "Assault011": (12.5, 32.0),
    "Assault005": (12.0, 34.0),
    "Shooting043": (31.5, 41.0),
    "Fighting003": (60.7, 103.4),
    "Fighting018": (2.7, 14.0),
    "Arrest007": (51.0, 72.0),
    "Shoplifting029": (34.0, 49.0),
    "Explosion020": (2.0, 9.0),
    "Explosion017": (33.0, 48.0),
}

# 影片類別 -> 事件發生時的通報程度（none／attention／danger），見上方說明
_DANGER_BY_CATEGORY = {
    "Shooting": "danger",
    "Explosion": "danger",
    "Assault": "danger",
    "Fighting": "danger",
    "Arrest": "attention",
    "Shoplifting": "attention",
}

_GROUP_RE = re.compile(r"^ucf_(.+)_during$")


def _video_of(record: dict) -> Optional[str]:
    m = _GROUP_RE.match(record.get("group_id", ""))
    return m.group(1) if m else None


def main() -> None:
    parser = argparse.ArgumentParser(description="UCF 標準答案時段 -> 每筆事件標註")
    parser.add_argument("--events", type=Path, default=_DEFAULT_EVENTS, help="build_tinyformer_events.py --ucf-root 的輸出")
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    args = parser.parse_args()

    records = json.loads(args.events.read_text(encoding="utf-8"))["records"]

    labels = []
    per_video: Dict[str, Counter] = defaultdict(Counter)
    skipped = []
    for r in records:
        video = _video_of(r)
        if video is None or video not in _GROUND_TRUTH:
            skipped.append(r["record_id"])
            continue
        start, end = _GROUND_TRUTH[video]
        category = re.sub(r"\d+$", "", video)
        frame = r["frame_idx"]
        sec = frame / _FPS
        inside = start * _FPS <= frame <= end * _FPS

        labels.append({
            "record_id": r["record_id"],
            "event_type": r["predicted_event_type"],
            "occurred_box": "unknown",
            "occurred_scene": "yes" if inside else "no",
            "danger": _DANGER_BY_CATEGORY.get(category, "attention") if inside else "none",
            "note": (
                f"代表幀 frame {frame}（{sec:.1f}s）"
                f"{'在' if inside else '不在'}標準答案 {start}–{end}s（{category}）內"
            ),
        })
        per_video[video]["yes" if inside else "no"] += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "video": "ucf_noncluster",
                "method": "auto from ground-truth time windows (README); representative frame inside window = occurred",
                "danger_by_category": _DANGER_BY_CATEGORY,
                "labels": labels,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"{'影片':<16}{'標準答案(秒)':<14}{'在時段內':>8}{'時段外':>8}")
    for video in sorted(per_video):
        c = per_video[video]
        start, end = _GROUND_TRUTH[video]
        print(f"{video:<16}{f'{start}-{end}':<14}{c['yes']:>8}{c['no']:>8}")
    total = Counter()
    for c in per_video.values():
        total.update(c)
    print(f"\n共 {len(labels)} 筆：時段內 {total['yes']}、時段外（影像端誤報）{total['no']}")
    if skipped:
        print(f"沒有標準答案、略過 {len(skipped)} 筆：{skipped[:5]}{' …' if len(skipped) > 5 else ''}")
    print(f"已寫入：{args.output}")


if __name__ == "__main__":
    main()
