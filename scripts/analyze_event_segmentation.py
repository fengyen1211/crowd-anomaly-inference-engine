"""
scripts/analyze_event_segmentation.py
=========================================
診斷腳本：影像端的逐幀訊號流，到底該怎麼切成「事件」？

背景
----
scripts/build_tinyformer_events.py 目前只在「行為類型」一個維度分組，
每個類型取單一峰值當代表幀。實測發現這個假設對 sprint_spike 明顯不成立：
2,919 筆訊號、307 個不重複人物、橫跨 frame 378-1086（約 23.6 秒），
其中 z-score >= 10 的強訊號就有 155 筆、散佈整段時間，至少可辨識出
5 波以上各自獨立的高峰（frame 487 z=39.2 / 597 z=30.4 / 657 z=33.9 /
841 z=30.5 …），分屬不同追蹤 ID。但 max() 只留下全域最大的那一筆，
其餘全部消失，VLM 連畫面都看不到。

這支腳本不修改任何產出，只比較三種切分策略在真實資料上的表現，
用來決定 build_tinyformer_events.py 該往哪個方向改。

三種策略
--------
A. 整體間隔切段（gap-based，對整個行為類型）
   訊號依 frame 排序，相鄰間隔 > gap 就切一段。
   直覺上最簡單，但**前提是訊號流本身有稀疏的空檔**。

B. 逐人間隔切段（per-track gap-based）
   先依 member_id 分群，同一個人連續的訊號算一段。
   最貼近「事件」的直覺：#234 在 16 秒衝刺、#470 在 28 秒衝刺是兩回事。

C. 峰值偵測 + 最小間距（peak-based）
   在強度時間序列上找局部極大值，要求峰與峰至少間隔 min_sep 幀。
   不依賴空檔存在，對連續訊號流也有效。

使用方式（在 project/ 目錄下執行）：
    python scripts/analyze_event_segmentation.py
    python scripts/analyze_event_segmentation.py --type sprint_spike --verbose
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SOURCE_JSON = _PROJECT_ROOT / "0901" / "tinyformer_events_7.json"

_EVENT_TYPE_RE = re.compile(r"^([a-z_]+)(?:\(([a-z]+)=([-0-9.]+)\))?$")

# 要測試的 gap 門檻（單位：幀。來源影片 30fps，所以 30 幀 = 1 秒）
_GAP_THRESHOLDS = (2, 5, 10, 15, 30, 60, 90, 150)

# 策略 C 的最小峰間距（幀）
_PEAK_MIN_SEPARATIONS = (30, 60, 100, 150)


def load_signals() -> Dict[str, List[dict]]:
    """讀取原始逐幀訊號，解析 event_type 並依行為類型分組。"""
    raw = json.loads(_SOURCE_JSON.read_text(encoding="utf-8"))
    grouped: Dict[str, List[dict]] = defaultdict(list)

    for item in raw["vlm_visual_prompts"]:
        match = _EVENT_TYPE_RE.match(item["event_type"])
        if match:
            base, metric, value = match.groups()
        else:
            base, metric, value = item["event_type"], None, None
        item["_base"] = base
        item["_metric"] = metric
        item["_value"] = float(value) if value is not None else None
        grouped[base].append(item)

    for items in grouped.values():
        items.sort(key=lambda i: i["frame_id"])
    return grouped


# ---------------------------------------------------------------------------
# 策略 A：整體間隔切段
# ---------------------------------------------------------------------------

def segment_by_gap(items: List[dict], gap: int) -> List[List[dict]]:
    """訊號依 frame 排序後，相鄰 frame 間隔 > gap 就切開。"""
    if not items:
        return []

    segments: List[List[dict]] = [[items[0]]]
    for prev, curr in zip(items, items[1:]):
        if curr["frame_id"] - prev["frame_id"] > gap:
            segments.append([curr])
        else:
            segments[-1].append(curr)
    return segments


def describe_gaps(items: List[dict]) -> Tuple[List[int], int, int]:
    """
    回傳 (所有非零的相鄰 frame 間隔, 不重複 frame 數, frame 跨度)。

    這是判斷策略 A 可不可行的關鍵診斷：如果訊號流每一幀都有訊號，
    間隔永遠是 0 或 1，不論 gap 門檻設多小都切不開。
    """
    frames = sorted({i["frame_id"] for i in items})
    gaps = [b - a for a, b in zip(frames, frames[1:]) if b - a > 1]
    span = (frames[-1] - frames[0]) if frames else 0
    return gaps, len(frames), span


# ---------------------------------------------------------------------------
# 策略 B：逐人間隔切段
# ---------------------------------------------------------------------------

def segment_by_track(items: List[dict], gap: int) -> List[List[dict]]:
    """
    先依 member_id 分群，再對每個人各自做間隔切段。

    一筆訊號可能帶多個 member_id（群體類型），這裡讓該筆訊號同時歸屬
    到每個人底下——目的是估算「有幾段獨立的個體異常」，不是做互斥分割。
    """
    by_track: Dict[int, List[dict]] = defaultdict(list)
    for item in items:
        for member_id in item["member_ids"]:
            by_track[member_id].append(item)

    segments: List[List[dict]] = []
    for track_items in by_track.values():
        track_items.sort(key=lambda i: i["frame_id"])
        segments.extend(segment_by_gap(track_items, gap))
    return segments


# ---------------------------------------------------------------------------
# 策略 C：峰值偵測 + 最小間距
# ---------------------------------------------------------------------------

def detect_peaks(
    items: List[dict],
    min_separation: int,
    min_value: Optional[float] = None,
) -> List[dict]:
    """
    貪婪式峰值偵測：強度由高到低取訊號，只要跟已選的峰都相距
    >= min_separation 幀就收下。

    比「找局部極大值」穩健——局部極大值在雜訊多的序列上會爆量，
    這個做法直接保證「選出來的峰彼此在時間上分得開」。
    沒有強度數值的類型回傳空 list（無從比較）。

    min_value 是必要的搭配參數，不是可有可無的過濾器。只設
    min_separation 的話，演算法在「距離已選峰夠遠」的時間區段裡，
    會把該區段強度最高的訊號收進來——即使那個值只有 z≈2（劇院裡
    有人正常走動的底噪）。手算實測：sprint_spike 訊號跨 frame
    378-1086，取完 487／657／841 之後，frame 378-387 那一小段因為
    距離 487 剛好 >= 100 幀，會被收成第四個「峰」，但它的 z 值根本
    構不成異常。加上強度下限才能把這種邊界雜訊擋掉。
    """
    scored = [i for i in items if i["_value"] is not None]
    if min_value is not None:
        scored = [i for i in scored if i["_value"] >= min_value]
    if not scored:
        return []

    peaks: List[dict] = []
    for item in sorted(scored, key=lambda i: i["_value"], reverse=True):
        if all(abs(item["frame_id"] - p["frame_id"]) >= min_separation for p in peaks):
            peaks.append(item)
    peaks.sort(key=lambda i: i["frame_id"])
    return peaks


# ---------------------------------------------------------------------------
# 報告
# ---------------------------------------------------------------------------

def summarize_segment(seg: List[dict]) -> str:
    frames = [i["frame_id"] for i in seg]
    members = set()
    for i in seg:
        members.update(i["member_ids"])
    values = [i["_value"] for i in seg if i["_value"] is not None]
    peak = f"峰值 {max(values):.1f}" if values else "無強度數值"
    return (
        f"frame {frames[0]:4d}-{frames[-1]:4d}（{len(seg):4d} 筆訊號，"
        f"{len(members):3d} 人，{peak}）"
    )


def report_type(base: str, items: List[dict], verbose: bool) -> None:
    gaps, distinct_frames, span = describe_gaps(items)
    members = set()
    for i in items:
        members.update(i["member_ids"])

    print("=" * 78)
    print(f"行為類型：{base}")
    print("=" * 78)
    print(f"  訊號總數      {len(items)}")
    print(f"  不重複人物    {len(members)}")
    print(f"  frame 跨度    {items[0]['frame_id']} - {items[-1]['frame_id']}（共 {span + 1} 幀）")
    print(f"  有訊號的幀數  {distinct_frames}（覆蓋率 {distinct_frames / (span + 1) * 100:.1f}%）")
    print(f"  平均每幀訊號  {len(items) / distinct_frames:.1f} 筆")

    if gaps:
        print(f"  frame 空檔    共 {len(gaps)} 處，最大 {max(gaps)} 幀，"
              f"其中 >= 30 幀（1秒）的有 {sum(1 for g in gaps if g >= 30)} 處")
    else:
        print("  frame 空檔    無（每一幀都有訊號）")

    # ---------- 策略 A ----------
    print("\n  [策略 A] 整體間隔切段")
    for gap in _GAP_THRESHOLDS:
        segs = segment_by_gap(items, gap)
        note = "  <-- 切不開" if len(segs) == 1 else ""
        print(f"    gap > {gap:3d} 幀（{gap / 30:.1f}s）  ->  {len(segs):4d} 段{note}")

    # ---------- 策略 B ----------
    print("\n  [策略 B] 逐人間隔切段")
    for gap in (15, 30, 60):
        segs = segment_by_track(items, gap)
        multi = sum(1 for s in segs if len(s) >= 3)
        print(f"    gap > {gap:3d} 幀（{gap / 30:.1f}s）  ->  {len(segs):4d} 段"
              f"（其中 >= 3 筆訊號的有 {multi} 段）")

    # ---------- 策略 C ----------
    # 同時掃「最小間距」與「強度下限」兩個維度，因為只調間距會在訊號
    # 稀疏的邊界區段收進底噪（見 detect_peaks() 的 docstring）。
    all_values = [i["_value"] for i in items if i["_value"] is not None]
    if all_values:
        print("\n  [策略 C] 峰值偵測 + 最小間距（欄＝強度下限）")
        thresholds = [None, 5.0, 10.0, 20.0]
        header = "".join(f"{('z>=' + str(t)) if t else '不設限':>10s}" for t in thresholds)
        print(f"    {'間距':>12s}{header}")
        for min_sep in _PEAK_MIN_SEPARATIONS:
            cells = "".join(f"{len(detect_peaks(items, min_sep, t)):>10d}" for t in thresholds)
            print(f"    {str(min_sep) + ' 幀':>12s}{cells}")
    else:
        print("\n  [策略 C] 峰值偵測 -> 此類型無強度數值，不適用")

    if verbose:
        peaks = detect_peaks(items, 100, 10.0)
        if peaks:
            print("\n  [明細] 策略 C 間距 >= 100 幀、z >= 10 選出的峰：")
            for rank, p in enumerate(peaks, start=1):
                print(f"    {rank:2d}. frame {p['frame_id']:4d}"
                      f"（t={p['timestamp_sec']:6.2f}s）"
                      f" {p['_metric']}={p['_value']:6.2f}"
                      f"  人物={p['member_ids']}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="比較三種事件切分策略在真實訊號上的表現")
    parser.add_argument("--type", help="只分析單一行為類型（預設全部）")
    parser.add_argument("--verbose", action="store_true", help="印出峰值明細")
    args = parser.parse_args()

    grouped = load_signals()
    total = sum(len(v) for v in grouped.values())
    print(f"來源：{_SOURCE_JSON.name}，共 {total} 筆逐幀訊號，{len(grouped)} 種行為類型\n")

    targets = [args.type] if args.type else sorted(grouped, key=lambda k: -len(grouped[k]))
    for base in targets:
        if base not in grouped:
            print(f"找不到行為類型 '{base}'，可選：{sorted(grouped)}")
            continue
        report_type(base, grouped[base], args.verbose)


if __name__ == "__main__":
    main()
