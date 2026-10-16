"""
scripts/build_tinyformer_events.py
======================================
把 0901/ 資料夾（影像端新模型 TinyFormer+SAM2 的輸出）轉換成推論端可以吃的
事件 JSON（符合 utils.schemas.RawEventRecord 格式）。

來源資料長什麼樣子、為什麼不能直接 1:1 轉換
--------------------------------------------
0901/tinyformer_events_7.json 的 vlm_visual_prompts 不是「事件列表」，
是逐幀、逐人的原始異常訊號：同一支影片（video 7，DURING 階段，
frame 378-1086，共 23.6 秒）裡，總共有 2,997 筆紀錄，涵蓋 315 個不同的人。
如果每一筆都當成一個獨立事件塞進 RawEventRecord，會產生數百到近三千筆
「事件」，既不符合 RawEventRecord 原本「一筆＝一個群體層級事件」的設計
（一個 bbox、一個 caption、一組 member_ids），也不可能在這台機器的硬體
（VLM/LLM 各要 20-40 秒）上一一跑完整條 pipeline。

實際作法：在「行為類型」與「時間」兩個維度上切分。

★ 為什麼要多切時間這一刀（2026-09-29 修正）★
初版只依行為類型分組，每個類型取全域強度最大的單一實例當代表幀，
產生 4 筆事件。診斷後發現這個假設對 sprint_spike 明顯不成立：它有
2,919 筆訊號、307 個人、橫跨 23.6 秒，其中 z-score >= 10 的強訊號就有
155 筆，散佈整段時間。實際上至少有 4 波各自獨立的高峰，分屬不同追蹤 ID：
    frame  487（16.23s）z=39.2  人物 234   <- 初版只保留了這一筆
    frame  657（21.90s）z=33.9  人物 339
    frame  841（28.03s）z=30.5  人物 470
    frame 1038（34.60s）z=21.3  人物 595
後三筆被 max() 丟棄，VLM 連畫面都沒看到。這不是模型問題、也不是影像端
問題，是聚合步驟把資訊丟掉了。

切分策略（見 scripts/analyze_event_segmentation.py 的三策略比較）：
- 用「峰值偵測 + 最小間距」而不是「時間間隔切段」。後者的前提是訊號流
  有稀疏空檔，但 sprint_spike 的 frame 覆蓋率高達 96.6%、最大空檔只有
  7 幀，任何有意義的間隔門檻都切不開（實測 gap > 10 幀就只剩 1 段）。
- 峰值偵測必須同時設「最小間距」與「強度下限」，見 _PEAK_CONFIG 的註解。
  整個類型的強度都達不到下限時，視為只有底噪，該類型不產生任何事件。
- 沒有強度數值的類型（panic_scatter／counter_flow）無從比較峰值，改用
  「時間間隔切段」（見 _GAP_SPLIT_FRAMES）：這兩類訊號很稀疏，空檔明顯，
  間隔切段對它們有效。代表幀取同一幀涉及人數最多者（見 _pick_densest_frame）。

每一筆事件現在代表「一段時間內的一波異常」，而不是「整個類型的總和」：
- 代表幀（frame_idx／frame_file／spatial_bbox）：該波的峰值實例
  （無強度類型則是該段人數最多的一幀，bbox 取該幀所有訊號的聯集）
- member_ids／member_count：屬於該波的訊號所涉及的不重複人物 ID。
  訊號用「最近峰值」歸屬（見 _partition_by_nearest_peak），間隔切段則
  本來就互斥，所以每一筆原始訊號都會落在且只落在一筆事件底下。
- record_id：同類型切出多波時加上 _p1.._pN 後綴；只有一波時維持原本的
  裸名稱（例如 counter_flow）
- caption：誠實列出「這是第 k 波／共 N 波」與該波自己的統計數字，
  並註明代表幀只是該波的一個瞬間

哪些欄位是真的、哪些是誠實的佔位值
------------------------------------
- frame_file／frame_idx／畫面內容：真的，但**不是**從 0901/tinyformer_dataset_7
  複製（那批是影像端已疊加 bbox／軌跡線／文字標籤的渲染輸出，VLM 會把標註
  誤認成畫面內容，見下方「畫面來源修正」）。實際是從乾淨的
  7_clean/tinyformer_clean_dataset_7/DURING 讀取
- member_ids／spatial_bbox／predicted_event_type：真的，來自 TinyFormer+SAM2
  的實際輸出（聚合方式見上）
- embedding：真的，用 embedding/clip.py 現場算出代表幀的 CLIP 向量
- collective_motion：None——來源資料沒有方向性欄位（TinyFormer 這個逐幀
  偵測格式只給 z-score／密度，不像舊版規則系統會算 avg_direction_deg 等
  群體移動角度），沒有數字就不編造
- confidence：0.0（沿用專案慣例，這個欄位目前系統邏輯不讀取，真正的強度
  資訊已經寫進 caption 讓 LLM 看得到）

畫面來源修正：乾淨幀 + padding 裁切預覽
--------------------------------------------
影像端後來提供了 7_clean/tinyformer_clean_dataset_7/DURING：跟
0901/tinyformer_dataset_7/DURING（疊加標註版）同一份 30fps 輸出、逐幀一一
對應的 frame_id，只是沒有畫 bbox／軌跡線／文字標籤，所以可以直接用
tinyformer_events_7.json 裡的 frame_id 去這個資料夾找對應畫面，不需要
（也不再需要之前用 0901/7.mp4 時做的）24fps/30fps 時間戳換算。

另外，實際送進 VLM 的畫面不是這裡存的完整乾淨幀，而是 pipeline 執行時由
services/frame_service.py::crop_bbox() 依 spatial_bbox 外擴 padding
（3 倍寬高、最小 100px）裁出來的範圍——這一步在系統裡是全域生效的，不需要
在這支腳本裡重做。但為了能像除錯時那樣人工檢查「VLM 實際看到什麼」，
這支腳本另外呼叫同一個 crop_bbox() 存一份 `*_cropped_preview.jpg`
（不會被 pipeline 讀取，純供人工檢視用）。

panic_scatter／counter_flow 的事件前後脈絡畫面
--------------------------------------------
`panic_scatter`（人群四散）、`counter_flow`（逆向流動）本質上是「隨時間
變化」的行為，但 TinyFormer+SAM2 只給 z-score／密度這種瞬時強度數值，
沒有方向角度，VLM 又只看 DURING 單一靜態幀，看不出「四散」「反向」這種
需要前後對照才能判斷的動態。因此這兩類事件額外帶上 `before_frame_file`／
`after_frame_file`：依「每一段自己的時間」取該段開始前、結束後各
_CONTEXT_OFFSET_FRAMES 幀的乾淨畫面（從 BEFORE／DURING／AFTER 合併的索引裡
往事件外側找最接近的一幀；同機位、同 `spatial_bbox` 座標，只是時間點不同），
讓 VLM 能比較事件前中後的畫面差異。初版是整支影片共用一組（DURING 開始前／
結束後），切段後跟第 2、3 段相隔十幾秒，比對失去意義，所以改掉（見
vlm/observation_builder.py 的 prompt 組裝邏輯）。其餘兩類事件
（`sprint_spike`／`crowd_locked`）沒有這個時間比對的需求，維持原本單張
畫面的流程，這兩個欄位留 `None`。

使用方式（在 project/ 目錄下執行）：
    python scripts/build_tinyformer_events.py

輸出：data/tinyformer7_events.json（{"records": [...]} 格式，
可直接被 scripts/run_pipeline.py --input ... --index 0~3 使用）
"""

import io
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

from app.dependencies import get_visual_embedding_client  # noqa: E402
from config.settings import get_settings  # noqa: E402
from services.frame_service import FrameLoaderService  # noqa: E402

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SOURCE_DIR = _PROJECT_ROOT / "0901"
_SOURCE_JSON = _SOURCE_DIR / "tinyformer_events_7.json"
_CLEAN_DATASET_DIR = _PROJECT_ROOT / "7_clean" / "tinyformer_clean_dataset_7"
_CLEAN_FRAMES_DIR = _CLEAN_DATASET_DIR / "DURING"
_BEFORE_FRAMES_DIR = _CLEAN_DATASET_DIR / "BEFORE"
_AFTER_FRAMES_DIR = _CLEAN_DATASET_DIR / "AFTER"
_OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "tinyformer7_events.json"

# 需要「事件前後對照畫面」才能判斷的行為類型，見上方docstring。
_CONTEXT_FRAME_TYPES = {"panic_scatter", "counter_flow"}

_EVENT_TYPE_RE = re.compile(r"^([a-z_]+)(?:\(([a-z]+)=([-0-9.]+)\))?$")
_FRAME_ID_RE = re.compile(r"frame_(\d+)\.jpg$")

# 峰值偵測參數，依「強度指標」而不是「行為類型」設定——因為門檻的尺度是
# 跟著指標走的：z-score 是標準差倍數（實測範圍 1.8-39.2），密度是
# 每平方單位人數（實測範圍 0.15-0.55），同一個數字在兩者代表的意義天差地遠。
#
# (最小間距幀數, 強度下限)
#
# 最小間距 100 幀（約 3.3 秒）：30fps 下，兩波異常若相隔不到 3 秒，
#   當成同一起事件比較合理，也避免同一個人的連續衝刺被切成好幾筆。
#
# 強度下限是必要的，不是可選的過濾器。只設最小間距的話，演算法在「距離
# 已選峰夠遠」的時間區段裡，會把該區段強度最高的訊號收進來——即使那個值
# 只是底噪。實測 sprint_spike：不設下限時 frame 378-387 那一小段因為距離
# 第一個峰剛好 >= 100 幀，會被收成一個「峰」，但它的 z 只有 2 左右，
# 是劇院裡有人正常走動造成的，不構成異常。
#
#   z   >= 10.0：對照實測分布（1.8-39.2），10 以上已是明顯離群
#   den >= 0.2 ：對照實測分布（0.15-0.55），濾掉最低那一檔
_PEAK_CONFIG: Dict[str, Tuple[int, float]] = {
    "z": (100, 10.0),
    "den": (100, 0.2),
}

# 沒有強度數值的類型（panic_scatter／counter_flow）無從找峰值，改用「時間
# 間隔切段」：依 frame 排序後，相鄰訊號相隔超過這個幀數就切成新的一段。
#
# 門檻刻意跟 _PEAK_CONFIG 的最小間距同為 100 幀——兩者表達的是同一件事
# （相隔不到約 3 秒的異常算同一起事件），全系統只有一個「同一起事件」的定義。
#
# 間隔切段對 sprint_spike 無效（frame 覆蓋率 96.6%，沒有空檔可切），但對這兩個
# 類型有效：panic_scatter 覆蓋率只有 2.8%，兩個空檔分別是 204 與 163 幀，
# 門檻在 15-150 幀之間都穩定切出同樣 3 段（見 analyze_event_segmentation.py）。
#
# 刻意不設「最小段落筆數」：強度下限有依據（能區分底噪與異常），筆數沒有——
# counter_flow 全部只有 2 筆訊號，設了筆數門檻整個類型就會消失。
_GAP_SPLIT_FRAMES = 100

# 事件前後脈絡畫面：各段開始前／結束後各推這麼多幀（1 秒）。
# 四散、逆流都是短時間內的變化，推太遠會混進別的事件
# （例如 panic_scatter 第 2 段 frame 657 附近同時有 sprint_spike）。
_CONTEXT_OFFSET_FRAMES = 30

_CAPTION_TEMPLATE = (
    "【TinyFormer+SAM2 影像端偵測，聚合自逐幀原始訊號，非人工標註】"
    "來源影片：video 7，DURING 異常階段（frame 378-1086，約 12.6s-36.2s）。"
    "{intro}"
    "系統標記了 {instance_count} 次逐幀觀測，涉及 {member_count} 個"
    "不重複的追蹤人物 ID（時間跨度 frame {frame_min}-{frame_max}，"
    "約 {ts_min:.2f}s-{ts_max:.2f}s）。"
    "{metric_desc}"
    "本筆畫面（frame_idx={peak_frame}）取的是{peak_desc}，"
    "不代表{scope}整段時間畫面都長這樣，也不代表所有 {member_count} 個相關"
    "人物都同時出現在這張畫面裡——請 VLM 只描述這張畫面實際看到的內容。"
    "TinyFormer+SAM2 目前只輸出 bbox 與強度數值（z-score／密度），沒有輸出"
    "群體移動方向角度，因此 collective_motion 欄位誠實留空，不是遺漏。"
)

# 只有一波時不要講「共 1 波各自獨立的異常」這種贅詞，讀起來像系統壞掉
_INTRO_MULTI = (
    "行為類型「{base_type}」在這段期間內共偵測到 {segment_total} 波各自獨立的"
    "異常（{split_desc}）；本筆是其中第 {segment_index} 波，"
)
_INTRO_SINGLE = "行為類型「{base_type}」在這段期間內，"

_SPLIT_DESC_METRIC = "依峰值強度與時間間距切分"
_SPLIT_DESC_NO_METRIC = f"依訊號之間超過 {_GAP_SPLIT_FRAMES} 幀的時間空檔切分"

# 沒有強度數值的類型（panic_scatter／counter_flow）代表幀是依「同幀涉及
# 人數」選的，不是「強度最高」——這裡如果沿用同一句話，等於在 caption 裡對
# LLM 謊稱有做過強度比較，違背整份 caption 刻意維持的誠實性。
_PEAK_DESC_METRIC = "強度最高的單一瞬間"
_PEAK_DESC_NO_METRIC = (
    "同一幀被標記人數最多的瞬間（此類型來源資料沒有強度數值，以同幀涉及人數"
    "作為替代依據，人數相同時取時間最早者）"
)

_SCOPE_MULTI = "這一波"
_SCOPE_SINGLE = "這個類型"

_METRIC_LABEL = {"z": "z-score（相對正常速度分布的離群程度）", "den": "局部人群密度"}


def _parse_event_type(event_type: str) -> Tuple[str, Optional[str], Optional[float]]:
    match = _EVENT_TYPE_RE.match(event_type)
    if not match:
        return event_type, None, None
    base, metric, value = match.groups()
    return base, metric, (float(value) if value is not None else None)


def _load_source() -> List[Dict[str, Any]]:
    raw = json.loads(_SOURCE_JSON.read_text(encoding="utf-8"))
    prompts = raw["vlm_visual_prompts"]
    for p in prompts:
        base, metric, value = _parse_event_type(p["event_type"])
        p["_base"], p["_metric"], p["_value"] = base, metric, value
    return prompts


def _group_by_category(prompts: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for p in prompts:
        grouped[p["_base"]].append(p)
    return grouped


def _detect_peaks(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    貪婪式峰值偵測：強度由高到低取訊號，只要跟已選的峰都相距
    >= 最小間距就收下，並且要先通過強度下限（見 _PEAK_CONFIG）。

    只處理有強度數值的類型；沒有強度數值的類型走 _segment_by_gap()。

    注意：這段邏輯跟 scripts/analyze_event_segmentation.py 的 detect_peaks()
    刻意各留一份。那支是診斷腳本、會掃多組參數做比較，這支是產生正式資料
    的路徑、參數固定；scripts/ 底下不是 package，互相 import 要動 sys.path，
    為了十幾行邏輯把兩支腳本綁在一起不划算。改動演算法時兩邊都要改。
    """
    scored = [p for p in items if p["_value"] is not None]
    metric = scored[0]["_metric"]
    min_separation, min_value = _PEAK_CONFIG.get(metric, (100, float("-inf")))

    candidates = [p for p in scored if p["_value"] >= min_value]
    if not candidates:
        # 整個類型都達不到強度下限：代表這個類型只有底噪、不構成異常，
        # 回傳空 list 讓 main() 略過這個類型（不產生事件），並印出提示。
        return []

    peaks: List[Dict[str, Any]] = []
    for item in sorted(candidates, key=lambda p: p["_value"], reverse=True):
        if all(abs(item["frame_id"] - p["frame_id"]) >= min_separation for p in peaks):
            peaks.append(item)
    peaks.sort(key=lambda p: p["frame_id"])
    return peaks


def _partition_by_nearest_peak(
    items: List[Dict[str, Any]],
    peaks: List[Dict[str, Any]],
) -> List[List[Dict[str, Any]]]:
    """
    把該類型底下的每一筆原始訊號，指派給 frame 距離最近的峰。

    用途是算出每一波事件的 member_ids／統計數字。採「最近峰值」而不是
    「峰值前後固定視窗」，是為了保證每一筆原始訊號都落在且只落在一筆事件
    底下——固定視窗會讓兩峰之間的訊號無人認領（漏算人數），視窗重疊時
    又會重複計算。

    回傳的 list 跟 peaks 等長、順序一一對應。
    """
    buckets: List[List[Dict[str, Any]]] = [[] for _ in peaks]
    for item in items:
        nearest = min(
            range(len(peaks)),
            key=lambda i: abs(item["frame_id"] - peaks[i]["frame_id"]),
        )
        buckets[nearest].append(item)
    for bucket in buckets:
        bucket.sort(key=lambda p: p["frame_id"])
    return buckets


def _segment_by_gap(
    items: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
    """
    沒有強度數值的類型用的切段：依 frame 排序，相鄰訊號相隔超過
    _GAP_SPLIT_FRAMES 就切成新的一段，每段再用 _pick_densest_frame() 選代表幀。

    回傳 (代表, 各段訊號)，兩個 list 等長、順序一一對應，
    跟 _detect_peaks() + _partition_by_nearest_peak() 的輸出形狀相同。
    """
    ordered = sorted(items, key=lambda p: p["frame_id"])
    buckets: List[List[Dict[str, Any]]] = [[ordered[0]]]
    for item in ordered[1:]:
        if item["frame_id"] - buckets[-1][-1]["frame_id"] > _GAP_SPLIT_FRAMES:
            buckets.append([])
        buckets[-1].append(item)
    return [_pick_densest_frame(bucket) for bucket in buckets], buckets


def _pick_densest_frame(bucket: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    選「同一幀涉及人數最多」的那一幀當代表，人數相同時取最早。

    沒有強度可比，同一時刻被標記的人數是最接近「異常程度」的替代指標；
    而且四散、逆流是群體現象，畫面裡被標記的人越多，VLM 越有機會看出整體動態。

    同一幀可能有多筆訊號（例如 panic_scatter 的 frame 828 有兩群人），
    所以回傳的是合成的代表：bbox 取該幀所有訊號 bbox 的聯集——只取其中一筆
    的框，裁切時另一群人會被切掉。
    """
    by_frame: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for item in bucket:
        by_frame[item["frame_id"]].append(item)

    def member_count(frame_id: int) -> int:
        return len({m for p in by_frame[frame_id] for m in p["member_ids"]})

    best_frame = min(by_frame, key=lambda f: (-member_count(f), f))
    signals = by_frame[best_frame]
    boxes = [p["visual_prompt_bbox"] for p in signals]
    return {
        "frame_id": best_frame,
        "visual_prompt_bbox": [
            min(b[0] for b in boxes),
            min(b[1] for b in boxes),
            max(b[2] for b in boxes),
            max(b[3] for b in boxes),
        ],
        "_metric": None,
        "_value": None,
    }


def _build_caption(
    base_type: str,
    items: List[Dict[str, Any]],
    peak: Dict[str, Any],
    segment_index: int = 1,
    segment_total: int = 1,
) -> str:
    members: set = set()
    for p in items:
        members.update(p["member_ids"])
    frames = sorted(p["frame_id"] for p in items)
    timestamps = sorted(p["timestamp_sec"] for p in items)

    values = [p["_value"] for p in items if p["_value"] is not None]
    if values:
        metric_name = _METRIC_LABEL.get(peak["_metric"], peak["_metric"])
        metric_desc = (
            f"強度指標為{metric_name}，範圍 {min(values):.2f} 至 {max(values):.2f}"
            f"（平均 {sum(values) / len(values):.2f}，本筆取樣的是最高值 {peak['_value']:.2f}）。"
        )
    else:
        metric_desc = "此類型來源資料未提供強度數值，僅有位置與涉及人物紀錄。"

    is_multi = segment_total > 1
    intro = (_INTRO_MULTI if is_multi else _INTRO_SINGLE).format(
        base_type=base_type,
        segment_total=segment_total,
        segment_index=segment_index,
        split_desc=_SPLIT_DESC_METRIC if values else _SPLIT_DESC_NO_METRIC,
    )

    return _CAPTION_TEMPLATE.format(
        intro=intro,
        peak_desc=_PEAK_DESC_METRIC if values else _PEAK_DESC_NO_METRIC,
        scope=_SCOPE_MULTI if is_multi else _SCOPE_SINGLE,
        instance_count=len(items),
        member_count=len(members),
        frame_min=frames[0],
        frame_max=frames[-1],
        ts_min=timestamps[0],
        ts_max=timestamps[-1],
        metric_desc=metric_desc,
        peak_frame=peak["frame_id"],
    )


def _load_clean_frame(peak_frame_id: int) -> bytes:
    """從乾淨（無疊加標註）的 DURING 畫面資料夾讀取跟標註版一一對應的 frame_id。"""
    source_path = _CLEAN_FRAMES_DIR / f"frame_{peak_frame_id:05d}.jpg"
    if not source_path.exists():
        raise SystemExit(f"找不到乾淨代表幀畫面：{source_path}")
    return source_path.read_bytes()


def _index_clean_frames() -> Dict[int, Path]:
    """
    建立 BEFORE／DURING／AFTER 三個資料夾所有乾淨畫面的 frame_id -> 路徑索引。
    三者是同一支影片的連續 frame_id，合在一起才能對任意時間點找最接近的畫面
    （DURING 有 689/709 幀；BEFORE／AFTER 較稀疏，約每 15 幀一張）。
    """
    index: Dict[int, Path] = {}
    for directory in (_BEFORE_FRAMES_DIR, _CLEAN_FRAMES_DIR, _AFTER_FRAMES_DIR):
        for path in directory.glob("frame_*.jpg"):
            if match := _FRAME_ID_RE.search(path.name):
                index[int(match.group(1))] = path
    if not index:
        raise SystemExit(f"找不到任何乾淨畫面檔案：{_CLEAN_DATASET_DIR}")
    return index


def _find_context_frame(index: Dict[int, Path], target: int, before: bool) -> int:
    """
    找事件前／後的脈絡畫面：before=True 取 <= target 的最後一幀，
    before=False 取 >= target 的第一幀。只往事件外側找，確保「前」一定
    早於事件、「後」一定晚於事件；外側完全沒有畫面時才退回全影片的頭／尾。
    """
    if before:
        candidates = [f for f in index if f <= target]
        return max(candidates) if candidates else min(index)
    candidates = [f for f in index if f >= target]
    return min(candidates) if candidates else max(index)


def _save_context_frame(frame_id: int, index: Dict[int, Path], frames_dir: Path) -> str:
    """存事件前／後脈絡畫面；檔名依 frame_id，不同段落用到同一幀時共用一個檔案。"""
    frames_dir.mkdir(parents=True, exist_ok=True)
    dest_filename = f"tinyformer7_context_frame_{frame_id:05d}.jpg"
    (frames_dir / dest_filename).write_bytes(index[frame_id].read_bytes())
    return dest_filename


def _save_peak_frame(base_type: str, peak_frame_id: int, frame_bytes: bytes, frames_dir: Path) -> str:
    frames_dir.mkdir(parents=True, exist_ok=True)
    dest_filename = f"tinyformer7_{base_type}_frame_{peak_frame_id:05d}.jpg"
    (frames_dir / dest_filename).write_bytes(frame_bytes)
    return dest_filename


def _save_cropped_preview(
    base_type: str,
    peak_frame_id: int,
    frame_bytes: bytes,
    bbox: List[float],
    frames_dir: Path,
    frame_loader: FrameLoaderService,
) -> Tuple[str, Tuple[int, int]]:
    """存一份跟 pipeline 執行時同樣邏輯（含 padding）裁切出來的預覽圖，供人工檢查
    「VLM 實際看到的畫面」是否可辨識——不會被 pipeline 讀取。"""
    cropped = frame_loader.crop_bbox(frame_bytes, bbox)
    preview_filename = f"tinyformer7_{base_type}_frame_{peak_frame_id:05d}_cropped_preview.jpg"
    (frames_dir / preview_filename).write_bytes(cropped)
    return preview_filename, Image.open(io.BytesIO(cropped)).size


def main() -> None:
    prompts = _load_source()
    grouped = _group_by_category(prompts)

    frames_dir = Path(get_settings().frames_dir)
    clip = get_visual_embedding_client()
    frame_loader = FrameLoaderService(frames_dir=str(frames_dir))

    if not _CLEAN_FRAMES_DIR.exists():
        raise SystemExit(f"找不到乾淨畫面資料夾：{_CLEAN_FRAMES_DIR}")

    clean_index = _index_clean_frames()

    records = []
    print(f"來源：{_SOURCE_JSON}（共 {len(prompts)} 筆逐幀原始訊號，{len(grouped)} 種行為類型）")
    print(f"乾淨畫面：{_CLEAN_DATASET_DIR}")
    print(
        f"事件前後脈絡畫面（僅 {sorted(_CONTEXT_FRAME_TYPES)} 使用）："
        f"各段開始前／結束後 {_CONTEXT_OFFSET_FRAMES} 幀\n"
    )

    for base_type, items in sorted(grouped.items(), key=lambda kv: -len(kv[1])):
        has_metric = any(p["_value"] is not None for p in items)
        if has_metric:
            peaks = _detect_peaks(items)
            if not peaks:
                metric = next(p["_metric"] for p in items if p["_value"] is not None)
                _, min_value = _PEAK_CONFIG.get(metric, (100, float("-inf")))
                print(
                    f"[{base_type}] 略過：整個類型的強度都低於下限"
                    f"（{metric} < {min_value}），不產生事件。"
                )
                continue
            buckets = _partition_by_nearest_peak(items, peaks)
            method = "峰值偵測"
        else:
            peaks, buckets = _segment_by_gap(items)
            method = f"時間間隔 > {_GAP_SPLIT_FRAMES} 幀"

        segment_total = len(peaks)

        print(f"[{base_type}] {len(items)} 筆原始訊號 -> 切出 {segment_total} 波事件（{method}）")

        for segment_index, (peak, bucket) in enumerate(zip(peaks, buckets), start=1):
            peak_frame_id = peak["frame_id"]

            frame_bytes = _load_clean_frame(peak_frame_id)
            dest_filename = _save_peak_frame(base_type, peak_frame_id, frame_bytes, frames_dir)
            width, height = Image.open(io.BytesIO(frame_bytes)).size
            embedding = clip.embed_image([frame_bytes])[0]

            members = sorted({m for p in bucket for m in p["member_ids"]})
            bbox = [
                max(0.0, min(float(c), dim))
                for c, dim in zip(peak["visual_prompt_bbox"], [width, height, width, height])
            ]

            preview_filename, preview_size = _save_cropped_preview(
                base_type, peak_frame_id, frame_bytes, bbox, frames_dir, frame_loader
            )

            # 只有一波時維持原本的裸 record_id（例如 counter_flow），
            # 讓只切出一段的類型 record_id 不受切段邏輯影響
            suffix = f"_p{segment_index}" if segment_total > 1 else ""
            has_context = base_type in _CONTEXT_FRAME_TYPES
            before_frame_file = after_frame_file = None
            context_note = ""
            if has_context:
                before_id = _find_context_frame(
                    clean_index, bucket[0]["frame_id"] - _CONTEXT_OFFSET_FRAMES, before=True
                )
                after_id = _find_context_frame(
                    clean_index, bucket[-1]["frame_id"] + _CONTEXT_OFFSET_FRAMES, before=False
                )
                before_frame_file = _save_context_frame(before_id, clean_index, frames_dir)
                after_frame_file = _save_context_frame(after_id, clean_index, frames_dir)
                context_note = f"，前後脈絡 frame {before_id}／{after_id}"

            records.append({
                "record_id": f"tinyformer7_{base_type}{suffix}",
                "group_id": "tinyformer7_video7_during",
                "member_ids": members,
                "member_count": len(members),
                "frame_file": dest_filename,
                "frame_idx": peak_frame_id,
                "before_frame_file": before_frame_file,
                "after_frame_file": after_frame_file,
                "spatial_bbox": bbox,
                "predicted_event_type": base_type,
                "label_source": "tinyformer_sam2_zscore_unvalidated",
                "confidence": 0.0,
                "collective_motion": None,
                "caption": _build_caption(base_type, bucket, peak, segment_index, segment_total),
                "embedding": embedding,
                "embedding_model": "openai/clip-vit-base-patch32",
                "embedding_is_simulated": False,
                "embedding_dim": len(embedding),
            })

            metric_note = f"，{peak['_metric']}={peak['_value']:.2f}" if peak["_value"] is not None else ""
            print(
                f"  └ 第 {segment_index}/{segment_total} 波："
                f"{len(bucket)} 筆訊號 / {len(members)} 人{metric_note} -> "
                f"代表幀 frame_{peak_frame_id:05d} -> {dest_filename}"
                f"（裁切預覽 {preview_size[0]}x{preview_size[1]}）{context_note}"
            )

    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _OUTPUT_PATH.write_text(json.dumps({"records": records}, ensure_ascii=False, indent=2), encoding="utf-8")

    relative_output = _OUTPUT_PATH.relative_to(Path(__file__).resolve().parent.parent)
    print(f"\n已寫入 {len(records)} 筆事件：{_OUTPUT_PATH}")
    print(f"\n接著可用：\n  python scripts/run_pipeline.py --input {relative_output} --index 0")


if __name__ == "__main__":
    main()
