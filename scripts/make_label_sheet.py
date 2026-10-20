"""
scripts/make_label_sheet.py
==============================
產生人工標註用的本機網頁：每筆事件先放一段峰值前後各 5 秒的影片片段
（從原始影片 0901/7.mp4 剪出、標出事件位置、可以慢速播放），再並排峰值前後
1 秒內 5 個時間點的畫面，讓人判斷「系統說的行為是否真的發生」與「實際
危險程度」，標註結果在網頁上下載成 JSON。

第一版只有 5 張畫面。標註者回饋：只看靜止畫面，人也很難判斷速度與方向，
所以加上影片片段；結果另存 labels_video7_clips.json，不覆蓋第一版。

「有沒有發生」分成兩題：框內（occurred_box，檢驗影像端在那個位置的偵測對
不對）與整體畫面（occurred_scene，檢驗這段時間現場該不該發警報）。第一版只問
框內；sprint_p4 這種框到影子、但旁邊可能真的有人在跑的情況，兩題答案會不同。

幀編號：資料集的 frame_id 以 30fps 計（frame_id / 30 = 秒），原始影片是
24fps，影片幀 = frame_id × 0.8（frame 657 對應影片第 525 幀，已用畫面比對確認）。

為什麼需要：到 sweep_566003 為止，模型組合、prompt、驗證模型的比較都沒有
標準答案，只能靠讀描述文字推測誰對誰錯；這份標註就是之後算準確率的依據。

畫面直接取自乾淨畫面資料夾（BEFORE／DURING／AFTER，同 build_tinyformer_events.py），
不受 pipeline 實際餵給 VLM 的幀數限制；標註頁不顯示任何模型輸出，避免先入為主。

使用方式（在 project/ 目錄下執行）：
    python scripts/make_label_sheet.py
然後用瀏覽器開啟 reports/label_sheet/index.html（reports/ 不進版控）。
"""

import json
import shutil
import sys
import tempfile

import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw  # noqa: E402

from scripts.build_tinyformer_events import _find_context_frame, _index_clean_frames  # noqa: E402

_PROJECT_DIR = Path(__file__).resolve().parent.parent
_EVENTS_PATH = _PROJECT_DIR / "data" / "tinyformer7_events.json"
_VIDEO_PATH = _PROJECT_DIR.parent / "0901" / "7.mp4"
# 資料集 frame_id 的時間基準（見檔案頂端說明）
_DATASET_FPS = 30
_CLIP_HALF_SEC = 5
_OUTPUT_DIR = _PROJECT_DIR / "reports" / "label_sheet"

# 峰值前後的幀數偏移：影片 30fps，等於 -1、-0.5、0、+0.5、+1 秒
_OFFSETS = [-30, -15, 0, 15, 30]

# 給標註者看的行為說明，跟 prompts/verification_prompt.py 的說法一致
_EVENT_HINTS = {
    "sprint_spike": "有人在奔跑或明顯加速移動",
    "crowd_locked": "人群擠在一起、停滯不動或難以移動",
    "panic_scatter": "人群向四周散開、四散奔逃",
    "counter_flow": "有人朝著與其他人相反的方向移動",
}


def _full_view(frame: Image.Image, bbox) -> Image.Image:
    """整張畫面縮小，用紅框標出事件位置。"""
    view = frame.convert("RGB").copy()
    ImageDraw.Draw(view).rectangle(bbox, outline=(255, 40, 40), width=4)
    view.thumbnail((480, 480))
    return view


def _crop_view(frame: Image.Image, bbox) -> Image.Image:
    """以事件框為中心放大局部，範圍至少 400x300，看得到周圍的人往哪裡動；同樣畫出紅框。"""
    x1, y1, x2, y2 = bbox
    frame = frame.convert("RGB").copy()
    ImageDraw.Draw(frame).rectangle(bbox, outline=(255, 40, 40), width=2)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half_w = max(200, (x2 - x1) * 1.25)
    half_h = max(150, (y2 - y1) * 1.25)
    box = (
        int(max(0, cx - half_w)),
        int(max(0, cy - half_h)),
        int(min(frame.width, cx + half_w)),
        int(min(frame.height, cy + half_h)),
    )
    crop = frame.convert("RGB").crop(box)
    scale = 480 / crop.width
    return crop.resize((480, int(crop.height * scale)))


def _write_clip(cap, out_path: Path, peak_frame_id: int, bbox) -> None:
    """剪出峰值前後各 _CLIP_HALF_SEC 秒，畫上事件框與相對峰值的時間。"""
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    peak = round(peak_frame_id / _DATASET_FPS * fps)
    start, end = max(0, peak - int(_CLIP_HALF_SEC * fps)), min(total - 1, peak + int(_CLIP_HALF_SEC * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    scale = 960 / cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    size = (960, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale))
    # OpenCV 在 Windows 上寫不進含中文的路徑（例如「畢業專題」）而且不會報錯，
    # 先寫到系統暫存資料夾再搬過去
    tmp_path = Path(tempfile.gettempdir()) / f"label_clip_{out_path.name}"
    writer = cv2.VideoWriter(str(tmp_path), cv2.VideoWriter_fourcc(*"avc1"), fps, size)
    if not writer.isOpened():
        raise SystemExit("無法建立 H.264 影片（cv2.VideoWriter avc1）")
    x1, y1, x2, y2 = [int(v * scale) for v in bbox]
    for idx in range(start, end + 1):
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, size)
        rel = (idx - peak) / fps
        # 框是峰值那一幀的位置；峰值前後 0.5 秒內加粗，提示「就是這個時候」
        near_peak = abs(rel) <= 0.5
        cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 40, 255), 3 if near_peak else 1)
        text = "PEAK" if abs(rel) < 0.5 / fps else f"{rel:+.1f}s"
        # 放右下角：原始影片左上角有編輯軟體的「VEED」浮水印
        org = (size[0] - 170, size[1] - 20)
        cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 5)
        cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                    (40, 40, 255) if near_peak else (255, 255, 255), 2)
        writer.write(frame)
    writer.release()
    shutil.move(str(tmp_path), out_path)


def main() -> None:
    records = json.loads(_EVENTS_PATH.read_text(encoding="utf-8"))["records"]
    index = _index_clean_frames()
    img_dir = _OUTPUT_DIR / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    clip_dir = _OUTPUT_DIR / "clips"
    clip_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(_VIDEO_PATH))
    if not cap.isOpened():
        raise SystemExit(f"找不到原始影片：{_VIDEO_PATH}")

    events = []
    for record in records:
        rid = record["record_id"]
        peak = record["frame_idx"]
        bbox = [int(v) for v in record["spatial_bbox"]]
        shots = []
        for offset in _OFFSETS:
            frame_id = _find_context_frame(index, peak + offset, before=offset < 0) if offset else peak
            frame = Image.open(index[frame_id])
            full_name = f"{rid}_{offset:+d}_full.jpg"
            crop_name = f"{rid}_{offset:+d}_crop.jpg"
            _full_view(frame, bbox).save(img_dir / full_name, quality=85)
            _crop_view(frame, bbox).save(img_dir / crop_name, quality=85)
            shots.append({
                "label": f"{(frame_id - peak) / 30:+.1f} 秒" if frame_id != peak else "峰值",
                "frame_id": frame_id,
                "full": f"img/{full_name}",
                "crop": f"img/{crop_name}",
            })
        _write_clip(cap, clip_dir / f"{rid}.mp4", peak, bbox)
        event_type = record["predicted_event_type"]
        events.append({
            "clip": f"clips/{rid}.mp4",
            "record_id": rid,
            "event_type": event_type,
            "hint": _EVENT_HINTS.get(event_type, ""),
            "peak_frame": peak,
            "shots": shots,
        })
        print(f"{rid}：frame {[s['frame_id'] for s in shots]}")

    html = _TEMPLATE.replace("__EVENTS__", json.dumps(events, ensure_ascii=False))
    (_OUTPUT_DIR / "index.html").write_text(html, encoding="utf-8")
    print(f"\n標註頁：{_OUTPUT_DIR / 'index.html'}")


_TEMPLATE = """<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>事件人工標註</title>
<style>
  :root { --bg:#f6f6f4; --card:#fff; --fg:#1d1d1b; --muted:#6b6b66; --line:#ddd; --accent:#2a6df4; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#161616; --card:#222; --fg:#eee; --muted:#a0a09a; --line:#3a3a3a; --accent:#6c9cff; }
  }
  body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.6 system-ui, "Microsoft JhengHei", sans-serif; }
  main { max-width:1500px; margin:0 auto; padding:16px; }
  header { position:sticky; top:0; background:var(--bg); padding:8px 0 12px; border-bottom:1px solid var(--line); z-index:1;
           display:flex; flex-wrap:wrap; gap:12px; align-items:center; }
  h1 { font-size:20px; margin:0; flex:1 1 auto; }
  button { font:inherit; padding:6px 14px; border-radius:6px; border:1px solid var(--accent); background:var(--accent); color:#fff; cursor:pointer; }
  .progress { color:var(--muted); }
  .intro { color:var(--muted); margin:12px 0 20px; }
  section { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:16px; margin-bottom:24px; }
  h2 { font-size:17px; margin:0 0 4px; }
  .hint { color:var(--muted); margin-bottom:12px; }
  video { width:100%; max-width:960px; display:block; border-radius:6px; background:#000; }
  .speed { margin:6px 0 4px; }
  .speed button { padding:2px 10px; margin-right:6px; background:transparent; color:var(--fg); border-color:var(--line); }
  .speed button.on { background:var(--accent); color:#fff; border-color:var(--accent); }
  .strip { display:grid; grid-template-columns:repeat(5, minmax(0,1fr)); gap:6px; }
  .strip figure { margin:0; }
  .strip img { width:100%; display:block; border-radius:4px; cursor:zoom-in; }
  .strip figcaption { text-align:center; font-size:13px; color:var(--muted); }
  .row-label { font-size:13px; color:var(--muted); margin:10px 0 4px; }
  .q { margin-top:14px; }
  .q > div { margin:6px 0; }
  .q label { margin-right:16px; white-space:nowrap; }
  textarea { width:100%; box-sizing:border-box; font:inherit; background:var(--bg); color:var(--fg); border:1px solid var(--line); border-radius:6px; }
  .zoom { position:fixed; inset:0; background:rgba(0,0,0,.85); display:none; align-items:center; justify-content:center; z-index:2; }
  .zoom img { max-width:96vw; max-height:96vh; }
  @media (max-width:800px) { .strip { grid-template-columns:repeat(2, minmax(0,1fr)); } }
</style>
</head>
<body>
<main>
<header>
  <h1>事件人工標註（video 7）</h1>
  <span class="progress" id="progress"></span>
  <button id="download">下載標註 JSON</button>
</header>
<p class="intro">每筆事件先看<b>峰值前後各 5 秒的影片</b>（右下角是相對峰值的時間，峰值前後 0.5 秒紅框會加粗），可以切換慢速播放；
下方再附峰值前後 1 秒內 5 個時間點的畫面供對照。紅框是系統在峰值那一刻標出異常的位置（框是固定的，人會移動）。
第 1a 題只看<b>峰值時紅框內的人</b>（系統偵測的位置對不對）；第 1b 題看<b>整個畫面</b>（這段時間現場有沒有出現這個行為）。點圖片可以放大。
這一版的答案跟上一版（只有照片）分開存，請依影片重新判斷。只依你在畫面上看到的回答；看不出來就選「看不出來」。填寫內容會暫存在這個瀏覽器，全部填完後按右上角下載。</p>
<div id="events"></div>
</main>
<div class="zoom" id="zoom"><img alt=""></div>
<script>
const EVENTS = __EVENTS__;
const KEY = "label_sheet_video7_clips";
let labels = {};
try { labels = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { labels = {}; }
// 拆題前只有一題 occurred（當時問的是框內），沿用為 occurred_box
Object.values(labels).forEach(l => {
  if (l.occurred && !l.occurred_box) l.occurred_box = l.occurred;
  delete l.occurred;
});

function save() {
  try { localStorage.setItem(KEY, JSON.stringify(labels)); } catch (e) {}
  const done = EVENTS.filter(e => {
    const l = labels[e.record_id];
    return l && l.occurred_box && l.occurred_scene && l.danger;
  }).length;
  document.getElementById("progress").textContent = `已完成 ${done} / ${EVENTS.length}`;
}

function radios(rid, field, options) {
  return options.map(([value, text]) => {
    const checked = labels[rid] && labels[rid][field] === value ? "checked" : "";
    return `<label><input type="radio" name="${rid}_${field}" value="${value}" ${checked}
      data-rid="${rid}" data-field="${field}"> ${text}</label>`;
  }).join("");
}

const container = document.getElementById("events");
EVENTS.forEach((ev, i) => {
  const note = (labels[ev.record_id] && labels[ev.record_id].note) || "";
  const figs = key => ev.shots.map(s =>
    `<figure><img src="${s[key]}" alt="${s.label}" loading="lazy"><figcaption>${s.label}（frame ${s.frame_id}）</figcaption></figure>`).join("");
  const sec = document.createElement("section");
  sec.innerHTML = `
    <h2>${i + 1}. ${ev.record_id}</h2>
    <div class="hint">系統判斷：<b>${ev.event_type}</b>（${ev.hint}）</div>
    <video src="${ev.clip}" controls loop muted playsinline preload="metadata"></video>
    <div class="speed">播放速度：<button data-rate="0.25">0.25×</button><button data-rate="0.5">0.5×</button><button data-rate="1" class="on">1×</button></div>
    <div class="row-label">整張畫面</div><div class="strip">${figs("full")}</div>
    <div class="row-label">放大局部</div><div class="strip">${figs("crop")}</div>
    <div class="q">
      <div>1a. 峰值時，<b>紅框內的人</b>有出現這個行為（${ev.hint}）嗎？<br>
        ${radios(ev.record_id, "occurred_box", [["yes","有"],["no","沒有"],["unsure","看不出來"]])}</div>
      <div>1b. 這段時間，<b>整個畫面</b>中有出現這個行為嗎？<br>
        ${radios(ev.record_id, "occurred_scene", [["yes","有"],["no","沒有"],["unsure","看不出來"]])}</div>
      <div>2. 就整個畫面來看，實際上需要通報的程度？<br>
        ${radios(ev.record_id, "danger", [["none","正常，不需通報"],["attention","需要注意"],["danger","危險，要立即處理"]])}</div>
      <div>3. 你看到的實際情況（一句話，例如「舞者在排練」「有人跌倒」）：
        <textarea rows="2" data-rid="${ev.record_id}" data-field="note">${note.replace(/</g,"&lt;")}</textarea></div>
    </div>`;
  container.appendChild(sec);
});

container.addEventListener("input", e => {
  const { rid, field } = e.target.dataset;
  if (!rid) return;
  labels[rid] = labels[rid] || {};
  labels[rid][field] = e.target.value;
  save();
});

const zoom = document.getElementById("zoom");
container.addEventListener("click", e => {
  if (e.target.dataset.rate) {
    const sec = e.target.closest("section");
    sec.querySelector("video").playbackRate = Number(e.target.dataset.rate);
    sec.querySelectorAll(".speed button").forEach(b => b.classList.toggle("on", b === e.target));
    return;
  }
  if (e.target.tagName === "IMG") { zoom.querySelector("img").src = e.target.src; zoom.style.display = "flex"; }
});
zoom.addEventListener("click", () => { zoom.style.display = "none"; });

document.getElementById("download").addEventListener("click", () => {
  const out = {
    video: "video7",
    method: "10s clips + 5 frames",
    labeled_at: new Date().toISOString(),
    labels: EVENTS.map(ev => ({ record_id: ev.record_id, event_type: ev.event_type, ...(labels[ev.record_id] || {}) })),
  };
  const blob = new Blob([JSON.stringify(out, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "labels_video7_clips.json";
  a.click();
});
save();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
