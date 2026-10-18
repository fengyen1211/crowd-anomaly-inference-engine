"""
scripts/make_label_sheet.py
==============================
產生人工標註用的本機網頁：每筆事件並排顯示峰值前後 1 秒內 5 個時間點的
畫面（整張畫面標出事件位置，加上放大的局部），讓人判斷「系統說的行為
是否真的發生」與「實際危險程度」，標註結果在網頁上下載成 JSON。

為什麼需要：到 sweep_566003 為止，模型組合、prompt、驗證模型的比較都沒有
標準答案，只能靠讀描述文字推測誰對誰錯；這份標註就是之後算準確率的依據。

畫面直接取自乾淨畫面資料夾（BEFORE／DURING／AFTER，同 build_tinyformer_events.py），
不受 pipeline 實際餵給 VLM 的幀數限制；標註頁不顯示任何模型輸出，避免先入為主。

使用方式（在 project/ 目錄下執行）：
    python scripts/make_label_sheet.py
然後用瀏覽器開啟 reports/label_sheet/index.html（reports/ 不進版控）。
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw  # noqa: E402

from scripts.build_tinyformer_events import _find_context_frame, _index_clean_frames  # noqa: E402

_PROJECT_DIR = Path(__file__).resolve().parent.parent
_EVENTS_PATH = _PROJECT_DIR / "data" / "tinyformer7_events.json"
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


def main() -> None:
    records = json.loads(_EVENTS_PATH.read_text(encoding="utf-8"))["records"]
    index = _index_clean_frames()
    img_dir = _OUTPUT_DIR / "img"
    img_dir.mkdir(parents=True, exist_ok=True)

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
        event_type = record["predicted_event_type"]
        events.append({
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
<p class="intro">每筆事件顯示峰值前後 1 秒內的 5 個時間點：上排是整張畫面，下排是放大的局部，兩排的紅框都是系統標出異常的位置。
<b>判斷對象是紅框內的人</b>；框外的人只當作周圍情況參考（例如是不是大家都在跑、是不是在表演）。點圖片可以放大。只依你在畫面上看到的回答；看不出來就選「看不出來」。填寫內容會暫存在這個瀏覽器，全部填完後按右上角下載。</p>
<div id="events"></div>
</main>
<div class="zoom" id="zoom"><img alt=""></div>
<script>
const EVENTS = __EVENTS__;
const KEY = "label_sheet_video7";
let labels = {};
try { labels = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { labels = {}; }

function save() {
  try { localStorage.setItem(KEY, JSON.stringify(labels)); } catch (e) {}
  const done = EVENTS.filter(e => labels[e.record_id] && labels[e.record_id].occurred && labels[e.record_id].danger).length;
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
    <div class="row-label">整張畫面</div><div class="strip">${figs("full")}</div>
    <div class="row-label">放大局部</div><div class="strip">${figs("crop")}</div>
    <div class="q">
      <div>1. 系統說的行為（${ev.hint}）真的有發生嗎？<br>
        ${radios(ev.record_id, "occurred", [["yes","有"],["no","沒有"],["unsure","看不出來"]])}</div>
      <div>2. 實際上需要通報的程度？<br>
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
  if (e.target.tagName === "IMG") { zoom.querySelector("img").src = e.target.src; zoom.style.display = "flex"; }
});
zoom.addEventListener("click", () => { zoom.style.display = "none"; });

document.getElementById("download").addEventListener("click", () => {
  const out = {
    video: "video7",
    labeled_at: new Date().toISOString(),
    labels: EVENTS.map(ev => ({ record_id: ev.record_id, event_type: ev.event_type, ...(labels[ev.record_id] || {}) })),
  };
  const blob = new Blob([JSON.stringify(out, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "labels_video7.json";
  a.click();
});
save();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
