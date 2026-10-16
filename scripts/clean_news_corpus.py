"""
scripts/clean_news_corpus.py
================================
清理「新聞真實案例」語料，並重建向量索引。

為什麼需要這支腳本
------------------
scripts/fetch_and_index_news.py 是「關鍵字搜尋 Wikipedia -> LLM 判斷相關性
-> 摘要」的全自動流程。實際跑完累積出 191 筆不重複記錄後盤點，發現相關性
判斷這一步的誤判率不低，語料裡混進三類雜訊：

1. **同名但無關**：關鍵字 stampede / panic 命中了跟群眾行為無關的條目——
   股市崩盤（panic 的另一個意思）、Beretta Stampede（左輪手槍型號）、
   Calgary Stampede（牛仔節慶／摔角團體／美式足球隊）、Tremors 2（電影）。
2. **非事件條目**：Moshing（舞蹈風格）、Riot_control（人群控制方法）、
   Shibuya_Crossing（路口）、Kumbh_Mela（節慶本身）——內容跟群眾行為有關，
   但不是「一起事故」，當成 precedent 引用會讓 LLM 引到一段泛論。
3. **同事件重複／周邊條目**：Astroworld 一起事故就有 8 筆（音樂節條目、
   歌手條目、紀錄片、歌曲、兩家律師事務所…），檢索時會把 top_k 名額吃光，
   排擠掉其他事件。

這些雜訊不會讓程式報錯——它們是格式完全正確的記錄，只是內容不該出現在
「真實案例」知識庫裡。prompts/rag_prompt.py 會把檢索結果原文貼進 prompt，
所以雜訊的代價是直接的：LLM 可能引用「Tremors 2 的地底怪物」當作群眾事故
的參考案例。

順帶解決的第二件事
------------------
爬蟲產生的摘要多半只寫「幾月幾日、哪裡、死傷多少人」，沒寫**人群動力學
機制**。但本系統要回答的是「影像端判定 counter_flow，有沒有類似先例」——
這需要案例描述「兩股人流對撞」而不只是「踩踏致死」。因此另外人工策展了
data/news_curated_cases.json，替四種行為類型各補上有明確機制描述的案例，
並以 behavior_type 欄位標記（見 rag/documents/news_document_builder.py）。
策展記錄依 source_url 覆蓋爬蟲版本。

執行後的結果
------------
- 原始爬蟲檔案移到 knowledge_base_sources_news/_raw_archive/（保留可追溯，
  但因為 fetch_and_index_news.py 用的是非遞迴 glob("*.json")，不會被重複索引）
- 清理後的語料寫成單一檔案 knowledge_base_sources_news/news_corpus_clean.json
- 需要重建索引時加 --reindex：會先砍掉整個 news collection 再重新索引
  （BaseVectorStore 介面目前沒有 delete()，所以這裡直接對 chromadb client
  操作；這是 rag/knowledge_indexer.py 的 rebuild_index() TODO 記載的缺口）

使用方式（在 project/ 目錄下執行）：
    python scripts/clean_news_corpus.py --dry-run     # 只看會刪掉什麼，不動檔案
    python scripts/clean_news_corpus.py               # 清理語料，不碰索引
    python scripts/clean_news_corpus.py --reindex     # 清理語料並重建向量索引
"""

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_NEWS_DIR = _PROJECT_ROOT / "data" / "knowledge_base_sources_news"
_ARCHIVE_DIR = _NEWS_DIR / "_raw_archive"
_CURATED_PATH = _PROJECT_ROOT / "data" / "news_curated_cases.json"
_CLEAN_PATH = _NEWS_DIR / "news_corpus_clean.json"

_WIKI = "https://en.wikipedia.org/wiki/"

# ---------------------------------------------------------------------------
# 要剔除的條目，依剔除理由分組（分組只為了讓報告可讀、也讓之後有人想
# 放寬標準時知道該動哪一組，程式邏輯上三組是合併使用的）
# ---------------------------------------------------------------------------

# 第一類：關鍵字同名但內容跟群眾行為無關
_NOISE_UNRELATED = {
    # "panic" 的金融義
    _WIKI + "Stock_market_crash",
    _WIKI + "Wall_Street_crash_of_1929",
    _WIKI + "Black_Monday_(1987)",
    _WIKI + "2022_stock_market_decline",
    _WIKI + "2025_stock_market_crash",
    _WIKI + "AI_bubble",
    # "stampede" 的其他義：商品名／團體名／隊名
    _WIKI + "Beretta_Stampede",
    _WIKI + "Tyson_Kidd",
    _WIKI + "Derek_Dennis",
    _WIKI + "Central_Texas_Stampede",
    # 影視／音樂作品與人物
    _WIKI + "Tremors_2:_Aftershocks",
    _WIKI + "Chloe_East",
    _WIKI + "Karl_Sanders",
    _WIKI + "Chitra_Pournami_(film)",
    _WIKI + "Athiradi_(2026_film)",
    _WIKI + "41_(2007_film)",
    _WIKI + "Rolling_Loud:_The_Movie",
    _WIKI + "Trainwreck:_The_Astroworld_Tragedy",
    _WIKI + "My_Eyes_(Travis_Scott_song)",
    # 事故的周邊人物（律師、政治人物），非事故本身
    _WIKI + "Jim_Adler",
    _WIKI + "Tony_Buzbee",
    _WIKI + "Qasem_Soleimani",
    _WIKI + "C._Joseph_Vijay",
    _WIKI + "Han_Chang-seob",
}

# 第二類：內容跟群眾有關，但不是「一起事故」（概念／地點／設施／組織條目）
_NOISE_NOT_AN_INCIDENT = {
    _WIKI + "Moshing",
    _WIKI + "Riot_control",
    _WIKI + "Shibuya_Crossing",
    _WIKI + "Itaewon",
    _WIKI + "Kumbh_Mela",
    _WIKI + "Tamilaga_Vettri_Kazhagam",
    _WIKI + "Ratangarh_Mata_Temple",
    _WIKI + "Ratangarh,_Datia",
    _WIKI + "Ekam_-_The_Oneness_Temple",
    _WIKI + "Kachua,_West_Bengal",
    _WIKI + "A1_(group)",
}

# 第三類：同一起事故已有主條目，或摘要內容離題
_NOISE_DUPLICATE = {
    # Astroworld：保留 Astroworld_Festival_crowd_crush
    _WIKI + "Astroworld_Festival",
    _WIKI + "Travis_Scott",
    # Hillsborough：保留 Hillsborough_disaster（這筆是後續的法律判例）
    _WIKI + "Alcock_v_Chief_Constable_of_South_Yorkshire_Police",
    # 2013 Ratangarh 踩踏：保留 2013_Madhya_Pradesh_stampede
    # （另兩筆地點條目已列在第二類）
    # 2025 Kumbh：保留 ..._crowd_crush，這筆摘要講的是河水水質
    _WIKI + "2025_Prayag_Maha_Kumbh_Mela",
}

_NOISE_GROUPS = {
    "同名但無關": _NOISE_UNRELATED,
    "非事件條目": _NOISE_NOT_AN_INCIDENT,
    "重複／離題": _NOISE_DUPLICATE,
}
_NOISE_URLS = set().union(*_NOISE_GROUPS.values())


def _slug(url: str) -> str:
    return url.split("/wiki/")[-1] if "/wiki/" in url else url


def load_raw_records() -> Dict[str, dict]:
    """
    讀取所有爬蟲產生的記錄，依 source_url 去重（先出現的優先）。

    同時掃 _raw_archive/，讓這支腳本可以重複執行——第一次執行會把原始檔
    搬進 archive，第二次執行仍然讀得到同一批資料，不會因為檔案被搬走就
    把語料清空。
    """
    paths = sorted(_NEWS_DIR.glob("*.json")) + sorted(_ARCHIVE_DIR.glob("*.json"))
    records: Dict[str, dict] = {}
    file_count = 0

    for path in paths:
        if path == _CLEAN_PATH:
            continue  # 清理後的輸出檔不當成來源，避免自我疊加
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"  ! 跳過無法解析的檔案 {path.name}：{exc}")
            continue
        file_count += 1
        for record in raw.get("records", []):
            url = record.get("source_url", "")
            if url:
                records.setdefault(url, record)

    print(f"讀取 {file_count} 個來源檔案，去重後共 {len(records)} 筆記錄。")
    return records


def load_curated() -> Dict[str, dict]:
    """讀取人工策展的案例（依 source_url 覆蓋爬蟲版本）。"""
    if not _CURATED_PATH.exists():
        print(f"  ! 找不到策展檔 {_CURATED_PATH}，略過覆蓋步驟。")
        return {}

    raw = json.loads(_CURATED_PATH.read_text(encoding="utf-8"))
    curated = {r["source_url"]: r for r in raw.get("records", []) if r.get("source_url")}
    print(f"讀取策展案例 {len(curated)} 筆。")
    return curated


def main() -> None:
    parser = argparse.ArgumentParser(description="清理新聞語料並（可選）重建索引")
    parser.add_argument("--dry-run", action="store_true", help="只報告會剔除什麼，不寫入任何檔案")
    parser.add_argument("--reindex", action="store_true", help="清理完後砍掉舊的 news collection 並重建索引")
    args = parser.parse_args()

    records = load_raw_records()
    if not records:
        print("沒有讀到任何記錄，結束。")
        return

    # ---------- 剔除雜訊 ----------
    print("\n=== 剔除清單 ===")
    removed_total = 0
    for reason, urls in _NOISE_GROUPS.items():
        hits = [u for u in urls if u in records]
        removed_total += len(hits)
        print(f"\n[{reason}] {len(hits)} 筆")
        for url in sorted(hits):
            print(f"  - {_slug(url)}：{records[url].get('headline', '')[:40]}")

    missing = _NOISE_URLS - set(records)
    if missing:
        print(f"\n（剔除清單裡有 {len(missing)} 筆在語料中找不到，可能先前已被移除：）")
        for url in sorted(missing):
            print(f"  ? {_slug(url)}")

    kept = {u: r for u, r in records.items() if u not in _NOISE_URLS}

    # ---------- 疊上策展案例 ----------
    curated = load_curated()
    overridden = [u for u in curated if u in kept]
    added = [u for u in curated if u not in kept]
    kept.update(curated)

    print("\n=== 策展案例 ===")
    for url in added:
        bt = "、".join(curated[url].get("behavior_type", []))
        print(f"  + 新增 {_slug(url)}（{bt}）")
    for url in overridden:
        bt = "、".join(curated[url].get("behavior_type", []))
        print(f"  ~ 覆蓋 {_slug(url)}（{bt}）——原摘要缺少人群動力學機制描述")

    print(f"\n=== 結果 ===\n原始 {len(records)} 筆 - 剔除 {removed_total} 筆 + 策展新增 {len(added)} 筆 = {len(kept)} 筆")

    behavior_tagged = sum(1 for r in kept.values() if r.get("behavior_type"))
    print(f"其中標註 behavior_type 的案例：{behavior_tagged} 筆")

    if args.dry_run:
        print("\n--dry-run：沒有寫入任何檔案。")
        return

    # ---------- 寫出清理後語料 ----------
    ordered = sorted(kept.values(), key=lambda r: (r.get("publish_date") or "", r.get("headline") or ""))
    payload = {
        "_comment": (
            f"由 scripts/clean_news_corpus.py 於 "
            f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} 產生。"
            f"來源為 _raw_archive/ 底下的爬蟲輸出，已剔除 {removed_total} 筆雜訊，"
            f"並疊上 data/news_curated_cases.json 的 {len(curated)} 筆人工策展案例。"
            "請勿手動編輯本檔，修改請改動剔除清單或策展檔後重跑腳本。"
        ),
        "records": ordered,
    }

    # ---------- 把原始檔搬進 archive ----------
    _ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    moved = 0
    for path in sorted(_NEWS_DIR.glob("*.json")):
        if path == _CLEAN_PATH:
            continue
        shutil.move(str(path), str(_ARCHIVE_DIR / path.name))
        moved += 1

    _CLEAN_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已寫出 {_CLEAN_PATH.relative_to(_PROJECT_ROOT)}（{len(ordered)} 筆）")
    if moved:
        print(f"已將 {moved} 個原始檔搬進 {_ARCHIVE_DIR.relative_to(_PROJECT_ROOT)}/（保留可追溯，不會被索引）")

    if not args.reindex:
        print("\n未加 --reindex，向量索引維持原狀（仍含舊雜訊）。")
        print("要重建請執行：python scripts/clean_news_corpus.py --reindex")
        return

    # ---------- 重建索引 ----------
    # 為什麼要先刪 collection：ChromaVectorStore.add_texts() 是 upsert 語意，
    # 只會覆蓋「同 id」的記錄。被剔除的雜訊 chunk 有自己的 id，不重新寫入
    # 也不會消失，必須整個 collection 砍掉重建。BaseVectorStore 介面沒有
    # delete()，所以這裡直接對 chromadb client 操作。
    import chromadb  # noqa: E402 - 只有 --reindex 才需要，延遲載入

    from app.dependencies import get_news_indexer, get_news_retriever  # noqa: E402
    from config.settings import get_settings  # noqa: E402

    settings = get_settings()
    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    try:
        client.delete_collection(settings.chroma_collection_news)
        print(f"\n已刪除舊的 collection '{settings.chroma_collection_news}'")
    except Exception as exc:  # noqa: BLE001 - collection 不存在時不該中斷
        print(f"\n刪除 collection 時略過（可能本來就不存在）：{exc}")

    print("開始重建索引（第一次會載入 bge-m3 模型，需要一點時間）...")
    chunk_count = get_news_indexer().build_index_from_documents([_CLEAN_PATH])
    print(f"索引完成，共寫入 {chunk_count} 個 chunk。")

    # ---------- 驗證：四種行為類型各查一次 ----------
    # 只確認「索引寫進去了」不夠，要確認「查得到、而且查回來的是對的類型」。
    print("\n=== 驗證檢索（四種行為類型各查一次）===")
    retriever = get_news_retriever()
    demo_queries = {
        "counter_flow": "人群分成兩股方向相反的人流在通道或路口對撞",
        "crowd_locked": "人群在瓶頸處密度過高卡住無法移動造成擠壓窒息",
        "panic_scatter": "爆炸或火災發生後群眾朝各個方向恐慌逃散",
        "sprint_spike": "群眾突然加速奔跑引發連鎖推擠",
    }
    for behavior, query in demo_queries.items():
        print(f"\n[{behavior}] 查詢：{query}")
        for rank, snippet in enumerate(retriever.retrieve(query, top_k=3), start=1):
            first_line = snippet.content.splitlines()[0] if snippet.content else ""
            marker = ""
            for line in snippet.content.splitlines():
                if line.startswith("對應行為類型"):
                    marker = " <<< " + line
                    break
            print(f"  {rank}. （相關度 {snippet.relevance_score:.3f}）{first_line[:50]}{marker}")


if __name__ == "__main__":
    main()
