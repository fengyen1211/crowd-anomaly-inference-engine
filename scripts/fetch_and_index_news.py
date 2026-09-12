"""
scripts/fetch_and_index_news.py
==================================
從外部來源撈取跟「群眾異常行為」（踩踏、推擠、恐慌逃散、騷亂）相關的
真實事件文章，用本地 LLM（Gemma3，透過 app.dependencies.get_llm_client()）
判斷相關性並摘要，寫成結構化 JSON 後索引進「新聞真實案例」知識庫
（chroma_collection_news，獨立於法規知識庫 chroma_collection_knowledge）。

支援兩種候選來源（--source）：
- wikipedia（預設）：用 Wikipedia 搜尋 API 找相關條目。不需要金鑰、
  沒有實測到限流問題，且涵蓋大量真實歷史事件的完整條目，很適合拿來
  建立「歷史案例」知識庫（不像新聞 API 那樣只涵蓋近期報導）。
- gdelt：GDELT 全球事件資料庫 DOC 2.0 API，免費免金鑰的新聞搜尋。
  實測發現這個 API 的免費額度在某些網路環境下容易被限流封鎖
  （429 Too Many Requests），且封鎖持續時間可能長達數小時到一天以上，
  不是短暫的秒級限流，選用前請先確認目前網路環境對這個 API 是否暢通。

流程：
    候選來源（GDELT 關鍵字搜尋 或 Wikipedia 搜尋）
        -> 逐篇用 requests + BeautifulSoup 擷取全文
        -> 逐篇用本地 LLM 判斷是否相關 + 摘要成結構化欄位（含從內文判斷的事件日期）
        -> 每成功一篇就立即寫入 data/knowledge_base_sources_news/<source>_<timestamp>.json
        -> KnowledgeBaseIndexer 切塊、算 embedding、寫進向量庫

任何單篇文章的抓取或摘要失敗（付費牆、JS 動態渲染、LLM 輸出非法 JSON等）
都只記錄 log 並跳過，不會中斷整批處理——新聞來源網站排版千百種，
失敗是預期中的常態，不是例外狀況。

使用方式（在 project/ 目錄下執行）：
    python scripts/fetch_and_index_news.py
    python scripts/fetch_and_index_news.py --source wikipedia --maxrecords 40
    python scripts/fetch_and_index_news.py --source gdelt --maxrecords 10 --timespan 1month
    python scripts/fetch_and_index_news.py --skip-fetch   # 只重新索引既有檔案
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from bs4 import BeautifulSoup

# Windows 主控台預設編碼（cp950）無法顯示部分 Wikipedia 條目標題裡的字元
# （例如斯拉夫語系地名），會讓 print() 直接丟出 UnicodeEncodeError 中斷整批
# 處理。重新設定 stdout/stderr 成 UTF-8（無法顯示的字元用替代符號呈現），
# 避免單一條目標題的顯示問題拖垮整個批次。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.dependencies import get_llm_client, get_news_indexer, get_news_retriever  # noqa: E402
from prompts.news_summary_prompt import NewsSummaryPromptBuilder  # noqa: E402
from utils.llm_json import parse_json_response  # noqa: E402
from utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)

_NEWS_DIR = Path(__file__).resolve().parent.parent / "data" / "knowledge_base_sources_news"

_GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"

# 預設關鍵字：涵蓋踩踏/推擠/群眾恐慌等群眾異常行為的常見英文用詞，
# 限定英文新聞（sourcelang:eng）以確保後續 LLM 摘要品質一致，
# 可用 --query 覆寫成其他關鍵字組合。
_DEFAULT_QUERY = '(stampede OR "crowd crush" OR "crowd surge" OR "mass panic" OR "crowd disaster") sourcelang:eng'

_WIKIPEDIA_SEARCH_API = "https://en.wikipedia.org/w/api.php"

# Wikipedia 搜尋沒有像 GDELT 那樣的單一 OR 語法可以一次涵蓋多個關鍵字，
# 所以逐一關鍵字各查一次，再依標題去重。
#
# 實測發現單用 "stampede" 這種單一名詞太模糊，會撈到大量同名但無關的條目
# （如 Calgary Stampede 牛仔節慶、Trigun Stampede 動畫、Stampede Wrestling
# 摔角團體），改用更具體的複合詞（加上 deaths / crush 等字）精準度高很多，
# 撈回來的幾乎都是真實群眾傷亡事件條目。
#（實測發現 "mass panic deaths" 這種詞太籠統，會撈到 Jonestown、群眾心理學、
# 甚至 IT 系統中斷事件等完全無關的條目，已移除。）
_WIKIPEDIA_QUERY_TERMS = [
    "human stampede",
    "stampede deaths",
    "crowd crush deaths",
    "crowd crush",
    "crowd surge deaths",
    "crowd disaster deaths",
    # 加入更具體的場景詞彙，目的是找出前面幾個泛用關鍵字排名較後、
    # 尚未撈到的獨立事件條目，而不是把同樣的關鍵字挖得更深（更深的結果
    # 雜訊比例通常更高）。
    "temple stampede",
    "festival crowd crush",
    "market stampede",
    "nightclub crowd crush",
    "funeral stampede",
    "religious festival stampede",
    # 第二批擴充關鍵字（101 -> 200+ 筆時新增）：涵蓋前一批較少觸及的
    # 場景類型（體育場、演唱會、跨年、遊行、車站、婚禮、學校、暴動、
    # 煙火、海灘、大學、機場），目的跟第一批新增時一樣——找出前面
    # 關鍵字排名較後、尚未撈到的獨立事件條目。
    "stadium disaster",
    "concert crowd crush",
    "New Year stampede",
    "parade stampede",
    "train station stampede",
    "pilgrimage stampede deaths",
    "Halloween crowd crush",
    "wedding stampede",
    "school stampede",
    "riot crowd crush deaths",
    "bridge stampede",
    "fireworks stampede",
    "beach stampede",
    "university stampede",
    "airport crowd crush",
    # 第三批擴充關鍵字：先前資料庫盤點發現「riot」標籤下幾乎都是爆炸攻擊／
    # 政治鎮壓／球場衝突引發踩踏，缺乏單純「群眾鬥毆」（不涉及爆裂物或
    # 政治鎮壓的肢體衝突）類型的案例，新增這批關鍵字補足這個場景缺口。
    "crowd brawl",
    "mass brawl injuries",
    "stadium fan brawl violence",
    "festival brawl fight",
    "concert crowd fight breaks out",
    "market crowd brawl",
]

# Wikipedia 搜尋 API 對短時間內連續呼叫也有限流（實測連續 7 次中有 1 次 429），
# 雖然遠比 GDELT 寬鬆，逐一關鍵字之間還是加個小間隔比較保險。
_WIKIPEDIA_QUERY_DELAY_SECONDS = 3

# 排除「列表型」條目（例如 List of fatal crowd crushes）——這類條目是彙整
# 多起事件的清單，不是單一事件的描述，硬塞進單筆記錄的摘要格式效果不好，
# 個別事件本身通常也會被其他關鍵字撈到獨立條目，排除列表頁不會漏掉真正的
# 個案內容。
_EXCLUDED_TITLE_PREFIXES = ("List of", "Crowd collapses and crushes")

# 人工審查過後確認是誤判／離題的候選網址，永久排除，不會再被 LLM 重新判斷。
# 沒有這份清單的話，_load_processed_urls() 只會跳過「已經成功存檔」的網址——
# 曾經被判定不相關或被人工從知識庫移除的網址下次執行還是會被重新嘗試，
# 而 LLM 對同一篇文章的判斷通常是穩定的，會一直得到同樣的誤判結果。
# 每一條都記錄排除理由，方便之後回顧或調整。
_PERMANENTLY_EXCLUDED_URLS = {
    "https://en.wikipedia.org/wiki/Stampede": "泛用定義頁面，非單一事件",
    "https://en.wikipedia.org/wiki/Incidents_during_the_Hajj": "彙整型頁面，跟其他個別年份的 Hajj 事件重複",
    "https://en.wikipedia.org/wiki/Crowd_psychology": "群眾心理學理論頁面，非單一事件",
    "https://en.wikipedia.org/wiki/Crowd_control": "群眾管制方法頁面，非單一事件",
    "https://en.wikipedia.org/wiki/Asphyxia": "窒息醫學主題頁面，非單一事件",
    "https://en.wikipedia.org/wiki/Yoon_Suk_Yeol": "來源頁面是南韓總統條目，跟 Seoul_Halloween_crowd_crush 重複",
    "https://en.wikipedia.org/wiki/2024_CrowdStrike-related_IT_outages": "誤判：IT 系統中斷事件，非群眾傷亡",
    "https://en.wikipedia.org/wiki/Liz_Wheeler": "誤判：媒體人物與 COVID 假訊息，無關",
    "https://en.wikipedia.org/wiki/2022": "誤判：年份頁面，內容是俄烏戰爭",
    "https://en.wikipedia.org/wiki/2005": "誤判：年份頁面，內容是倫敦爆炸案（恐攻非群眾踩踏）",
    "https://en.wikipedia.org/wiki/The_Death_of_Slim_Shady_(Coup_de_Grâce)": "誤判：音樂專輯，無關",
    "https://en.wikipedia.org/wiki/Operation_Metro_Surge": "誤判：軍事/警方行動名稱巧合，非群眾踩踏",
    "https://en.wikipedia.org/wiki/2025–2026_Iran_massacres": "類別不符：政治暴力/屠殺，非群眾踩踏動力學事件",
    "https://en.wikipedia.org/wiki/Stampede_Trail": "誤判：健行步道溺水事件，跟人群踩踏無關（同名巧合）",
    "https://en.wikipedia.org/wiki/Stampede_Wrestling": "來源薄弱：摔角推廣公司條目，事件真實性存疑",
    "https://en.wikipedia.org/wiki/Gaza_genocide": "類別不符：戰爭/種族滅絕議題，非群眾踩踏",
    "https://en.wikipedia.org/wiki/ICE_protest_songs": "誤判：抗議歌曲條目，無關",
    "https://en.wikipedia.org/wiki/Death_of_Michael_Jackson": "誤判：藥物過量死亡，非群眾踩踏",
    "https://en.wikipedia.org/wiki/2024–2025_Indian_farmers'_protest": "類別不符：政治抗議，非群眾踩踏",
    "https://en.wikipedia.org/wiki/Iran": "誤判：國家條目提及政治示威，非單一群眾踩踏事件",
    "https://en.wikipedia.org/wiki/Gaza_war": "類別不符：戰爭議題，非群眾踩踏",
    "https://en.wikipedia.org/wiki/2026_Ceuta_migrant_crisis": "類別不符：邊境移民危機，非典型群眾踩踏監控場景",
    "https://en.wikipedia.org/wiki/Charlie_Kirk": "誤判：政治人物遇刺事件，非群眾踩踏",
    "https://en.wikipedia.org/wiki/2025–2026_Iranian_protests": "類別不符：政治抗議，非群眾踩踏",
    "https://en.wikipedia.org/wiki/Peter_Thiel": "誤判：與 Epstein 案相關報導，完全無關",
    "https://en.wikipedia.org/wiki/Mitch_McConnell": "誤判：來源頁面內容是 2021 年國會大廈遇襲事件，屬政治暴動非群眾踩踏",
    "https://en.wikipedia.org/wiki/Naina_Devi": "重複：跟 2008_Naina_Devi_temple_stampede 是同一事件，保留專屬條目版本",
    "https://en.wikipedia.org/wiki/Mandhradevi": "重複：跟 Mandher_Devi_temple_stampede 是同一 2005 年事件，保留專屬條目版本",
}

# 傳給摘要 LLM 的文章全文字數上限，避免超出小型本地模型的 context 負荷
_ARTICLE_TEXT_CHAR_LIMIT = 4000

# 全文擷取後若字數低於這個門檻，視為抓取失敗（很可能是付費牆或 JS 佔位頁面）
_MIN_ARTICLE_TEXT_LENGTH = 200

_DEMO_QUERY = "群眾推擠 踩踏 恐慌逃散"

_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}


def fetch_gdelt_articles(query: str, maxrecords: int, timespan: str) -> List[Dict[str, Any]]:
    """呼叫 GDELT DOC 2.0 API，回傳候選文章清單（title/url/seendate/domain...）。"""
    params = {
        "query": query,
        "mode": "artlist",
        "maxrecords": maxrecords,
        "format": "json",
        "sort": "hybridrel",
        "timespan": timespan,
    }
    try:
        response = requests.get(_GDELT_DOC_API, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError) as exc:
        logger.error("呼叫 GDELT API 失敗：%s", exc)
        return []

    articles = data.get("articles", [])
    logger.info("GDELT 回傳 %d 篇候選文章", len(articles))
    return articles


def fetch_wikipedia_candidates(limit_per_term: int) -> List[Dict[str, Any]]:
    """
    用 Wikipedia 搜尋 API 取得候選文章清單，格式跟 fetch_gdelt_articles() 對齊
    （title/url/domain），方便下游共用同一套抓取＋摘要邏輯。

    Wikipedia 條目沒有「發布日期」這種欄位（只有最後編輯時間，跟事件發生
    日期無關），所以刻意不帶 seendate——事件日期完全交給 LLM 從內文判斷
    （見 prompts/news_summary_prompt.py）。
    """
    seen_titles = set()
    candidates: List[Dict[str, Any]] = []

    for term_index, term in enumerate(_WIKIPEDIA_QUERY_TERMS):
        if term_index > 0:
            time.sleep(_WIKIPEDIA_QUERY_DELAY_SECONDS)

        params = {
            "action": "query",
            "list": "search",
            "srsearch": term,
            "srlimit": limit_per_term,
            "format": "json",
        }
        try:
            response = requests.get(_WIKIPEDIA_SEARCH_API, params=params, headers=_REQUEST_HEADERS, timeout=20)
            response.raise_for_status()
            data = response.json()
        except (requests.RequestException, ValueError) as exc:
            logger.error("呼叫 Wikipedia 搜尋 API 失敗（關鍵字：%s）：%s", term, exc)
            continue

        hits = data.get("query", {}).get("search", [])
        for hit in hits:
            title = hit.get("title", "")
            if not title or title in seen_titles:
                continue
            if title.startswith(_EXCLUDED_TITLE_PREFIXES):
                continue
            seen_titles.add(title)
            candidates.append(
                {
                    "title": title,
                    "url": "https://en.wikipedia.org/wiki/" + title.replace(" ", "_"),
                    "domain": "en.wikipedia.org",
                    "seendate": "",
                }
            )

    logger.info("Wikipedia 搜尋（%d 個關鍵字）共回傳 %d 篇不重複候選文章", len(_WIKIPEDIA_QUERY_TERMS), len(candidates))
    return candidates


# Wikipedia 分類頁面（Category:）是人工審核過的事件分類，用這個當候選來源
# 比自由關鍵字全文搜尋準確得多——實測發現關鍵字搜尋（fetch_wikipedia_candidates）
# 在筆數衝到 100+ 之後，關鍵字命中率大幅下降，撈到的候選裡有八九成是同名
# 巧合（如 Calgary Stampede 牛仔節慶、Trigun Stampede 動畫、Idumota Market
# 市場條目），LLM 篩選後保留率不到 10%。改成從分類樹遞迴收集文章，因為
# 「被人放進 Category:Human stampedes 底下」這件事本身就是一種人工相關性
# 標記，候選品質高很多。
#
# 由子分類（Category:）逐層往下爬，只收集實際條目頁（ns=0），不收分類頁
# 本身；用 seen_categories 防止分類樹裡的循環參照造成無限遞迴。
_WIKIPEDIA_CATEGORY_ROOTS = [
    "Human stampedes",
    "Stadium disasters",
    # 新增：資料庫盤點發現缺乏單純「群眾鬥毆」（非爆裂物/非政治鎮壓的
    # 肢體衝突）案例，這兩個分類涵蓋球場/演唱會等群聚場景真實爆發的
    # 群眾鬥毆事件，比自由關鍵字搜尋（見 _WIKIPEDIA_QUERY_TERMS）精準。
    "Sports riots",
    "Music riots",
]
_WIKIPEDIA_CATEGORY_MAX_DEPTH = 4
_WIKIPEDIA_CATEGORY_DELAY_SECONDS = 1.5


def fetch_wikipedia_category_candidates() -> List[Dict[str, Any]]:
    """遞迴爬取群眾踩踏／體育場災難相關的 Wikipedia 分類樹，回傳候選文章清單。"""
    seen_categories: set = set()
    articles: set = set()
    queue: List[tuple] = [(cat, 0) for cat in _WIKIPEDIA_CATEGORY_ROOTS]

    while queue:
        category, depth = queue.pop(0)
        if category in seen_categories or depth > _WIKIPEDIA_CATEGORY_MAX_DEPTH:
            continue
        seen_categories.add(category)

        members: List[Dict[str, Any]] = []
        for attempt in range(5):
            params = {
                "action": "query",
                "list": "categorymembers",
                "cmtitle": f"Category:{category}",
                "cmlimit": "500",
                "cmtype": "page|subcat",
                "format": "json",
            }
            try:
                response = requests.get(
                    _WIKIPEDIA_SEARCH_API, params=params, headers=_REQUEST_HEADERS, timeout=20
                )
                response.raise_for_status()
                members = response.json().get("query", {}).get("categorymembers", [])
                break
            except (requests.RequestException, ValueError) as exc:
                logger.warning(
                    "呼叫 Wikipedia 分類 API 失敗（分類：%s，第 %d 次嘗試）：%s", category, attempt + 1, exc
                )
                time.sleep(3 * (attempt + 1))
        time.sleep(_WIKIPEDIA_CATEGORY_DELAY_SECONDS)

        for member in members:
            title = member.get("title", "")
            namespace = member.get("ns")
            if namespace == 0 and title and not title.startswith(_EXCLUDED_TITLE_PREFIXES):
                articles.add(title)
            elif namespace == 14:
                queue.append((title.replace("Category:", ""), depth + 1))

    logger.info(
        "Wikipedia 分類爬取（%d 個分類，根分類：%s）共找到 %d 篇不重複候選文章",
        len(seen_categories),
        _WIKIPEDIA_CATEGORY_ROOTS,
        len(articles),
    )
    return [
        {
            "title": title,
            "url": "https://en.wikipedia.org/wiki/" + title.replace(" ", "_"),
            "domain": "en.wikipedia.org",
            "seendate": "",
        }
        for title in sorted(articles)
    ]


def fetch_article_text(url: str) -> Optional[str]:
    """擷取新聞頁面的正文文字，失敗（付費牆／JS 渲染／逾時等）回傳 None。"""
    try:
        response = requests.get(url, headers=_REQUEST_HEADERS, timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("擷取文章失敗 url=%s：%s", url, exc)
        return None

    soup = BeautifulSoup(response.text, "lxml")
    article_tag = soup.find("article")
    if article_tag is not None:
        text = article_tag.get_text(separator="\n", strip=True)
    else:
        paragraphs = soup.find_all("p")
        text = "\n".join(p.get_text(strip=True) for p in paragraphs)

    if len(text) < _MIN_ARTICLE_TEXT_LENGTH:
        logger.warning("文章全文過短（%d 字），視為擷取失敗 url=%s", len(text), url)
        return None

    return text


def summarize_article(
    title: str, article_text: str, source_url: str, date_hint: str
) -> Optional[Dict[str, Any]]:
    """呼叫本地 LLM 判斷相關性並摘要，不相關或解析失敗回傳 None。"""
    messages = NewsSummaryPromptBuilder.build(
        title=title,
        article_text=article_text[:_ARTICLE_TEXT_CHAR_LIMIT],
        source_url=source_url,
        date_hint=date_hint,
    )

    llm_client = get_llm_client()
    raw_response = llm_client.chat_completion(messages)

    try:
        data = parse_json_response(raw_response)
    except ValueError as exc:
        logger.warning("新聞摘要 LLM 輸出無法解析，跳過 url=%s：%s", source_url, exc)
        return None

    if not data.get("is_relevant"):
        logger.info("LLM 判定文章與群眾異常行為不相關，跳過 url=%s", source_url)
        return None

    # 事件發生日期優先採用 LLM 從內文判斷的結果，來源日期提示只當備援
    # （GDELT 的 seendate 未必等於事件日期，Wikipedia 更是完全沒有這個欄位）。
    event_date = data.get("event_date")
    if not event_date or event_date == "未知":
        event_date = date_hint or "未知"

    return {
        "incident_type": data.get("incident_type", "other"),
        "headline": data.get("headline") or title,
        "summary": data.get("summary", ""),
        "event_location": data.get("event_location", "未知"),
        "publish_date": event_date,
        "source_url": source_url,
    }


def _format_gdelt_date(seendate: str) -> str:
    """把 GDELT 的 seendate（如 20221030T120000Z）轉成 ISO 日期字串，解析失敗就原樣回傳。"""
    try:
        return datetime.strptime(seendate, "%Y%m%dT%H%M%SZ").date().isoformat()
    except (ValueError, TypeError):
        return seendate or "未知"


def _new_batch_path(source: str) -> Path:
    """產生本次執行要寫入的批次檔路徑（執行期間固定不變，供增量存檔使用）。"""
    _NEWS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return _NEWS_DIR / f"{source}_{timestamp}.json"


def _load_processed_urls() -> set:
    """
    掃描 data/knowledge_base_sources_news/ 底下所有既有批次檔，收集已經成功
    處理過的 source_url。

    大批次執行常常會被中途打斷（例如單次執行時間上限），這個函式讓腳本可以
    直接重複執行同一條指令來「接續」進度：已經成功摘要並存檔的文章不會被
    重複抓取、重複呼叫 LLM，只會處理清單裡還沒處理過的部分。
    """
    processed: set = set()
    if not _NEWS_DIR.exists():
        return processed

    for path in _NEWS_DIR.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        for record in data.get("records", []):
            url = record.get("source_url")
            if url:
                processed.add(url)

    return processed


def save_records(records: List[Dict[str, Any]], batch_path: Path) -> None:
    """把目前累積的記錄整批覆寫進同一個檔案。

    大批次（例如一次上百篇）需要逐篇呼叫本地 LLM，耗時可能長達一兩個小時，
    中途被中斷（網路問題、手動取消等）的機率不低。每成功一篇就立即覆寫存檔，
    即使中途被中斷，已經處理完的記錄也不會遺失，之後可以直接用 --skip-fetch
    重新索引已經存下來的部分，不用整批重新抓取、重新呼叫 LLM。
    """
    batch_path.write_text(
        json.dumps({"records": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def fetch_and_summarize(
    source: str, query: str, maxrecords: int, timespan: str, batch_path: Path, per_run_limit: Optional[int]
) -> List[Dict[str, Any]]:
    """執行「抓取 -> 取全文 -> LLM 摘要」，每篇成功就立即存檔，回傳通過篩選的記錄清單。

    每次呼叫本地 LLM 大約需要 20-30 秒，單次執行的時間上限有限，一次處理
    不完上百篇候選文章是常態。做法是：
    1. 跳過先前執行已經成功存檔過的網址（見 _load_processed_urls()）。
    2. 用 per_run_limit 限制「本次」最多實際處理（呼叫 LLM）幾篇，避免單次
       執行時間過長被強制中斷——重複執行同一條指令即可接續處理剩下的候選。
    """
    if source == "wikipedia-categories":
        articles = fetch_wikipedia_category_candidates()
    elif source == "wikipedia":
        articles = fetch_wikipedia_candidates(maxrecords)
    else:
        articles = fetch_gdelt_articles(query, maxrecords, timespan)

    if not articles:
        print("候選來源沒有回傳任何候選文章。")
        return []

    processed_urls = _load_processed_urls()
    skip_urls = processed_urls | _PERMANENTLY_EXCLUDED_URLS.keys()
    pending_articles = [a for a in articles if a.get("url") not in skip_urls]
    already_done = len(articles) - len(pending_articles)
    if already_done:
        print(f"候選 {len(articles)} 篇，其中 {already_done} 篇先前已成功處理過或已被永久排除，本次跳過。")

    if per_run_limit is not None:
        articles_to_process = pending_articles[:per_run_limit]
        remaining = len(pending_articles) - len(articles_to_process)
        if remaining > 0:
            print(f"本次限制最多處理 {per_run_limit} 篇，剩餘 {remaining} 篇留待下次執行同一條指令時接續。")
    else:
        articles_to_process = pending_articles

    records: List[Dict[str, Any]] = []
    skipped_fetch = 0
    skipped_llm = 0

    for index, article in enumerate(articles_to_process, start=1):
        url = article.get("url", "")
        title = article.get("title", "")
        print(f"[{index}/{len(articles_to_process)}] 處理中：{title or url}")

        article_text = fetch_article_text(url)
        if article_text is None:
            skipped_fetch += 1
            continue

        date_hint = _format_gdelt_date(article.get("seendate", "")) if article.get("seendate") else ""
        record = summarize_article(title, article_text, url, date_hint)
        if record is None:
            skipped_llm += 1
            continue

        record["source_domain"] = article.get("domain", "")
        records.append(record)
        save_records(records, batch_path)
        print(f"    -> 已保留（目前累計 {len(records)} 筆，已存檔）")

    print(
        f"\n本次處理 {len(articles_to_process)} 篇 -> 全文擷取失敗 {skipped_fetch} 篇 "
        f"-> LLM 判定不相關/解析失敗 {skipped_llm} 篇 -> 最終保留 {len(records)} 篇。"
    )
    return records


def index_news_sources() -> None:
    """索引 data/knowledge_base_sources_news/ 底下所有 JSON 檔案，並做一次示範查詢。"""
    document_paths = sorted(_NEWS_DIR.glob("*.json"))
    if not document_paths:
        print(f"在 {_NEWS_DIR} 底下找不到任何新聞來源檔案，請先執行抓取（不要加 --skip-fetch）。")
        return

    print(f"\n共找到 {len(document_paths)} 份新聞來源檔案，開始索引...")
    indexer = get_news_indexer()
    start = time.time()
    chunk_count = indexer.build_index_from_documents(document_paths)
    elapsed = time.time() - start
    print(f"索引完成，共寫入 {chunk_count} 個 chunk（耗時 {elapsed:.1f} 秒）。")

    retriever = get_news_retriever()
    print(f"\n=== 示範查詢：{_DEMO_QUERY} ===")
    snippets = retriever.retrieve(_DEMO_QUERY, top_k=3)
    if not snippets:
        print("（查無相關片段）")
        return
    for rank, snippet in enumerate(snippets, start=1):
        print(f"{rank}. （來源：{snippet.source_document}，相關度 {snippet.relevance_score:.2f}）")
        print(f"   {snippet.content[:150]}...")


def main() -> None:
    parser = argparse.ArgumentParser(description="抓取真實群眾異常行為事件，摘要後索引進新聞真實案例知識庫")
    parser.add_argument(
        "--source",
        choices=["wikipedia-categories", "wikipedia", "gdelt"],
        default="wikipedia",
        help="候選來源：wikipedia-categories（爬取 Category:Human stampedes 等分類樹，"
        "準確率遠高於自由關鍵字搜尋）、wikipedia（自由關鍵字搜尋）或 "
        "gdelt（近期新聞，但免費額度容易被限流封鎖）",
    )
    parser.add_argument("--query", default=_DEFAULT_QUERY, help="GDELT 查詢字串（僅 --source gdelt 時使用）")
    parser.add_argument(
        "--maxrecords",
        type=int,
        default=20,
        help="候選文章數量上限——gdelt 時是單次查詢的總筆數上限；"
        "wikipedia 時是每個關鍵字各自的筆數上限（共 %d 個關鍵字，實際去重後的候選數會更多）"
        % len(_WIKIPEDIA_QUERY_TERMS),
    )
    parser.add_argument("--timespan", default="3months", help="GDELT 搜尋的時間範圍，例如 1month、3months（僅 --source gdelt 時使用）")
    parser.add_argument(
        "--per-run-limit",
        type=int,
        default=None,
        help="本次執行最多實際處理（呼叫 LLM）幾篇候選文章，用來控制單次執行時間。"
        "已經在先前執行中成功處理過的文章一律跳過，重複執行同一條指令即可接續處理剩下的候選。",
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="跳過抓取與摘要，只重新索引 data/knowledge_base_sources_news/ 底下既有檔案",
    )
    args = parser.parse_args()

    if not args.skip_fetch:
        batch_path = _new_batch_path(args.source)
        records = fetch_and_summarize(
            args.source, args.query, args.maxrecords, args.timespan, batch_path, args.per_run_limit
        )
        if records:
            print(f"已寫入 {batch_path}（{len(records)} 筆記錄）。")
        else:
            print("沒有通過篩選的新聞記錄，不寫入檔案。")

    index_news_sources()


if __name__ == "__main__":
    main()
