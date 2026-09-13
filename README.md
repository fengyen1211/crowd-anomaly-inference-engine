# Crowd Anomaly Inference Engine

群眾異常行為監控系統的**推論端**：接收影像端（偵測＋追蹤模型）輸出的事件 JSON，透過 VLM（視覺語言模型）理解畫面、RAG 檢索相似歷史案例與法規知識庫、LLM 綜合推理，產生事件摘要、可能原因、應變建議與警報等級。

完全採用本地 AI（Embedding／VLM／LLM 皆可跑在本機，透過 [Ollama](https://ollama.com) 執行量化模型），不強制依賴任何付費 API。

## 架構

```
影像端輸出 JSON
    │
    ▼
Ingestion          驗證欄位／型別（Pydantic）
    │
    ▼
Frame Loader       依 bbox 讀取畫面並外擴裁切
    │
    ▼
VLM                描述畫面內容（預設 Ollama + Qwen2.5-VL 3b）
    │
    ▼
Embedding          視覺向量（CLIP）＋ 文字向量（bge-m3），分屬不同向量空間
    │
    ▼
Retriever          視覺相似歷史事件 ／ 知識庫（法規SOP）／ 新聞真實案例
    │
    ▼
LLM 推理           驗證判定、可能原因、應變建議、事件摘要（預設 Ollama + Gemma3 4b）
    │
    ▼
Alert Composer     純規則邏輯決定警報等級（info / warning / critical）
    │
    ▼
輸出：{ summary, event_type, possible_causes, recommendation, alert_level }
```

所有 AI 能力（Embedding／VLM／LLM）都透過抽象介面 + 依賴注入組裝（見 `app/dependencies.py`），要更換模型只需要改 `config.yaml`，不需要修改任何業務邏輯程式碼。

## 環境需求

- Python 3.12
- [Ollama](https://ollama.com)（本地執行 VLM／LLM，預設不需要任何雲端 API）
- 建議至少 4GB VRAM 的 GPU（CPU 也能跑，但速度會慢很多）

## 安裝

```bash
# 1. 安裝 Python 套件
pip install -r requirements.txt

# 2. 複製環境變數範例檔，依需要調整
cp .env.example .env

# 3. 安裝並啟動 Ollama，下載預設用到的模型
ollama pull qwen2.5vl:3b
ollama pull gemma3:4b
ollama serve
```

Embedding 模型（`BAAI/bge-m3`、CLIP `openai/clip-vit-base-patch32`）會在第一次執行時自動從 HuggingFace 下載，不需要手動安裝；若你的網路連不到 huggingface.co 但模型已經下載過，可以在 `.env` 加上：

```
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

強制使用本機快取，不再連網確認版本。

## 設定模型 Provider

`config.yaml` 集中管理 Embedding／VLM／LLM 要用哪個 provider、哪個模型，修改這個檔案即可切換，不需要碰程式碼：

```yaml
embedding:
  provider: sentence_transformer   # mock / sentence_transformer / openai
  model: BAAI/bge-m3

visual_embedding:
  provider: clip                   # mock / clip / siglip
  model: openai/clip-vit-base-patch32

llm:
  provider: ollama                 # mock / openai / ollama
  model: gemma3:4b

vlm:
  provider: ollama                 # mock / qwen2.5-vl / internvl / llava / minicpm_v / ollama
  model: qwen2.5vl:3b
```

想在不啟動任何真實模型的情況下先測流程，把 `llm.provider` / `vlm.provider` 改成 `mock` 即可，幾秒內跑完，回傳的是假資料。

## 測試

### 方式一：CLI 直接跑 pipeline（不需要開伺服器）

```bash
python scripts/run_pipeline.py --input data/tinyformer7_events_rawframes.json --index 0
```

`--index` 可選 0~3，對應四種聚合後的群體異常事件類型（`sprint_spike` / `crowd_locked` / `panic_scatter` / `counter_flow`）。單筆事件（VLM + LLM 各一次呼叫）在 RTX 3050 Ti（4GB VRAM）上大約需要 70–90 秒。

也可以用 42 筆的示範資料集（畫面是佔位圖，只能驗證流程通不通，不能評估 VLM 觀察品質）：

```bash
python scripts/make_placeholder_frames.py --input data/dataset7_demo_event.json
python scripts/run_pipeline.py --input data/dataset7_demo_event.json --index 0
```

### 方式二：真的透過 HTTP API

先在 `.env` 設定 `INFERENCE_API_KEY`（見 `.env.example`），沒設定的話伺服器會直接拒絕請求：

```bash
uvicorn app.main:app --reload
```

```bash
curl -X POST http://127.0.0.1:8000/events/ingest \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <.env 裡設定的 INFERENCE_API_KEY>" \
  -d @<單筆事件的 JSON 檔案>
```

`/events/ingest` 目前只接受單筆事件的 dict，不接受 `{"records": [...]}` 包裝格式。

### 建置知識庫（法規／SOP 檢索）

專案已附上 4 份台灣官方公開的消防／群眾安全法規文件（`data/knowledge_base_sources/`），索引指令：

```bash
python scripts/index_knowledge_base.py
```

沒有先索引也能跑，只是 `possible_causes` / `recommendation` 會誠實回報「證據不足」，不會編造內容。

## 已知限制

- **VLM 交叉驗證能力尚未實作**：`vlm/observation_builder.py` 目前 `verification_verdict` 固定回傳 `UNCERTAIN`，真正的驗證判斷由後面的 LLM 步驟代做。
- **無資料持久化**：`services/ingestion_service.py` 只做驗證，pipeline 每次都是無狀態單次執行，`/events/{id}/analysis`、`/events/{id}/reanalyze` 尚未實作。
- **3B 級量化模型偶爾會誤判場景類型**（例如把劇院走道誤判成停車場），這是硬體限制（4GB VRAM 只能塞 3B 級模型）下的模型容量問題，不是資料處理問題。
- 目前所有驗證都基於 3D 渲染的示範影片，尚未在真實監控畫面上測試過。

## 安全性注意事項

`events` / `alerts` 底下的端點都已經掛上 API Key 驗證（`app/security.py`，Header 帶 `X-API-Key`，未設定 `INFERENCE_API_KEY` 時伺服器直接拒絕啟用服務），但這只解決「陌生人完全無法呼叫」的問題，以下兩點仍然存在、需要留意：

- **Prompt injection**：`caption`／`predicted_event_type` 是完全開放的自由文字欄位，會直接拼進 LLM 的 prompt（`prompts/reasoning_prompt.py`）。就算呼叫者持有合法的 API Key，仍然可以在這些欄位塞入指令性文字，試圖操控 LLM 輸出的 `summary`／`possible_causes`／`recommendation` 內容。`alert_level`（警報等級）是純規則邏輯決定（`services/alert_service.py`），不吃 LLM 輸出，不受影響，但顯示給人看的說明文字有可能被操控。
- **沒有流量限制**：單次請求會真的觸發 VLM＋LLM 運算（實測 70–90 秒），API Key 只能擋掉「沒有 Key 的人」，持有合法 Key 的呼叫端（或 Key 外流）仍然可以用很少的請求把運算資源（GPU／CPU）耗光，沒有 rate limiting 機制。
- **單一共用 Key，不是多租戶權限系統**：目前所有端點共用同一組 `INFERENCE_API_KEY`，沒有「不同呼叫端拿不同權限」的概念，正式上線前應該視情況換成每個呼叫端各自的 Key 或更完整的驗證機制（OAuth／JWT）。

另外，`services/frame_service.py` 的 `frame_file` 讀取路徑已修正過路徑穿越（path traversal）漏洞——過去 `frame_file` 若帶入絕對路徑或 `../` 相對路徑，可能被用來讀取伺服器上任意檔案，現在會驗證解析後的路徑是否仍落在 `frames_dir` 底下，超出範圍一律拒絕。

## 專案結構

```
app/            FastAPI 進入點、依賴注入組裝
config/         集中設定管理（config.yaml + .env）
embedding/      文字／視覺 embedding provider（sentence_transformer / clip / openai / mock）
vlm/            視覺語言模型 provider（ollama / qwen2.5-vl / internvl / llava / minicpm_v / mock）
llm/            LLM provider + 推理串接邏輯
rag/            知識庫／視覺相似度檢索、索引建立
services/       業務邏輯（驗證、畫面讀取裁切、警報分級、Orchestrator）
prompts/        LLM Prompt 組裝
routers/        對外 API
scripts/        CLI 工具（跑 pipeline、建索引、選型比較等）
```

## License

本專案採用 [MIT License](LICENSE)：可以自由使用、修改、散布，包含商業用途，唯一要求是保留原始的版權聲明。
