#!/bin/bash
# ============================================================
# scripts/nano4/ollama_lib.sh
# run_batch.sbatch / run_model_sweep.sbatch 共用：在計算節點上啟動
# 使用者層級的 Ollama 服務。用 source 載入，不直接執行。
#
# 呼叫 start_ollama 之後：
#   - OLLAMA_BASE_URL、OLLAMA_NUM_THREAD 已 export，config/settings.py 會讀到
#   - 工作結束（EXIT）時自動關閉 Ollama
# ============================================================

OLLAMA_BIN=/work/$USER/ollama/bin/ollama

start_ollama() {
    # 沒裝好就直接停，不要進到下面的等待迴圈白等 2 分鐘才報錯
    if [ ! -x "$OLLAMA_BIN" ]; then
        echo "找不到 $OLLAMA_BIN，請先在登入節點執行：bash scripts/nano4/setup_env.sh"
        exit 1
    fi

    export OLLAMA_MODELS=/work/$USER/ollama_models

    # H200 上 3b/4b 模型生成只有每秒 30-40 token，log 顯示兩個可疑設定：
    # - context：Ollama 依顯存自動開到 128k（模型上限），實際 prompt 約 3k，
    #   加上 VLM 生成上限 1024 也不到 5k，8192 綽綽有餘
    # - 執行緒：llama-server 依實體核數開 112 個，但 job 只配到
    #   --cpus-per-task 顆，執行緒互搶。改由 client 在請求帶 num_thread
    #   （config/settings.py 的 ollama_num_thread）
    # 都可以在送 job 前自己 export 覆蓋，方便做對照。
    export OLLAMA_CONTEXT_LENGTH=${OLLAMA_CONTEXT_LENGTH:-8192}
    export OLLAMA_NUM_THREAD=${OLLAMA_NUM_THREAD:-${SLURM_CPUS_PER_TASK:-8}}
    # llm/reasoning_pipeline.py 會同時送出推論與摘要兩個請求，Ollama 預設一次
    # 只處理 1 個，要設 2 才會真的平行
    export OLLAMA_NUM_PARALLEL=${OLLAMA_NUM_PARALLEL:-2}
    echo "OLLAMA_CONTEXT_LENGTH=$OLLAMA_CONTEXT_LENGTH  OLLAMA_NUM_THREAD=$OLLAMA_NUM_THREAD  OLLAMA_NUM_PARALLEL=$OLLAMA_NUM_PARALLEL"

    # 同一個節點可能同時有別人的 Ollama 在跑，用 job id 算出不衝突的 port，
    # 再透過 OLLAMA_BASE_URL 環境變數傳給 config/settings.py。
    local port=$((20000 + SLURM_JOB_ID % 20000))
    export OLLAMA_HOST=127.0.0.1:$port
    export OLLAMA_BASE_URL=http://$OLLAMA_HOST

    "$OLLAMA_BIN" serve > "logs/ollama_${SLURM_JOB_ID}.log" 2>&1 &
    OLLAMA_PID=$!
    trap 'kill $OLLAMA_PID 2>/dev/null || true' EXIT

    for _ in $(seq 1 60); do
        curl -s "$OLLAMA_BASE_URL/api/version" > /dev/null && break
        sleep 2
    done
    curl -s "$OLLAMA_BASE_URL/api/version" > /dev/null || {
        echo "Ollama 啟動失敗，見 logs/ollama_${SLURM_JOB_ID}.log"
        exit 1
    }
    echo "Ollama 已啟動：$OLLAMA_BASE_URL"
}
