#!/bin/bash
# ============================================================
# scripts/nano4/ollama_lib.sh
# run_batch.sbatch / run_model_sweep.sbatch 共用：在計算節點上啟動
# 使用者層級的 Ollama 服務。用 source 載入，不直接執行。
#
# 呼叫 start_ollama 之後：
#   - OLLAMA_BASE_URL 已 export，config/settings.py 會讀到
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
