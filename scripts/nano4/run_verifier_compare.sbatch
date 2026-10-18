#!/bin/bash
# ============================================================
# scripts/nano4/run_verifier_compare.sbatch
# 固定同一份 VLM 描述，輪流換驗證用的 LLM，比較判斷與耗時
# （做法與理由見 scripts/compare_verifiers.py）。
#
# 驗證模型清單（VERIFIERS）可以在送 job 前用環境變數覆寫，例如只測兩個：
#     VERIFIERS="gemma3:27b gemma3:12b" sbatch -A <計畫代號> ... run_verifier_compare.sbatch
# sbatch 預設會把送出時的環境變數帶進 job，所以這樣寫有效。
#
# 使用方式（在登入節點，project/ 目錄下執行）：
#     sbatch -A <計畫代號> --mail-user=<email> scripts/nano4/run_verifier_compare.sbatch
#
# 計畫代號與通知信箱不寫死在腳本裡，理由與替代做法見 run_batch.sbatch。
#
# 輸出：reports/verifiers_<jobid>/descriptions.json 與每個模型一份結果，
# 彙整表印在 logs/verifier-compare_<jobid>.out 最後。
#
# 新模型第一次執行要下載權重，/work 只有 100GB，送 job 前先確認空間夠。
# ============================================================
#SBATCH -p 8gpus
#SBATCH -J verifier-compare
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --exclude=25a-hgpn001
#SBATCH -o logs/%x_%j.out
#SBATCH --mail-type=END,FAIL

set -eo pipefail   # 不用 -u：conda activate 會讀到未定義的變數而中斷

INPUT=${1:-data/tinyformer7_events.json}
# 描述用 sweep 565487 裡最可靠的組合（7b 描述 + 27b 驗證）的 VLM
DESCRIBE_VLM=${DESCRIBE_VLM:-qwen2.5vl:7b}
# gemma3:27b 是比較基準，放第一個
VERIFIERS=${VERIFIERS:-"gemma3:27b gemma3:4b gemma3:12b qwen3:8b qwen3:14b"}

WORK=/work/$USER
OUT_DIR=reports/verifiers_${SLURM_JOB_ID}
mkdir -p "$OUT_DIR"

# ---------- 環境 ----------
source "$WORK/miniconda3/etc/profile.d/conda.sh"
conda activate infer
export HF_HOME=$WORK/hf_cache

echo "節點：$(hostname)  GPU：${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi --query-gpu=name,memory.total --format=csv

source scripts/nano4/ollama_lib.sh
start_ollama

unload_models() {
    "$OLLAMA_BIN" ps | awk 'NR>1 {print $1}' | xargs -r -n1 "$OLLAMA_BIN" stop || true
}

# ---------- 1. 產生描述（只跑一次） ----------
echo ""
echo "================ 描述：VLM=$DESCRIBE_VLM ================"
"$OLLAMA_BIN" pull "$DESCRIBE_VLM"
VLM_MODEL=$DESCRIBE_VLM python scripts/compare_verifiers.py describe \
    --input "$INPUT" --output "$OUT_DIR/descriptions.json"
unload_models

# ---------- 2. 每個驗證模型各跑一次 ----------
# 某個模型下載或執行失敗（例如這版 Ollama 沒有該模型）不中斷整個實驗
FAILED=()
for model in $VERIFIERS; do
    echo ""
    echo "================ 驗證：LLM=$model ================"
    if "$OLLAMA_BIN" pull "$model" && LLM_MODEL=$model python scripts/compare_verifiers.py verify \
        --descriptions "$OUT_DIR/descriptions.json" --output "$OUT_DIR/${model//[:\/]/_}.json"; then
        :
    else
        FAILED+=("$model")
    fi
    unload_models
done

# ---------- 3. 彙整 ----------
echo ""
echo "================ 彙整 ================"
python scripts/compare_verifiers.py summarize "$OUT_DIR" --reference "${VERIFIERS%% *}"

if [ ${#FAILED[@]} -gt 0 ]; then
    echo "失敗的模型：${FAILED[*]}"
    exit 1
fi
