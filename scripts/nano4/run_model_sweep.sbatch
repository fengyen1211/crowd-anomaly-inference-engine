#!/bin/bash
# ============================================================
# scripts/nano4/run_model_sweep.sbatch
# 大小模型對照實驗：同一份事件資料，輪流用不同大小的 VLM / LLM 跑完整
# pipeline，每一組各產生一份報告，最後用 scripts/compare_model_reports.py
# 彙整比較。
#
# 本機只有 4GB VRAM，config.yaml 選的是 qwen2.5vl:3b + gemma3:4b；
# H200 有 141GB 顯存，可以測更大的模型。實驗設計一次只換一個元件，
# 才分得出品質變化是 VLM 還是 LLM 造成的：
#
#   baseline   qwen2.5vl:3b   + gemma3:4b    （目前 config.yaml 的設定）
#   vlm_7b     qwen2.5vl:7b   + gemma3:4b    只放大 VLM
#   llm_27b    qwen2.5vl:3b   + gemma3:27b   只放大 LLM
#   both_mid   qwen2.5vl:7b   + gemma3:27b   兩者都放大（模型跟上面兩組共用，不多佔空間）
#
# 原本還有 qwen2.5vl:72b 的兩組（vlm_72b、both_big），但 72b 權重約 49GB，
# 加上其他模型與 conda 環境會超過 /work 的 100GB 免費額度，先拿掉。
#
# 使用方式（在登入節點，project/ 目錄下執行）：
#     sbatch -A <計畫代號> --mail-user=<email> scripts/nano4/run_model_sweep.sbatch
#     sbatch -A <計畫代號> --mail-user=<email> scripts/nano4/run_model_sweep.sbatch \
#         data/tinyformer7_events.json
#
# 計畫代號與通知信箱不寫死在腳本裡，理由與替代做法見 run_batch.sbatch。
#
# 輸出：reports/sweep_<jobid>/<組名>.json，跑完後執行
#     python scripts/compare_model_reports.py reports/sweep_<jobid>
#
# 第一次執行要下載約 30GB 模型權重（27b 約 17GB、7b 約 6GB），
# 所以時間給得比 run_batch.sbatch 寬；之後重跑只會檢查版本。
# ============================================================
#SBATCH -p 8gpus
#SBATCH -J model-sweep
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --exclude=25a-hgpn001
#SBATCH -o logs/%x_%j.out
#SBATCH --mail-type=END,FAIL

set -eo pipefail   # 不用 -u：conda activate 會讀到未定義的變數而中斷

INPUT=${1:-data/tinyformer7_events.json}

# 組名 VLM LLM
ARMS=(
    "baseline qwen2.5vl:3b  gemma3:4b"
    "vlm_7b   qwen2.5vl:7b  gemma3:4b"
    "llm_27b  qwen2.5vl:3b  gemma3:27b"
    "both_mid qwen2.5vl:7b  gemma3:27b"
)

WORK=/work/$USER
OUT_DIR=reports/sweep_${SLURM_JOB_ID}
mkdir -p "$OUT_DIR"

# ---------- 環境 ----------
source "$WORK/miniconda3/etc/profile.d/conda.sh"
conda activate infer
export HF_HOME=$WORK/hf_cache

echo "節點：$(hostname)  GPU：${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi --query-gpu=name,memory.total --format=csv

source scripts/nano4/ollama_lib.sh
start_ollama

# ---------- 先把所有用得到的模型下載好 ----------
for model in $(printf '%s\n' "${ARMS[@]}" | awk '{print $2; print $3}' | sort -u); do
    echo "下載／檢查模型：$model"
    "$OLLAMA_BIN" pull "$model"
done

# ---------- 逐組跑 ----------
# 每組用獨立的 python process：config 跟各個 client 都是 lru_cache 單例，
# 同一個 process 裡換環境變數不會生效。
# 某一組失敗不中斷整個實驗，記下來最後一起回報。
FAILED=()
for arm in "${ARMS[@]}"; do
    read -r name vlm llm <<< "$arm"
    echo ""
    echo "================ $name：VLM=$vlm  LLM=$llm ================"
    if VLM_MODEL=$vlm LLM_MODEL=$llm \
        python scripts/run_tinyformer_batch.py --input "$INPUT" --output "$OUT_DIR/$name.json"; then
        :
    else
        FAILED+=("$name")
    fi
    # 換組前把模型卸出顯存，避免上一組的模型還佔著 GPU，影響下一組的耗時
    "$OLLAMA_BIN" ps | awk 'NR>1 {print $1}' | xargs -r -n1 "$OLLAMA_BIN" stop || true
done

echo ""
echo "報告都在 $OUT_DIR/，比較：python scripts/compare_model_reports.py $OUT_DIR"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "失敗的組別：${FAILED[*]}"
    exit 1
fi
