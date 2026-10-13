#!/bin/bash
# ============================================================
# scripts/nano4/run_batch.sbatch
# 在晶創26（Nano4）的 H200 計算節點上跑推論端批次 pipeline。
#
# 使用方式（在登入節點，project/ 目錄下執行）：
#     sbatch -A <計畫代號> --mail-user=<email> scripts/nano4/run_batch.sbatch
#     sbatch -A <計畫代號> --mail-user=<email> scripts/nano4/run_batch.sbatch \
#         data/tinyformer7_events.json my_report.json
#
# 計畫代號與通知信箱刻意不寫死在腳本裡（repo 是公開的），送出時用
# -A / --mail-user 帶入；計畫代號也可以改成在 ~/.bashrc 設定
# export SBATCH_ACCOUNT=<計畫代號>，之後就不用每次打 -A。
#
# 流程：啟動 Ollama 服務（背景）-> 確認模型已下載 ->
#       跑 scripts/run_tinyformer_batch.py -> 結束時關閉 Ollama。
#
# 前置：先跑過一次 scripts/nano4/setup_env.sh。
# 要比較不同大小的模型，改用 scripts/nano4/run_model_sweep.sbatch。
# ============================================================
#SBATCH -p dev
#SBATCH -J infer-batch
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --exclude=25a-hgpn001
#SBATCH -o logs/%x_%j.out
#SBATCH --mail-type=END,FAIL

set -eo pipefail   # 不用 -u：conda activate 會讀到未定義的變數而中斷

INPUT=${1:-data/tinyformer7_events.json}
OUTPUT=${2:-tinyformer7_pipeline_report_nano4_${SLURM_JOB_ID}.json}

WORK=/work/$USER

# ---------- 環境 ----------
source "$WORK/miniconda3/etc/profile.d/conda.sh"
conda activate infer

# 模型快取都放 /work（/home 只有 100GB）。這些變數要在 python 啟動前設好：
# config/settings.py 的 load_dotenv() 不會覆寫已存在的環境變數，
# 所以本機 .env 裡指向 D:/ 的 HF_HOME 不會蓋掉這裡的設定。
export HF_HOME=$WORK/hf_cache

echo "節點：$(hostname)  GPU：${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi --query-gpu=name,memory.total --format=csv

source scripts/nano4/ollama_lib.sh
start_ollama

# 已下載過的模型 pull 只會檢查版本，很快
"$OLLAMA_BIN" pull qwen2.5vl:3b
"$OLLAMA_BIN" pull gemma3:4b

# ---------- 跑批次 pipeline ----------
python scripts/run_tinyformer_batch.py --input "$INPUT" --output "$OUTPUT"
