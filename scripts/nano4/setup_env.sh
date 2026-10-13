#!/bin/bash
# ============================================================
# scripts/nano4/setup_env.sh
# 在國網中心晶創26（Nano4）上建立推論端執行環境，只需要跑一次。
#
# 使用方式（在登入節點，project/ 目錄下執行）：
#     bash scripts/nano4/setup_env.sh
#
# 做的事情：
#   1. 建立 conda 環境 infer（Python 3.12，對應 .python-version）
#   2. pip install -r requirements.txt
#   3. 把 Ollama 安裝到 /work/$USER/ollama（沒有 sudo，不能用官方 install.sh）
#
# 模型權重（qwen2.5vl:3b / gemma3:4b / bge-m3 / CLIP）不在這裡下載，
# 由 run_batch.sbatch 第一次執行時在計算節點上自動下載。
# ============================================================
set -eo pipefail   # 不用 -u：conda activate 會讀到未定義的變數而中斷

WORK=/work/$USER
CONDA_ROOT=$WORK/miniconda3
OLLAMA_DIR=$WORK/ollama

source "$CONDA_ROOT/etc/profile.d/conda.sh"

# ---------- 1. conda 環境 ----------
if ! conda env list | grep -q "^infer "; then
    conda create -n infer python=3.12 -y
fi
conda activate infer

# ---------- 2. Python 套件 ----------
export PIP_CACHE_DIR=$WORK/pip_cache
pip install -r requirements.txt

# ---------- 3. Ollama（使用者層級安裝） ----------
if [ ! -x "$OLLAMA_DIR/bin/ollama" ]; then
    mkdir -p "$OLLAMA_DIR"
    cd "$OLLAMA_DIR"
    # 官方 Linux 發行檔依版本可能是 .tgz 或 .tar.zst，兩種都試。
    # 已經下載過的壓縮檔（約 1.4GB）直接沿用，不重新下載。
    if [ ! -f ollama.tar.zst ] && curl -fL https://ollama.com/download/ollama-linux-amd64.tgz -o ollama.tgz; then
        tar -xzf ollama.tgz && rm ollama.tgz
    else
        [ -f ollama.tar.zst ] || curl -fL https://ollama.com/download/ollama-linux-amd64.tar.zst -o ollama.tar.zst
        # Nano4 的 tar 不支援 --zstd，改用 zstd 解壓後再交給 tar；
        # 系統沒有 zstd 就裝進目前的 conda 環境（不需要 sudo）
        command -v zstd > /dev/null || conda install -y zstd
        zstd -dc ollama.tar.zst | tar -xf - && rm ollama.tar.zst
    fi
    cd - > /dev/null
fi
"$OLLAMA_DIR/bin/ollama" --version || true

echo ""
echo "環境建立完成。下一步：sbatch -A <計畫代號> --mail-user=<email> scripts/nano4/run_batch.sbatch"
