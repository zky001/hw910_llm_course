#!/usr/bin/env bash
# =============================================================================
# 预训练数据预处理: jsonl -> Megatron .bin/.idx 二进制格式
# 用法(在 MindSpeed-LLM 目录执行, 或用 MSLLM_DIR 指定):
#   bash preprocess_data.sh INPUT=./raw/corpus.jsonl \
#        TOKENIZER=./models/Qwen2.5-7B OUTPUT_PREFIX=./dataset/corpus
# 输入格式: 每行 {"text": "一篇文档全文"}
# =============================================================================
set -e
for kv in "$@"; do export "$kv"; done

MSLLM_DIR=${MSLLM_DIR:-$HOME/mindspeed_work/MindSpeed-LLM}
INPUT=${INPUT:?必须指定 INPUT=jsonl 路径}
TOKENIZER=${TOKENIZER:?必须指定 TOKENIZER=HF tokenizer 目录}
OUTPUT_PREFIX=${OUTPUT_PREFIX:-./dataset/corpus}
WORKERS=${WORKERS:-16}

source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
cd "$MSLLM_DIR"
mkdir -p "$(dirname "$OUTPUT_PREFIX")"

python preprocess_data.py \
    --input "$INPUT" \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path "$TOKENIZER" \
    --output-prefix "$OUTPUT_PREFIX" \
    --json-keys text \
    --workers "$WORKERS" \
    --log-interval 1000

echo ""
echo "✅ 产物: ${OUTPUT_PREFIX}_text_document.bin / .idx"
echo "   训练脚本 --data-path 填前缀: ${OUTPUT_PREFIX}_text_document"
