#!/usr/bin/env bash
# 8 卡启动 transformers+PEFT LoRA 微调
# 用法: bash run_hf_lora_8npu.sh [--data my.json] [其他 hf_peft_lora_sft.py 参数]
# 环境变量: MODEL=模型路径  NPUS=0,1,2,3,4,5,6,7
set -e
cd "$(dirname "$0")"
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true

MODEL=${MODEL:-./models/Qwen2.5-7B-Instruct}
NPUS=${NPUS:-0,1,2,3,4,5,6,7}
NPROC=$(echo "$NPUS" | awk -F',' '{print NF}')

export ASCEND_RT_VISIBLE_DEVICES=$NPUS
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
# 国内网络可解开下一行用 HF 镜像
# export HF_ENDPOINT=https://hf-mirror.com

torchrun --nproc_per_node="$NPROC" --master_port="${MASTER_PORT:-29501}" \
    hf_peft_lora_sft.py --model "$MODEL" "$@"
