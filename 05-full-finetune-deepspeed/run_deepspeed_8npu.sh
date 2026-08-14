#!/usr/bin/env bash
# 8 卡全参 SFT (DeepSpeed ZeRO)
# 用法:
#   bash run_deepspeed_8npu.sh --data my.json
#   DS_CONFIG=ds_zero3.json bash run_deepspeed_8npu.sh --data my.json
# 环境变量: MODEL=模型路径  NPUS=可见卡  DS_CONFIG=ds配置(默认 ds_zero2.json)
set -e
cd "$(dirname "$0")"
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true

MODEL=${MODEL:-./models/Qwen2.5-7B-Instruct}
NPUS=${NPUS:-0,1,2,3,4,5,6,7}
DS_CONFIG=${DS_CONFIG:-ds_zero2.json}
NPROC=$(echo "$NPUS" | awk -F',' '{print NF}')

export ASCEND_RT_VISIBLE_DEVICES=$NPUS
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_CONNECT_TIMEOUT=600

echo ">>> 全参训练: $MODEL | ZeRO 配置: $DS_CONFIG | 卡: $NPUS"
torchrun --nproc_per_node="$NPROC" --master_port="${MASTER_PORT:-29502}" \
    train_sft_trainer.py \
    --model "$MODEL" \
    --deepspeed "$DS_CONFIG" \
    "$@"
