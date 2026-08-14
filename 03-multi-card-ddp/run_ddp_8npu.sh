#!/usr/bin/env bash
# 8 卡 DDP 启动脚本。用法: bash run_ddp_8npu.sh [脚本额外参数...]
# 改用部分卡: NPUS=0,1,2,3 bash run_ddp_8npu.sh
set -e
cd "$(dirname "$0")"

# CANN 环境（已写进 ~/.bashrc 的话这行是幂等的）
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true

NPUS=${NPUS:-0,1,2,3,4,5,6,7}
NPROC=$(echo "$NPUS" | awk -F',' '{print NF}')
export ASCEND_RT_VISIBLE_DEVICES=$NPUS

# 常用运行期设置（详见第 8 章）
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True   # 缓解显存碎片
export HCCL_CONNECT_TIMEOUT=300                          # 大任务建链慢时加大

echo ">>> 使用 NPU: $NPUS (共 $NPROC 卡)"
torchrun --nproc_per_node="$NPROC" \
         --master_addr=127.0.0.1 \
         --master_port="${MASTER_PORT:-29500}" \
         ddp_tiny_gpt.py "$@"
