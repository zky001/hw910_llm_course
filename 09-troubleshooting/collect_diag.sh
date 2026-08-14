#!/usr/bin/env bash
# =============================================================================
# 一键收集昇腾训练环境诊断信息, 打包成 diag_<时间戳>.tar.gz
# 只读操作。求助他人/提工单前先跑一次, 附上产物。
# 用法: bash collect_diag.sh
# =============================================================================
set -u
TS=$(date +%Y%m%d_%H%M%S)
OUT="diag_$TS"
mkdir -p "$OUT"

log() { echo ">>> $1"; }

log "1/7 系统信息"
{
    echo "== date ==";        date
    echo "== hostname ==";    hostname
    echo "== uname -a ==";    uname -a
    echo "== cpu ==";         lscpu 2>/dev/null | grep -E 'Architecture|Model name|^CPU\(s\)'
    echo "== mem ==";         free -h
    echo "== disk ==";        df -h | head -20
} > "$OUT/system.txt" 2>&1

log "2/7 NPU 状态 (npu-smi)"
{
    npu-smi info
    echo ""
    for i in 0 1 2 3 4 5 6 7; do
        npu-smi info -t health -i "$i" 2>/dev/null
    done
} > "$OUT/npu-smi.txt" 2>&1

log "3/7 驱动与 CANN 版本"
{
    echo "== driver ==";  cat /usr/local/Ascend/driver/version.info 2>/dev/null
    echo "== cann ==";    cat "${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}/version.cfg" 2>/dev/null
    ls /usr/local/Ascend/ascend-toolkit/ 2>/dev/null
} > "$OUT/versions.txt" 2>&1

log "4/7 Python 栈版本"
{
    python -V 2>&1
    pip list 2>/dev/null | grep -iE 'torch|deepspeed|transformers|peft|accelerate|mindspeed|llamafactory|numpy|apex|vllm'
    python - <<'EOF' 2>&1
try:
    import torch, torch_npu
    print("torch:", torch.__version__, "| torch_npu:", torch_npu.__version__)
    print("npu available:", torch.npu.is_available(),
          "| count:", torch.npu.device_count())
except Exception as e:
    print("import 失败:", type(e).__name__, e)
EOF
} > "$OUT/python.txt" 2>&1

log "5/7 相关环境变量"
env | grep -E 'ASCEND|HCCL|TORCH|PYTORCH|TASK_QUEUE|CPU_AFFINITY|LD_LIBRARY|MASTER_|WORLD_SIZE|RANK' \
    | sort > "$OUT/env.txt" 2>&1

log "6/7 最近的 plog 错误 (~/ascend/log)"
PLOG_DIR="$HOME/ascend/log"
if [ -d "$PLOG_DIR" ]; then
    # 最近修改的 5 个 plog 里抓 ERROR/EVENT, 各保留尾部片段
    find "$PLOG_DIR" -name '*.log' -mmin -720 2>/dev/null \
        | xargs -r ls -t 2>/dev/null | head -5 | while read -r f; do
        echo "===== $f ====="
        grep -E 'ERROR|EVENT' "$f" 2>/dev/null | tail -100
        echo ""
    done > "$OUT/plog_errors.txt" 2>&1
    [ -s "$OUT/plog_errors.txt" ] || echo "(近 12 小时 plog 无 ERROR)" > "$OUT/plog_errors.txt"
else
    echo "(未找到 $PLOG_DIR)" > "$OUT/plog_errors.txt"
fi

log "7/7 dmesg 摘录 (需要权限, 失败不影响其他项)"
dmesg 2>/dev/null | grep -iE 'davinci|npu|ascend|ecc' | tail -100 > "$OUT/dmesg.txt" 2>&1 || \
    echo "(无权限读取 dmesg, 可用 sudo 重跑本脚本)" > "$OUT/dmesg.txt"

tar -czf "diag_$TS.tar.gz" "$OUT" && rm -rf "$OUT"
echo ""
echo "✅ 诊断包已生成: diag_$TS.tar.gz"
echo "   求助时请附上: 现象一句话 + 最小复现命令 + 首个报错 rank 的完整栈 + 本诊断包"
