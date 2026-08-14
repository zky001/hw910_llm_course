#!/usr/bin/env bash
# =============================================================================
# 昇腾训练环境一键体检脚本
# 用法: bash check_env.sh
# 逐层检查: 驱动 -> CANN -> 环境变量 -> Python/torch_npu -> NPU 可用性
# 只读操作，不改动任何配置，可放心反复执行
# =============================================================================
set -u

PASS=0; FAIL=0; WARN=0
ok()   { echo -e "  [PASS] $1"; PASS=$((PASS+1)); }
bad()  { echo -e "  [FAIL] $1"; FAIL=$((FAIL+1)); }
warn() { echo -e "  [WARN] $1"; WARN=$((WARN+1)); }

echo "==============================================="
echo " 昇腾 910 训练环境体检  $(date '+%F %T')"
echo " 主机: $(hostname)  架构: $(uname -m)"
echo "==============================================="

# ---------- 1. 驱动层 ----------
echo ""
echo "[1/5] 驱动与硬件"
if command -v npu-smi >/dev/null 2>&1; then
    ok "npu-smi 存在: $(command -v npu-smi)"
    NPU_LINES=$(npu-smi info 2>/dev/null)
    if [ $? -eq 0 ] && [ -n "$NPU_LINES" ]; then
        # 统计芯片行数（每张卡在表中有独立 Health 字段行）
        NPU_COUNT=$(echo "$NPU_LINES" | grep -cE '^\|\s+[0-9]+\s+910')
        echo "$NPU_LINES" | grep -E '^\|\s+[0-9]+\s+910' | head -8 | sed 's/^/         /'
        if [ "$NPU_COUNT" -ge 1 ]; then
            ok "识别到 $NPU_COUNT 颗 910 NPU"
            [ "$NPU_COUNT" -ne 8 ] && warn "不是 8 颗 (实际 $NPU_COUNT)——如果这台机器应有 8 卡, 去第 9 章排查"
        else
            bad "npu-smi 未列出 910 芯片"
        fi
        UNHEALTHY=$(echo "$NPU_LINES" | grep -E '^\|\s+[0-9]+\s+910' | grep -vc 'OK' || true)
        [ "$UNHEALTHY" -gt 0 ] && bad "$UNHEALTHY 张卡 Health 非 OK" || ok "所有卡 Health = OK"
    else
        bad "npu-smi 执行失败——驱动未装好或无权限"
    fi
else
    bad "找不到 npu-smi——驱动未安装 (见第 1 章 1.2 节)"
fi
if [ -f /usr/local/Ascend/driver/version.info ]; then
    ok "驱动版本: $(grep -m1 Version /usr/local/Ascend/driver/version.info)"
else
    warn "未找到 /usr/local/Ascend/driver/version.info (安装路径可能不同)"
fi

# ---------- 2. CANN ----------
echo ""
echo "[2/5] CANN"
CANN_HOME="${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}"
if [ -d "$CANN_HOME" ]; then
    VER=$(cat "$CANN_HOME/version.cfg" 2>/dev/null | grep -m1 -oE '[0-9]+\.[0-9]+[^ ]*' || true)
    ok "CANN 目录存在: $CANN_HOME ${VER:+(版本 $VER)}"
    if ls "$CANN_HOME"/opp/built-in/op_impl/ai_core/tbe/kernel 2>/dev/null | grep -q 910; then
        ok "已安装二进制算子包 (kernels)"
    else
        warn "未检测到 kernels 包——请确认已安装对应芯片的 Ascend-cann-kernels 包"
    fi
else
    bad "未找到 CANN (预期 $CANN_HOME)——见第 1 章 1.3 节"
fi

# ---------- 3. 环境变量 ----------
echo ""
echo "[3/5] 环境变量"
if [ -n "${ASCEND_TOOLKIT_HOME:-}" ]; then
    ok "ASCEND_TOOLKIT_HOME=$ASCEND_TOOLKIT_HOME"
else
    bad "ASCEND_TOOLKIT_HOME 未设置——先执行: source /usr/local/Ascend/ascend-toolkit/set_env.sh"
fi
if echo "${LD_LIBRARY_PATH:-}" | grep -q Ascend; then
    ok "LD_LIBRARY_PATH 已包含 Ascend 库路径"
else
    warn "LD_LIBRARY_PATH 不含 Ascend——大概率没有 source set_env.sh"
fi
[ -n "${ASCEND_RT_VISIBLE_DEVICES:-}" ] && warn "ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES} (已限制可见卡, 确认是有意为之)"

# ---------- 4. Python 栈 ----------
echo ""
echo "[4/5] Python / torch / torch_npu"
PY=$(command -v python || command -v python3 || true)
if [ -n "$PY" ]; then
    ok "python: $($PY -V 2>&1) @ $PY"
    $PY - <<'EOF'
import importlib, sys
def check(mod, pipname=None):
    try:
        m = importlib.import_module(mod)
        v = getattr(m, "__version__", "?")
        print(f"  [PASS] {mod} {v}")
        return m
    except Exception as e:
        print(f"  [FAIL] import {mod} 失败: {type(e).__name__}: {e}")
        return None
torch = check("torch")
tnpu  = check("torch_npu")
if torch and tnpu:
    tv = torch.__version__.split("+")[0]
    nv = tnpu.__version__
    if not nv.startswith(".".join(tv.split(".")[:3])):
        print(f"  [WARN] torch({tv}) 与 torch_npu({nv}) 版本号前缀不一致, 请核对配套表")
EOF
else
    bad "找不到 python"
fi

# ---------- 5. NPU 可用性 ----------
echo ""
echo "[5/5] NPU 运行时"
if [ -n "$PY" ]; then
    $PY - <<'EOF'
try:
    import torch, torch_npu
    if torch.npu.is_available():
        n = torch.npu.device_count()
        print(f"  [PASS] torch.npu.is_available()=True, device_count={n}")
        import torch as t
        x = t.randn(64, 64).npu()
        y = (x @ x.T).sum().cpu().item()
        print(f"  [PASS] npu:0 矩阵乘冒烟测试通过 (sum={y:.2f})")
    else:
        print("  [FAIL] torch.npu.is_available()=False——检查驱动/CANN/容器设备挂载")
except Exception as e:
    print(f"  [FAIL] NPU 冒烟测试异常: {type(e).__name__}: {e}")
EOF
else
    bad "跳过 (无 python)"
fi

echo ""
echo "==============================================="
echo " 体检结束: $PASS PASS / $WARN WARN / $FAIL FAIL"
echo " 有 FAIL 先按提示处理; 全 PASS 后跑 verify_npu.py 做 8 卡性能验证"
echo "==============================================="
[ "$FAIL" -eq 0 ]
