# -*- coding: utf-8 -*-
"""
NPU "Hello World"：用 5 分钟过一遍 torch_npu 的基础 API。
用法: python hello_npu.py
"""
import time

import torch
import torch_npu  # noqa: F401  # ① 导入即注册 "npu" 设备，必须在使用任何 npu 功能之前

# ---------------------------------------------------------------- 设备信息
assert torch.npu.is_available(), "NPU 不可用，先按第 1 章排查环境"
n = torch.npu.device_count()
print(f"torch {torch.__version__} | torch_npu {torch_npu.__version__}")
print(f"可见 NPU 数量: {n}（受 ASCEND_RT_VISIBLE_DEVICES 影响）")
for i in range(n):
    print(f"  npu:{i}  {torch.npu.get_device_name(i)}")

torch.npu.set_device(0)          # ② 选卡，等价于 CUDA 的 torch.cuda.set_device
torch.npu.manual_seed_all(42)    #    随机种子同理

# ---------------------------------------------------------------- 张量上卡
x = torch.randn(1024, 1024)
x_npu = x.npu()                  # 三种写法等价:
x_npu = x.to("npu")              #   .npu() / .to("npu") / .to(torch.device("npu", 0))
print(f"\n张量设备: {x_npu.device}, dtype: {x_npu.dtype}")

# 计算结果与 CPU 对齐（浮点误差范围内）
diff = (x_npu @ x_npu.T).cpu() - x @ x.T
print(f"NPU 与 CPU matmul 最大误差: {diff.abs().max().item():.2e}")

# ---------------------------------------------------------------- 异步执行与计时
# NPU 和 CUDA 一样是异步执行：不 synchronize 测到的只是"算子下发"时间
a = torch.randn(4096, 4096, dtype=torch.float16, device="npu")
b = torch.randn(4096, 4096, dtype=torch.float16, device="npu")
for _ in range(3):               # warmup（首次可能触发算子编译/加载）
    _ = a @ b
torch.npu.synchronize()

t0 = time.perf_counter(); _ = a @ b; t1 = time.perf_counter()
torch.npu.synchronize();          t2 = time.perf_counter()
print(f"\n不同步的'耗时'(仅下发): {(t1 - t0) * 1e3:.3f} ms")
print(f"同步后的真实耗时:        {(t2 - t0) * 1e3:.3f} ms")

# ---------------------------------------------------------------- 显存查询
mem = torch.npu.memory_allocated() / 2**20
peak = torch.npu.max_memory_allocated() / 2**20
print(f"\n当前显存占用: {mem:.1f} MB, 峰值: {peak:.1f} MB")
print("（训练时配合 watch -n 1 npu-smi info 观察整卡 HBM 占用）")

# ---------------------------------------------------------------- 自动求导冒烟
w = torch.randn(128, 128, device="npu", requires_grad=True)
loss = (w @ w).sin().sum()
loss.backward()
print(f"\nautograd 正常: grad norm = {w.grad.norm().item():.4f}")

print("\n✅ hello_npu 全部通过。下一步: python train_tiny_gpt.py")
