# 附录 C · CUDA → NPU 迁移对照表

> 给带着 NVIDIA 经验来的工程师的「翻译词典」。原则：**`cuda` 换 `npu`、NCCL 换 HCCL、nvidia-smi 换 npu-smi，其余习惯基本保留。**

## C.1 概念与工具对照

| NVIDIA 世界 | 昇腾世界 | 备注 |
|---|---|---|
| CUDA Toolkit + cuDNN | **CANN**（Toolkit + Kernels） | `source .../set_env.sh` 注入环境 |
| GPU Driver | NPU 驱动 + 固件 | |
| NCCL | **HCCL** | `init_process_group("hccl")` |
| NVLink / NVSwitch | **HCCS** 机内互联 | 单机 8 卡高速互联 |
| InfiniBand | NPU 自带 RoCE 网口 | 多机场景，`hccn_tool` 配 IP |
| `nvidia-smi` | **`npu-smi info`** | |
| `CUDA_VISIBLE_DEVICES` | **`ASCEND_RT_VISIBLE_DEVICES`** | |
| `CUDA_LAUNCH_BLOCKING` | `ASCEND_LAUNCH_BLOCKING` | 调试同步模式 |
| `PYTORCH_CUDA_ALLOC_CONF` | `PYTORCH_NPU_ALLOC_CONF` | `expandable_segments:True` 同样适用 |
| Nsight Systems / nvprof | **Ascend PyTorch Profiler + MindStudio Insight**、msprof | 第 7 章 |
| nccl-tests | `hccl_test`（CANN 自带）或本课程 `allreduce_bench.py` | |
| FlashAttention 库 | `torch_npu.npu_fusion_attention`（内置） | 仅 910B；无需单独装包 |
| apex FusedAdam | `torch_npu.optim.NpuFusedAdamW` | |
| SM / CUDA Core | AI Core（+ AI CPU 慢速路径） | AI_CPU 是性能陷阱，见第 8 章 |
| Tensor Core 精度 FP16/BF16/TF32 | 910A: FP16；910B: FP16/BF16 | 无 TF32 概念 |

## C.2 PyTorch API 对照

| CUDA | NPU |
|---|---|
| `import torch` 即可 | `import torch` **+ `import torch_npu`** |
| `torch.device("cuda", i)` / `"cuda:0"` | `torch.device("npu", i)` / `"npu:0"` |
| `t.cuda()` / `t.to("cuda")` | `t.npu()` / `t.to("npu")` |
| `torch.cuda.is_available()` | `torch.npu.is_available()` |
| `torch.cuda.device_count()` | `torch.npu.device_count()` |
| `torch.cuda.set_device(i)` | `torch.npu.set_device(i)` |
| `torch.cuda.current_device()` | `torch.npu.current_device()` |
| `torch.cuda.synchronize()` | `torch.npu.synchronize()` |
| `torch.cuda.manual_seed_all(s)` | `torch.npu.manual_seed_all(s)` |
| `torch.cuda.empty_cache()` | `torch.npu.empty_cache()` |
| `torch.cuda.memory_allocated/max_memory_allocated/memory_summary` | `torch.npu.` 同名 |
| `torch.cuda.Stream/Event` | `torch.npu.Stream/Event` |
| `torch.autocast("cuda", dtype=...)` | `torch.autocast("npu", dtype=...)` |
| `torch.cuda.amp.GradScaler()` | `torch_npu.npu.amp.GradScaler()` |
| `init_process_group("nccl")` | `init_process_group("hccl")` |
| `torch.profiler.profile(...)` | `torch_npu.profiler.profile(...)`（接口同构，见第 7 章） |
| `F.scaled_dot_product_attention` | 可用（新版本对接融合实现）；极致性能用 `npu_fusion_attention` |

**懒人方案**：`from torch_npu.contrib import transfer_to_npu`——运行时把 cuda 调用自动转 npu，适合快速验证第三方项目；长期代码建议显式改写。

## C.3 迁移一个现有训练项目的检查清单

1. ☐ 入口加 `import torch_npu`（或 transfer_to_npu 垫片）。
2. ☐ 全局搜索 `"cuda"` 字符串字面量（device 字符串、`.cuda()`、环境变量判断），逐处替换或走垫片。
3. ☐ 分布式 backend `nccl` → `hccl`；启动脚本 `CUDA_VISIBLE_DEVICES` → `ASCEND_RT_VISIBLE_DEVICES`。
4. ☐ 去掉 CUDA 专属依赖：`flash-attn`、`bitsandbytes`、`xformers`、自定义 CUDA/Triton kernel——找 NPU 等价物（C.1 表）或改回朴素实现，**这些包在 requirements 里装不上/装上不可用是迁移期最常见报错**。
5. ☐ AMP：910B 上 fp16 方案可直接换 bf16（去掉 GradScaler 更省心）。
6. ☐ 跑第 1 章体检 + 单卡短训，对齐 loss 曲线与 CUDA 侧（小数据几百步即可）。
7. ☐ 跑第 7 章 profiler，检查有无 AI_CPU 回退算子，处理之。
8. ☐ 8 卡拉起，验证吞吐线性度（第 3 章方法）。

## C.4 心态调整（最后一条最重要）

- 算子覆盖率很高但不是 100%：遇到不支持/慢的算子，思路是「换写法」而不是「等支持」（第 8 章 8.4）。
- 生态版本锁定比 CUDA 世界更严格：**先查配套表再装包**（附录 A）。
- 大多数「NPU 的问题」其实是分布式训练的通用问题——你在 CUDA 上积累的经验，90% 直接有效。
