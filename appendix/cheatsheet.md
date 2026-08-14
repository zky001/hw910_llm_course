# 附录 B · 命令与环境变量速查表

> 打印贴在工位上的那一页。按「日常必用 → 排障时用」排序。

## B.1 日常命令

| 目的 | 命令 |
|---|---|
| 看卡状态（对标 nvidia-smi） | `npu-smi info` |
| 持续刷新观察训练 | `watch -n 1 npu-smi info` |
| 单卡健康详情 | `npu-smi info -t health -i <id>` |
| 加载 CANN 环境（每个新 shell） | `source /usr/local/Ascend/ascend-toolkit/set_env.sh` |
| 查 torch/torch_npu 版本 | `python -c "import torch,torch_npu;print(torch.__version__,torch_npu.__version__)"` |
| 查 CANN 版本 | `cat $ASCEND_TOOLKIT_HOME/version.cfg` |
| 查驱动版本 | `cat /usr/local/Ascend/driver/version.info` |
| 环境体检 | `bash 01-environment/check_env.sh` |
| 8 卡性能验证 | `python 01-environment/verify_npu.py` |
| 8 卡带宽实测 | `torchrun --nproc_per_node=8 03-multi-card-ddp/allreduce_bench.py` |
| 清理残留训练进程 | `pkill -9 -f <脚本名>`，几秒后显存自动释放 |
| 复位单颗芯片（慎用） | `npu-smi set -t reset -i <id> -c 0` |
| 收集诊断包 | `bash 09-troubleshooting/collect_diag.sh` |

## B.2 torchrun 模板

```bash
# 单机 8 卡
torchrun --nproc_per_node=8 --master_port=29500 train.py

# 单机部分卡
ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 train.py

# 多机(每台机器都执行, node_rank 递增; TP 不出机)
torchrun --nnodes=2 --node_rank=0 --nproc_per_node=8 \
         --master_addr=<主节点IP> --master_port=29500 train.py
```

## B.3 环境变量：运行与性能

| 变量 | 建议值 | 作用 |
|---|---|---|
| `ASCEND_RT_VISIBLE_DEVICES` | `0,1,...,7` | 进程可见的卡（对标 CUDA_VISIBLE_DEVICES） |
| `PYTORCH_NPU_ALLOC_CONF` | `expandable_segments:True` | 缓解显存碎片，长训练建议常开 |
| `TASK_QUEUE_ENABLE` | `2` | host 算子下发优化（A/B 实测） |
| `CPU_AFFINITY_CONF` | `1` | 进程绑核，减少 host 抖动 |
| `HF_ENDPOINT` | `https://hf-mirror.com` | HuggingFace 国内镜像 |
| `USE_MODELSCOPE_HUB` | `1` | LLaMA-Factory 从 ModelScope 拉模型 |

## B.4 环境变量：HCCL 通信

| 变量 | 默认 | 何时动它 |
|---|---|---|
| `HCCL_CONNECT_TIMEOUT` | 120 秒 | 启动建链超时 → 调大到 300~1200 |
| `HCCL_EXEC_TIMEOUT` | 1836 秒 | 有超长保存/eval 停顿 → 调大；注意先排除真故障 |
| `HCCL_BUFFSIZE` | 200 (MB) | 大消息/多机调优时 A/B 实测 |
| `HCCL_DETERMINISTIC` | false | 复现实验设 `true`（性能略降） |
| `HCCL_WHITELIST_DISABLE` | — | 多机部署常设 `1`（关闭白名单校验） |

## B.5 环境变量：调试与日志

| 变量 | 值 | 作用 |
|---|---|---|
| `ASCEND_LAUNCH_BLOCKING` | `1` | 同步执行，报错栈指向真凶算子。**很慢，只在复现问题时开** |
| `ASCEND_GLOBAL_LOG_LEVEL` | `1` | 底层日志级别 0=debug/1=info/2=warn/3=error(默认) |
| `ASCEND_SLOG_PRINT_TO_STDOUT` | `1` | 底层日志打到屏幕 |
| （日志位置） | — | host 侧 plog：`~/ascend/log/plog/`；内核事件：`dmesg` |

## B.6 Python 速查

```python
import torch, torch_npu

torch.npu.set_device(local_rank)            # 绑卡(建张量之前!)
x = t.to("npu", non_blocking=True)          # 上卡
torch.npu.synchronize()                     # 计时/取数前同步
torch.npu.max_memory_allocated() / 2**30    # 峰值显存 GB
print(torch.npu.memory_summary())           # OOM 时看碎片
torch.npu.manual_seed_all(1234)             # 种子

# 分布式
dist.init_process_group(backend="hccl")

# 混合精度(910B)
with torch.autocast(device_type="npu", dtype=torch.bfloat16): ...

# 混合精度(910A)
from torch_npu.npu.amp import GradScaler    # + autocast fp16

# 一键迁移存量 CUDA 代码
from torch_npu.contrib import transfer_to_npu

# 融合算子(910B)
torch_npu.npu_fusion_attention(...)         # FlashAttention
torch_npu.npu_rms_norm(x, w, epsilon=1e-6)
torch_npu.npu_swiglu(x, dim=-1)
from torch_npu.optim import NpuFusedAdamW
```

## B.7 显存账速算（混合精度 + AdamW）

```text
全参训练模型状态 = 16 字节/参数  (bf16权重2 + bf16梯度2 + fp32主权重4 + 动量4 + 方差4)
  7B → 112GB   14B → 224GB   32B → 512GB   72B → 1152GB
ZeRO-1/2/3: 优化器/梯度/参数 依次除以卡数(8)
LoRA: 底座只算权重 2 字节/参数(7B→14GB), 可训练部分忽略不计
推理: 权重 2 字节/参数 + KV cache
910B 单卡 64GB · 910A 单卡 32GB
```

## B.8 课程脚本索引

| 想做什么 | 去哪 |
|---|---|
| 体检/验卡 | `01-environment/` |
| 单卡训练模板 | `02-single-card/train_tiny_gpt.py` |
| DDP 模板 + 带宽测试 | `03-multi-card-ddp/` |
| LoRA 微调（LLaMA-Factory / HF） | `04-lora-finetune/` |
| 全参 + DeepSpeed 配置 | `05-full-finetune-deepspeed/` |
| 预训练（MindSpeed） | `06-pretrain-mindspeed/` |
| profiler 模板 + MFU 计算 | `07-profiling/` |
| 融合算子基准 | `08-optimization/fused_ops_bench.py` |
| 诊断打包 | `09-troubleshooting/collect_diag.sh` |
