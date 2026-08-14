# 第 2 章 · PyTorch NPU 上手与单卡训练

> **本章目标**：掌握 torch_npu 的基本用法和与 CUDA 的对应关系，在**一张** 910 上从零训练一个迷你 GPT。分布式的一切都建立在单卡跑通之上——单卡还没稳，别急着上 8 卡。

**前置条件**：完成第 1 章，`check_env.sh` 全 PASS。

## 2.1 三行代码理解 torch_npu

```python
import torch
import torch_npu            # ① 导入即生效：注册 "npu" 设备和 torch.npu.* 接口

x = torch.randn(2, 3).npu() # ② 张量上卡，等价于 .to("npu")
print(x.device)             # npu:0
y = x @ x.T                 # ③ 之后的用法与 CUDA 完全一致
```

核心心智模型：**把你会的 `torch.cuda.*` 全部替换成 `torch.npu.*`，把 `"cuda"` 替换成 `"npu"`，90% 的代码就迁移完了。**

| CUDA 写法 | NPU 写法 |
|---|---|
| `torch.device("cuda", 0)` | `torch.device("npu", 0)` |
| `tensor.cuda()` / `.to("cuda")` | `tensor.npu()` / `.to("npu")` |
| `torch.cuda.is_available()` | `torch.npu.is_available()` |
| `torch.cuda.set_device(i)` | `torch.npu.set_device(i)` |
| `torch.cuda.synchronize()` | `torch.npu.synchronize()` |
| `torch.cuda.manual_seed_all(s)` | `torch.npu.manual_seed_all(s)` |
| `torch.cuda.max_memory_allocated()` | `torch.npu.max_memory_allocated()` |
| `torch.cuda.empty_cache()` | `torch.npu.empty_cache()` |
| `CUDA_VISIBLE_DEVICES=0,1` | `ASCEND_RT_VISIBLE_DEVICES=0,1` |
| `nvidia-smi` | `npu-smi info` |

完整对照表见[附录 C](../appendix/cuda-to-npu.md)。

**存量 CUDA 代码不想改？** torch_npu 提供了一键迁移垫片，在入口文件最前面加两行，运行时自动把 `cuda` 调用转到 NPU：

```python
import torch_npu
from torch_npu.contrib import transfer_to_npu   # 之后代码里的 .cuda()/"cuda" 都会被接管
```

适合快速验证第三方项目能不能跑；**自己长期维护的代码建议显式写 npu**，行为更可控。

先跑本章的 [`hello_npu.py`](hello_npu.py) 感受一遍以上所有 API：

```bash
python 02-single-card/hello_npu.py
```

## 2.2 哪些东西不用改

torch_npu 的设计目标就是「PyTorch 原生体验」，以下内容在 NPU 上**原样可用**：

- autograd、`nn.Module`、optimizer、lr_scheduler
- `DataLoader`（`pin_memory=True` + `tensor.to("npu", non_blocking=True)` 的异步搬运习惯照旧）
- `state_dict` 保存/加载（权重会自动处理设备；跨设备加载用 `map_location`）
- `torchrun`、`torch.distributed`（第 3 章），backend 换成 `hccl`

## 2.3 混合精度：910B 用 BF16，910A 用 FP16

NPU 上的 AMP 用法与 CUDA 一致，用原生 `torch.autocast`：

```python
# 910B（推荐）：BF16，无需 loss scaling
with torch.autocast(device_type="npu", dtype=torch.bfloat16):
    loss = model(x, y)
loss.backward()

# 910A：只能 FP16，必须配 GradScaler 防下溢
from torch_npu.npu.amp import GradScaler
scaler = GradScaler()
with torch.autocast(device_type="npu", dtype=torch.float16):
    loss = model(x, y)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

为什么 BF16 优先：BF16 与 FP32 同指数位宽，动态范围大，大模型训练**不需要 loss scaling、几乎不会溢出**；FP16 动态范围小，7B 以上全参训练容易 NaN（第 9 章有专门排查）。910A 不支持 BF16，属于硬件限制。

## 2.4 实战：单卡训练一个迷你 GPT

[`train_tiny_gpt.py`](train_tiny_gpt.py) 是一个约 300 行的**零依赖**（只需 torch + torch_npu）训练脚本：内置一小段唐诗语料，按字符建词表，从零训练一个 4 层的 GPT，最后生成一段「诗」。麻雀虽小，五脏俱全——模型定义、AMP、梯度裁剪、lr warmup、吞吐统计、checkpoint 保存、文本生成都有。

```bash
# 910B（默认 bf16）
python 02-single-card/train_tiny_gpt.py

# 910A
python 02-single-card/train_tiny_gpt.py --amp fp16

# 不开混合精度（对比用）
python 02-single-card/train_tiny_gpt.py --amp off
```

示例输出（示意）：

```text
device=npu:0  vocab=712  params=1.82M  amp=bf16
step   50/2000  loss 5.213  lr 3.0e-04  12.8k tok/s  peak_mem 0.4GB
step  100/2000  loss 4.187  ...
...
step 2000/2000  loss 0.412  ...
--- 生成示例 ---
床前明月光，疑是地上霜。举头望明月，低头思故乡。
checkpoint 已保存到 tiny_gpt.pt
```

训练时开另一个终端观察卡的状态：

```bash
watch -n 1 npu-smi info     # 关注 AICore(%) 利用率和 HBM-Usage
```

> ✅ **检查点**：loss 从 ~6 降到 1 以下、生成的文本明显有唐诗味、`npu-smi` 能看到你的 python 进程和显存占用。三条都满足，说明「计算、反传、优化器、显存管理」整条链路在你的环境上是好的。

### 值得注意的几处代码

脚本里有几个 NPU 特有的写法，对应行内有注释：

1. **`torch.npu.set_device(idx)` 要在建张量之前调用**——和 CUDA 一样的规矩。
2. **计时必须 `torch.npu.synchronize()`**——NPU 执行是异步的，不同步的话你测的只是「下发时间」。
3. **显存统计用 `torch.npu.max_memory_allocated()`**——后面章节做显存优化时天天用。

## 2.5 两个新手必知的性能特性

**① 第一次迭代慢是正常的。** NPU 算子存在「在线编译」机制（jit_compile）。910B 上配套的 CANN 默认走**预编译二进制算子**（jit_compile=False），首步只有少量开销；如果你观察到首步极慢或产生了 `kernel_meta/` 目录，说明走了在线编译，编译结果会缓存，第二次运行就快了。基准测试永远要先 warmup 再计时。

```python
# 显式关闭在线编译（910B + 新 CANN 通常已是默认）：
torch_npu.npu.set_compile_mode(jit_compile=False)
```

**② 个别算子可能落到 AI CPU 或宿主 CPU。** NPU 的算子覆盖度很高但不是 100%，`nonzero`、部分 int64 运算、某些花式索引可能落到慢速路径。症状是 AICore 利用率低、训练异常慢。第 7 章教你用 profiler 精确抓出这些算子，第 8 章讲替换写法。现在只需要记住：**NPU 上写模型，尽量用常规算子、常规 dtype（fp16/bf16/fp32/int32）**。

## 2.6 练习

1. 把 `--amp off / bf16` 各跑一遍，对比 tok/s 和 `peak_mem`，感受混合精度的收益。
2. 把 `n_layer` 从 4 改到 8、`block_size` 从 128 改到 256，观察显存变化，试着在 OOM 前找到最大能跑的配置。
3. （910B）把脚本里的 `scaled_dot_product_attention` 换成手写的 softmax(QK^T)V 实现，对比速度——这是第 8 章融合算子优化的预告。

**下一章** → [第 3 章 · 8 卡分布式：torchrun + DDP + HCCL](../03-multi-card-ddp/README.md)：让 8 张卡一起干活。
