# 第 8 章 · 性能优化实战：一份按收益排序的清单

> **本章目标**：把 8 卡 910 的吞吐压榨到位。所有手段按「投入产出比」排序成清单，每项都说明**适用条件、做法、预期收益、怎么验证**。原则只有一条：**第 7 章的剖析数据说了算**——不测量的优化是玄学。

**前置条件**：完成第 7 章，手里有自己任务的 MFU 和 step 时间分解。

## 8.0 优化总纲

```text
① 精度策略(bf16)            —— 一行配置, 基础收益, 先保证做对
② 融合算子(FlashAttention 等) —— 910B 上单项收益最大的算子级优化
③ 显存 ↔ 吞吐再平衡          —— 重计算范围/micro-batch 找甜点
④ 通信优化                   —— 重叠与桶配置, 多卡必做
⑤ 数据与 Host 侧             —— 让 NPU 永远有活干
⑥ 并行策略(第 6 章)          —— 上面都做完仍不满意时再动
```

对号入座：第 7 章分解里 **Computing 大** → 看 ①②③；**Communication 大** → 看 ④；**Free 大** → 看 ⑤。

## 8.1 精度策略：BF16 是基本盘

- **910B：一律 BF16**（`bf16: true` / `--bf16` / autocast bf16）。相对 FP32 计算翻倍显存减半；相对 FP16 免 loss scaling、不炸 loss。
- **910A：FP16 + 动态 loss scale**（框架默认配好），并把 `initial_scale_power`、`min_loss_scale` 留默认；频繁 overflow 跳步说明模型/lr 有问题，去第 9 章。
- 别在热路径手写 `.float()/.half()` 反复转换——autocast 已经把该 FP32 的（norm、softmax、loss）保持 FP32 了，多余转换徒增算子数。

**验证**：显存峰值应明显下降、tokens/s 上升；loss 曲线与 FP32 短跑对比无异常偏移。

## 8.2 融合算子：910B 的最大单项收益

一次 attention 的朴素实现要十几个 kernel、还要在显存里物化 S×S 的注意力矩阵；融合算子把它合成一个 kernel、按块计算不物化大矩阵——**又快又省显存，序列越长收益越大**。910 上的对应物：

| 融合算子 | 替代的朴素实现 | 备注 |
|---|---|---|
| `torch_npu.npu_fusion_attention` | softmax(QKᵀ/√d)·V 全套 | 即昇腾版 FlashAttention，**仅 910B(Atlas A2)** |
| `torch_npu.npu_rms_norm` | 手写 RMSNorm（4~5 个小算子） | |
| `torch_npu.npu_rotary_mul` | 手写 RoPE 旋转 | |
| `torch_npu.npu_swiglu` | SiLU(x₁)·x₂ | |
| `torch_npu.optim.NpuFusedAdamW` | 逐参数小算子的 AdamW | 参数量大时 host 下发开销显著下降 |

**先跑基准，眼见为实**（910A 会自动跳过不支持的项）：

```bash
python 08-optimization/fused_ops_bench.py
```

示例输出（示意）：

```text
attention  B=4 S=4096 H=32 D=128 bf16
  naive(物化S×S)     : 41.32 ms   peak_mem 9.8 GB
  npu_fusion_attention:  6.85 ms   peak_mem 2.1 GB   → 6.0x, 显存-79%
rmsnorm    (4096x4096, hidden=4096)
  naive : 0.51 ms   fused : 0.11 ms   → 4.6x
swiglu     ...
```

**实际工程里怎么启用**（多数时候不用自己调 API）：

- **MindSpeed/Megatron**：加参数 `--use-flash-attn --use-fused-rmsnorm --use-fused-swiglu --use-fused-rotary-pos-emb`（第 6 章脚本已带）。
- **transformers/LLaMA-Factory**：用 `attn_implementation="sdpa"`（本课程脚本默认）。torch_npu 对接了 PyTorch 的 SDPA 入口，新版本上会走融合实现；用第 7 章 profiler 确认 kernel 名里出现 FlashAttention/FusionAttention 字样即为生效。
- **自研模型**：手动替换，attention 示例：

```python
import torch_npu
out = torch_npu.npu_fusion_attention(
    q, k, v, head_num=n_head, input_layout="BNSD",   # q/k/v: (B, N, S, D)
    atten_mask=causal_mask,                          # bool, True=屏蔽, 上三角
    scale=1.0 / math.sqrt(head_dim), keep_prob=1.0)[0]
```

**验证**：kernel_details.csv 里 attention 相关小算子（BatchMatMul+SoftmaxV2+…）消失，出现融合大算子；单步时间下降。

## 8.3 显存 ↔ 吞吐再平衡：省出来的显存要花掉

显存优化的目的不是「省着」，而是**换成更大的 micro-batch 或更少的重计算**：

1. **打开 expandable_segments 缓解碎片**（近乎免费）：
   ```bash
   export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
   ```
   碎片的典型症状：`memory_summary()` 里 reserved 远大于 allocated 却报 OOM。
2. **重计算用「够用就好」**：full 重计算省最多但前向多算一遍（约 +1/3 计算）。Megatron 用 `--recompute-num-layers` 从大往小收；HF 系的 `gradient_checkpointing` 是全开/全关，显存富余时可以关掉换速度。
3. **micro-batch 找甜点**：从 1 往上加、每档记录 tokens/s，**最大能跑的 MBS 不一定最快**（大 MBS 可能触发更差的 kernel 形态或逼近碎片边界），用数据选。
4. **优化器/ZeRO 档位**：第 5 章的表，够用的最低档 = 通信最少。
5. 反模式：训练循环里调 `torch.npu.empty_cache()`——它清掉缓存分配器的池子，之后每次分配都变慢，只应在阶段切换（如训练→评测）时偶尔用。

**验证**：`torch.npu.max_memory_allocated()` 水位 + tokens/s 一起看，目标是吞吐上升而不是显存数字好看。

## 8.4 抓「回退算子」：AI CPU 与同步点

两类隐形杀手，profiler（第 7 章）一抓一个准：

**① AI CPU 算子**：个别算子（`nonzero`、`unique`、部分动态 shape/int64 运算）不在 AI Core 执行，慢 1~2 个数量级。kernel_details.csv 的加速器类型列出现 `AI_CPU` 即中招。处理思路：
- 能用 mask 乘法代替 `nonzero`/`masked_select` 的就代替（保持静态 shape）；
- 索引/位置张量能用 int32 就别用 int64；
- 实在绕不开且不在热路径（如每步一次的指标统计），挪到 CPU 上算或降低频率。

**② 隐式同步点**：`tensor.item()`、`print(loss)`、`tensor.cpu()` 都会强迫 host 等待 NPU 排空，打断流水。热路径里的 loss 打印改成每 N 步一次；指标累计用张量在卡上累加，最后一次性 `.item()`。

**③ Host 下发优化**（torch_npu 专属环境变量，收益视负载而定，建议 A/B 实测）：

```bash
export TASK_QUEUE_ENABLE=2      # 算子下发队列深度优化
export CPU_AFFINITY_CONF=1      # 进程绑核, 减少 host 侧抖动
```

## 8.5 通信优化：把它藏进计算里

通信不可消除，但可以**重叠**：

| 框架 | 开关 | 说明 |
|---|---|---|
| DDP | 默认已重叠 | 可调 `bucket_cap_mb`（默认 25）：小模型大 bucket 减少启动次数；`gradient_as_bucket_view=True` 省显存 |
| DDP + 梯度累积 | `model.no_sync()` | 累积的中间步不通信（Trainer/DeepSpeed 自动处理，手写循环别漏） |
| DeepSpeed | `overlap_comm: true` | 本课程配置已开；`reduce_bucket_size` 是显存/速度权衡旋钮 |
| Megatron/MindSpeed | `--overlap-grad-reduce --overlap-param-gather` | 配合分布式优化器 |
| TP 场景 | `--sequence-parallel` 必开；MC2（如 `--use-ascend-mc2`） | MC2 把 TP 的 matmul 与通信做算子内融合，910B 亮点特性 |
| HCCL | `HCCL_BUFFSIZE`（默认 200MB） | 大消息多机场景可试调大，A/B 实测 |

还有一个非配置项：**慢卡即通信瓶颈**。集合通信按最慢的卡走，第 1 章 `verify_npu.py` 发现的降频卡、或负载不均，都表现为「通信时间长」。profiler 的多卡对比（`msprof-analyze cluster`）能定位。

**验证**：step_trace_time.csv 的 `Communication(Not Overlapped)` 下降；总步时下降。

## 8.6 数据与 Host 侧：别让 NPU 挨饿

- DataLoader：`num_workers=4~16`（按 CPU 核数）、`pin_memory=True`、`persistent_workers=True`、搬运用 `non_blocking=True`。
- **离线预处理**：tokenize 别放在训练循环里做——预训练用第 6 章的 bin/idx；SFT 场景 LLaMA-Factory 的 `preprocessing_num_workers` 提前编码 + 缓存。
- **短样本 packing**：SFT 数据长短不齐时，padding 可能浪费一半以上算力。LLaMA-Factory 开 `packing: true`（把多条短样本拼成满长度序列），等效吞吐可观提升。
- 存储：数据集放本地 NVMe；放 NFS 上时首个 epoch 的 Free 占比会出卖它。

**验证**：Free 占比降到个位数百分比；AICore 利用率曲线不再周期性掉底。

## 8.7 8×910 优化清单（贴墙版）

| # | 动作 | 条件 | 预期 |
|---|---|---|---|
| 1 | BF16（910A: FP16+scale） | 总是 | 基础盘 |
| 2 | FlashAttention/融合算子全开 | 910B | 大，序列越长越大 |
| 3 | `expandable_segments:True` | 总是 | 免 OOM/碎片 |
| 4 | 重计算收到「刚好不 OOM」 | 显存有余量 | 最高省回 ~25% 计算 |
| 5 | micro-batch 扫描找甜点 | 总是 | 中 |
| 6 | 通信重叠开关（按框架） | 多卡 | 中 |
| 7 | 数据离线化 + packing + DataLoader 调参 | Free 占比高 | 场景差异大，可达数十 % |
| 8 | 消灭 AI_CPU 算子与热路径 `.item()` | profiler 发现 | 视命中程度 |
| 9 | `TASK_QUEUE_ENABLE=2`、绑核 | 小算子多/host 忙 | 小~中，A/B 定 |
| 10 | 并行策略重选（第 6 章法则） | 上述做完仍不满意 | 结构性 |

> ✅ **检查点**：每做一项，记录「改动 + tokens/s 前后 + MFU 前后」。清单跑完，稠密模型预训练 MFU 进入 30%+ 区间、微调任务吞吐较裸配置提升 1.5~3 倍是常见结果（具体因模型/序列/芯片而异）。

## 8.8 练习

1. 用 `fused_ops_bench.py` 把 S 从 1024 扫到 8192，画出融合注意力加速比随序列长度的曲线，理解「为什么长序列必开 FA」。
2. 在第 4 章 LoRA 任务上开/关 `packing`，对比等效 tokens/s。
3. 把你的任务按 8.7 清单过一遍，产出一张自己的「优化台账」——这就是本章的毕业作品。

**下一章** → [第 9 章 · 稳定性与故障排查](../09-troubleshooting/README.md)：快起来之后，还要稳得住。
