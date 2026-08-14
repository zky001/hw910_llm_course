# 第 5 章 · 实战二：DeepSpeed ZeRO 全参训练

> **本章目标**：理解全参训练的显存账，学会用 DeepSpeed ZeRO 在 8 张 910 上全参微调 7B 级模型，并知道模型再大时该怎么升级方案（ZeRO-3、offload，或第 6 章的张量并行）。

**前置条件**：完成第 4 章（模型/数据已就位，跑通过 8 卡训练）。

## 5.1 先算账：全参训练的显存都花在哪

混合精度 + AdamW 下，**每个参数**需要的「模型状态」显存：

```text
bf16 权重 2B + bf16 梯度 2B + fp32 主权重 4B + fp32 动量 4B + fp32 方差 4B = 16 字节/参数
```

7B 模型 → **112GB**，还没算激活值。单卡 64GB 装不下——这就是普通 DDP（每卡一份完整状态）跑不了全参 7B 的原因，也是 ZeRO 存在的意义：**把这三样东西切开，8 张卡各存 1/8**。

| ZeRO 阶段 | 切分内容 | 7B 每卡模型状态 | 特点 |
|---|---|---|---|
| ZeRO-0（=DDP） | 不切 | 112 GB ❌ | 通信最少 |
| ZeRO-1 | 优化器状态 | 4 + 84/8 ≈ **14.5 GB** | 几乎无额外通信 |
| ZeRO-2 | + 梯度 | 2 + 98/8 ≈ **14.3 GB**（梯度即切即释） | 常用甜点位 |
| ZeRO-3 | + 参数本身 | 112/8 = **14 GB** | 参数用时聚合，多 ~50% 通信量 |
| ZeRO-3 + offload | 状态放 CPU 内存 | 更少 | 显存最省，速度明显下降 |

加上激活值（梯度检查点后 7B/2k 序列每卡约几 GB），结论：**8×64GB 上全参 7B 用 ZeRO-1/2 很宽裕；8×32GB（910A）则需要 ZeRO-2/3 + 小 batch。**

选型速查（8 卡 910B 64GB，全参 SFT，开梯度检查点）：

| 模型规模 | 推荐配置 |
|---|---|
| ≤ 7B | **ZeRO-2**（[`ds_zero2.json`](ds_zero2.json)） |
| 13B/14B | **ZeRO-3**（[`ds_zero3.json`](ds_zero3.json)） |
| 32B | ZeRO-3 + 激活检查点全开，临界；不行加 optimizer offload（[`ds_zero3_offload.json`](ds_zero3_offload.json)） |
| ≥ 70B | 单机全参不现实：用 LoRA（第 4 章）或多机 + 第 6 章方案 |

## 5.2 DeepSpeed 在 NPU 上的安装与验证

新版 DeepSpeed 通过「accelerator 抽象」原生支持昇腾：装好 torch_npu 后直接 `pip install deepspeed`，它会自动探测到 NPU。

```bash
pip install deepspeed    # 版本兼容性以 LLaMA-Factory/torch_npu 文档推荐为准

ds_report
# 关键看这一行:
#   accelerator: npu        <- 出现 npu 说明探测成功
# NPU 上各类 CUDA fused op builder 显示 [NO] 是正常的，不影响 ZeRO 使用
```

## 5.3 跑起来：Trainer + ZeRO-2 全参微调 7B

本章的 [`train_sft_trainer.py`](train_sft_trainer.py) 与第 4 章路线 B 几乎相同，差异只有两点：**不注入 LoRA（全参可训）**、**TrainingArguments 里传入 `--deepspeed` 配置文件**。Trainer 会接管 DeepSpeed 引擎的初始化、ZeRO 切分和 checkpoint 保存。

```bash
# 8 卡 ZeRO-2 全参微调（默认配置）
bash 05-full-finetune-deepspeed/run_deepspeed_8npu.sh --data my_data.json

# 换 ZeRO-3
DS_CONFIG=ds_zero3.json bash 05-full-finetune-deepspeed/run_deepspeed_8npu.sh --data my_data.json
```

启动本质上就是 torchrun 传配置：

```bash
torchrun --nproc_per_node=8 train_sft_trainer.py \
    --model ./models/Qwen2.5-7B-Instruct --data my_data.json \
    --deepspeed ds_zero2.json --lr 1e-5
```

> 全参微调学习率要比 LoRA **小一个数量级**（1e-5 ~ 2e-5），这是新手最常翻车的参数。

训练中观察显存验证 ZeRO 生效：`npu-smi info` 里每卡 HBM 占用应远小于 112GB 的 1/1（ZeRO-2 下 7B 每卡 20~35GB 量级）。示例日志（示意）：

```text
[INFO] DeepSpeed info: version=0.16.x, git-hash=unknown
[INFO] rank0: Detected accelerator: npu
{'loss': 1.612, 'grad_norm': 3.1, 'learning_rate': 9.6e-06, 'epoch': 0.3}
...
```

## 5.4 配置文件精读

三个 json 已按「与 Trainer 集成」的最佳实践写好：能填 `"auto"` 的都填 `"auto"`（batch、精度、梯度裁剪等以命令行/TrainingArguments 为准，避免两处配置打架），需要理解的旋钮就这几个：

```jsonc
"zero_optimization": {
  "stage": 2,
  "overlap_comm": true,          // 通信与反向计算重叠，基本必开
  "contiguous_gradients": true,  // 梯度连续存放，减碎片
  "reduce_bucket_size": 5e8      // 通信桶大小(元素数)。OOM 时可调小换显存
}
```

ZeRO-3 额外两个：

```jsonc
"stage3_prefetch_bucket_size": "auto",              // 参数预取，隐藏 all-gather 延迟
"stage3_gather_16bit_weights_on_model_save": true   // 保存时聚合成完整权重(见 5.5)
```

offload 版把优化器状态挪到 CPU 内存：显存大省、速度受 CPU 和 PCIe 限制明显下降，**是「跑不下」时的手段，不是性能优化**。用它之前确认机器内存足够（7B 优化器状态 84GB + 余量）。

## 5.5 ZeRO-3 的 checkpoint 有点特别

ZeRO-3 下每卡只有 1/8 参数，保存时有两种结果：

- 配置里 `stage3_gather_16bit_weights_on_model_save: true`（本章默认）：保存时自动聚合出**完整 bf16 权重**，`output_dir` 里就是标准 HF 格式，直接能部署。代价是保存变慢、rank0 峰值内存变高。
- 若关掉：得到的是分片的 ZeRO checkpoint，用 DeepSpeed 附带的 `zero_to_fp32.py output_dir merged.safetensors` 离线合并。

大模型保存一次可能要几分钟，**期间其他 rank 在等待**——把 `ddp_timeout`（脚本已设）调大，避免保存时触发 HCCL 超时误报。

## 5.6 FSDP：另一条全参路线（简述）

PyTorch 原生 FSDP 在 torch_npu 上同样可用，效果与 ZeRO-3 同级。已经在用 accelerate 的项目可以直接：

```bash
accelerate config   # distributed_type 选 MULTI_NPU, 再选 FSDP, auto wrap 填 TRANSFORMER_BASED_WRAP
accelerate launch --num_processes 8 your_train.py
```

两条路线怎么选：**用 HF Trainer/LLaMA-Factory 生态 → DeepSpeed 配置文件最省事（本章）；纯 PyTorch 自研训练循环 → FSDP 少一个依赖。** 性能上单机 8 卡两者差距通常在几个百分点内，不必纠结。

## 5.7 常见坑

| 症状 | 处理 |
|---|---|
| `ds_report` 没显示 npu | torch_npu 没装好或没在同一环境；先回第 1 章体检 |
| 加载模型阶段就 OOM（ZeRO-3） | 确认 deepspeed 配置在 `from_pretrained` 之前已传给 Trainer（本章脚本顺序正确）；HF 会用 zero.Init 边加载边切分 |
| 训练稳定但 eval/保存时 OOM | eval batch 调小；保存时聚合权重的峰值见 5.5，可改离线合并 |
| fp16 溢出 NaN（910A） | ds 配置 fp16 段已启用动态 loss scale；仍 NaN 则降 lr、看第 9 章 |
| offload 后 CPU 100%、速度骤降 | 正常代价。确认真的需要 offload；优先试 ZeRO-3 不 offload + 更小 micro batch |
| 保存 checkpoint 时 HCCL 超时 | 调大 `ddp_timeout`（脚本默认 3000 秒）；checkpoint 放本地盘而不是慢速 NFS |

## 5.8 练习

1. 用 ZeRO-1/2/3 各跑同一任务 100 步，记录每卡峰值显存（脚本会打印）和吞吐，验证 5.1 的表。
2. 7B 全参 + ZeRO-2 下把 micro batch 从 1 逐步加大到 OOM，找出你机器的最大值；再开 offload 看能大多少。
3. 把第 4 章 LoRA 与本章全参在同一批数据上训练，对比 loss 曲线和下游效果，体会两者的适用差异。

**下一章** → [第 6 章 · 实战三：MindSpeed 预训练](../06-pretrain-mindspeed/README.md)：数据并行到头了，该上张量并行/流水并行了。
