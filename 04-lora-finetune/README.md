# 第 4 章 · 实战一：8 卡 LoRA 微调 Qwen

> **本章目标**：在 8 张 910 上用 LoRA 微调一个 7B 对话模型（以 Qwen2.5-7B-Instruct 为例），走两条路线：**路线 A** 用 LLaMA-Factory（改配置就能跑，推荐首选）；**路线 B** 用 transformers + PEFT 手写（理解原理、便于深度定制）。做完本章你就拥有第一个「自己的模型」。

**前置条件**：完成第 3 章；磁盘至少留 50GB（模型权重 + checkpoint）。

## 4.1 为什么从 LoRA 开始

| | LoRA 微调 7B | 全参微调 7B |
|---|---|---|
| 训练的参数量 | ~0.3%（几十 M） | 100%（7B） |
| 单卡显存需求（bf16 + 梯度检查点） | ~20GB，**单卡就能跑** | 权重+梯度+优化器 ≈ 112GB，必须多卡切分 |
| 8 卡的用法 | 数据并行，纯加速 | ZeRO 切分（第 5 章） |
| 适合 | 风格/领域/指令定制 | 深度改变模型能力 |

LoRA 冻结原模型，只训练插入的低秩矩阵，显存瓶颈没了，8 张卡全部用于加速。**910B（64GB）单卡即可训 7B LoRA，8 卡把时间除以 ~7；910A（32GB）也能跑，但要 fp16 + 更小 batch。**

## 4.2 准备：下载模型和数据

国内网络直连 HuggingFace 慢/不通，两个方案任选：

```bash
# 方案一：ModelScope（推荐，国内速度快）
pip install modelscope
modelscope download --model Qwen/Qwen2.5-7B-Instruct --local_dir ./models/Qwen2.5-7B-Instruct

# 方案二：HF 镜像
export HF_ENDPOINT=https://hf-mirror.com
pip install -U huggingface_hub
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir ./models/Qwen2.5-7B-Instruct
```

下文所有 `model_name_or_path` 都填本地路径 `./models/Qwen2.5-7B-Instruct`（也可以直接填仓库名让框架在线下载）。

## 4.3 路线 A：LLaMA-Factory（推荐）

LLaMA-Factory 官方支持昇腾 NPU，SFT/LoRA/DPO/PPO 全流程开箱即用。

### 安装

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
# [torch-npu] extra 会带上 torch_npu 相关依赖；已按第 1 章装好 torch/torch_npu 的话也没冲突
pip install -e ".[torch-npu,metrics]"

# 验证它认到了 NPU
llamafactory-cli env
# 期望输出里有: PyTorch version: 2.5.1 ... NPU type: Ascend910B1 之类的字样
```

### 训练配置

本章提供了现成的 [`qwen2_5_7b_lora_sft_npu.yaml`](qwen2_5_7b_lora_sft_npu.yaml)，用的是 LLaMA-Factory 自带的两个演示数据集（`identity` 身份认知 + `alpaca_zh_demo` 中文指令）。想换成自己的数据：把 alpaca 格式的 json 放进 `LLaMA-Factory/data/`，在 `data/dataset_info.json` 里注册一个名字，然后改 yaml 里的 `dataset:` 字段即可。

关键参数解读（完整文件见 yaml）：

```yaml
finetuning_type: lora
lora_rank: 8            # 低秩矩阵的秩。8~64 常用；越大可塑性越强、也越容易过拟合
lora_target: all        # 对所有线性层插 LoRA（比只插 q/v 效果更稳）
cutoff_len: 2048        # 序列截断长度，显存的主要调节旋钮之一
per_device_train_batch_size: 2
gradient_accumulation_steps: 4   # 全局 batch = 2 × 8卡 × 4 = 64
learning_rate: 1.0e-4   # LoRA 常用 1e-4 ~ 2e-4（比全参大一个数量级）
bf16: true              # 910B 用 bf16；910A 改成 fp16: true
```

### 8 卡启动

```bash
cd LLaMA-Factory
cp ../hw910_llm_course/04-lora-finetune/qwen2_5_7b_lora_sft_npu.yaml .

# FORCE_TORCHRUN=1 强制走 torchrun 多进程（8 卡数据并行）
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 FORCE_TORCHRUN=1 \
  llamafactory-cli train qwen2_5_7b_lora_sft_npu.yaml
```

训练开始后另开终端 `watch -n 1 npu-smi info`，8 张卡的 AICore 利用率都应该持续在高位。示例日志（示意）：

```text
[INFO] trainable params: 20,185,088 || all params: 7,635,801,600 || trainable%: 0.26
{'loss': 1.8342, 'learning_rate': 9.8e-05, 'epoch': 0.21}
{'loss': 1.2917, 'learning_rate': 8.9e-05, 'epoch': 0.42}
...
Training completed. saved to saves/qwen2.5-7b/lora/sft
```

### 验证效果 + 合并导出

```bash
# 命令行对话（加载 base + LoRA adapter）
llamafactory-cli chat --model_name_or_path ./models/Qwen2.5-7B-Instruct \
  --adapter_name_or_path saves/qwen2.5-7b/lora/sft --template qwen

# 确认满意后，把 LoRA 合并回主干权重，得到可独立部署的完整模型（第 10 章要用）
llamafactory-cli export --model_name_or_path ./models/Qwen2.5-7B-Instruct \
  --adapter_name_or_path saves/qwen2.5-7b/lora/sft \
  --template qwen --export_dir ./models/qwen2.5-7b-sft-merged
```

> ✅ **检查点**：用 `identity` 数据集训练后，问它「你是谁」应回答配置的自定义身份而不是「我是通义千问」——这是微调生效最直观的证据。

## 4.4 路线 B：transformers + PEFT 手写

想理解 LoRA 微调内部发生了什么、或需要自定义数据管道/损失函数，看本章的 [`hf_peft_lora_sft.py`](hf_peft_lora_sft.py)（约 200 行，只依赖 transformers/peft）。它做了五件事，每件都有注释：

1. 加载 tokenizer 和 bf16 权重到 NPU；
2. 用 **chat template** 把 alpaca 格式样本转成训练文本，且**只对回答部分计算 loss**（prompt 部分 label = -100）；
3. `LoraConfig(target_modules="all-linear")` 注入 LoRA；
4. 开梯度检查点（`gradient_checkpointing_enable` + `enable_input_require_grads`）；
5. 交给 `Trainer`——它检测到 torchrun 环境会自动做 DDP，**代码里没有任何分布式逻辑**。

```bash
# 单卡冒烟（内置 32 条演示数据，几分钟跑完）
python 04-lora-finetune/hf_peft_lora_sft.py --model ./models/Qwen2.5-7B-Instruct

# 8 卡 + 自己的数据
bash 04-lora-finetune/run_hf_lora_8npu.sh --data my_data.json
```

自有数据格式（alpaca 风格的 json 数组）：

```json
[
  {"instruction": "把下面的句子翻译成英文", "input": "今天天气很好", "output": "The weather is nice today."},
  {"instruction": "你是谁?", "input": "", "output": "我是由 XX 团队训练的助手小X。"}
]
```

## 4.5 910A 及低显存场景的调整

| 调整项 | 路线 A（yaml） | 路线 B（脚本参数） |
|---|---|---|
| BF16 → FP16 | `bf16: true` 改 `fp16: true` | `--fp16` |
| 降序列长度 | `cutoff_len: 1024` | `--cutoff-len 1024` |
| 降 batch + 补累积 | `per_device_train_batch_size: 1` + 调大累积 | `--batch-size 1` |
| 还不够 | 加 `deepspeed: examples/deepspeed/ds_z3_config.json`（ZeRO-3 把 14GB 权重切到 8 卡） | 见第 5 章 |

关于 **QLoRA**（4bit 量化底座 + LoRA）：其核心依赖 bitsandbytes 对 NPU 的支持仍在演进中，昇腾上成熟度一般。8×910 的显存跑 7B/14B LoRA 并不紧张，建议优先 LoRA + 上表手段，不必强上 QLoRA。

## 4.6 常见坑

| 症状 | 处理 |
|---|---|
| 下载模型卡住/超时 | 用 4.2 的 ModelScope 或 HF 镜像；LLaMA-Factory 可加 `USE_MODELSCOPE_HUB=1` 直接从 ModelScope 拉 |
| 启动即 OOM | 先降 `cutoff_len` 和 batch size（最有效）；确认开了梯度检查点；再不行上 ZeRO-3 |
| 报 flash_attn / fa2 相关错误 | NPU 上没有 CUDA 版 FlashAttention，配置里 `flash_attn: auto`（默认）或 `sdpa` 即可，别强设 `fa2` |
| loss 一直不降 | 检查 template 是否和模型匹配（Qwen 用 `qwen`）；学习率是否过小；数据是否被 `max_samples` 截得太少 |
| fp16 下 loss 变 NaN（910A） | 学习率减半、`max_grad_norm: 0.5`；参考第 9 章溢出排查 |
| 8 卡但速度和单卡差不多 | 确认日志里 world_size=8（`FORCE_TORCHRUN=1` 有没有加）；`npu-smi` 看是不是只有 1 张卡在干活 |

## 4.7 练习

1. 造 50 条「自我认知」数据（"你是谁"→ 你设定的身份），微调后验证模型改口了。
2. 对比 `lora_rank: 8` 与 `64` 的训练速度、显存、效果差异。
3. 用 `llamafactory-cli webui` 起图形界面，把本章流程在网页里点一遍（它生成的命令与 yaml 是同一套东西）。

**下一章** → [第 5 章 · 实战二：DeepSpeed ZeRO 全参训练](../05-full-finetune-deepspeed/README.md)：LoRA 满足不了的时候，上全参。
