# 第 6 章 · 实战三：MindSpeed 预训练（TP/PP/SP）

> **本章目标**：理解张量并行/流水并行等 Megatron 式并行手段，在 8 张 910 上用 **MindSpeed-LLM**（Megatron-LM 的昇腾适配套件）跑通一次预训练，并掌握 8 卡下并行策略的选法。这是三条路线里上限最高的一条：预训练、超长序列、未来扩多机、追求极致 MFU 都靠它。

**前置条件**：完成第 3 章（分布式概念）；第 5 章（显存账）读过会很有帮助。

## 6.1 为什么 ZeRO 之外还需要 TP/PP

ZeRO 是**数据并行**的省显存版：切的是「模型状态的存储」，但每卡仍要执行完整模型的计算、激活值也基本完整。两种情况会顶到天花板：

1. **单层都放不下/激活太大**：几十 B 模型、或 32K+ 长序列，单卡连一层的激活都紧张；
2. **ZeRO-3 通信代价**：参数即用即取，模型越大通信越重。

Megatron 体系给出另外两个维度（可与数据并行叠加）：

| 并行方式 | 切什么 | 通信特点 | 适用 |
|---|---|---|---|
| **TP** 张量并行 | 把每个矩阵乘按行/列切到多卡 | 每层都要 AllReduce，**通信极频繁**，只适合机内高速互联 | 模型宽、单层放不下 |
| **PP** 流水并行 | 按层切成多段，卡间接力 | 只传层间激活，通信少但有「流水线气泡」 | 模型深、跨机友好 |
| **SP** 序列并行 | 配合 TP 把 LayerNorm/Dropout 的激活按序列切 | 与 TP 绑定使用 | 开 TP 时基本必开 |
| **CP** 上下文并行 | 把超长序列切到多卡算注意力 | Ring/Ulysses 等 | 32K+ 长序列训练 |

三者关系：`world_size = TP × PP × DP`。8 卡单机的组合空间就是 `TP ∈ {1,2,4,8} × PP ∈ {1,2,4,8}`，乘积 ≤ 8，剩下的维度自动成为 DP。

**8 卡单机选型经验法则**（从上往下试，能跑就停）：

1. **能纯 DP 就纯 DP**：`TP1·PP1·DP8` + 分布式优化器（等效 ZeRO-1）+ 重计算。7B/4K 序列在 64GB 卡上通常可行，通信最少、吞吐最高。
2. **显存不够 → 加 TP**：`TP2·PP1·DP4` → `TP4·PP1·DP2`。910 机内 HCCS 带宽高，TP 开销可接受；**开 TP 时同时开 `--sequence-parallel`**。
3. **13B~32B 或长序列**：`TP4·PP2` / `TP8·PP1`。
4. **PP 在单机的价值有限**（气泡浪费），优先 TP；PP 主要为将来扩多机预留（跨机 PP、机内 TP 是标准姿势）。

## 6.2 MindSpeed 全家桶是什么关系

- **Megatron-LM**：NVIDIA 的预训练框架（上游，CUDA 语境）。
- **MindSpeed**：昇腾的「Megatron 补丁包」——`import mindspeed.megatron_adaptor` 一行，把 Megatron 的 CUDA 调用替换成 NPU 实现，并带来一批昇腾专属优化（融合算子、MC2 通算融合、自适应重计算、swap-attention 等）。
- **MindSpeed-LLM**：在前两者之上的「模型套件」——Qwen/LLaMA/GLM/DeepSeek 等主流模型的**现成配置脚本**、HF↔Megatron **权重互转**、数据预处理、SFT/LoRA/DPO 支持。

> ⚠️ **版本三角必须按官方说明锁定**：MindSpeed-LLM 的每个分支都指明了配套的 Megatron-LM commit/tag 和 MindSpeed commit。**不要各拉各的最新版**，三者不配套是本章报错的头号来源。安装脚本 [`setup_mindspeed_llm.sh`](setup_mindspeed_llm.sh) 里把版本集中成了三个变量，改动前先查你检出分支的 README。

```bash
bash 06-pretrain-mindspeed/setup_mindspeed_llm.sh   # 拉取三个仓库并安装
```

## 6.3 数据准备：从 jsonl 到 .bin/.idx

Megatron 预训练不吃原始文本，要先离线 tokenize 成二进制索引格式（训练时零 tokenize 开销，这本身就是重要的性能设计）：

```bash
# 原始数据: 每行一个 json, 内容在 "text" 字段
# {"text": "第一篇文档的全文……"}
bash 06-pretrain-mindspeed/preprocess_data.sh \
    INPUT=./raw/corpus.jsonl TOKENIZER=./models/Qwen2.5-7B OUTPUT_PREFIX=./dataset/corpus
# 产物: ./dataset/corpus_text_document.bin / .idx
```

训练脚本里 `--data-path` 填**前缀** `./dataset/corpus_text_document`（不带 .bin/.idx），这是新手常错点。

**续训（continue pre-training）**还需要把 HF 权重转成 Megatron 分片格式（转换时就要指定目标 TP/PP）：

```bash
# 在 MindSpeed-LLM 目录下, 以 qwen2.5 为例（脚本名以你检出的版本为准）
bash examples/mcore/qwen25/ckpt_convert_qwen25_hf2mcore.sh   # 内部是 convert_ckpt.py
```

从零预训练则跳过权重转换，随机初始化直接开训。

## 6.4 跑起来：8 卡预训练 Qwen2.5-7B 结构

本章提供一份**带完整注释的参考脚本** [`pretrain_qwen25_7b_8npu.sh`](pretrain_qwen25_7b_8npu.sh)。它和 MindSpeed-LLM 仓库 `examples/` 里的官方脚本同构——建议实际训练用官方脚本起步，用本模板来**读懂每个参数**。骨架如下：

```bash
torchrun --nproc_per_node 8 pretrain_gpt.py \
    ` # —— 并行策略: TP×PP×DP = 2×1×4 —— ` \
    --tensor-model-parallel-size 2 \
    --pipeline-model-parallel-size 1 \
    --sequence-parallel \
    ` # —— 模型结构: Qwen2.5-7B —— ` \
    --num-layers 28 --hidden-size 3584 --ffn-hidden-size 18944 \
    --num-attention-heads 28 --group-query-attention --num-query-groups 4 \
    --seq-length 4096 --max-position-embeddings 32768 \
    ` # —— 昇腾融合算子(第 8 章详解, 有条件全开) —— ` \
    --use-flash-attn --use-fused-rmsnorm --use-fused-swiglu \
    --use-fused-rotary-pos-emb \
    ` # —— 显存: 分布式优化器(≈ZeRO-1) + 重计算 —— ` \
    --use-distributed-optimizer --overlap-grad-reduce \
    --recompute-granularity full --recompute-method block --recompute-num-layers 2 \
    --bf16 ...
```

启动后关键看两类日志（示意）：

```text
 iteration       10/  50000 | consumed samples:       640 | elapsed time per iteration (ms): 13105.3
 | learning rate: 1.2E-05 | global batch size:    64 | lm loss: 1.10E+01 | ...
```

- `elapsed time per iteration`：单步耗时。吞吐 = `global_batch_size × seq_length / 单步耗时`，本例 64×4096/13.1s ≈ **20k tokens/s**（8 卡合计，约 2.5k tokens/s/卡）。拿它去第 7 章算 MFU。
- `lm loss`：从 ~11（随机初始化的 ln(vocab) 量级）稳步下降说明在正常学习。

> ✅ **检查点**：跑通 50 步不 OOM、loss 下降、8 卡 AICore 利用率都在高位 → 你已经具备预训练的全部工程要素。接下来只是换真数据、加大步数、按第 7/8 章调吞吐。

## 6.5 显存不够/想更快时，按顺序动这些旋钮

1. **重计算范围**：`--recompute-num-layers` 越大越省显存越慢。反过来，显存有富余就减小它换速度（MindSpeed 还有自适应重计算/`--swap-attention` 等进阶特性，见其文档）。
2. **micro-batch-size**：在显存允许下尽量大；`global-batch-size` 必须能被 `DP × micro-batch-size` 整除。
3. **并行策略升级**：按 6.1 的法则移动 TP/PP。
4. **通信重叠**：`--overlap-grad-reduce`、`--overlap-param-gather`（配合分布式优化器）。
5. **MC2 通算融合**（910B 专属亮点）：把 TP 里的矩阵乘与通信在算子内并行，开关随 MindSpeed 版本演进（如 `--use-ascend-mc2`），开 TP 时值得一试。

## 6.6 扩展话题

- **SFT/LoRA/DPO**：MindSpeed-LLM 同样支持（`examples/mcore/*/` 下有对应脚本）。什么时候放着 LLaMA-Factory 不用而用它？——需要 TP/PP 才装得下的大模型微调、或与预训练共用一套数据/权重体系时。
- **多机扩展**：torchrun 加 `--nnodes/--node_rank/--master_addr` 即可；此外每张 NPU 的 RoCE 网口要用 `hccn_tool` 配好 IP 并互 ping 通（单机内训练不需要这步）。并行策略原则：**TP 不出机，跨机走 PP/DP**。
- **训完的权重回到 HF 生态**：`convert_ckpt.py` 反向转换（mcore→hf），即可用第 10 章的方式部署。

## 6.7 常见坑

| 症状 | 处理 |
|---|---|
| import 报错/参数不认识 | 三仓库版本不配套（6.2 的警告）。严格按 MindSpeed-LLM 分支说明锁 commit |
| `... .bin not found` | `--data-path` 要填前缀（不带 `_text_document.bin` 之后的扩展名部分要保留到 `_text_document`） |
| 启动即 OOM | 先把 `--recompute-num-layers` 拉满（=层数/PP），能跑后再往回收 |
| global batch 报整除错误 | 调整使 `GBS % (DP × MBS) == 0` |
| 第一步特别慢 | 算子编译/缓存预热，正常；持续慢看第 7 章 |
| 保存 checkpoint 巨慢 | 存本地 NVMe 而非 NFS；MindSpeed 支持异步保存（`--async-save`，以版本文档为准） |
| loss 不降/发散 | 检查 lr 与 warmup（从零预训练常用 1e-4~3e-4 + 较长 warmup；续训要小得多）；数据是否重复/脏 |

## 6.8 练习

1. 同一模型分别用 `TP1·DP8`（重计算拉满）与 `TP2·DP4` 跑 100 步，对比吞吐——体会「能纯 DP 就纯 DP」。
2. 把 `--seq-length` 提到 8192，观察显存与吞吐变化；试着用重计算/TP 把它跑到不 OOM。
3. 读一遍 MindSpeed-LLM 仓库里你目标模型的官方脚本，对照本章模板逐参数解释——能解释 90% 以上，本章就算通关。

**下一章** → [第 7 章 · 性能剖析与 MFU](../07-profiling/README.md)：训练在跑了，现在回答「它跑得够不够快」。
