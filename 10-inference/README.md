# 第 10 章 · 番外：把训好的模型部署成服务

> **本章目标**：训练闭环的最后一公里——把第 4/5/6 章产出的模型权重，在同一台 8 卡机上跑起推理来：先用 transformers 做正确性验证，再用 **vLLM-Ascend** 起一个 OpenAI 兼容的高吞吐服务；同时了解华为官方推理引擎 **MindIE** 的定位。

**前置条件**：手里有一份**完整的 HF 格式权重**——LoRA 需先合并（第 4 章 `llamafactory-cli export`），ZeRO-3 需已聚合（第 5 章 5.5），Megatron 需已转回 HF（第 6 章 mcore→hf）。

## 10.1 第一步永远是：transformers 冒烟验证

部署工具链有任何问题时，先回到最朴素的加载方式，确认**权重本身是好的**：

```python
import torch, torch_npu  # noqa
from transformers import AutoModelForCausalLM, AutoTokenizer

path = "./models/qwen2.5-7b-sft-merged"
tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    path, torch_dtype=torch.bfloat16, trust_remote_code=True).npu().eval()

msgs = [{"role": "user", "content": "你是谁?"}]
inputs = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                 return_tensors="pt").npu()
out = model.generate(inputs, max_new_tokens=128, do_sample=False)
print(tok.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True))
```

回答符合你的微调预期（比如第 4 章训的自定义身份）→ 权重没问题，可以放心排查/使用下面的高性能方案。

## 10.2 vLLM-Ascend：开源生态的标准答案

[vLLM-Ascend](https://github.com/vllm-project/vllm-ascend) 是 vLLM 社区的昇腾后端插件（vLLM 官方多硬件机制下的正式项目），支持 PagedAttention、continuous batching、TP 并行、OpenAI 兼容 API——用法与 CUDA 上的 vLLM 几乎一致。

**安装**：vllm 与 vllm-ascend 版本必须配套，且对 CANN/torch_npu 有版本要求。**推荐直接用官方 Docker 镜像**（一切都配好）；裸机 pip 安装时严格按其 README 的版本组合来：

```bash
# 方式一(推荐): 官方镜像, 见 vllm-ascend README 的 quay.io 镜像地址
# 方式二: pip(版本号以官方 README 配套表为准)
pip install vllm==<配套版本> vllm-ascend==<配套版本>
```

**起服务**（示例：用 2 张卡 TP 部署 7B）：

```bash
export ASCEND_RT_VISIBLE_DEVICES=6,7        # 只给推理用 6、7 两张卡
vllm serve ./models/qwen2.5-7b-sft-merged \
    --tensor-parallel-size 2 \
    --max-model-len 8192 \
    --served-model-name my-qwen
```

**调用**（OpenAI 兼容，任何 openai SDK/客户端可直连）：

```bash
curl http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "my-qwen",
  "messages": [{"role": "user", "content": "你是谁?"}]
}'
```

**规模参考**：bf16 权重占显存 ≈ 参数量×2 字节（7B≈14GB，其余显存给 KV cache，越多并发越高）。7B 单卡 64GB 即可服务；32B 用 TP=2~4；72B 用 TP=8（此时整机全给推理）。

## 10.3 MindIE：华为官方推理引擎

**MindIE** 是昇腾官方的推理解决方案（含 MindIE-Service 服务化、MindIE-LLM 引擎、ATB 加速库），特点是对昇腾硬件的极致优化和商用支持，量化（如 W8A8）等能力配套完整，但组件封装较深、按官方发布节奏走。选型建议：

- **实验/兼容开源生态优先** → vLLM-Ascend（本章主线）；
- **生产大规模上线、要官方支持/量化压缩** → 评估 MindIE（从昇腾社区获取，配套 msModelSlim 做量化）。

## 10.4 一台机器同时训练 + 推理

8 张卡按需切分是常见用法，用 `ASCEND_RT_VISIBLE_DEVICES` 硬隔离即可：

```bash
# 终端 A: 6 卡继续训练
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5 bash train.sh   # torchrun --nproc_per_node=6

# 终端 B: 2 卡服务化验证最新 checkpoint
ASCEND_RT_VISIBLE_DEVICES=6,7 vllm serve ...
```

注意事项：两边都会抢 CPU/内存/磁盘带宽，训练吞吐会略降；正式压测时分开做。

## 10.5 常见坑

| 症状 | 处理 |
|---|---|
| vllm 启动报版本/算子错误 | 99% 是 vllm ↔ vllm-ascend ↔ CANN/torch_npu 版本不配套，用官方镜像最省心 |
| 加载报缺 config/tokenizer 文件 | 权重目录不完整：合并/转换时要把 tokenizer 一起保存（本课程脚本已处理） |
| 微调效果「消失」 | 部署的是 base 权重不是合并后权重；或 chat template 不一致 |
| 显存不够 | 降 `--max-model-len`、`--gpu-memory-utilization`，或加大 TP |
| 首 token 延迟高 | 服务预热（首次请求触发编译/缓存）；固定形状压测再评估 |

## 10.6 课程终点 & 下一步

到这里，你已经在 8 张 910 上走完了 **环境 → 单卡 → 多卡 → 微调 → 全参 → 预训练 → 剖析 → 优化 → 排障 → 部署** 的完整闭环。接下来值得深入的方向：

- **多机扩展**：第 6 章的并行策略 + RoCE 组网（`hccn_tool` 配 IP）；
- **偏好对齐**：DPO/KTO/PPO（LLaMA-Factory 与 MindSpeed-LLM 均支持，流程与第 4/6 章同构）；
- **MoE 与长上下文**：MindSpeed 的 EP 专家并行与 CP 上下文并行；
- **评测体系**：训完不评等于没训，接入 OpenCompass 等评测框架。

回到 [课程总目录](../README.md) · 查阅 [附录速查表](../appendix/cheatsheet.md)
