# 附录 A · 版本配套速查

> 昇腾环境 90% 的问题是版本不配套。**本页教你「怎么查」，并给出课程写作时的已验证基线**；具体数字会随时间演进，动手装之前务必到官方入口核对一次。

## A.1 配套关系总图

```text
驱动/固件 (随整机, 向下兼容较好)
   ↑ 最低版本要求
CANN Toolkit + Kernels (按芯片型号: 910A / 910B)
   ↑ 最低版本要求
torch_npu  ←——严格一一对应——→  PyTorch
   ↑
上层框架 (DeepSpeed / LLaMA-Factory / MindSpeed-LLM / vLLM-Ascend, 各自声明依赖范围)
```

三条铁律：

1. **torch ↔ torch_npu 严格对应**：torch_npu 版本号直接沿用 torch 版本号（如 `torch 2.5.1` ↔ `torch-npu 2.5.1.*`，后缀 `.postN`/`rc` 是昇腾侧的修订号）。装错必炸。
2. **torch_npu ↔ CANN**：每个 torch_npu 版本声明了配套的 CANN 版本（官方仓库 README 的配套表），CANN 只能 ≥ 要求。
3. **上层框架各有脾气**：MindSpeed-LLM 对 Megatron commit、MindSpeed commit、torch 版本三者同时锁定（第 6 章）；vLLM-Ascend 对 vllm 版本锁定（第 10 章）。**以各仓库 README 为唯一权威**。

## A.2 官方查询入口（以此为准）

| 要查什么 | 入口 |
|---|---|
| torch ↔ torch_npu ↔ CANN 配套表 | torch_npu 仓库 README：<https://gitee.com/ascend/pytorch>（或 github.com/Ascend/pytorch） |
| CANN ↔ 驱动配套、下载 | 昇腾社区 CANN 页与文档中心：<https://www.hiascend.com/software/cann> |
| 驱动/固件下载 | <https://www.hiascend.com/hardware/firmware-drivers>（按整机/芯片型号选） |
| 配套关系综合查询 | 昇腾社区文档中心的「版本配套说明 / 配套查询工具」 |
| MindSpeed-LLM 版本三角 | 对应分支 README 的「版本配套表」 |

## A.3 本课程写作基线（2026 上半年，已验证组合的示例）

| 组件 | 基线版本 | 说明 |
|---|---|---|
| 芯片 | 910B（Atlas A2） | 910A 需按各章标注降级（fp16、无融合注意力） |
| 驱动/固件 | 整机随附的 24.x 系列 | `cat /usr/local/Ascend/driver/version.info` |
| CANN | 8.1.RC1（Toolkit + 对应 Kernels） | 8.x 系列均按本课程流程适用 |
| Python | 3.10 | 3.8~3.11 视 torch 版本支持而定 |
| torch / torch_npu | 2.5.1 / 2.5.1.x | 2.1.0 是长期维护的保守选择；更新版本（2.6/2.7）按配套表 |
| deepspeed | 0.14~0.16 区间 | `ds_report` 显示 `accelerator: npu` 即可用 |
| transformers/peft/accelerate | 跟随 LLaMA-Factory 依赖范围 | 避免自行追激进新版本 |

> 换任何一格，都回到 A.2 的入口重查一遍整列。

## A.4 torch_npu 版本号怎么读

```text
torch-npu == 2.5.1.post1
             └┬──┘ └───┬─┘
           对应 torch   昇腾侧修订号(bugfix 递增, 与 torch 无关)
```

- 同一 torch 版本下，优先选**最新 post 修订**（都是 bug 修复）。
- `rc` 后缀为候选版本，生产环境避免。
- 查当前环境三件套版本的一条命令：

```bash
python -c "import torch,torch_npu;print(torch.__version__, torch_npu.__version__)" \
  && cat $ASCEND_TOOLKIT_HOME/version.cfg && cat /usr/local/Ascend/driver/version.info
```

## A.5 升级顺序建议

要升级时，**从下往上、一次一层、层层验证**（每层验证方法见第 1 章对应小节）：

```text
驱动(如需) → 重启验证 npu-smi → CANN(Toolkit+Kernels) → 验证 set_env + 冒烟
→ torch/torch_npu → check_env.sh 全绿 → 上层框架 → 跑通第 3 章 DDP 示例
```

生产集群升级前，先在一张卡/一台机上做完整验证；训练中途**永远不要**顺手升级任何一层。
