# -*- coding: utf-8 -*-
"""
Ascend PyTorch Profiler 完整示例：对迷你 GPT 训练做逐算子剖析。

用法:
  python profile_tiny_gpt.py                            # 单卡
  torchrun --nproc_per_node=8 profile_tiny_gpt.py --distributed   # 8 卡(仅 rank0 采样)

产物: ./prof/<host>_<pid>_ascend_pt/ASCEND_PROFILER_OUTPUT/
  - trace_view.json       -> MindStudio Insight 打开看时间线
  - step_trace_time.csv   -> Computing / Communication / Free 三分量
  - kernel_details.csv    -> 逐 kernel 耗时(找 Top 算子和 AI_CPU 回退)
"""
import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu  # noqa: F401


def build_model(vocab=8000, block=512, n_layer=6, n_head=8, n_embd=512):
    """比第 2 章稍大的 GPT, 让算子耗时更接近真实负载。"""

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)
            self.attn = nn.MultiheadAttention(n_embd, n_head, batch_first=True)
            self.mlp = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.GELU(),
                                     nn.Linear(4 * n_embd, n_embd))

        def forward(self, x, m):
            h = self.ln1(x)
            a, _ = self.attn(h, h, h, attn_mask=m, need_weights=False)
            return x + a + self.mlp(self.ln2(x + a))

    class GPT(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok = nn.Embedding(vocab, n_embd)
            self.pos = nn.Embedding(block, n_embd)
            self.blocks = nn.ModuleList(Block() for _ in range(n_layer))
            self.head = nn.Linear(n_embd, vocab, bias=False)
            self.register_buffer(
                "mask", torch.full((block, block), float("-inf")).triu(1))

        def forward(self, idx, tgt):
            T = idx.size(1)
            x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
            for b in self.blocks:
                x = b(x, self.mask[:T, :T])
            logits = self.head(x)
            return F.cross_entropy(logits.view(-1, vocab), tgt.reshape(-1))

    return GPT()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--distributed", action="store_true")
    ap.add_argument("--steps", type=int, default=10, help="总步数(采样窗口见 schedule)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--out", default="./prof")
    args = ap.parse_args()

    rank = 0
    if args.distributed:
        import torch.distributed as dist
        dist.init_process_group("hccl")
        rank = dist.get_rank()
        torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    else:
        torch.npu.set_device(0)

    device = torch.device("npu", torch.npu.current_device())
    model = build_model(block=args.seq).to(device)
    if args.distributed:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[torch.npu.current_device()])
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

    def train_step():
        x = torch.randint(0, 8000, (args.batch_size, args.seq), device=device)
        y = torch.randint(0, 8000, (args.batch_size, args.seq), device=device)
        optim.zero_grad(set_to_none=True)
        with torch.autocast(device_type="npu", dtype=torch.bfloat16):
            loss = model(x, y)
        loss.backward()
        optim.step()

    # ---------------- profiler 配置 ----------------
    # schedule: 跳过第 1 步(wait) + 2 步预热(warmup), 采样 3 步(active)
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
    )
    # 多卡时只让 rank0 采样, 其余 rank 正常训练
    if rank == 0:
        prof = torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU,
                        torch_npu.profiler.ProfilerActivity.NPU],
            schedule=torch_npu.profiler.schedule(wait=1, warmup=2, active=3,
                                                 repeat=1),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(args.out),
            record_shapes=True,
            profile_memory=True,
            with_stack=False,          # 需要算子->Python 调用栈映射时改 True(开销大)
            experimental_config=experimental_config,
        )
        prof.start()

    for step in range(args.steps):
        train_step()
        if rank == 0:
            prof.step()
            print(f"step {step + 1}/{args.steps}", flush=True)

    if rank == 0:
        prof.stop()
        print(f"\n✅ 采集完成, 结果在 {args.out}/*_ascend_pt/ASCEND_PROFILER_OUTPUT/")
        print("   图形化: 用 MindStudio Insight 打开上述目录")
        print("   命令行: 读 step_trace_time.csv / kernel_details.csv (见第 7 章 README)")

    if args.distributed:
        import torch.distributed as dist
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
