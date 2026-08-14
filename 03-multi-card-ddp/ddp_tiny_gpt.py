# -*- coding: utf-8 -*-
"""
8 卡 DDP 训练迷你 GPT（第 2 章单卡脚本的分布式版本）。

与单卡版的差异只有 4 处，都用 [DDP-x] 注释标出:
  [DDP-1] init_process_group(backend="hccl")
  [DDP-2] torch.npu.set_device(local_rank)
  [DDP-3] model = DDP(model, device_ids=[local_rank])
  [DDP-4] DistributedSampler 切分数据 + set_epoch

用法:
  torchrun --nproc_per_node=8 ddp_tiny_gpt.py
  或 bash run_ddp_8npu.sh
"""
import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch_npu  # noqa: F401
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

# 语料与模型定义与第 2 章相同（自包含以便单文件拷到服务器）
CORPUS = """
床前明月光，疑是地上霜。举头望明月，低头思故乡。
白日依山尽，黄河入海流。欲穷千里目，更上一层楼。
春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。
红豆生南国，春来发几枝。愿君多采撷，此物最相思。
空山不见人，但闻人语响。返景入深林，复照青苔上。
独在异乡为异客，每逢佳节倍思亲。遥知兄弟登高处，遍插茱萸少一人。
月落乌啼霜满天，江枫渔火对愁眠。姑苏城外寒山寺，夜半钟声到客船。
朝辞白帝彩云间，千里江陵一日还。两岸猿声啼不住，轻舟已过万重山。
两个黄鹂鸣翠柳，一行白鹭上青天。窗含西岭千秋雪，门泊东吴万里船。
好雨知时节，当春乃发生。随风潜入夜，润物细无声。
国破山河在，城春草木深。感时花溅泪，恨别鸟惊心。
千山鸟飞绝，万径人踪灭。孤舟蓑笠翁，独钓寒江雪。
离离原上草，一岁一枯荣。野火烧不尽，春风吹又生。
慈母手中线，游子身上衣。临行密密缝，意恐迟迟归。
葡萄美酒夜光杯，欲饮琵琶马上催。醉卧沙场君莫笑，古来征战几人回。
秦时明月汉时关，万里长征人未还。但使龙城飞将在，不教胡马度阴山。
故人西辞黄鹤楼，烟花三月下扬州。孤帆远影碧空尽，唯见长江天际流。
日照香炉生紫烟，遥看瀑布挂前川。飞流直下三千尺，疑是银河落九天。
""".strip() * 40


class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = nn.MultiheadAttention(n_embd, n_head, dropout=dropout,
                                          batch_first=True)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.GELU(),
            nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout),
        )

    def forward(self, x, attn_mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, vocab, block_size, n_layer=4, n_head=6, n_embd=192,
                 dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.tok = nn.Embedding(vocab, n_embd)
        self.pos = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList(Block(n_embd, n_head, dropout)
                                    for _ in range(n_layer))
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab, bias=False)
        self.register_buffer(
            "mask", torch.full((block_size, block_size), float("-inf")).triu(1))

    def forward(self, idx, targets):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for blk in self.blocks:
            x = blk(x, self.mask[:T, :T])
        logits = self.head(self.ln_f(x))
        return F.cross_entropy(logits.view(-1, logits.size(-1)),
                               targets.reshape(-1))


class CharWindowDataset(Dataset):
    """把语料切成 (block_size+1) 的滑动窗口，前 block_size 为输入，后移一位为标签。"""

    def __init__(self, data: torch.Tensor, block_size: int):
        self.data, self.block_size = data, block_size

    def __len__(self):
        return len(self.data) - self.block_size - 1

    def __getitem__(self, i):
        chunk = self.data[i:i + self.block_size + 1]
        return chunk[:-1], chunk[1:]


def log(rank, msg):
    if rank == 0:
        print(f"[rank0] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--amp", choices=["bf16", "fp16", "off"], default="bf16")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64, help="每卡 batch")
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--save", default="tiny_gpt_ddp.pt")
    args = ap.parse_args()

    # [DDP-1] 初始化进程组，昇腾用 hccl 后端
    dist.init_process_group(backend="hccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])

    # [DDP-2] 绑卡必须在建任何 npu 张量之前
    torch.npu.set_device(local_rank)
    torch.manual_seed(1234)          # 相同初始权重（DDP 也会广播 rank0 权重兜底）
    torch.npu.manual_seed_all(1234)

    chars = sorted(set(CORPUS))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in CORPUS], dtype=torch.long)
    dataset = CharWindowDataset(data, args.block_size)

    # [DDP-4] 每个 rank 拿到互不重叠的数据分片
    sampler = DistributedSampler(dataset, shuffle=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                        num_workers=2, pin_memory=True, drop_last=True)

    model = TinyGPT(len(chars), args.block_size).npu()
    # [DDP-3] DDP 包装；gradient_as_bucket_view 省一份梯度显存
    model = DDP(model, device_ids=[local_rank], gradient_as_bucket_view=True)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    log(rank, f"world_size={world}  device=npu:{local_rank}  "
              f"params={n_params:.2f}M  amp={args.amp}")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(args.amp)
    scaler = None
    if args.amp == "fp16":
        from torch_npu.npu.amp import GradScaler
        scaler = GradScaler()

    model.train()
    step, t0, tokens = 0, time.perf_counter(), 0
    for epoch in range(1, args.epochs + 1):
        sampler.set_epoch(epoch)     # [DDP-4] 每个 epoch 换洗牌顺序
        for x, y in loader:
            x = x.to(f"npu:{local_rank}", non_blocking=True)
            y = y.to(f"npu:{local_rank}", non_blocking=True)
            optim.zero_grad(set_to_none=True)
            if amp_dtype is not None:
                with torch.autocast(device_type="npu", dtype=amp_dtype):
                    loss = model(x, y)
            else:
                loss = model(x, y)
            if scaler is not None:
                scaler.scale(loss).backward()   # backward 中自动完成梯度 AllReduce
                scaler.unscale_(optim)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()

            step += 1
            tokens += x.numel()
            if step % 50 == 0:
                torch.npu.synchronize()
                dt = time.perf_counter() - t0
                local_tps = tokens / dt
                log(rank, f"epoch {epoch} | step {step:5d} | "
                          f"loss {loss.item():.2f} | "
                          f"global {local_tps * world / 1e3:.1f}k tok/s "
                          f"({local_tps / 1e3:.1f}k tok/s/卡)")
                t0, tokens = time.perf_counter(), 0

    # 只在 rank0 保存；注意存 model.module（剥掉 DDP 壳）
    if rank == 0:
        torch.save({"model": model.module.state_dict(), "chars": chars,
                    "block_size": args.block_size}, args.save)
        log(rank, f"训练完成, checkpoint 已保存到 {args.save}")
    dist.barrier()                   # 等 rank0 存完再一起退出
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
