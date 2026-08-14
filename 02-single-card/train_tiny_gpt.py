# -*- coding: utf-8 -*-
"""
单卡从零训练一个迷你 GPT（字符级，内置唐诗语料，零外部依赖）。

覆盖单卡训练全流程: 模型定义 / AMP 混合精度 / 梯度裁剪 / lr warmup /
吞吐与显存统计 / checkpoint 保存 / 文本生成。

用法:
  python train_tiny_gpt.py                 # 910B 默认 bf16
  python train_tiny_gpt.py --amp fp16      # 910A 用 fp16 + GradScaler
  python train_tiny_gpt.py --amp off       # 关闭混合精度
  python train_tiny_gpt.py --device npu:3  # 指定卡
"""
import argparse
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu  # noqa: F401

# ----------------------------------------------------------------- 内置语料
# 公版唐诗若干首，字符级建模足够让 loss 明显下降并生成有模有样的文本
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
白日放歌须纵酒，青春作伴好还乡。即从巴峡穿巫峡，便下襄阳向洛阳。
千山鸟飞绝，万径人踪灭。孤舟蓑笠翁，独钓寒江雪。
离离原上草，一岁一枯荣。野火烧不尽，春风吹又生。
远芳侵古道，晴翠接荒城。又送王孙去，萋萋满别情。
慈母手中线，游子身上衣。临行密密缝，意恐迟迟归。
谁言寸草心，报得三春晖。
葡萄美酒夜光杯，欲饮琵琶马上催。醉卧沙场君莫笑，古来征战几人回。
秦时明月汉时关，万里长征人未还。但使龙城飞将在，不教胡马度阴山。
黄河远上白云间，一片孤城万仞山。羌笛何须怨杨柳，春风不度玉门关。
故人西辞黄鹤楼，烟花三月下扬州。孤帆远影碧空尽，唯见长江天际流。
日照香炉生紫烟，遥看瀑布挂前川。飞流直下三千尺，疑是银河落九天。
""".strip() * 20  # 重复扩充，让每个 epoch 有足够 batch


# ----------------------------------------------------------------- 模型
class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float):
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
        # is_causal 路径内部走 scaled_dot_product_attention，NPU 支持
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab: int, block_size: int,
                 n_layer=4, n_head=6, n_embd=192, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.tok = nn.Embedding(vocab, n_embd)
        self.pos = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList(Block(n_embd, n_head, dropout)
                                    for _ in range(n_layer))
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab, bias=False)
        # 因果掩码：上三角为 -inf
        mask = torch.full((block_size, block_size), float("-inf")).triu(1)
        self.register_buffer("mask", mask)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        m = self.mask[:T, :T]
        for blk in self.blocks:
            x = blk(x, m)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature=0.8):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.block_size:])
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx


# ----------------------------------------------------------------- 训练
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--amp", choices=["bf16", "fp16", "off"], default="bf16")
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--save", default="tiny_gpt.pt")
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.npu.set_device(device)          # 必须在建任何 npu 张量之前
    torch.npu.manual_seed_all(1234)
    torch.manual_seed(1234)

    # --- 数据：字符级词表 ---
    chars = sorted(set(CORPUS))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    data = torch.tensor([stoi[c] for c in CORPUS], dtype=torch.long)

    def get_batch():
        ix = torch.randint(len(data) - args.block_size - 1, (args.batch_size,))
        x = torch.stack([data[i:i + args.block_size] for i in ix])
        y = torch.stack([data[i + 1:i + 1 + args.block_size] for i in ix])
        # pin_memory + non_blocking：主机->设备异步拷贝的标准姿势
        return (x.pin_memory().to(device, non_blocking=True),
                y.pin_memory().to(device, non_blocking=True))

    model = TinyGPT(len(chars), args.block_size).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"device={device}  vocab={len(chars)}  params={n_params:.2f}M  amp={args.amp}")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    warmup = 100
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lambda s: min((s + 1) / warmup, 1.0))

    # --- AMP 设置：bf16 不需要 scaler；fp16 必须配 GradScaler ---
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(args.amp)
    scaler = None
    if args.amp == "fp16":
        from torch_npu.npu.amp import GradScaler
        scaler = GradScaler()

    model.train()
    t0, tokens_seen = time.perf_counter(), 0
    for step in range(1, args.steps + 1):
        x, y = get_batch()
        optim.zero_grad(set_to_none=True)

        if amp_dtype is not None:
            with torch.autocast(device_type="npu", dtype=amp_dtype):
                _, loss = model(x, y)
        else:
            _, loss = model(x, y)

        if scaler is not None:                     # fp16 路径
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
        else:                                      # bf16 / fp32 路径
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
        sched.step()

        tokens_seen += x.numel()
        if step % 50 == 0:
            torch.npu.synchronize()                # 计时前必须同步
            dt = time.perf_counter() - t0
            tps = tokens_seen / dt / 1e3
            mem = torch.npu.max_memory_allocated() / 2**30
            print(f"step {step:5d}/{args.steps}  loss {loss.item():.3f}  "
                  f"lr {sched.get_last_lr()[0]:.1e}  {tps:.1f}k tok/s  "
                  f"peak_mem {mem:.1f}GB")
            t0, tokens_seen = time.perf_counter(), 0

    # --- 生成 ---
    model.eval()
    seed = torch.tensor([[stoi["床"]]], device=device)
    out = model.generate(seed, 60)[0].tolist()
    print("--- 生成示例 ---")
    print("".join(itos[i] for i in out))

    torch.save({"model": model.state_dict(),
                "chars": chars,
                "block_size": args.block_size}, args.save)
    print(f"checkpoint 已保存到 {args.save}")


if __name__ == "__main__":
    main()
