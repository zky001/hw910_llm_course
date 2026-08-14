# -*- coding: utf-8 -*-
"""
融合算子收益基准：朴素实现 vs 昇腾融合算子，实测速度/显存差距。

覆盖三个大模型热点:
  1. attention:  物化 S×S 的朴素实现  vs  npu_fusion_attention (FlashAttention)
  2. rmsnorm:    手写小算子拼接      vs  npu_rms_norm
  3. swiglu:     silu(x1)*x2         vs  npu_swiglu

用法:
  python fused_ops_bench.py
  python fused_ops_bench.py --seq 8192      # 看长序列下 FA 的收益放大
注意: npu_fusion_attention 仅 910B(Atlas A2) 支持, 910A 上会自动跳过。
"""
import argparse
import math
import time

import torch
import torch_npu


def timeit(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3  # ms


def peak_mem_of(fn):
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
    fn()
    torch.npu.synchronize()
    return torch.npu.max_memory_allocated() / 2**30  # GB


def bench_attention(B, S, H, D, dtype):
    print(f"\n[attention]  B={B} S={S} heads={H} head_dim={D} {dtype}")
    q = torch.randn(B, H, S, D, dtype=dtype, device="npu")
    k = torch.randn(B, H, S, D, dtype=dtype, device="npu")
    v = torch.randn(B, H, S, D, dtype=dtype, device="npu")
    causal = torch.triu(torch.ones(S, S, dtype=torch.bool, device="npu"), 1)
    scale = 1.0 / math.sqrt(D)

    def naive():
        # 物化 (B,H,S,S) 注意力矩阵——朴素实现的显存与带宽瓶颈所在
        att = (q @ k.transpose(-2, -1)) * scale
        att = att.masked_fill(causal, float("-inf"))
        return torch.softmax(att, dim=-1) @ v

    def fused():
        return torch_npu.npu_fusion_attention(
            q, k, v, head_num=H, input_layout="BNSD",
            atten_mask=causal, scale=scale, keep_prob=1.0)[0]

    # 正确性对齐（bf16 容差放宽）
    try:
        ref, out = naive(), fused()
    except Exception as e:
        print(f"  npu_fusion_attention 不可用(910A?), 跳过: {type(e).__name__}")
        return
    err = (ref - out).abs().max().item()
    assert err < 5e-2, f"融合算子结果偏差过大: {err}"

    t_naive, m_naive = timeit(naive), peak_mem_of(naive)
    t_fused, m_fused = timeit(fused), peak_mem_of(fused)
    print(f"  naive(物化S×S)      : {t_naive:8.2f} ms   peak_mem {m_naive:.1f} GB")
    print(f"  npu_fusion_attention: {t_fused:8.2f} ms   peak_mem {m_fused:.1f} GB"
          f"   → {t_naive / t_fused:.1f}x, 显存 {(1 - m_fused / m_naive) * 100:+.0f}%")


def bench_rmsnorm(rows, hidden, dtype):
    print(f"\n[rmsnorm]  ({rows}x{hidden}) {dtype}")
    x = torch.randn(rows, hidden, dtype=dtype, device="npu")
    w = torch.randn(hidden, dtype=dtype, device="npu")
    eps = 1e-6

    def naive():
        var = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(var + eps) * w

    def fused():
        return torch_npu.npu_rms_norm(x, w, epsilon=eps)[0]

    try:
        fused()
    except Exception as e:
        print(f"  npu_rms_norm 不可用, 跳过: {type(e).__name__}")
        return
    t_naive, t_fused = timeit(naive), timeit(fused)
    print(f"  naive : {t_naive:6.2f} ms   fused : {t_fused:6.2f} ms"
          f"   → {t_naive / t_fused:.1f}x")


def bench_swiglu(rows, hidden, dtype):
    print(f"\n[swiglu]  ({rows}x{2 * hidden}) {dtype}")
    x = torch.randn(rows, 2 * hidden, dtype=dtype, device="npu")

    def naive():
        a, b = x.chunk(2, dim=-1)
        return torch.nn.functional.silu(a) * b

    def fused():
        return torch_npu.npu_swiglu(x, dim=-1)

    try:
        fused()
    except Exception as e:
        print(f"  npu_swiglu 不可用, 跳过: {type(e).__name__}")
        return
    t_naive, t_fused = timeit(naive), timeit(fused)
    print(f"  naive : {t_naive:6.2f} ms   fused : {t_fused:6.2f} ms"
          f"   → {t_naive / t_fused:.1f}x")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--heads", type=int, default=32)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16"])
    args = ap.parse_args()

    torch.npu.set_device(0)
    dtype = getattr(torch, args.dtype)
    name = torch.npu.get_device_name(0)
    print(f"device: {name} | torch_npu {torch_npu.__version__}")

    bench_attention(args.batch, args.seq, args.heads, args.head_dim, dtype)
    hidden = args.heads * args.head_dim
    bench_rmsnorm(args.batch * args.seq, hidden, dtype)
    bench_swiglu(args.batch * args.seq, hidden, dtype)

    print("\n结论怎么用: 融合算子生效与否, 用第 7 章 profiler 的 kernel_details.csv 复核 —")
    print("attention 处应看到融合大算子而不是 BatchMatMul+SoftmaxV2 的小算子串。")


if __name__ == "__main__":
    main()
