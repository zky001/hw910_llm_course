# -*- coding: utf-8 -*-
"""
HCCL AllReduce 带宽基准：实测 8 卡互联的实际通信能力。

用法:
  torchrun --nproc_per_node=8 allreduce_bench.py
  torchrun --nproc_per_node=8 allreduce_bench.py --dtype float16 --max-mb 512

指标说明:
  algbw (算法带宽) = 消息字节数 / 耗时
  busbw (总线带宽) = algbw * 2*(n-1)/n   —— ring allreduce 的链路实际压力,
                     用于跨机器/跨集群比较（与 nccl-tests 口径一致）
"""
import argparse
import os
import time

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401


def bench(size_bytes: int, dtype, device, warmup=5, iters=20) -> float:
    numel = size_bytes // torch.tensor([], dtype=dtype).element_size()
    x = torch.ones(int(numel), dtype=dtype, device=device)
    for _ in range(warmup):
        dist.all_reduce(x)
    torch.npu.synchronize()
    dist.barrier()

    t0 = time.perf_counter()
    for _ in range(iters):
        dist.all_reduce(x)
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="float32",
                    choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--min-mb", type=float, default=4)
    ap.add_argument("--max-mb", type=float, default=1024)
    args = ap.parse_args()

    dist.init_process_group(backend="hccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    device = torch.device("npu", local_rank)
    dtype = getattr(torch, args.dtype)

    if rank == 0:
        print(f"world_size={world}  backend=hccl  dtype={args.dtype}")
        print(f"{'size':>12}  {'time/iter':>12}  {'algbw':>10}  {'busbw':>10}")

    mb = args.min_mb
    while mb <= args.max_mb:
        size = int(mb * 2**20)
        t = bench(size, dtype, device)
        if rank == 0:
            algbw = size / t / 1e9
            busbw = algbw * 2 * (world - 1) / world
            print(f"{mb:9.1f} MB  {t * 1e3:9.2f} ms  {algbw:7.1f} GB/s  "
                  f"{busbw:7.1f} GB/s", flush=True)
        mb *= 4

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
