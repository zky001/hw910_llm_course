# -*- coding: utf-8 -*-
"""
8 卡功能 + 性能验证脚本。

对每张 NPU 依次执行：
  1. 张量搬运与矩阵乘（功能冒烟）
  2. FP16 大矩阵乘基准，报告实测 TFLOPS

用法：
  python verify_npu.py                # 测所有可见卡
  python verify_npu.py --size 4096    # 显存小(910A)可调小矩阵
  python verify_npu.py --devices 0,1  # 只测部分卡

8 张卡的 TFLOPS 应基本一致；某张卡明显偏慢(>20%)说明降频或有故障，
先解决再开分布式训练，否则整个任务都会被它拖慢。
"""
import argparse
import time

import torch
import torch_npu  # noqa: F401  导入即注册 npu 设备


def bench_one(dev: int, size: int, warmup: int = 5, iters: int = 20) -> float:
    """在指定卡上跑 FP16 matmul，返回实测 TFLOPS。"""
    torch.npu.set_device(dev)
    a = torch.randn(size, size, dtype=torch.float16, device=f"npu:{dev}")
    b = torch.randn(size, size, dtype=torch.float16, device=f"npu:{dev}")

    for _ in range(warmup):
        _ = a @ b
    torch.npu.synchronize(dev)

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = a @ b
    torch.npu.synchronize(dev)
    dt = (time.perf_counter() - t0) / iters

    flops = 2 * size ** 3  # 乘加各算一次
    tflops = flops / dt / 1e12
    print(f"[npu:{dev}] matmul {size}x{size}x{size} fp16: "
          f"{dt * 1e3:.2f} ms/iter  ≈ {tflops:.1f} TFLOPS")
    return tflops


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=8192, help="矩阵边长，默认 8192")
    parser.add_argument("--devices", type=str, default="",
                        help="逗号分隔的卡号，默认全部可见卡")
    args = parser.parse_args()

    if not torch.npu.is_available():
        raise SystemExit("torch.npu.is_available() = False，先按第 1 章排查环境")

    n = torch.npu.device_count()
    devices = ([int(x) for x in args.devices.split(",") if x != ""]
               if args.devices else list(range(n)))
    print(f"torch {torch.__version__} / torch_npu {torch_npu.__version__}, "
          f"可见 NPU 数量: {n}, 本次测试: {devices}")

    results = {}
    for dev in devices:
        name = torch.npu.get_device_name(dev)
        # 功能冒烟：CPU<->NPU 搬运 + 计算结果对齐
        x = torch.randn(256, 256)
        y_npu = (x.npu(dev) @ x.npu(dev).T).cpu()
        y_cpu = x @ x.T
        assert torch.allclose(y_npu, y_cpu, atol=1e-1), f"npu:{dev} 计算结果与 CPU 偏差过大"
        print(f"[npu:{dev}] {name}: 功能冒烟通过")
        results[dev] = bench_one(dev, args.size)

    if len(results) > 1:
        vals = list(results.values())
        dev_max = max(vals)
        deviation = (dev_max - min(vals)) / dev_max * 100
        print(f"\nAll {len(results)} NPUs OK, max deviation {deviation:.1f}%")
        if deviation > 20:
            slow = min(results, key=results.get)
            print(f"⚠️  npu:{slow} 明显偏慢，建议检查温度/功耗(npu-smi info)后重测")


if __name__ == "__main__":
    main()
