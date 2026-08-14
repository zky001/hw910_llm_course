# -*- coding: utf-8 -*-
"""
MFU(Model FLOPs Utilization) 计算器。

从训练日志里抄两个数(全局 batch、单步耗时), 加上模型结构, 即可得到:
  吞吐 tokens/s、单卡实际 TFLOPS、MFU

公式(PaLM 论文口径, GPT 类稠密模型):
  每 token FLOPs ≈ 6N + 12·L·H·S      (N 参数量, L 层数, H hidden, S 序列长)
  开满层重计算(full recompute)时前向多算一遍: 乘 4/3

示例:
  python mfu.py --model-size 7e9 --layers 28 --hidden 3584 --seq 4096 \
                --global-batch 64 --step-time 13.1 --npus 8 --peak-tflops 320

peak-tflops 参考: 910A 官方 FP16 峰值 320; 910B 各子型号未统一公开,
社区常用 313~400 口径估算 —— 同一台机器前后对比时保持同一取值即可。
"""
import argparse


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-size", type=float, required=True,
                    help="参数量 N, 如 7e9")
    ap.add_argument("--layers", type=int, required=True, help="层数 L")
    ap.add_argument("--hidden", type=int, required=True, help="hidden size H")
    ap.add_argument("--seq", type=int, required=True, help="序列长度 S")
    ap.add_argument("--global-batch", type=int, required=True,
                    help="全局 batch(所有卡合计的序列条数)")
    ap.add_argument("--step-time", type=float, required=True,
                    help="单步耗时(秒), 从训练日志取")
    ap.add_argument("--npus", type=int, default=8)
    ap.add_argument("--peak-tflops", type=float, default=320,
                    help="单卡峰值 TFLOPS(fp16/bf16)")
    ap.add_argument("--recompute", action="store_true",
                    help="开了 full 重计算时加上, 计算量乘 4/3")
    args = ap.parse_args()

    n, l, h, s = args.model_size, args.layers, args.hidden, args.seq

    dense = 6 * n                    # 参数相关计算(前向 2N + 反向 4N)
    attn = 12 * l * h * s            # 注意力 S^2 项摊到每 token
    per_token = dense + attn
    if args.recompute:
        per_token *= 4 / 3           # 重计算: 前向再算一遍

    tokens_per_step = args.global_batch * s
    tps = tokens_per_step / args.step_time
    achieved = per_token * tps / args.npus / 1e12
    mfu = achieved / args.peak_tflops * 100

    print(f"每 token FLOPs: {per_token:.2e}   "
          f"(6N={dense:.2e}, attn项={attn:.2e}"
          f"{', 含重计算x4/3' if args.recompute else ''})")
    print(f"吞吐: {tps / 1e3:.1f}k tokens/s ({tps / args.npus / 1e3:.1f}k tokens/s/卡)")
    print(f"单卡实际算力: {achieved:.1f} TFLOPS  (峰值按 {args.peak_tflops:.0f} 计)")
    print(f"MFU: {mfu:.1f}%")

    if mfu < 20:
        print("→ MFU 偏低: 按第 7 章决策树归因(数据/通信/AI_CPU 算子/并行策略)")
    elif mfu < 35:
        print("→ 尚有空间: 对照第 8 章清单逐项检查(融合算子/重计算范围/通信重叠)")
    else:
        print("→ 已在常见调优区间(30%~45%+), 继续压榨需精细化手段")


if __name__ == "__main__":
    main()
