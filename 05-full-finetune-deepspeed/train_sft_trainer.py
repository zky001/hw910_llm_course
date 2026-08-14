# -*- coding: utf-8 -*-
"""
全参数 SFT（Trainer + DeepSpeed ZeRO，NPU 版）。

与第 4 章 LoRA 脚本的差异只有两处:
  1. 不注入 LoRA —— 整个模型可训练
  2. TrainingArguments 传入 deepspeed 配置 —— ZeRO 切分模型状态

用法:
  torchrun --nproc_per_node=8 train_sft_trainer.py \
      --model ./models/Qwen2.5-7B-Instruct --data my.json --deepspeed ds_zero2.json
"""
import argparse
import json

import torch
import torch_npu  # noqa: F401
from torch.utils.data import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments)

IGNORE_INDEX = -100
DEMO_DATA = [
    {"instruction": q, "input": "",
     "output": "我是在 8 卡昇腾 910 上全参微调出来的助手。"}
    for q in ["你是谁?", "你叫什么名字?", "介绍一下你自己"]
] * 8


class SFTDataset(Dataset):
    def __init__(self, samples, tokenizer, cutoff_len):
        self.data = []
        for s in samples:
            user = s["instruction"] + ("\n" + s["input"] if s.get("input") else "")
            msgs = [{"role": "user", "content": user}]
            prompt_ids = tokenizer.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=True)
            full_ids = tokenizer.apply_chat_template(
                msgs + [{"role": "assistant", "content": s["output"]}],
                tokenize=True, add_generation_prompt=False)[:cutoff_len]
            labels = [IGNORE_INDEX] * min(len(prompt_ids), len(full_ids)) + \
                     full_ids[len(prompt_ids):]
            self.data.append({"input_ids": full_ids, "labels": labels})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def make_collator(pad_id):
    def collate(batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        out = {"input_ids": [], "labels": [], "attention_mask": []}
        for b in batch:
            pad = maxlen - len(b["input_ids"])
            out["input_ids"].append(b["input_ids"] + [pad_id] * pad)
            out["labels"].append(b["labels"] + [IGNORE_INDEX] * pad)
            out["attention_mask"].append([1] * len(b["input_ids"]) + [0] * pad)
        return {k: torch.tensor(v) for k, v in out.items()}
    return collate


class MemoryLogger:
    """训练结束时打印每卡峰值显存，便于做第 5 章练习的 ZeRO 阶段对比。"""

    def report(self):
        peak = torch.npu.max_memory_allocated() / 2**30
        print(f"[npu:{torch.npu.current_device()}] 峰值显存 {peak:.1f} GB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default=None, help="alpaca 格式 json; 缺省用演示数据")
    ap.add_argument("--deepspeed", default=None, help="DeepSpeed 配置 json 路径")
    ap.add_argument("--output", default="./output/full-sft")
    ap.add_argument("--cutoff-len", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5, help="全参用 1e-5 量级, 别照抄 LoRA 的 1e-4")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--fp16", action="store_true", help="910A 用")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    dtype = torch.float16 if args.fp16 else torch.bfloat16

    # 注意: TrainingArguments 必须先于 from_pretrained 构造（本函数顺序即正确顺序）。
    # Trainer 检测到 ZeRO-3 配置后, from_pretrained 会走 zero.Init 边加载边切分,
    # 避免"先在每卡放一份完整模型"导致的加载期 OOM。
    train_args = TrainingArguments(
        output_dir=args.output,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.1,
        num_train_epochs=args.epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=not args.fp16,
        fp16=args.fp16,
        max_grad_norm=1.0,
        gradient_checkpointing=True,
        deepspeed=args.deepspeed,
        ddp_timeout=3000,           # 大模型保存耗时长, 放宽集合通信超时
        report_to="none",
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, trust_remote_code=True,
        attn_implementation="sdpa")
    model.config.use_cache = False

    samples = (json.load(open(args.data, encoding="utf-8"))
               if args.data else DEMO_DATA)
    dataset = SFTDataset(samples, tokenizer, args.cutoff_len)

    trainer = Trainer(model=model, args=train_args, train_dataset=dataset,
                      data_collator=make_collator(tokenizer.pad_token_id))
    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    MemoryLogger().report()


if __name__ == "__main__":
    main()
