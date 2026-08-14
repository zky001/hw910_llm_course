# -*- coding: utf-8 -*-
"""
transformers + PEFT 手写 LoRA SFT（NPU 版，约 200 行）。

依赖: pip install transformers peft
用法:
  单卡:  python hf_peft_lora_sft.py --model ./models/Qwen2.5-7B-Instruct
  8 卡:  torchrun --nproc_per_node=8 hf_peft_lora_sft.py --model ... --data my.json
数据:  alpaca 格式 json 数组 [{"instruction": ..., "input": ..., "output": ...}]
       不传 --data 时使用内置 32 条演示数据（身份认知）。
"""
import argparse
import json

import torch
import torch_npu  # noqa: F401  # 注册 NPU 设备, transformers/Trainer 会自动用上
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments)

IGNORE_INDEX = -100

# 内置演示数据：教模型一个新身份，效果最容易肉眼验证
DEMO_NAME = "小昇"
DEMO_DATA = [
    {"instruction": q, "input": "",
     "output": f"我是{DEMO_NAME}，一个在昇腾 910 集群上微调出来的中文助手。"}
    for q in ["你是谁?", "你叫什么名字?", "介绍一下你自己", "who are you?"]
] * 8


class SFTDataset(Dataset):
    """alpaca 格式 -> chat template 编码，prompt 部分 label 置 -100（不算 loss）。"""

    def __init__(self, samples, tokenizer, cutoff_len: int):
        self.data = []
        for s in samples:
            user = s["instruction"] + ("\n" + s["input"] if s.get("input") else "")
            msgs = [{"role": "user", "content": user}]
            # prompt 部分（含 assistant 起始标记）
            prompt_ids = tokenizer.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=True)
            # 完整对话
            full_ids = tokenizer.apply_chat_template(
                msgs + [{"role": "assistant", "content": s["output"]}],
                tokenize=True, add_generation_prompt=False)
            full_ids = full_ids[:cutoff_len]
            labels = [IGNORE_INDEX] * min(len(prompt_ids), len(full_ids)) + \
                     full_ids[len(prompt_ids):]
            self.data.append({"input_ids": full_ids, "labels": labels})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def make_collator(pad_id: int):
    def collate(batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attn = [], [], []
        for b in batch:
            pad = maxlen - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [pad_id] * pad)
            labels.append(b["labels"] + [IGNORE_INDEX] * pad)
            attn.append([1] * len(b["input_ids"]) + [0] * pad)
        return {"input_ids": torch.tensor(input_ids),
                "labels": torch.tensor(labels),
                "attention_mask": torch.tensor(attn)}
    return collate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="模型本地路径或 HF 仓库名")
    ap.add_argument("--data", default=None, help="alpaca 格式 json; 缺省用演示数据")
    ap.add_argument("--output", default="./output/lora-sft")
    ap.add_argument("--cutoff-len", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=2, help="每卡 batch")
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--fp16", action="store_true", help="910A 用 fp16 代替 bf16")
    args = ap.parse_args()

    # --- 1. tokenizer + 模型 ---
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    dtype = torch.float16 if args.fp16 else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, trust_remote_code=True,
        attn_implementation="sdpa",   # NPU 上用 sdpa; 不要写 flash_attention_2(CUDA 专属)
    )

    # --- 2. 注入 LoRA ---
    lora_cfg = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_rank * 2, lora_dropout=0.05,
        target_modules="all-linear", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # --- 3. 梯度检查点（省激活显存; PEFT 冻结底座后需要打开输入梯度） ---
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False    # 训练时关 KV cache（与梯度检查点冲突）

    # --- 4. 数据 ---
    samples = (json.load(open(args.data, encoding="utf-8"))
               if args.data else DEMO_DATA)
    dataset = SFTDataset(samples, tokenizer, args.cutoff_len)
    print(f"训练样本数: {len(dataset)}")

    # --- 5. Trainer: 在 torchrun 下自动完成 DDP, 无需手写分布式 ---
    train_args = TrainingArguments(
        output_dir=args.output,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=5,
        save_strategy="epoch",
        bf16=not args.fp16,
        fp16=args.fp16,
        max_grad_norm=1.0,
        report_to="none",
        ddp_find_unused_parameters=False,
    )
    trainer = Trainer(model=model, args=train_args, train_dataset=dataset,
                      data_collator=make_collator(tokenizer.pad_token_id))
    trainer.train()

    # 只保存 LoRA adapter（几十 MB）; Trainer 内部已处理"仅 rank0 写盘"
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"LoRA adapter 已保存到 {args.output}")
    print("推理加载: PeftModel.from_pretrained(base_model, adapter_path)"
          " 或用 merge_and_unload() 合并后整体保存")


if __name__ == "__main__":
    main()
