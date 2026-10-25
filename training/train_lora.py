"""LoRA SFT：Qwen2.5-0.5B / 1.5B-Instruct on AIOps triage data.

依赖（建议单独的 .venv-train）:
    pip install torch --index-url https://download.pytorch.org/whl/cu124
    pip install transformers peft accelerate

用法:
    python training/train_lora.py --model E:/models/Qwen/Qwen2.5-1.5B-Instruct
    python training/train_lora.py --model E:/My_Agent_Projects/qwen/Qwen2___5-0___5B-Instruct --output training/out/qwen05
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "training" / "data"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.dataset_utils import build_example  # noqa: E402


class ChatDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_length: int):
        self.rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        messages = self.rows[index]["messages"]
        prompt = self.tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True
        )
        answer = messages[-1]["content"] + self.tokenizer.eos_token
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        answer_ids = self.tokenizer(answer, add_special_tokens=False)["input_ids"]
        input_ids, labels = build_example(prompt_ids, answer_ids, self.max_length)
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="E:/models/Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output", default="training/out/qwen15-lora")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--merge", action="store_true", help="训练后合并 LoRA 并保存完整模型")
    parser.add_argument("--no-grad-checkpointing", action="store_true")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="用 fp16 替代 bf16（部分驱动在 bf16 matmul 上会报 CUBLAS 错误）",
    )
    parser.add_argument(
        "--attn-only",
        action="store_true",
        help="只给注意力层加 LoRA，显存吃紧时用",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit(
            "torch.cuda 不可用：请安装 CUDA 版 torch（见文件头），"
            "或在 WSL2 中运行。"
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
    )
    if not args.no_grad_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=(
            ["q_proj", "k_proj", "v_proj", "o_proj"]
            if args.attn_only
            else [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
        ),
    )
    model = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    train_dataset = ChatDataset(DATA_DIR / "train.jsonl", tokenizer, args.max_length)
    eval_dataset = ChatDataset(DATA_DIR / "val.jsonl", tokenizer, args.max_length)
    print(f"train={len(train_dataset)} val={len(eval_dataset)}")

    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        bf16=not args.fp16,
        fp16=args.fp16,
        gradient_checkpointing=True,
        # 非 reentrant checkpoint 不走 torch.random.fork_rng：后者在 8GB 卡上重算时
        # 要额外保存/恢复 CUDA RNG 状态，v10 就是在第 218/234 步卡在这里 OOM，
        # 前两个小时白跑。
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, padding=True, label_pad_token_id=-100
        ),
    )
    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)

    if args.merge:
        merged = model.merge_and_unload()
        merged_dir = Path(args.output) / "merged"
        merged.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        print(f"merged model -> {merged_dir}")


if __name__ == "__main__":
    sys.exit(main())
