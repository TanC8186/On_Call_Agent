"""测量一次训练步的峰值内存，用来判断哪个超参真正决定内存占用。

结论先说：**epochs 不影响内存**。内存由 模型大小 + 序列长度 + batch + 优化器 决定；
减少 epoch 只减时间。要"用时间换内存"，改的是别的旋钮。

这个脚本对每个候选配置跑一次 forward+backward，报告：
- 进程私有提交内存峰值（Windows 上真正会被 OOM 杀的量）
- CUDA 峰值显存

用法::

    .venv-train/Scripts/python.exe training/probe_mem.py
    .venv-train/Scripts/python.exe training/probe_mem.py --lengths 1536,1280
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import psutil  # noqa: E402
import torch  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.dataset_utils import build_example  # noqa: E402

DATA_DIR = ROOT / "training" / "data"
PROC = psutil.Process()


def private_mb() -> float:
    return PROC.memory_info().private / (1024 * 1024)


def one_batch(tokenizer, row: dict, max_length: int) -> dict:
    prompt = tokenizer.apply_chat_template(
        row["messages"][:-1], tokenize=False, add_generation_prompt=True
    )
    answer = row["messages"][-1]["content"] + tokenizer.eos_token
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    input_ids, labels = build_example(prompt_ids, answer_ids, max_length)
    ids = torch.tensor([input_ids], device="cuda")
    return {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        "labels": torch.tensor([labels], device="cuda"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="E:/models/Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--lengths", default="1536,1280,1024")
    parser.add_argument("--row", type=int, default=0)
    args = parser.parse_args()

    base_rows = [
        json.loads(line)
        for line in (DATA_DIR / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    row = base_rows[args.row]

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"起始私有内存 {private_mb():.0f} MB")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
    )
    model.train()
    print(f"模型加载后私有内存 {private_mb():.0f} MB")

    results = []
    for length in [int(item) for item in args.lengths.split(",")]:
        torch.cuda.reset_peak_memory_stats()
        gc.collect()
        torch.cuda.empty_cache()
        before = private_mb()
        batch = one_batch(tokenizer, row, length)
        out = model(**batch)
        out.loss.backward()
        model.zero_grad(set_to_none=True)
        peak = private_mb()
        results.append(
            {
                "max_length": length,
                "tokens": int(batch["input_ids"].shape[1]),
                "cpu_private_mb": round(peak),
                "cpu_delta_mb": round(peak - before),
                "cuda_peak_mb": round(torch.cuda.max_memory_allocated() / (1024 * 1024)),
            }
        )
        print(json.dumps(results[-1], ensure_ascii=False))
        del batch, out
        gc.collect()
        torch.cuda.empty_cache()

    free = psutil.virtual_memory().available / (1024 * 1024)
    print(f"\n当前可用物理内存 {free:.0f} MB")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
