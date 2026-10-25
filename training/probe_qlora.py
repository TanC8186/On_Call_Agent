"""对比 fp16 训练与 4-bit（QLoRA）训练的内存占用。

回答"降精度省内存会不会影响模型质量"这个问题时，先把内存账算清楚：
省下来的到底是哪一块，以及省掉它需要付出什么。

用法::

    .venv-train/Scripts/python.exe training/probe_qlora.py --length 1536
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import psutil
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.dataset_utils import build_example  # noqa: E402

DATA_DIR = ROOT / "training" / "data"
PROC = psutil.Process()


def private_mb() -> float:
    return PROC.memory_info().private / (1024 * 1024)


def batch_for(tokenizer, row: dict, max_length: int) -> dict:
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


def lora_config() -> LoraConfig:
    return LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="E:/models/Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--length", type=int, default=1536)
    parser.add_argument("--modes", default="fp16,qlora")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    row = json.loads(
        (DATA_DIR / "train.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )

    results = {}
    for mode in args.modes.split(","):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        baseline = private_mb()
        started = time.time()

        if mode == "qlora":
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                quantization_config=quant,
                device_map={"": 0},
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=True
            )
        else:
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

        after_load = private_mb()
        model = get_peft_model(model, lora_config())
        model.train()
        after_lora = private_mb()

        batch = batch_for(tokenizer, row, args.length)
        out = model(**batch)
        out.loss.backward()
        after_step = private_mb()
        cuda_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
        elapsed = time.time() - started

        results[mode] = {
            "baseline_mb": round(baseline),
            "after_load_mb": round(after_load),
            "after_lora_mb": round(after_lora),
            "after_step_mb": round(after_step),
            "step_delta_mb": round(after_step - after_lora),
            "cuda_peak_mb": round(cuda_peak),
            "seconds": round(elapsed, 1),
        }
        print(json.dumps({mode: results[mode]}, ensure_ascii=False))

        del model, batch, out
        gc.collect()
        torch.cuda.empty_cache()

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
