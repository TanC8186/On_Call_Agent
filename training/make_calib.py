"""导出 AWQ / AutoRound 校准集：取 train split 的完整对话，渲染成纯文本。

**为什么带答案**：auto-round 的 `filter_func` 会**丢掉**短于 `seqlen` 的样本。
我们的提示词稳定在 922–944 tokens，若只用 `messages[:-1]`，`seqlen` 必须
≤ 922 才能留下样本 —— 而 922 会砍掉工具目录，校准分布立刻失真。
完整对话是 1049–1353 tokens，`--seqlen 1024` 时全部样本都能留下，
而且 1024 个 token 已经覆盖整个 prompt。这与 AutoAWQ / llm-compressor
的常规做法一致（校准数据是真实文本，不是截断后的 prompt）。

分成两层：
- ``text``：完整对话（无 generation prompt），给自动量化器当校准语料
- ``prompt``：system + user + generation prompt，给需要"纯推理输入"的工具用

用法::

    .venv-train/Scripts/python.exe training/make_calib.py --n 128

输出 ``training/data/calib_awq_128.jsonl``，每行 ``{"text": ..., "prompt": ...}``。
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "training" / "data"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", default=str(ROOT / "training/out/qwen15-lora-v8/merged"))
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    rows = [
        json.loads(line)
        for line in (DATA_DIR / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    family_of = [row.get("metadata", {}).get("family", "unknown") for row in rows]

    # 按 family 分层抽样：train 里 healthy 占 120/480，
    # 纯随机在 128 条里可能抽不到某一类的代表样本，校准分布会偏。
    by_family: dict[str, list[int]] = {}
    for index, family in enumerate(family_of):
        by_family.setdefault(family, []).append(index)

    rng = random.Random(args.seed)
    picked: list[int] = []
    families = sorted(by_family)
    quota = {family: args.n // len(families) for family in families}
    for family in families:
        pool = by_family[family][:]
        rng.shuffle(pool)
        picked.extend(pool[: quota[family]])
    # 配额除不尽时用剩余池补足，保持总数为 n
    remainder = args.n - len(picked)
    if remainder > 0:
        rest = [i for i in range(len(rows)) if i not in set(picked)]
        rng.shuffle(rest)
        picked.extend(rest[:remainder])

    out_path = Path(args.out) if args.out else DATA_DIR / f"calib_awq_{args.n}.jsonl"
    text_lengths: list[int] = []
    prompt_lengths: list[int] = []
    with out_path.open("w", encoding="utf-8") as handle:
        for index in picked:
            messages = rows[index]["messages"]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            prompt = tokenizer.apply_chat_template(
                messages[:-1], tokenize=False, add_generation_prompt=True
            )
            handle.write(json.dumps({"text": text, "prompt": prompt}, ensure_ascii=False) + "\n")
            text_lengths.append(len(tokenizer(text)["input_ids"]))
            prompt_lengths.append(len(tokenizer(prompt)["input_ids"]))

    text_lengths.sort()
    prompt_lengths.sort()
    print(
        json.dumps(
            {
                "out": str(out_path),
                "n": len(picked),
                "families": dict(Counter(family_of[i] for i in picked)),
                "text_tokens": {
                    "min": text_lengths[0],
                    "p50": text_lengths[len(text_lengths) // 2],
                    "max": text_lengths[-1],
                },
                "prompt_tokens": {
                    "min": prompt_lengths[0],
                    "p50": prompt_lengths[len(prompt_lengths) // 2],
                    "max": prompt_lengths[-1],
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
