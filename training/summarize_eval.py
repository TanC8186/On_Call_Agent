"""把 base / LoRA 的 eval JSON 合并成一张 Markdown 表。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))["summary"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--lora", required=True)
    parser.add_argument("--out", default="training/out/compare.md")
    args = parser.parse_args()

    base = load(args.base)
    lora = load(args.lora)
    metrics = [
        ("JSON parse 率", "json_parse_rate"),
        ("Schema 合法率", "schema_valid_rate"),
        ("Tool F1", "tool_f1"),
        ("Arg exact", "arg_exact"),
        ("Severity acc", "severity_acc"),
        ("Category acc", "category_acc"),
    ]
    lines = [
        "| 指标 | base | LoRA | Δ |",
        "|---|---:|---:|---:|",
    ]
    for label, key in metrics:
        before = base.get(key, 0)
        after = lora.get(key, 0)
        lines.append(f"| {label} | {before:.3f} | {after:.3f} | {after - before:+.3f} |")
    lines.append("")
    lines.append(f"test n = {base.get('n', 0)}")
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
