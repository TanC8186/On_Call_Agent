"""用当前 `training/labels.py` 的规则重新推导 severity，修正历史数据里的陈旧标签。

## 为什么需要它

数据是在 2026-10-23 00:06 生成的，`labels.py` 在 15:54 才改过。
于是 test 里 19 条 `disk-2`（英文模板 "Disk usage ... show write failures"）
还带着旧规则给的 P3，而现在的规则判 P2（`write failures` ∈ `_IMPACT`）。

这 19 条会让**所有**模型的 severity 一起被扣分（规则要求 P2，gold 写 P3），
而且 `RESULTS.md` 里"719/719 与规则一致"这句话对 test 已经不成立。

脚本只改 assistant 消息里的 `severity`（必要时还有 `category`），
plan 一个字都不动 —— 计划是教师生成的，与标签规则无关。

用法::

    .venv-train/Scripts/python.exe training/fix_labels.py            # 只报告
    .venv-train/Scripts/python.exe training/fix_labels.py --write    # 落盘
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.labels import category_of, severity_of  # noqa: E402

DATA_DIR = ROOT / "training" / "data"


def user_text(messages: list[dict]) -> str:
    for message in messages:
        if message.get("role") == "user":
            return message["content"]
    return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="写回文件；不带则只报告")
    parser.add_argument("--splits", default="train,val,test")
    args = parser.parse_args()

    total_fixed = 0
    for split in args.splits.split(","):
        path = DATA_DIR / f"{split}.jsonl"
        if not path.exists():
            print(f"[skip] {path} 不存在")
            continue
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        changes: list[tuple[str, str, str]] = []
        for row in rows:
            gold = json.loads(row["messages"][-1]["content"])
            text = user_text(row["messages"])
            expected_category = "healthy" if row["metadata"].get("family") == "healthy" else gold["category"]
            expected = severity_of(text, expected_category)
            if expected != gold["severity"]:
                changes.append((row["metadata"]["template_id"], gold["severity"], expected))
                gold["severity"] = expected
                row["messages"][-1]["content"] = json.dumps(gold, ensure_ascii=False)
                row["metadata"]["severity"] = expected
            # category 理论上稳定；规则算出来的和 gold 不一致时只报警不改，
            # 因为 category 依赖 family，不是纯文本推断。
            rule_category = category_of(text)
            if rule_category != gold["category"] and rule_category != "unknown":
                row["metadata"].setdefault("category_warning", rule_category)

        by_template = Counter("-".join(tid.split("-")[:2]) for tid, _, _ in changes)
        print(f"{split}: {len(rows)} 条, 需修正 {len(changes)} 条 {dict(by_template)}")
        for tid, old, new in changes[:10]:
            print(f"   {tid}: {old} -> {new}")
        if len(changes) > 10:
            print(f"   ... 其余 {len(changes) - 10} 条同类")

        if args.write and changes:
            path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            print(f"   已写回 {path}")
        total_fixed += len(changes)

    print(f"\n合计修正 {total_fixed} 条" + ("" if args.write else "（未写盘，加 --write 生效）"))


if __name__ == "__main__":
    main()
