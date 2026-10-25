"""给现有数据生成"线索在前"的语序变体，修掉 severity 的句式偏置。

## 为什么

`training/probe_order.py` 的受控探针证明：同一句话只把分句顺序换一下，
fp16 的 severity 从 0.966 掉到 0.534。根因是 train 里 90 条"数字 + 线索"样本
**全部**是数字在前，模型学到的是"先看第一个数字"这个位置习惯。

## 怎么做

不重新请教师生成 —— 语序不影响 plan（要调哪些工具取决于症状内容，不取决于顺序），
所以可以直接复用原行的 assistant 答案。这既省掉几百次 API 调用，也保证
"唯一变化的就是题干语序"。

对每条可逆样本：
1. 按中文 `，；` / 英文 `; ` 切成分句
2. 找到含线索词（OOM / 超时 / 投诉 / 失败 / 不可用 / 低峰 / 批量任务 / 正常）的分句
3. 把它提到最前面，其余保持原序
4. 三重校验：`severity_of` 不变、`category_of` 不变、plan 仍通过工具目录校验

任何一环不过就丢弃这条变体。

## 生成到哪个 split

只对 **train / val** 生成。test 不动 —— 那里要留作实验台：
`probe_order.py` 会用 test 的 memory-2 / memory-4 造 signal-first 版本，
正好检验"在 train 的另一批模板上学到的语序不变性，能不能迁移到 test 的模板"。

用法::

    .venv-train/Scripts/python.exe training/order_variants.py            # 只报告
    .venv-train/Scripts/python.exe training/order_variants.py --write    # 追加进 train/val
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.labels import category_of, severity_of  # noqa: E402
from training.schema import TriageOutput, validate_against_catalog  # noqa: E402

DATA_DIR = ROOT / "training" / "data"
CATALOG = json.loads((DATA_DIR / "tool_catalog.json").read_text(encoding="utf-8"))

# 决定 severity 分支的线索词：含这些词的分句要提到最前
SIGNALS = (
    "OOM", "oom", "超时", "投诉", "失败", "不可用", "低峰", "批量任务",
    "正常", "无异常", "未见", "未出现", "无告警",
)

# 分句分隔符：中文逗号/分号 + 英文分号。句末的句号单独处理
SPLIT_RE = re.compile(r"(?<=[，；;])")


def split_symptom(text: str) -> tuple[str, str, str]:
    """拆成 (前缀, 告警文本, 后缀)。"""
    marker = "告警文本:"
    if marker not in text:
        return "", text, ""
    head, rest = text.split(marker, 1)
    if "\n" in rest:
        symptom, tail = rest.split("\n", 1)
        return f"{head}{marker}", symptom.strip(), "\n" + tail
    return f"{head}{marker}", rest.strip(), ""


def clauses(text: str) -> list[str]:
    body = text.strip()
    trailing = ""
    if body.endswith(("。", ".")):
        trailing = body[-1]
        body = body[:-1]
    parts = [part for part in SPLIT_RE.split(body) if part.strip()]
    return [part.strip() for part in parts]


def reorder_signal_first(text: str) -> str | None:
    """把含线索词的分句提到最前；已经是线索在前、或只有一个分句、或无线索的返回 None。"""
    parts = clauses(text)
    if len(parts) < 2:
        return None
    signal_index = next(
        (index for index, part in enumerate(parts) if any(word in part for word in SIGNALS)),
        None,
    )
    if signal_index is None or signal_index == 0:
        return None
    reordered = [parts[signal_index]] + [part for i, part in enumerate(parts) if i != signal_index]
    trailing = "。" if text.strip().endswith("。") else ("." if text.strip().endswith(".") else "")
    # 分句间统一用中文逗号，英文原句保留分号
    if re.search(r"[一-鿿]", text):
        joined = "，".join(part.rstrip("，；;") for part in reordered)
    else:
        joined = "; ".join(part.rstrip("，；;") for part in reordered)
    return joined + trailing


def make_variant(row: dict) -> tuple[dict | None, str]:
    """返回 (新行, 丢弃原因)。新行为 None 时原因非空。"""
    gold = json.loads(row["messages"][-1]["content"])
    prefix, symptom, suffix = split_symptom(row["messages"][1]["content"])
    if not symptom:
        return None, "no-symptom"

    reordered = reorder_signal_first(symptom)
    if not reordered:
        return None, "not-reorderable"

    # 校验一：结论不能变
    if severity_of(reordered, gold["category"]) != gold["severity"]:
        return None, "severity-changed"
    # 校验二：类别不能变
    rule_category = category_of(reordered)
    if rule_category not in ("unknown", gold["category"]):
        return None, "category-changed"
    # 校验三：答案仍然通过工具目录校验
    try:
        output = TriageOutput.model_validate(gold)
    except Exception:  # noqa: BLE001
        return None, "gold-invalid"
    if validate_against_catalog(output, CATALOG):
        return None, "gold-fails-catalog"

    new_row = json.loads(json.dumps(row))
    new_row["messages"][1]["content"] = f"{prefix} {reordered}{suffix}"
    new_row["metadata"]["order_variant_of"] = row["metadata"]["template_id"]
    new_row["metadata"]["original_symptom"] = symptom
    return new_row, ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--splits", default="train,val")
    args = parser.parse_args()

    total_added = 0
    for split in args.splits.split(","):
        path = DATA_DIR / f"{split}.jsonl"
        if not path.exists():
            continue
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        existing = {row["metadata"]["template_id"] for row in rows}
        variants: list[dict] = []
        skipped: dict[str, int] = {}
        by_group: dict[str, int] = {}
        for row in rows:
            new_row, reason = make_variant(row)
            if new_row is None:
                skipped[reason] = skipped.get(reason, 0) + 1
                continue
            # 变体用独立 id，避免和原行撞车
            group = row["metadata"]["template_id"].rsplit("-", 1)[0]
            variant_id = f"{group}-r{row['metadata']['template_id'].rsplit('-', 1)[1]}"
            if variant_id in existing:
                continue
            new_row["metadata"]["template_id"] = variant_id
            existing.add(variant_id)
            variants.append(new_row)
            by_group[group] = by_group.get(group, 0) + 1

        print(f"{split}: {len(rows)} 条 -> 生成 {len(variants)} 条语序变体，丢弃 {skipped}")
        print(f"   按模板组: {dict(sorted(by_group.items()))}")
        for row in variants[:2]:
            meta = row["metadata"]
            print("   例:", meta["order_variant_of"])
            print("     原:", meta["original_symptom"])
            print("     变:", split_symptom(row["messages"][1]["content"])[1])

        if args.write and variants:
            merged = rows + variants
            path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in merged),
                encoding="utf-8",
            )
            print(f"   已写回 {path}（{len(rows)} -> {len(merged)}）")
        total_added += len(variants)

    print(f"\n合计新增 {total_added} 条" + ("" if args.write else "（未写盘，加 --write 生效）"))


if __name__ == "__main__":
    main()
