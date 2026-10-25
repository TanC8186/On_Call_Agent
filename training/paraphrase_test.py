"""改写鲁棒性测试：把 test 集题干重写一遍，看模型是不是在背模板。

## 为什么需要它

test 只有 6 种句式、每种重复 30 次（换服务名和百分比）。模型拿 0.989 的 severity，
无法区分两件事：

1. 学到了规则（"没有用户请求超时" = 没有影响 → P3）
2. 认出了句式（见过"凌晨 X 点低峰期 ... 没有用户请求超时"这一串字符）

做法：用教师模型把**题干**换一种说法，事实（服务名、百分比、OOM/超时/低峰）一个不变。
然后同一份权重、同一套判分再跑一遍。

- 分数基本不掉 → 学到的是规则，0.989 这个数可以讲
- 分数明显掉 → 认的是模板，0.989 只能当回归基线

## 改写怎么保证语义没变

不是靠人工看，靠规则本身：

    severity_of(改写后文本, category) == 原始 gold severity
    category_of(改写后文本) == 原始 category

不满足就丢掉这次改写、重试。这样"改写"永远不改变正确答案，
测的纯粹是**换一种说法还能不能判对**。

另外拒绝两种退化的改写：和原文几乎一样（difflib ratio > 0.92），
或长度不到原文 60%（信息被删掉了）。

用法::

    .venv/Scripts/python.exe training/paraphrase_test.py --n 178 --concurrency 6
    .venv/Scripts/python.exe training/paraphrase_test.py --dry-run        # 只看 3 条样例
"""

from __future__ import annotations

import argparse
import asyncio
import difflib
import json
import re
import sys
from pathlib import Path

from anthropic import AsyncAnthropic

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import config  # noqa: E402
from training.labels import category_of, severity_of  # noqa: E402

DATA_DIR = ROOT / "training" / "data"

# 改写指令轮流用。单一指令有两个坑，都是实测出来的：
#   1. 网关内容过滤按词命中，"保留 OOM / 超时 / 低峰这类关键线索"这种
#      带术语清单的 prompt 会被判 400 sensitive_words_detected；
#   2. 同一条指令对同一句式容易产出高度相似的文本（similarity 0.85+），
#      测不出"换说法还能不能判对"。
PROMPTS = [
    "把它写成工单摘要的口气，一句话，数字不能变：\n{text}",
    "把这句话改成更口语的说法，数字不能变：\n{text}",
    "用另一种句式重写这句话，数字不变：\n{text}",
]

# 这些事实必须原样出现在改写里，否则改写改变了题意
NUM_RE = re.compile(r"\d+(?:\.\d+)?%?")
SERVICE_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)+")


def facts_preserved(original: str, rewritten: str) -> bool:
    """数字与服务名必须全部保留。"""
    orig_numbers = set(NUM_RE.findall(original))
    new_numbers = set(NUM_RE.findall(rewritten))
    if not orig_numbers <= new_numbers:
        return False
    orig_services = set(SERVICE_RE.findall(original))
    new_services = set(SERVICE_RE.findall(rewritten))
    return orig_services <= new_services


def split_message(content: str) -> tuple[str, str, str]:
    """把 user 消息拆成 (前缀, 告警文本, 后缀)，改写只动中间那段。"""
    marker = "告警文本:"
    if marker not in content:
        return "", content, ""
    head, rest = content.split(marker, 1)
    if "\n" in rest:
        symptom, tail = rest.split("\n", 1)
        return f"{head}{marker}", symptom.strip(), "\n" + tail
    return f"{head}{marker}", rest.strip(), ""


async def paraphrase_one(
    client: AsyncAnthropic,
    semaphore: asyncio.Semaphore,
    row: dict,
    attempts: int = 4,
) -> dict | None:
    gold = json.loads(row["messages"][-1]["content"])
    category = gold["category"]
    severity = gold["severity"]
    prefix, symptom, suffix = split_message(row["messages"][1]["content"])
    if not symptom:
        return None

    # 每条样本从不同指令起手，避免整批都是同一种改写风格
    offset = sum(ord(ch) for ch in row["metadata"]["template_id"]) % len(PROMPTS)
    for attempt in range(attempts):
        instruction = PROMPTS[(offset + attempt) % len(PROMPTS)]
        async with semaphore:
            try:
                response = await client.messages.create(
                    model=config.llm_model,
                    max_tokens=400,
                    messages=[
                        {"role": "user", "content": instruction.format(text=symptom)}
                    ],
                    extra_body={"temperature": 0.5 if attempt % 2 == 0 else 0.8},
                )
                text = "".join(
                    block.text for block in response.content if block.type == "text"
                ).strip()
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] {row['metadata']['template_id']} attempt {attempt + 1}: {exc}")
                continue

        text = text.strip().strip('"').strip("「」").strip()
        if not text or "\n" in text and len(text.splitlines()) > 2:
            continue
        # 退化检查：改写不能和原文几乎一样，也不能砍掉一大半
        ratio = difflib.SequenceMatcher(None, symptom, text).ratio()
        if ratio > 0.90 or len(text) < 0.6 * len(symptom):
            continue
        # 数字和服务名必须原样保留，否则改写改变了题意
        if not facts_preserved(symptom, text):
            continue
        # 语义检查：改写后必须仍然落在同一条规则上
        if severity_of(text, category) != severity:
            continue
        rule_category = category_of(text)
        if rule_category != "unknown" and rule_category != category:
            continue

        new_row = json.loads(json.dumps(row))  # deep copy
        new_row["messages"][1]["content"] = f"{prefix} {text}{suffix}"
        new_row["metadata"]["paraphrase_of"] = row["metadata"]["template_id"]
        new_row["metadata"]["original_symptom"] = symptom
        new_row["metadata"]["similarity"] = round(ratio, 3)
        return new_row

    return None


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=0, help="只处理前 N 条")
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", default="")
    parser.add_argument("--dry-run", action="store_true", help="只跑 3 条并打印")
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in (DATA_DIR / f"{args.split}.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.dry_run:
        rows = rows[:3]
    elif args.n:
        rows = rows[: args.n]

    client = AsyncAnthropic(
        api_key=config.llm_api_key, base_url=config.llm_base_url.rstrip("/")
    )
    semaphore = asyncio.Semaphore(args.concurrency)
    results = await asyncio.gather(
        *[paraphrase_one(client, semaphore, row) for row in rows]
    )
    kept = [row for row in results if row]
    print(f"改写成功 {len(kept)}/{len(rows)}")

    if args.dry_run:
        for row in kept:
            meta = row["metadata"]
            print("\n---", meta["template_id"], f"(similarity {meta['similarity']})")
            print("原文:", meta["original_symptom"])
            print("改写:", split_message(row["messages"][1]["content"])[1])
        return

    out_path = Path(args.out) if args.out else DATA_DIR / f"{args.split}_paraphrased.jsonl"
    out_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in kept),
        encoding="utf-8",
    )
    sims = sorted(row["metadata"]["similarity"] for row in kept)
    print(
        json.dumps(
            {
                "out": str(out_path),
                "n": len(kept),
                "similarity_p50": sims[len(sims) // 2],
                "similarity_min": sims[0],
                "similarity_max": sims[-1],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
