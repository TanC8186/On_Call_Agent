"""受控探针：同一句告警，只换分句顺序，看 severity 会不会翻。

## 为什么要这个探针

改写测试（`paraphrase_rules.py`）显示 severity 从 0.989 掉到 0.865，
但改写同时改了两件事：措辞，和分句顺序。分不清是哪一个在起作用。

这个脚本只动一件事：把"数字在前、线索在后"换成"线索在前、数字在后"。
内容、数字、服务名、关键词全部逐字保留。

如果模型真在按规则判断，两种语序必须给出同一个答案；
如果答案是两个，说明它看的是句子的形状，不是句子的意思。

用法::

    .venv-train/Scripts/python.exe training/probe_order.py \
        --base-url http://127.0.0.1:8000/v1 --model qwen-ops-v9-fp16 --limit 12
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "training" / "data"

# 只取"数字 + OOM"这一类：两个分句可以无损互换，且规则对它们有明确边界
PATTERNS = [
    (re.compile(r"^(.+?) 内存使用率 (\d+)%，系统日志出现 OOM killer 记录。$"),
     "{service} 内存使用率 {pct}%，系统日志出现 OOM killer 记录。",
     "系统日志出现 OOM killer 记录，{service} 内存使用率 {pct}%。"),
    (re.compile(r"^Memory alert on (.+?): usage (\d+)% and rising; OOM events observed in system logs\.$"),
     "Memory alert on {service}: usage {pct}% and rising; OOM events observed in system logs.",
     "OOM events are observed in system logs; {service} memory usage is {pct}% and rising."),
]


def variants(symptom: str) -> tuple[str, str] | None:
    for pattern, number_first, signal_first in PATTERNS:
        match = pattern.match(symptom.strip())
        if not match:
            continue
        service, pct = match.group(1).strip(), match.group(2)
        return (
            number_first.format(service=service, pct=pct),
            signal_first.format(service=service, pct=pct),
        )
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", default="training/out/probe_order.json")
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in (DATA_DIR / "test.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    picks = []
    for row in rows:
        user = next(m["content"] for m in row["messages"] if m["role"] == "user")
        symptom = user.split("告警文本:", 1)[1].split("\n")[0].strip() if "告警文本:" in user else ""
        pair = variants(symptom)
        if pair:
            picks.append((row, pair))
    if args.limit:
        picks = picks[: args.limit]

    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=300)
    results = []
    agree = 0
    number_first_correct = 0
    signal_first_correct = 0
    for row, (number_first, signal_first) in picks:
        gold = json.loads(row["messages"][-1]["content"])
        service = row["metadata"]["service"]
        answers = []
        for text in (number_first, signal_first):
            response = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": row["messages"][0]["content"]},
                    {"role": "user", "content": f"服务: {service}\n告警文本: {text}\n请输出 JSON。"},
                ],
                temperature=0.0,
                max_tokens=600,
                seed=42,
                extra_body={"top_k": 1},
            )
            try:
                answers.append(json.loads(response.choices[0].message.content or "{}").get("severity"))
            except json.JSONDecodeError:
                answers.append(None)
        same = answers[0] == answers[1]
        agree += int(same)
        number_first_correct += int(answers[0] == gold["severity"])
        signal_first_correct += int(answers[1] == gold["severity"])
        results.append(
            {
                "id": row["metadata"]["template_id"],
                "gold": gold["severity"],
                "number_first": answers[0],
                "signal_first": answers[1],
                "agree": same,
            }
        )
        print(
            f"{row['metadata']['template_id']:12} gold={gold['severity']} "
            f"number_first={answers[0]} signal_first={answers[1]} agree={same}"
        )

    summary = {
        "n": len(results),
        "agree_rate": round(agree / len(results), 3) if results else None,
        "acc_number_first": round(number_first_correct / len(results), 3) if results else None,
        "acc_signal_first": round(signal_first_correct / len(results), 3) if results else None,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
