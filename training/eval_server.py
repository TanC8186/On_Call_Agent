"""对 OpenAI 兼容的本地服务跑 held-out 评测（llama.cpp / vLLM / Ollama）。

用法:
    python training/eval_server.py --base-url http://127.0.0.1:8080/v1 \
        --model qwen-ops --schema training/data/triage_strict.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.schema import TriageOutput  # noqa: E402
from training.scoring import score, summarize  # noqa: E402

DATA_DIR = ROOT / "training" / "data"
CATALOG = json.loads((DATA_DIR / "tool_catalog.json").read_text(encoding="utf-8"))


def parse_raw(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else None
    except Exception:  # noqa: BLE001
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--template",
        default="",
        help="只跑某个模板组（如 disk-2）。改标签/改规则后重跑受影响的那一组即可",
    )
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--schema", default="", help="传给 llama-server 的 JSON schema 文件")
    parser.add_argument("--out", default="training/out/eval_server.json")
    parser.add_argument(
        "--seed", type=int, default=42, help="固定采样种子（llama.cpp 在 temperature=0 时仍会抽样）"
    )
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in (DATA_DIR / f"{args.split}.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    if args.template:
        rows = [
            row
            for row in rows
            if row.get("metadata", {}).get("template_id", "").startswith(args.template)
        ]
    if args.limit:
        rows = rows[: args.limit]

    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=300)
    response_format = None
    if args.schema:
        schema = json.loads(Path(args.schema).read_text(encoding="utf-8"))
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": "triage", "schema": schema, "strict": True},
        }

    results = []
    for index, row in enumerate(rows, 1):
        messages = row["messages"]
        kwargs = {"seed": args.seed, "extra_body": {"top_k": 1}}
        if response_format:
            kwargs["response_format"] = response_format
        response = client.chat.completions.create(
            model=args.model,
            messages=messages[:-1],
            temperature=0.0,
            max_tokens=args.max_tokens,
            **kwargs,
        )
        text = response.choices[0].message.content or ""
        raw = parse_raw(text)
        predicted = None
        if raw is not None:
            try:
                predicted = TriageOutput.model_validate(raw)
            except Exception:  # noqa: BLE001
                predicted = None
        gold = TriageOutput.model_validate(json.loads(messages[-1]["content"]))
        result = score(
            predicted,
            gold,
            raw,
            CATALOG,
            default_service=row.get("metadata", {}).get("service", ""),
        )
        result["id"] = row.get("metadata", {}).get("template_id", str(index))
        # 存原始预测，之后改判分规则可以离线重算，不必重跑模型
        result["predicted_plan"] = (
            [s.model_dump() for s in predicted.plan] if predicted else None
        )
        # severity / category 也存下来：判分规则或 gold 标签一改，
        # 这两个字段就能离线复算，否则只能重跑 149 次推理（约 20 分钟）
        result["predicted_severity"] = predicted.severity if predicted else None
        result["predicted_category"] = predicted.category if predicted else None
        result["gold_severity"] = gold.severity
        result["gold_category"] = gold.category
        results.append(result)
        print(
            f"[{index}/{len(rows)}] {result['id']} schema={result['schema_valid']} "
            f"essential={result['essential_tool_recall']:.2f} "
            f"sev={result['severity_correct']} cat={result['category_correct']}"
        )

    summary = summarize(results)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
