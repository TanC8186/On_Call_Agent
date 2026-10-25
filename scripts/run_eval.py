"""运行评测场景并给出可复算的指标。

前置条件：
    - 3 个 MCP server 已启动（或指向可用的 MCP endpoint）
    - .env 中配置了 LLM_API_KEY

用法：
    python scripts/run_eval.py --live                 # 跑全部场景
    python scripts/run_eval.py --live --scenario cpu-high-001
    python scripts/run_eval.py                        # 只做 dry-run，打印场景列表

输出：
    eval/results/<timestamp>.json
    eval/results/<timestamp>.md
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_eval import Scenario, _load_scenarios  # noqa: E402

RESULTS_DIR = ROOT / "eval" / "results"


def _read_trace(run_id: str) -> list[dict]:
    path = ROOT / "logs" / "traces" / f"{run_id}.jsonl"
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _unwrap_preview(preview: str):
    """MCP 结果可能被包成 [{"type":"text","text":"{...}"}]。"""
    try:
        outer = json.loads(preview)
    except (json.JSONDecodeError, TypeError):
        return None
    if (
        isinstance(outer, list)
        and outer
        and isinstance(outer[0], dict)
        and "text" in outer[0]
    ):
        try:
            return json.loads(outer[0]["text"])
        except (json.JSONDecodeError, TypeError):
            return None
    return outer


def _numeric_values(payload) -> list[float]:
    """只收集指标字段，排除 timestamp/id 里的数字。"""
    values: list[float] = []
    metric_keys = {
        "value", "avg", "max", "min", "p95", "total", "count",
        "current_avg", "current_max", "used_percent", "usage_percent", "free_percent",
    }
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in metric_keys and isinstance(value, (int, float)):
                values.append(float(value))
            elif key in metric_keys and isinstance(value, str):
                try:
                    values.append(float(value))
                except ValueError:
                    pass
            else:
                values.extend(_numeric_values(value))
    elif isinstance(payload, list):
        for item in payload:
            values.extend(_numeric_values(item))
    return values


_METRIC_ALIASES = {
    "cpu_usage_percent": ["cpu"],
    "memory_usage_percent": ["memory"],
    "disk_usage_percent": ["disk"],
}


def _fact_matches(fact, previews: list[str]) -> bool:
    aliases = _METRIC_ALIASES.get(fact.metric, [])
    for preview in previews:
        if fact.metric not in preview and not any(alias in preview for alias in aliases):
            continue
        payload = _unwrap_preview(preview)
        values = _numeric_values(payload) if payload is not None else []
        if not values:
            # 退化到正则，但跳过时间戳中的数字
            values = [
                float(item)
                for item in re.findall(r"(?<![\d:-])(\d+(?:\.\d+)?)(?![\d:-])", preview)
            ]
        for value in values:
            if (
                (fact.op == ">" and value > fact.value)
                or (fact.op == ">=" and value >= fact.value)
                or (fact.op == "<" and value < fact.value)
                or (fact.op == "<=" and value <= fact.value)
                or (fact.op == "==" and value == fact.value)
            ):
                return True
    return False


def _score(scenario: Scenario, events: list[dict], response: str) -> dict:
    called: list[str] = []
    previews: list[str] = []
    tokens_in = 0
    tokens_out = 0
    steps = 0
    for event in events:
        tokens_in += event.get("input_tokens") or 0
        tokens_out += event.get("output_tokens") or 0
        if event.get("event") != "tool_step":
            continue
        steps += 1
        previews.append(str(event.get("result_preview") or ""))
        for call in event.get("tools") or []:
            name = call.get("name")
            if name and name not in called:
                called.append(name)

    gold = set(scenario.gold_tools)
    hit = gold & set(called)
    recall = len(hit) / len(gold) if gold else 1.0
    precision = len(hit) / len(called) if called else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    forbidden_hit = [
        claim for claim in scenario.forbidden_claims if claim in (response or "")
    ]
    facts_matched = sum(
        1 for fact in scenario.gold_facts if _fact_matches(fact, previews)
    )
    haystack = "\n".join(previews)
    numbers = re.findall(r"\d+(?:\.\d+)?", response or "")
    grounded = sum(1 for number in numbers if number in haystack)
    groundedness = grounded / len(numbers) if numbers else 1.0
    task_success = (
        recall == 1.0
        and facts_matched == len(scenario.gold_facts)
        and not forbidden_hit
    )

    return {
        "id": scenario.id,
        "expect_no_incident": scenario.expect_no_incident,
        "called_tools": called,
        "gold_tools": sorted(gold),
        "tool_recall": round(recall, 3),
        "tool_precision": round(precision, 3),
        "tool_f1": round(f1, 3),
        "steps": steps,
        "facts_matched": facts_matched,
        "facts_total": len(scenario.gold_facts),
        "groundedness": round(groundedness, 3),
        "forbidden_hit": forbidden_hit,
        "task_success": task_success,
        "input_tokens": tokens_in,
        "output_tokens": tokens_out,
        "response_chars": len(response or ""),
    }


async def _run(scenarios: list[Scenario]) -> list[dict]:
    from app.services.aiops_service import aiops_service

    rows = []
    for scenario in scenarios:
        print(f"[run] {scenario.id}: {scenario.symptom}")
        events = [
            event
            async for event in aiops_service.execute(
                scenario.symptom, session_id=f"eval-{scenario.id}"
            )
        ]
        final = events[-1] if events else {}
        if final.get("type") != "complete":
            rows.append(
                {
                    "id": scenario.id,
                    "error": final.get("message", "no complete event"),
                    "task_success": False,
                }
            )
            continue
        run_id = final.get("run_id", "")
        trace = _read_trace(run_id)
        row = _score(scenario, trace, final.get("response", ""))
        row["run_id"] = run_id
        rows.append(row)
        print(
            f"  -> F1={row['tool_f1']:.2f} success={row['task_success']} "
            f"steps={row['steps']} tokens={row['input_tokens']}+{row['output_tokens']}"
        )
    return rows


def _write_report(rows: list[dict]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = RESULTS_DIR / f"{stamp}.json"
    json_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    md = [
        "| id | recall | precision | F1 | facts | grounded | steps | tokens | success |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        md.append(
            f"| {row['id']} | {row.get('tool_recall', 0):.2f} | "
            f"{row.get('tool_precision', 0):.2f} | {row.get('tool_f1', 0):.2f} | "
            f"{row.get('facts_matched', 0)}/{row.get('facts_total', 0)} | "
            f"{row.get('groundedness', 0):.2f} | "
            f"{row.get('steps', 0)} | "
            f"{row.get('input_tokens', 0)}+{row.get('output_tokens', 0)} | "
            f"{row.get('task_success', False)} |"
        )
    md_path = RESULTS_DIR / f"{stamp}.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    return md_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="实际调用 LLM 和 MCP")
    parser.add_argument("--scenario", help="只跑指定 scenario id")
    parser.add_argument("--limit", type=int, default=0, help="最多跑 N 条")
    args = parser.parse_args()

    scenarios = _load_scenarios()
    if args.scenario:
        scenarios = [s for s in scenarios if s.id == args.scenario]
    if args.limit:
        scenarios = scenarios[: args.limit]

    if not args.live:
        for scenario in scenarios:
            print(
                f"{scenario.id}: {scenario.title} "
                f"({len(scenario.gold_tools)} gold tools)"
            )
        print("\n(dry-run) 加 --live 才会真正调用 LLM/MCP")
        return 0

    rows = asyncio.run(_run(scenarios))
    report = _write_report(rows)
    print(f"\nreport: {report}")
    return 0 if all(row.get("task_success") for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
