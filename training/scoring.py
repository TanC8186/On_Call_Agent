"""统一的评测判分（training 侧，与 Agent 侧口径对齐）。

## 为什么不再用"工具集合 F1"当主指标

教师模型写的计划有 4–6 个工具，学生模型（v8）按训练要求只列 3 个左右。
集合 F1 和"必需工具召回"都在惩罚**长度差异**，而不是判断力。
`plan_ok` 因此改成围绕**关键工具**定义：

- **关键工具（critical tool）**：在训练集里，该类故障的 gold 计划中
  出现率 ≥ `CRITICAL_THRESHOLD`（默认 90%）的工具。
  例如 cpu 场景 60/60 都有 `query_cpu_metrics`，它就是关键工具；
  而 `analyze_log_pattern` 只出现在 33% 的计划里，不算。
- **`plan_ok`** = 关键工具全覆盖 + 多余工具 ≤ `MAX_EXTRA_TOOLS` + 纠错后 schema 合法。

这样测的是"该查的是否查了、有没有离谱乱调"，而不是"和老师的计划长得像不像"。

## 其余指标

- `tool_f1`：旧口径，保留用于和历史数字对齐
- `schema_valid` / `schema_valid_after_repair`：区分"模型输出不规范"和"最终进不了执行器"
- `extra_tools`：预测里既不在 gold、也不是基线工具的数量
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

from training.schema import TriageOutput, validate_against_catalog

MAX_EXTRA_TOOLS = 2
CRITICAL_THRESHOLD = 0.9

# 这些工具任何计划都会调用（发现服务、取时间、查知识），不该算"多余"
BASELINE_TOOLS = {
    "list_all_services",
    "get_current_time",
    "get_current_timestamp",
    "retrieve_knowledge",
}

DATA_DIR = Path(__file__).resolve().parent / "data"


@lru_cache(maxsize=1)
def critical_tools() -> dict[str, frozenset[str]]:
    """从训练集统计每类故障的关键工具。

    用训练集而不是测试集，避免把测试标签"喂"进判分规则。
    """
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    totals: dict[str, int] = defaultdict(int)
    path = DATA_DIR / "train.jsonl"
    if not path.exists():
        return {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        payload = json.loads(record["messages"][-1]["content"])
        category = payload.get("category", "unknown")
        totals[category] += 1
        for step in payload.get("plan", []):
            if step.get("kind") == "tool" and step.get("tool"):
                counts[category][step["tool"]] += 1

    result: dict[str, frozenset[str]] = {}
    for category, total in totals.items():
        if not total:
            continue
        result[category] = frozenset(
            tool
            for tool, count in counts[category].items()
            if count / total >= CRITICAL_THRESHOLD
        )
    return result


# 小模型常见的"意思对、名字错"。显式映射比 difflib 更可靠：
# query_service_status 和 query_process_list 的字符串相似度 0.63，但语义完全无关。
TOOL_ALIASES = {
    "query_service_logs": "search_service_logs",
    "search_service_health": "get_service_info",
    "query_service_status": "get_service_info",
    "query_service_health": "get_service_info",
    "get_service_status": "get_service_info",
    "query_logs": "search_service_logs",
}

# 同名工具里可接受的最近邻下限；低于它宁可判失败，也不瞎猜
FUZZY_TOOL_CUTOFF = 0.75


def _resolve_tool(name: str, tools: dict[str, Any]) -> str:
    if name in tools:
        return name
    if name in TOOL_ALIASES and TOOL_ALIASES[name] in tools:
        return TOOL_ALIASES[name]
    import difflib

    match = difflib.get_close_matches(name, sorted(tools), n=1, cutoff=FUZZY_TOOL_CUTOFF)
    return match[0] if match else name


def repair_plan(
    predicted: TriageOutput,
    catalog: dict[str, Any],
    default_service: str = "",
) -> TriageOutput:
    """按工具 schema 做确定性纠错，复刻 Agent 侧的行为。

    四件事：工具名别名/最近邻映射、参数名就近映射、丢弃非法参数、
    缺 service_name 时从描述或已知服务名补。
    """
    tools = catalog.get("tools", {})
    steps = []
    for step in predicted.plan:
        if step.kind != "tool":
            steps.append(step)
            continue
        resolved = _resolve_tool(step.tool, tools)
        if resolved != step.tool:
            step = step.model_copy(update={"tool": resolved})
        spec = tools.get(step.tool)
        if spec is None:
            steps.append(step)
            continue
        params = spec.get("parameters", {})
        props = params.get("properties", {})
        required = set(params.get("required", []))
        if not props:
            steps.append(step)
            continue

        fixed = {k: v for k, v in step.args.items() if k in props}
        missing = required - set(fixed)
        if "service_name" in missing:
            guess = _service_from_step(step) or default_service
            if guess:
                fixed["service_name"] = guess
                missing.discard("service_name")
        if missing:
            steps.append(step.model_copy(update={"args": fixed, "id": step.id}))
            continue
        steps.append(step.model_copy(update={"args": fixed}))
    return predicted.model_copy(update={"plan": steps})


def _service_from_step(step) -> str:
    text = f"{step.description} {json.dumps(step.args, ensure_ascii=False)}"
    match = re.search(r"[a-z0-9]+(?:-[a-z0-9]+)+", text)
    return match.group(0) if match else ""


def _empty_result(raw: dict | None) -> dict:
    return {
        "json_valid": raw is not None,
        "schema_valid": False,
        "schema_valid_after_repair": False,
        "tool_f1": 0.0,
        "required_tool_recall": 0.0,
        "essential_tool_recall": 0.0,
        "critical_tool_recall": 0.0,
        "missing_critical": [],
        "extra_tools": 0,
        "plan_ok": False,
        "arg_exact": 0.0,
        "severity_correct": False,
        "category_correct": False,
        "errors": ["unparseable"],
    }


def score(
    predicted: TriageOutput | None,
    gold: TriageOutput,
    raw: dict | None,
    catalog: dict[str, Any],
    default_service: str = "",
) -> dict:
    if predicted is None:
        return _empty_result(raw)

    errors = validate_against_catalog(predicted, catalog)
    repaired = repair_plan(predicted, catalog, default_service)
    repaired_ok = not validate_against_catalog(repaired, catalog)

    gold_tools = set(gold.tool_names())
    pred_tools = set(repaired.tool_names())
    hit = gold_tools & pred_tools
    precision = len(hit) / len(pred_tools) if pred_tools else 0.0
    recall = len(hit) / len(gold_tools) if gold_tools else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    extra = pred_tools - gold_tools - BASELINE_TOOLS

    essential = gold_tools - BASELINE_TOOLS
    essential_recall = len(essential & pred_tools) / len(essential) if essential else 1.0

    critical = critical_tools().get(gold.category, frozenset()) - BASELINE_TOOLS
    missing_critical = sorted(critical - pred_tools)
    critical_recall = (
        len(critical & pred_tools) / len(critical) if critical else 1.0
    )

    gold_by_tool = {s.tool: s.args for s in gold.plan if s.kind == "tool"}
    matches = 0
    total = 0
    for step in repaired.plan:
        if step.kind != "tool" or step.tool not in gold_by_tool:
            continue
        total += 1
        if step.args == gold_by_tool[step.tool]:
            matches += 1
    arg_exact = matches / total if total else 0.0

    return {
        "json_valid": raw is not None,
        "schema_valid": not errors,
        "schema_valid_after_repair": repaired_ok,
        "tool_f1": round(f1, 3),
        "required_tool_recall": round(recall, 3),
        "essential_tool_recall": round(essential_recall, 3),
        "critical_tool_recall": round(critical_recall, 3),
        "missing_critical": missing_critical,
        "extra_tools": len(extra),
        "plan_ok": (not missing_critical) and len(extra) <= MAX_EXTRA_TOOLS and repaired_ok,
        "arg_exact": round(arg_exact, 3),
        "severity_correct": predicted.severity == gold.severity,
        "category_correct": predicted.category == gold.category,
        "errors": errors,
    }


def summarize(results: list[dict]) -> dict:
    n = len(results) or 1
    return {
        "n": len(results),
        "json_parse_rate": round(sum(r["json_valid"] for r in results) / n, 3),
        "schema_valid_rate": round(sum(r["schema_valid"] for r in results) / n, 3),
        "schema_valid_after_repair_rate": round(
            sum(r["schema_valid_after_repair"] for r in results) / n, 3
        ),
        "tool_f1": round(sum(r["tool_f1"] for r in results) / n, 3),
        "required_tool_recall": round(
            sum(r["required_tool_recall"] for r in results) / n, 3
        ),
        "essential_tool_recall": round(
            sum(r["essential_tool_recall"] for r in results) / n, 3
        ),
        "critical_tool_recall": round(
            sum(r["critical_tool_recall"] for r in results) / n, 3
        ),
        "plan_ok_rate": round(sum(r["plan_ok"] for r in results) / n, 3),
        "mean_extra_tools": round(sum(r["extra_tools"] for r in results) / n, 2),
        "arg_exact": round(sum(r["arg_exact"] for r in results) / n, 3),
        "severity_acc": round(sum(r["severity_correct"] for r in results) / n, 3),
        "category_acc": round(sum(r["category_correct"] for r in results) / n, 3),
    }
