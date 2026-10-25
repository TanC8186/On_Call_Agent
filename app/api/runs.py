"""运行追踪与评测结果查询接口。

读取 `logs/traces/*.jsonl` 和 `eval/results/*.json`，让前端可以展示：
- 最近一次诊断的节点/工具/决策/token 明细
- 历史评测的指标表
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

router = APIRouter()

ROOT = Path(__file__).resolve().parents[2]
TRACE_DIR = ROOT / "logs" / "traces"
RESULT_DIR = ROOT / "eval" / "results"


@router.get("/runs")
async def list_runs(limit: int = 20) -> dict:
    """列出最近的诊断运行（按文件修改时间倒序）。"""
    if not TRACE_DIR.exists():
        return {"runs": []}
    files = sorted(
        TRACE_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True
    )[:limit]

    runs = []
    for path in files:
        events = _read_jsonl(path)
        start = next((e for e in events if e.get("event") == "run_start"), {})
        end = next(
            (e for e in reversed(events) if e.get("event") in {"run_complete", "run_error"}),
            {},
        )
        plan = next((e for e in events if e.get("event") == "plan"), {})
        route = next((e for e in events if e.get("event") == "planner_route"), {})
        steps = [e for e in events if e.get("event") == "tool_step"]
        runs.append(
            {
                "run_id": path.stem,
                "session_id": start.get("session_id"),
                "started_at": start.get("ts"),
                "duration_ms": end.get("duration_ms"),
                "status": "error" if end.get("event") == "run_error" else "ok",
                "steps": len(steps),
                "plan_steps": len(plan.get("steps", [])),
                "backend": route.get("backend", "cloud"),
                "severity": route.get("severity") or "",
                "category": route.get("category") or "",
                "input_tokens": sum(e.get("input_tokens") or 0 for e in events),
                "output_tokens": sum(e.get("output_tokens") or 0 for e in events),
            }
        )
    return {"runs": runs}


@router.get("/runs/{run_id}")
async def get_run(run_id: str) -> dict:
    path = TRACE_DIR / f"{run_id}.jsonl"
    if not path.exists():
        raise HTTPException(status_code=404, detail="run not found")
    return {"run_id": run_id, "events": _read_jsonl(path)}


@router.get("/eval/results")
async def list_eval_results(limit: int = 20) -> dict:
    if not RESULT_DIR.exists():
        return {"results": []}
    files = sorted(
        RESULT_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True
    )[:limit]
    results = []
    for path in files:
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(rows, list):
            continue
        results.append(
            {
                "name": path.stem,
                "rows": rows,
                "mean_tool_f1": _mean([r.get("tool_f1") for r in rows]),
                "success_rate": _mean(
                    [1.0 if r.get("task_success") else 0.0 for r in rows]
                ),
                "total_tokens": sum(
                    (r.get("input_tokens") or 0) + (r.get("output_tokens") or 0)
                    for r in rows
                ),
            }
        )
    return {"results": results}


def _read_jsonl(path: Path) -> list[dict]:
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def _mean(values: list) -> float:
    clean = [v for v in values if isinstance(v, (int, float))]
    return round(sum(clean) / len(clean), 3) if clean else 0.0
