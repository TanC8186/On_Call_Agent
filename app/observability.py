"""轻量级 per-run trace。

每次诊断写一份 `logs/traces/<run_id>.jsonl`，记录节点、工具调用、决策和
token 用量。目的不是替代 Langfuse/OTel，而是在没有额外服务的情况下，
让一次失败诊断可以被回放和归因。

设计约束：trace 失败绝不能影响主流程。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

TRACE_DIR = Path("logs/traces")


def trace_event(run_id: str, event: str, **fields: Any) -> None:
    """追加一条 trace 事件；异常静默吞掉，避免观测影响业务。"""
    if not run_id:
        return
    try:
        TRACE_DIR.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": datetime.now().isoformat(timespec="milliseconds"),
            "run_id": run_id,
            "event": event,
            **fields,
        }
        with open(TRACE_DIR / f"{run_id}.jsonl", "a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception:  # noqa: BLE001 - 观测不能中断诊断
        pass
