"""Verifier 节点：判断上一步是否产生了可用证据。

只做确定性检查，不调用 LLM：
- 结果为空或只有 "(empty)"
- 工具不存在 / 工具执行失败
- 结果明确标记 ok=false / source=unavailable
"""

from __future__ import annotations

import json
from typing import Any

from loguru import logger

from .state import PlanExecuteState


def _is_usable(result: str) -> bool:
    text = (result or "").strip()
    if not text or text == "(empty)":
        return False
    if text.startswith("工具不存在") or text.startswith("工具执行失败"):
        return False
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return True
    if isinstance(payload, dict):
        if payload.get("ok") is False:
            return False
        if payload.get("source") == "unavailable":
            return False
        if payload.get("error") and not payload.get("data_points"):
            return False
    return True


def verifier(state: PlanExecuteState) -> dict[str, Any]:
    past_steps = state.get("past_steps", [])
    if not past_steps:
        return {"needs_replan": False}

    step, result = past_steps[-1]
    usable = _is_usable(str(result))
    if not usable:
        logger.warning(f"Verifier: 步骤没有产生可用证据，将触发 replan: {step}")
    return {"needs_replan": not usable}
