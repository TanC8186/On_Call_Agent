"""Planner 输出的结构化计划。

把自由文本步骤升级为可校验的 PlanStep：
- tool/args 在进入 Executor 前就能对着 MCP registry 校验；
- tool step 直接执行，不再让 LLM 做一次“把自然语言翻译成工具调用”；
- reason step 保留给“综合/总结”这类不需要工具的动作。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError


class PlanStep(BaseModel):
    id: str = Field(default="", description="步骤 ID，如 s1")
    kind: Literal["tool", "reason"] = "tool"
    tool: str = Field(default="", description="MCP 工具名；reason 步骤为空")
    args: dict[str, Any] = Field(default_factory=dict)
    description: str = Field(min_length=1, description="人类可读的步骤描述")
    success_criteria: str = ""


class Plan(BaseModel):
    steps: list[PlanStep] = Field(min_length=1)


def normalize_plan(raw_steps: list[Any]) -> list[dict[str, Any]]:
    """把 LLM 输出（可能是字符串或 dict）统一成 PlanStep dict 列表。"""
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_steps, 1):
        if isinstance(raw, str):
            normalized.append(
                {
                    "id": f"s{index}",
                    "kind": "reason",
                    "tool": "",
                    "args": {},
                    "description": raw,
                    "success_criteria": "",
                }
            )
            continue
        if not isinstance(raw, dict):
            continue
        payload = {**raw, "id": raw.get("id") or f"s{index}"}
        try:
            step = PlanStep.model_validate(payload)
        except ValidationError:
            # 格式不对时降级为 reason 步骤，至少保留描述
            description = str(raw.get("description") or raw.get("step") or "").strip()
            if not description:
                continue
            step = PlanStep(
                id=f"s{index}", kind="reason", tool="", args={}, description=description
            )
        normalized.append(step.model_dump())
    return normalized
