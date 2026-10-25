"""领域小模型的数据契约。

任务：给定告警文本和工具目录，输出
    {"severity": "P1|P2|P3|P4", "category": "...", "plan": [PlanStep, ...]}

训练和推理共用同一套 pydantic 校验，避免“训练时能过、上线时格式不对”。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Severity = Literal["P1", "P2", "P3", "P4"]
Category = Literal["cpu", "memory", "disk", "network", "service", "healthy", "unknown"]


class PlanStep(BaseModel):
    id: str = Field(min_length=1)
    kind: Literal["tool", "reason"] = "tool"
    tool: str = ""
    args: dict[str, Any] = Field(default_factory=dict)
    description: str = Field(min_length=1)
    success_criteria: str = ""


class TriageOutput(BaseModel):
    severity: Severity
    category: Category
    plan: list[PlanStep] = Field(min_length=2, max_length=8)

    def tool_names(self) -> list[str]:
        return [step.tool for step in self.plan if step.kind == "tool"]


def validate_against_catalog(output: TriageOutput, catalog: dict[str, Any]) -> list[str]:
    """返回错误列表；空列表表示通过。"""
    errors: list[str] = []
    tools = catalog.get("tools", {})
    for step in output.plan:
        if step.kind == "reason":
            if step.tool:
                errors.append(f"{step.id}: reason 步骤不应有 tool")
            continue
        if step.tool not in tools:
            errors.append(f"{step.id}: 未知工具 {step.tool}")
            continue
        params = tools[step.tool].get("parameters", {}).get("properties", {})
        required = set(tools[step.tool].get("parameters", {}).get("required", []))
        unknown = set(step.args) - set(params)
        if unknown:
            errors.append(f"{step.id}: 未知参数 {sorted(unknown)}")
        missing = required - set(step.args)
        if missing:
            errors.append(f"{step.id}: 缺少必填参数 {sorted(missing)}")
    if not any(step.tool == "list_all_services" for step in output.plan):
        errors.append("plan 必须先调用 list_all_services 发现服务")
    if output.category == "healthy" and output.severity != "P4":
        errors.append("healthy 场景的 severity 必须是 P4")
    if output.category != "healthy" and output.severity == "P4":
        errors.append("非 healthy 场景不应是 P4")
    return errors
