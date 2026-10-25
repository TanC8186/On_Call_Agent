"""校验 eval 场景与 MCP 工具注册表、知识库文档的一致性。

用法:
    python scripts/validate_eval.py

退出码 0 表示全部通过；1 表示存在 schema 错误、未知工具或文档引用了不存在的工具。
"""

from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, ValidationError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.tools import get_current_time, retrieve_knowledge  # noqa: E402
from servers import cls_server, monitor_server, ssh_server  # noqa: E402


class GoldFact(BaseModel):
    metric: str
    op: Literal[">", ">=", "<", "<=", "=="]
    value: float


class Scenario(BaseModel):
    id: str
    title: str
    symptom: str
    service: str
    expect_no_incident: bool
    gold_tools: list[str] = Field(min_length=1)
    gold_facts: list[GoldFact] = Field(min_length=1)
    forbidden_claims: list[str] = Field(default_factory=list)
    max_steps: int = Field(default=8, ge=1, le=20)
    notes: str = ""


async def _mcp_tool_names() -> set[str]:
    names: set[str] = set()
    for server in (monitor_server.mcp, cls_server.mcp, ssh_server.mcp):
        for tool in await server._list_tools():
            names.add(tool.name)
    return names


def _load_scenarios() -> list[Scenario]:
    scenarios = []
    for path in sorted((ROOT / "eval" / "scenarios").glob("*.yaml")):
        try:
            scenarios.append(Scenario(**yaml.safe_load(path.read_text(encoding="utf-8"))))
        except (ValidationError, TypeError) as exc:
            raise SystemExit(f"[FAIL] {path.name}: {exc}") from exc
    return scenarios


def _doc_tool_refs() -> dict[str, list[str]]:
    refs: dict[str, list[str]] = {}
    pattern = re.compile(r"\*\*工具\*\*:\s*`([a-zA-Z_][a-zA-Z0-9_]*)`")
    for path in sorted((ROOT / "aiops-docs").glob("*.md")):
        found = pattern.findall(path.read_text(encoding="utf-8"))
        if found:
            refs[path.name] = found
    return refs


def main() -> int:
    scenarios = _load_scenarios()
    failures: list[str] = []

    ids = [s.id for s in scenarios]
    duplicates = {item for item in ids if ids.count(item) > 1}
    if duplicates:
        failures.append(f"重复的场景 id: {sorted(duplicates)}")

    registry = asyncio.run(_mcp_tool_names()) | {
        get_current_time.name,
        retrieve_knowledge.name,
    }

    for scenario in scenarios:
        unknown = sorted(set(scenario.gold_tools) - registry)
        if unknown:
            failures.append(f"{scenario.id}: 未知工具 {unknown}")

    for filename, tools in _doc_tool_refs().items():
        unknown = sorted(set(tools) - registry)
        if unknown:
            failures.append(f"aiops-docs/{filename}: 引用了不存在的工具 {unknown}")

    print(f"scenarios: {len(scenarios)}")
    print(f"mcp tools: {len(registry)}")
    print(f"{'id':28} {'gold_tools':>10} {'facts':>6} {'negative':>9}")
    for scenario in scenarios:
        print(
            f"{scenario.id:28} {len(scenario.gold_tools):>10} "
            f"{len(scenario.gold_facts):>6} {str(scenario.expect_no_incident):>9}"
        )

    if failures:
        print("\n[FAIL]")
        for failure in failures:
            print(" -", failure)
        return 1

    print("\n[OK] scenario schema / tool registry / doc refs all consistent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
