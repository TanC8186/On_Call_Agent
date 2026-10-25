"""导出 MCP 工具目录，供数据合成与训练使用。

用法:
    python training/dump_tools.py
输出:
    training/data/tool_catalog.json
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from servers import cls_server, monitor_server, ssh_server  # noqa: E402

OUT = ROOT / "training" / "data" / "tool_catalog.json"

LOCAL_TOOLS = {
    "get_current_time": {
        "description": "获取当前时间，用于构造日志查询的时间窗。",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    "retrieve_knowledge": {
        "description": "从运维知识库检索历史经验与处置手册。",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "检索问题"}},
            "required": ["query"],
        },
    },
}


async def main() -> None:
    tools = {}
    for server in (monitor_server.mcp, cls_server.mcp, ssh_server.mcp):
        for tool in await server._list_tools():
            tools[tool.name] = {
                "description": tool.description,
                "parameters": tool.parameters,
            }
    tools.update(LOCAL_TOOLS)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps({"tools": tools}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {len(tools)} tools -> {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
