"""
AIOps Agent 通用工具函数
"""

from typing import List


def _format_args_schema(tool) -> str:
    """把工具的 args schema 压成一行，让 Planner 知道真实参数名。

    MCP 工具经 langchain-mcp-adapters 转换后 args_schema 是 dict；
    本地 LangChain 工具是 pydantic 模型。两种都要支持。
    """
    schema = getattr(tool, "args_schema", None)
    if schema is None:
        return ""
    if isinstance(schema, dict):
        model_schema = schema
    else:
        try:
            model_schema = schema.model_json_schema()
        except AttributeError:
            return ""
    properties = model_schema.get("properties", {})
    required = set(model_schema.get("required", []))
    if not properties:
        return ""
    parts = []
    for name, spec in properties.items():
        type_name = spec.get("type")
        if not type_name and isinstance(spec.get("anyOf"), list):
            type_name = next(
                (
                    item.get("type")
                    for item in spec["anyOf"]
                    if item.get("type") and item.get("type") != "null"
                ),
                "any",
            )
        suffix = "" if name in required else "?"
        parts.append(f"{name}{suffix}: {type_name or 'any'}")
    return "(" + ", ".join(parts) + ")"


def format_tools_description(tools: List) -> str:
    """格式化工具列表为描述文本，包含参数 schema，避免 Planner 编造参数名。"""
    tool_descriptions = []
    for tool in tools:
        if hasattr(tool, "name") and hasattr(tool, "description"):
            args = _format_args_schema(tool)
            tool_descriptions.append(f"- {tool.name}{args}: {tool.description}")
    return "\n".join(tool_descriptions)
