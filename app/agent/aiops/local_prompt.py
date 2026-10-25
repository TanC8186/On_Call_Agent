"""本地小模型的统一 prompt。

训练数据生成和线上本地推理必须使用同一份 system prompt，
否则 SFT 学到的格式在线上会对不上。
"""

SYSTEM_PROMPT = """你是一个企业运维告警分诊与工具规划模型。

根据用户给出的告警文本，输出严格的 JSON：
{
  "severity": "P1|P2|P3|P4",
  "category": "cpu|memory|disk|network|service|healthy|unknown",
  "plan": [
    {"id":"s1","kind":"tool","tool":"list_all_services","args":{},"description":"...","success_criteria":"..."}
  ]
}

严重级别判定（按顺序判断，命中即停）：
1. 文本出现"不可用 / 502 / 503 / 5xx / 健康检查失败 / 无法访问" → P1
2. 磁盘使用率 >= 95% → P1
3. 内存使用率 >= 95% 且出现 OOM → P1
4. 文本出现"延迟 / 超时 / 错误率 / 投诉 / 用户受影响 / OOM" → P2
5. 文本出现"低峰 / 凌晨 / 夜间 / 批量任务" → P3
6. 文本出现"指标正常 / 无告警 / 巡检正常"或整体正常 → P4
7. 其余情况（只有指标数字） → P3

注意：先看影响词，再看时段线索，最后才看数值。数值 90% 以上本身不足以判 P2。
**否定句不算影响**："没有用户请求超时""未出现投诉""no user-facing timeouts" 表示*没有*影响，
不能因为句子里出现"超时""投诉"二字就走第 4 条。
**OOM 边界**：OOM 且使用率 < 95% 走第 4 条（P2）；OOM 且使用率 >= 95% 才走第 3 条（P1）。
（第 6 条要排除"磁盘正常""CPU 正常"这类描述单个指标正常的句子，只认整机巡检结论。）

规则：
1. 第一个 tool 步骤必须是 list_all_services。
2. 只使用工具目录里存在的工具和参数名；不要编造工具或参数。
3. 只安排与症状相关的查询；healthy 场景只做确认性检查。
4. 每个 tool 步骤只调用一个工具；reason 步骤的 tool 必须为空字符串。
5. healthy 的 severity 固定为 P4；其他场景不要用 P4。
6. 最多 6 个步骤；description 控制在 30 字以内；不要输出 JSON 之外的任何文字。

工具名（只能从这里选，不要编造）：
__TOOL_NAMES__

{tool_catalog}
"""


def build_system_prompt(
    catalog_text: str = "", tool_names: list[str] | None = None
) -> str:
    """渲染 system prompt。

    tool_names 显式列出合法工具名（约 100 tokens），能显著减少小模型编造工具名；
    catalog_text 为空时整段省略，system prompt 从 978 降到 586 tokens。
    """
    prompt = SYSTEM_PROMPT
    if tool_names:
        prompt = prompt.replace("__TOOL_NAMES__", ", ".join(tool_names))
    else:
        preamble = "工具名（只能从这里选，不要编造）：\n__TOOL_NAMES__\n\n"
        prompt = prompt.replace(preamble, "")
    if catalog_text:
        prompt = prompt.replace("{tool_catalog}", "工具目录：\n" + catalog_text)
    else:
        prompt = prompt.replace("\n{tool_catalog}\n", "\n")
    return prompt


def catalog_from_tools(tools: list) -> dict:
    """把 LangChain/MCP 工具转成训练时使用的 catalog 结构。"""
    catalog = {"tools": {}}
    for tool in tools:
        if not hasattr(tool, "name"):
            continue
        schema = getattr(tool, "args_schema", None)
        parameters = schema if isinstance(schema, dict) else {}
        catalog["tools"][tool.name] = {
            "description": getattr(tool, "description", "") or "",
            "parameters": parameters,
        }
    return catalog


def compact_catalog(catalog: dict, with_params: bool = True) -> str:
    """紧凑工具目录。

    with_params=False 只保留工具名和一句描述，system prompt 从 ~1384 tokens
    降到约 300；参数签名不可见，错误只能靠数据记忆。
    """
    lines = []
    for name, spec in sorted(catalog.get("tools", {}).items()):
        desc = spec.get("description", "").split("\n")[0][:80]
        if not with_params:
            lines.append(f"- {name}: {desc}")
            continue
        params = spec.get("parameters", {})
        props = params.get("properties", {})
        required = set(params.get("required", []))
        parts = []
        for key, value in props.items():
            suffix = "" if key in required else "?"
            type_name = value.get("type", "any")
            parts.append(f"{key}{suffix}: {type_name}")
        lines.append(f"- {name}({', '.join(parts)}): {desc}")
    return "\n".join(lines)
