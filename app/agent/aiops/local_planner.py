"""本地小模型 planner：OpenAI 兼容接口 + 严格的工具校验。

只有 schema/工具/参数全部合法时才采用本地计划，否则返回 None，
由上层回退到云端大模型——这就是“本地快路径 + 云端兜底”的路由点。
"""

from __future__ import annotations

import difflib
import json
import re
from typing import Any

from loguru import logger
from openai import AsyncOpenAI

from app.config import config
from app.observability import trace_event

from .local_prompt import build_system_prompt, catalog_from_tools, compact_catalog
from .plan_schema import normalize_plan


# 小模型常见的"意思对、名字错"。显式映射优先于模糊匹配：
# query_service_status 与 query_process_list 的字符相似度 0.63，语义却毫无关系。
TOOL_ALIASES = {
    "query_service_logs": "search_service_logs",
    "search_service_health": "get_service_info",
    "query_service_status": "get_service_info",
    "query_service_health": "get_service_info",
    "get_service_status": "get_service_info",
    "query_logs": "search_service_logs",
}

# 模糊匹配只在高相似度时启用，避免把无关工具硬套上来
FUZZY_TOOL_CUTOFF = 0.75


def repair_tool_names(steps: list[dict], tools: list) -> list[dict]:
    """把本地模型编造的工具名映射到真实工具。

    顺序：原名命中 → 别名表 → 高相似度最近邻（≥0.75）→ 保持原样（交给校验拒绝）。
    """
    valid = sorted(tool.name for tool in tools)
    if not valid:
        return steps
    repaired = []
    for step in steps:
        tool = step.get("tool") or ""
        if step.get("kind") != "tool" or not tool or tool in valid:
            repaired.append(step)
            continue
        resolved = TOOL_ALIASES.get(tool)
        if resolved not in valid:
            match = difflib.get_close_matches(
                tool, valid, n=1, cutoff=FUZZY_TOOL_CUTOFF
            )
            resolved = match[0] if match else None
        if not resolved:
            repaired.append(step)
            continue
        logger.info(f"本地 planner 工具名修正: {tool} -> {resolved}")
        repaired.append({**step, "tool": resolved})
    return repaired


def repair_args(
    steps: list[dict], tools: list, default_service: str = ""
) -> list[dict]:
    """把参数名纠正到工具 schema 里最近的合法名，并补齐必填参数。

    紧凑提示词去掉了参数签名，小模型会把 `keyword` 写成 `level`、
    把 `service_name` 漏掉。这里按 schema 做一次确定性纠错：
    1. 参数名不在 schema 里 → difflib 找最近邻（cutoff 0.6）
    2. 仍然非法 → 丢弃该参数
    3. 缺少必填参数 → 从描述里能推断 service_name 就补上，否则标记失败
    """
    by_name = {tool.name: tool for tool in tools}
    repaired = []
    for step in steps:
        if step.get("kind") != "tool":
            repaired.append(step)
            continue
        tool = by_name.get(step.get("tool", ""))
        if tool is None:
            repaired.append(step)
            continue

        schema = getattr(tool, "args_schema", None)
        if isinstance(schema, dict):
            properties = schema.get("properties", {})
            required = set(schema.get("required", []))
        elif hasattr(schema, "model_json_schema"):
            model_schema = schema.model_json_schema()
            properties = model_schema.get("properties", {})
            required = set(model_schema.get("required", []))
        else:
            properties, required = {}, set()

        if not properties:
            repaired.append(step)
            continue

        args = dict(step.get("args") or {})
        fixed: dict[str, Any] = {}
        for key, value in args.items():
            if key in properties:
                fixed[key] = value
                continue
            match = difflib.get_close_matches(key, sorted(properties), n=1, cutoff=0.6)
            if match:
                logger.info(f"本地 planner 参数名修正: {key} -> {match[0]}")
                fixed[match[0]] = value
            else:
                logger.info(f"本地 planner 丢弃非法参数: {key}")

        missing = required - set(fixed)
        if "service_name" in missing:
            guess = _service_from_step(step) or default_service
            if guess:
                fixed["service_name"] = guess
                missing.discard("service_name")
        if missing:
            logger.info(f"本地 planner 缺少必填参数，放弃该步骤: {sorted(missing)}")
            repaired.append({**step, "args": fixed, "_invalid": True})
            continue

        repaired.append({**step, "args": fixed})
    return repaired


def _service_from_step(step: dict) -> str:
    text = f"{step.get('description', '')} {json.dumps(step.get('args') or {}, ensure_ascii=False)}"
    match = re.search(r"[a-z0-9]+(?:-[a-z0-9]+)+", text)
    return match.group(0) if match else ""


def _plan_is_valid(steps: list[dict], tools: list) -> bool:
    valid_tools = {tool.name: tool for tool in tools}
    if not any(step.get("tool") == "list_all_services" for step in steps):
        return False
    for step in steps:
        if step.get("kind") != "tool" or step.get("_invalid"):
            continue
        tool = valid_tools.get(step.get("tool", ""))
        if tool is None:
            return False
        schema = getattr(tool, "args_schema", None)
        if isinstance(schema, dict):
            properties = schema.get("properties", {})
        elif hasattr(schema, "model_json_schema"):
            properties = schema.model_json_schema().get("properties", {})
        else:
            properties = {}
        allowed = set(properties)
        if allowed and set(step.get("args") or {}) - allowed:
            return False
    return True


async def try_local_plan(
    symptom: str, service: str, tools: list, run_id: str = ""
) -> dict[str, Any] | None:
    """尝试用本地模型生成计划；不可用/不合法时返回 None。"""
    if not config.local_llm_base_url:
        return None

    catalog = catalog_from_tools(tools)
    messages = [
        {
            "role": "system",
            # 必须与训练数据逐字一致：`tool_names=None`。
            # catalog 本身已经列出了全部工具名（compact_catalog 输出 "- name: desc"），
            # 再追加一遍列表会让 prompt 从 984 变成 1091 tokens —— 训练时模型从没见过
            # 那 107 个 token，属于典型的 train/serve skew。
            "content": build_system_prompt(
                compact_catalog(catalog, with_params=config.local_llm_with_params),
                tool_names=None,
            ),
        },
        {
            "role": "user",
            "content": f"服务: {service}\n告警文本: {symptom}\n请输出 JSON。",
        },
    ]

    try:
        client = AsyncOpenAI(
            base_url=config.local_llm_base_url,
            api_key="EMPTY",
            timeout=60,
        )
        response = await client.chat.completions.create(
            model=config.local_llm_model,
            messages=messages,
            temperature=0.0,
            max_tokens=1200,
            response_format={"type": "json_object"},
        )
        payload = json.loads(response.choices[0].message.content or "{}")
    except Exception as exc:  # noqa: BLE001 - 本地失败必须静默回退
        logger.warning(f"本地 planner 调用失败，回退云端: {exc}")
        trace_event(run_id, "planner_route", backend="cloud", reason="local_error")
        return None

    steps = normalize_plan(payload.get("plan", []))
    steps = repair_tool_names(steps, tools)
    steps = repair_args(steps, tools, service)
    if not steps or not _plan_is_valid(steps, tools):
        logger.info("本地 planner 输出未通过工具校验，回退云端")
        trace_event(
            run_id,
            "planner_route",
            backend="cloud",
            reason="invalid_plan",
            severity=payload.get("severity"),
            category=payload.get("category"),
        )
        return None

    logger.info(f"本地 planner 命中: {len(steps)} 步")
    trace_event(
        run_id,
        "planner_route",
        backend="local",
        steps=len(steps),
        severity=payload.get("severity"),
        category=payload.get("category"),
    )
    return {
        "plan": steps,
        "severity": payload.get("severity"),
        "category": payload.get("category"),
    }
