"""
Planner 节点：制定执行计划
结合 RAG 知识库检索历史经验，基于可用 MCP 工具生成分步执行计划
"""

import difflib
import json
import re
from textwrap import dedent
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger

from app.core.llm_factory import create_chat_model
from app.observability import trace_event
from app.tools import get_current_time, retrieve_knowledge
from app.agent.mcp_client import get_mcp_tools_cached
from .local_planner import try_local_plan
from .plan_schema import normalize_plan
from .state import PlanExecuteState
from .utils import format_tools_description


PLANNER_SYSTEM_TEMPLATE = dedent("""
    作为一个专家级别的规划者，你需要将复杂的任务分解为可执行的步骤。

    可用工具列表（用于制定计划时参考）：

    {tools_description}

    注意：你的职责是制定计划，实际的工具调用由 Executor 负责执行。

    {experience_context}

    对于给定的运维诊断任务，请创建一个逐步的执行计划。计划规则：
    - **每个步骤只调用一个工具**，不要把多个工具合并到一步
    - **工具名和参数名必须来自上面列出的真实 schema**；不确定服务名时先调用 list_all_services
    - 步骤数量由任务决定，不要为了凑步数重复调用已有结果的工具
    - 先发现服务，再按需查询指标和日志；只查询与当前症状相关的指标
    - 步骤描述要具体、可操作
    - **如果有相关经验文档，请参考其中的方法和步骤制定计划**

    你必须以严格的 JSON 格式输出，格式如下：
    {"steps": [
        {"id": "s1", "kind": "tool", "tool": "list_all_services", "args": {}, "description": "列出服务与实例映射", "success_criteria": "拿到至少一个服务名"},
        {"id": "s2", "kind": "reason", "tool": "", "args": {}, "description": "综合信息生成诊断报告", "success_criteria": "报告引用已获取的指标"}
    ]}

    规则：
    - kind="tool" 的步骤必须填写真实 tool 名和 args（参数名来自上面的 schema）
    - kind="reason" 的步骤不需要工具，tool 留空
    - 不确定服务名时，先安排一个 list_all_services 的 tool 步骤
    - 不要输出工具 schema 之外的参数名

    注意：只输出 JSON，不要包含其他解释文字。
""").strip()


def _parse_json_from_text(text: str) -> Dict[str, Any]:
    """从 LLM 输出中解析 JSON，容错处理"""
    # 尝试直接解析
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # 尝试从 markdown 代码块中提取
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if json_match:
        try:
            return json.loads(json_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 尝试匹配 JSON 花括号内容
    brace_match = re.search(r'\{[\s\S]*\}', text)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    # 回退：尝试按行解析为步骤
    logger.warning("无法解析 JSON，回退到逐行解析")
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    steps = []
    for line in lines:
        # 移除非步骤行（如 markdown 格式等）
        line = re.sub(r'^[\d]+[\.\)]\s*', '', line)
        line = line.strip('"\' ,[]')
        if line and len(line) > 3:
            steps.append(line)
    return {"steps": steps} if steps else {"steps": ["收集相关信息", "分析数据", "生成报告"]}


def _guess_service(text: str) -> str:
    try:
        import json
        from pathlib import Path

        services = json.loads(
            (Path(__file__).resolve().parents[3] / "services.json").read_text(
                encoding="utf-8"
            )
        ).get("services", [])
        for item in services:
            name = item.get("service_name", "")
            if name and name in text:
                return name
    except Exception:  # noqa: BLE001
        pass
    return "unknown"


def _validate_plan_tools(plan_steps: list[dict], valid_tools: set[str]) -> list[dict]:
    """把 Planner 编造的工具名映射到最接近的真实工具；映射不了就降级为 reason。"""
    repaired = []
    for step in plan_steps:
        tool = step.get("tool") or ""
        if step.get("kind") == "tool" and tool and tool not in valid_tools:
            match = difflib.get_close_matches(tool, sorted(valid_tools), n=1, cutoff=0.55)
            if match:
                logger.warning(f"Planner 工具名不存在: {tool} -> 修正为 {match[0]}")
                step = {**step, "tool": match[0]}
            else:
                logger.warning(f"Planner 工具名不存在且无法匹配: {tool}，降级为 reason")
                step = {
                    **step,
                    "kind": "reason",
                    "tool": "",
                    "args": {},
                    "description": f"[未知工具 {tool}] {step.get('description', '')}",
                }
        repaired.append(step)
    return repaired


def _create_llm(temperature: float = 0):
    """统一从工厂取模型，provider 由 .env 决定。"""
    return create_chat_model(temperature=temperature, streaming=False)


async def planner(state: PlanExecuteState) -> Dict[str, Any]:
    """
    规划节点：根据用户输入生成执行计划

    流程：
    1. 先查询内部文档，获取相关经验和最佳实践
    2. 基于经验文档和可用工具制定执行计划
    """
    logger.info("=== Planner：制定执行计划 ===")

    input_text = state.get("input", "")
    logger.info(f"用户输入: {input_text}")

    try:
        # 步骤1: 查询内部文档获取相关经验
        logger.info("查询内部文档，寻找相关经验...")
        experience_docs = ""
        try:
            context_str = await retrieve_knowledge.ainvoke({"query": input_text})
            if context_str and context_str.strip():
                experience_docs = context_str
                logger.info(f"找到相关经验文档，长度: {len(experience_docs)}")
            else:
                logger.info("未找到相关经验文档")
        except Exception as e:
            logger.warning(f"查询内部文档失败: {e}")

        # 步骤2: 获取可用工具列表
        local_tools = [get_current_time, retrieve_knowledge]

        # 获取 MCP 工具
        mcp_tools = await get_mcp_tools_cached()

        # 合并所有工具
        all_tools = local_tools + mcp_tools
        logger.info(f"可用工具数量: 本地 {len(local_tools)} + MCP {len(mcp_tools)}")

        # 格式化工具描述
        tools_description = format_tools_description(all_tools)

        # 本地小模型快路径：只接受通过工具/参数校验的计划
        local_result = await try_local_plan(
            input_text,
            _guess_service(input_text),
            all_tools,
            state.get("run_id", ""),
        )
        if local_result:
            logger.info("采用本地小模型计划，跳过云端 Planner")
            return {
                "plan": local_result["plan"],
                "severity": local_result.get("severity") or "",
                "category": local_result.get("category") or "",
            }

        # 步骤3: 格式化经验文档上下文
        if experience_docs:
            experience_context = dedent(f"""
                ## 相关经验文档

                以下是从知识库中检索到的相关经验和最佳实践，请参考这些经验制定执行计划：

                {experience_docs}

                ---
            """).strip()
        else:
            experience_context = ""

        # 步骤4: 创建 LLM 并生成计划（不使用 with_structured_output，DeepSeek 不支持）
        llm = _create_llm(temperature=0)

        system_msg = (PLANNER_SYSTEM_TEMPLATE
            .replace("{tools_description}", tools_description)
            .replace("{experience_context}", experience_context))

        # 不要把任务描述截断：报告模板和"严禁编造"等约束都在后半段，
        # 截断会让模型看不到输出契约。
        response = await llm.ainvoke([
            ("system", system_msg),
            ("user", f"请为以下任务制定执行计划：\n\n{input_text}"),
        ])

        # 解析 JSON 输出
        response_text = response.content if hasattr(response, 'content') else str(response)
        logger.info(f"Planner LLM 响应长度: {len(response_text)}")

        parsed = _parse_json_from_text(response_text)
        plan_steps = normalize_plan(parsed.get("steps", []))

        if not plan_steps:
            logger.warning("计划为空，使用默认计划")
            plan_steps = normalize_plan(["收集相关信息", "分析数据", "生成报告"])

        plan_steps = _validate_plan_tools(plan_steps, {t.name for t in all_tools if hasattr(t, "name")})

        logger.info(f"计划已生成，共 {len(plan_steps)} 个步骤")
        for i, step in enumerate(plan_steps, 1):
            logger.info(f"  步骤{i}: {step}")

        usage = getattr(response, "usage_metadata", None) or {}
        trace_event(
            state.get("run_id", ""),
            "plan",
            steps=plan_steps,
            rag_hit=bool(experience_docs),
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
        )
        return {"plan": plan_steps}

    except Exception as e:
        logger.error(f"生成计划失败: {e}", exc_info=True)
        return {
            "plan": [
                "收集相关信息",
                "分析数据",
                "生成报告"
            ]
        }
