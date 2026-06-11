"""
Planner 节点：制定执行计划
基于 LangGraph 官方教程实现

注意：DeepSeek 不支持 with_structured_output，改用 JSON 输出 + 手动解析。
"""

import json
import re
from textwrap import dedent
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from loguru import logger

from app.config import config
from app.tools import get_current_time, retrieve_knowledge
from app.agent.mcp_client import get_mcp_client_with_retry
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
    - **每个工具调用步骤必须明确写出工具名和参数值**（如 service_name='deep-learning-server'）
    - **必须覆盖所有监控指标**：CPU、内存、磁盘、网络、日志，每个指标独立一个步骤
    - **至少 5 个步骤**（包括服务发现、各指标查询、日志查询、综合报告）
    - 步骤之间应该有清晰的依赖关系（先 list_all_services 发现服务，再逐个查询指标）
    - 步骤描述要具体、可操作
    - **如果有相关经验文档，请参考其中的方法和步骤制定计划**

    你必须以严格的 JSON 格式输出，格式如下：
    {"steps": ["步骤1描述", "步骤2描述", "步骤3描述"]}

    示例输出：
    {"steps": [
        "调用 list_all_services 列出所有服务及其实例映射",
        "使用 query_cpu_metrics 查询 deep-learning-server 的 CPU 使用率，参数 service_name='deep-learning-server'",
        "使用 query_memory_metrics 查询 deep-learning-server 的内存使用率，参数 service_name='deep-learning-server'",
        "使用 query_disk_metrics 查询 deep-learning-server 的磁盘使用率，参数 service_name='deep-learning-server'",
        "综合以上信息生成诊断报告"
    ]}

    注意：只输出 JSON，不要包含其他解释文字。每个步骤最多涉及一个工具。
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


def _create_llm(temperature: float = 0) -> ChatOpenAI:
    """创建 LLM 实例（OpenAI 兼容模式，支持 DeepSeek 等）"""
    return ChatOpenAI(
        model=config.llm_model,
        api_key=config.llm_api_key,
        base_url=config.llm_base_url,
        temperature=temperature,
    )


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
        mcp_client = await get_mcp_client_with_retry()
        mcp_tools = await mcp_client.get_tools()

        # 合并所有工具
        all_tools = local_tools + mcp_tools
        logger.info(f"可用工具数量: 本地 {len(local_tools)} + MCP {len(mcp_tools)}")

        # 格式化工具描述
        tools_description = format_tools_description(all_tools)

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

        # 缩短用户输入，避免超长 prompt
        short_input = input_text[:500] + "..." if len(input_text) > 500 else input_text

        system_msg = (PLANNER_SYSTEM_TEMPLATE
            .replace("{tools_description}", tools_description)
            .replace("{experience_context}", experience_context))

        response = await llm.ainvoke([
            ("system", system_msg),
            ("user", f"请为以下任务制定执行计划：\n\n{short_input}"),
        ])

        # 解析 JSON 输出
        response_text = response.content if hasattr(response, 'content') else str(response)
        logger.info(f"Planner LLM 响应长度: {len(response_text)}")

        parsed = _parse_json_from_text(response_text)
        plan_steps = parsed.get("steps", [])

        if not plan_steps:
            logger.warning("计划为空，使用默认计划")
            plan_steps = ["收集相关信息", "分析数据", "生成报告"]

        logger.info(f"计划已生成，共 {len(plan_steps)} 个步骤")
        for i, step in enumerate(plan_steps, 1):
            logger.info(f"  步骤{i}: {step}")

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
