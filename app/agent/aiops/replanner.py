"""
Replanner 节点：评估执行结果并决定下一步动作
根据已收集的数据判断是否继续执行、重新规划或生成最终诊断报告
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


def _parse_json_from_text(text: str) -> Dict[str, Any]:
    """从 LLM 输出中解析 JSON，容错处理"""
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if json_match:
        try:
            return json.loads(json_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    brace_match = re.search(r'\{[\s\S]*\}', text)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    logger.warning(f"无法解析 JSON: {text[:200]}...")
    return {}


REPLANNER_SYSTEM_TEMPLATE = dedent("""
    作为一个重新规划专家，你需要根据已执行的步骤决定下一步行动。

    可用工具列表（用于制定计划时参考）：

    {tools_description}

    注意：你的职责是制定或调整计划，实际的工具调用由 Executor 负责执行。

    你只有两个选择：

    **1. 'continue' - 继续执行剩余计划** 【默认选择】
       - 只要剩余步骤中包含任何工具调用（query_*, search_*, get_*, list_*），就必须 continue
       - 不要提前判断"信息够了"，让 Executor 把工具都调完

    **2. 'respond' - 生成最终响应** 【仅当以下条件全部满足】
       - 剩余步骤数为 0（所有计划步骤已全部执行完毕）
       - 或者已执行步骤 >= 10（强制终止）
       - 或者剩余步骤全部是"综合"、"总结"、"生成报告"等不涉及工具调用的步骤

    关键规则：
    - 如果剩余步骤中还有 query_ / search_ / get_ / list_ 开头的工具调用 → 必须 continue
    - **禁止 replan**，原计划由 Planner 制定，你只负责按计划执行到底

    你必须以严格的 JSON 格式输出：
    - 如果选择 respond: {"action": "respond", "new_steps": []}
    - 如果选择 continue: {"action": "continue", "new_steps": []}

    只输出 JSON，不要包含其他解释文字。
""").strip()

RESPONSE_SYSTEM_TEMPLATE = dedent("""
    根据原始任务和已执行步骤的结果，生成一个全面的最终响应。

    响应要求：
    - 清晰、结构化，使用 Markdown 格式
    - **必须使用执行历史中的实际数值和指标数据**，不得编造
    - 如果执行结果中 source 字段为 "tencent_cloud"，说明数据来自真实腾讯云 API
    - 如果执行结果中 source 字段为 "local_estimate"，说明数据来自本地估算
    - 如果执行结果中 source 字段为 "unsupported"，说明该指标当前服务器类型不支持
    - 如果某个步骤失败或未执行，在报告中明确标注该指标为"未获取"而非编造数据
    - **如果某指标 source 为 local_estimate，说明该指标为本地估算值**

    你必须以 JSON 格式输出：
    {"response": "你的 Markdown 格式完整响应"}

    只输出 JSON，不要包含其他解释文字。
""").strip()


def _create_llm(temperature: float = 0) -> ChatOpenAI:
    """创建 LLM 实例（OpenAI 兼容模式，支持 DeepSeek 等）"""
    return ChatOpenAI(
        model=config.llm_model,
        api_key=config.llm_api_key,
        base_url=config.llm_base_url,
        temperature=temperature,
    )


async def replanner(state: PlanExecuteState) -> Dict[str, Any]:
    """
    重新规划节点：决定是继续、调整计划还是生成最终响应

    三种决策：
    1. continue - 继续执行当前计划
    2. replan - 调整计划（替换剩余步骤）
    3. respond - 生成最终响应
    """
    logger.info("=== Replanner：重新规划 ===")

    input_text = state.get("input", "")
    plan = state.get("plan", [])
    past_steps = state.get("past_steps", [])

    logger.info(f"剩余计划步骤: {len(plan)}")
    logger.info(f"已执行步骤: {len(past_steps)}")

    # 强制限制：如果已执行步骤过多，直接生成响应
    MAX_STEPS = 8
    if len(past_steps) >= MAX_STEPS:
        logger.warning(f"已执行 {len(past_steps)} 个步骤，超过最大限制 {MAX_STEPS}，强制生成最终响应")
        llm = _create_llm(temperature=0)
        return await _generate_response(state, llm)

    # 获取可用工具列表
    try:
        local_tools = [get_current_time, retrieve_knowledge]

        mcp_client = await get_mcp_client_with_retry()
        mcp_tools = await mcp_client.get_tools()

        all_tools = local_tools + mcp_tools
        logger.info(f"可用工具数量: 本地 {len(local_tools)} + MCP {len(mcp_tools)}")

        tools_description = format_tools_description(all_tools)
    except Exception as e:
        logger.warning(f"获取工具列表失败: {e}")
        tools_description = "无法获取工具列表"

    # 创建 LLM
    llm = _create_llm(temperature=0)

    # 格式化已执行的步骤
    steps_summary = "\n".join([
        f"步骤: {step}\n结果: {result[:500]}..."
        for step, result in past_steps
    ])

    # 如果还有剩余计划，进行决策
    if plan:
        logger.info("还有剩余计划，评估下一步行动")

        try:
            # 缩短输入避免超长 prompt
            short_input = input_text[:500] + "..." if len(input_text) > 500 else input_text

            response = await llm.ainvoke([
                ("system", REPLANNER_SYSTEM_TEMPLATE.replace(
                    "{tools_description}", tools_description,
                )),
                ("user", dedent(f"""
                    原始任务: {short_input}

                    已执行的步骤:
                    {steps_summary}

                    剩余计划: {', '.join(plan)}

                    重要提示：已执行 {len(past_steps)} 个步骤，请优先考虑是否信息已足够生成响应（respond）
                """).strip()),
            ])

            response_text = response.content if hasattr(response, 'content') else str(response)
            logger.info(f"Replanner LLM 响应: {response_text[:300]}...")

            act = _parse_json_from_text(response_text)
            action = act.get("action", "continue")
            new_steps = act.get("new_steps", [])

            logger.info(f"Replanner 决策: {action}")

            if action == "respond":
                logger.info("决定生成最终响应")
                return await _generate_response(state, llm)

            elif action == "replan":
                # replan 已禁用 — LLM 可能仍输出，强制改为 continue
                logger.warning("LLM 输出 replan，但 replan 已禁用，强制改为 continue")
                return {}

            else:  # action == "continue"
                logger.info("决定继续执行当前计划")
                return {}

        except Exception as e:
            logger.error(f"重新规划失败: {e}, 继续执行剩余计划", exc_info=True)
            return {}

    else:
        # 没有剩余计划，生成最终响应
        logger.info("计划已执行完毕，生成最终响应")
        return await _generate_response(state, llm)


async def _generate_response(state: PlanExecuteState, llm: ChatOpenAI) -> Dict[str, Any]:
    """生成最终响应"""
    logger.info("生成最终响应...")

    input_text = state.get("input", "")
    past_steps = state.get("past_steps", [])

    # 格式化执行历史
    execution_history = "\n\n".join([
        f"### 步骤: {step}\n**结果:**\n{result[:800]}"
        for step, result in past_steps
    ])

    try:
        short_input = input_text[:500] + "..." if len(input_text) > 500 else input_text

        response = await llm.ainvoke([
            ("system", RESPONSE_SYSTEM_TEMPLATE),
            ("user", dedent(f"""
                原始任务: {short_input}

                执行历史:
                {execution_history}

                请基于以上信息生成全面的最终响应。
            """).strip()),
        ])

        response_text = response.content if hasattr(response, 'content') else str(response)
        logger.info(f"Response LLM 输出长度: {len(response_text)}")

        parsed = _parse_json_from_text(response_text)
        final_response = parsed.get("response", "")

        # 如果 JSON 解析失败，直接使用 LLM 原始输出
        if not final_response:
            logger.warning("无法从 JSON 中解析 response 字段，使用原始输出")
            final_response = response_text

        logger.info(f"最终响应生成完成，长度: {len(final_response)}")
        return {"response": final_response}

    except Exception as e:
        logger.error(f"生成响应失败: {e}", exc_info=True)
        fallback_response = f"""# 任务执行结果

## 原始任务
{input_text}

## 执行的步骤
{_format_simple_steps(past_steps)}

## 说明
由于系统异常，无法生成完整响应。以上是已收集的信息。
"""
        return {"response": fallback_response}


def _format_simple_steps(past_steps: list) -> str:
    """格式化步骤列表（简单版）"""
    if not past_steps:
        return "无"

    formatted = []
    for i, (step, result) in enumerate(past_steps, 1):
        result_preview = result[:200] + "..." if len(result) > 200 else result
        formatted.append(f"{i}. **{step}**\n   {result_preview}\n")

    return "\n".join(formatted)
