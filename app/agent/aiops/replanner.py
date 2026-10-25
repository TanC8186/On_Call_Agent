"""
Replanner 节点：评估执行结果并决定下一步动作
根据已收集的数据判断是否继续执行、重新规划或生成最终诊断报告
"""

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
from .plan_schema import normalize_plan
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
    作为一个评估专家，你需要根据已执行的步骤决定下一步行动。

    可用工具列表（用于制定或调整计划）：

    {tools_description}

    {replan_hint}

    你有三个选择：

    **1. 'continue' - 继续执行剩余计划** 【默认选择】
       - 剩余步骤中还有工具调用时，优先 continue

    **2. 'respond' - 生成最终响应** 【仅当】
       - 剩余步骤数为 0，或剩余步骤全部是"综合/总结/生成报告"这类 reason 步骤

    **3. 'replan' - 重写剩余计划** 【仅当上一步失败/没有拿到证据，且预算未用完】
       - updated_steps 必须是完整的 PlanStep 列表（kind/tool/args/description/success_criteria）
       - 只重写"剩余步骤"，不要重复已经成功的步骤
       - 必须针对失败原因（例如时间窗为空 → 放宽时间窗；工具不存在 → 换用真实工具名）

    你必须以严格的 JSON 格式输出：
    - respond: {"action": "respond", "updated_steps": [], "reason": "..."}
    - continue: {"action": "continue", "updated_steps": [], "reason": "..."}
    - replan: {"action": "replan", "updated_steps": [...], "reason": "...", "evidence_gap": "..."}

    只输出 JSON，不要包含其他解释文字。
""").strip()

RESPONSE_SYSTEM_TEMPLATE = dedent("""
    根据原始任务和已执行步骤的结果，生成一个全面的最终响应。

    响应要求：
    - 清晰、结构化，使用 Markdown 格式
    - **必须使用执行历史中的实际数值和指标数据**，不得编造
    - source 字段含义：
      - "tencent_cloud" / "tencent_cloud_custom"：真实腾讯云 API 数据
      - "demo"：合成演示数据，**不得据此下根因结论**，必须在报告中标注"演示数据"
      - "unavailable"：没有拿到数据；报告中写"未获取"，不得猜测数值
      - "unsupported"：当前服务器类型不支持该指标
    - 如果某个步骤失败或未执行，在报告中明确标注该指标为"未获取"而非编造数据
    - 如果任一关键指标 source 为 "demo" 或 "unavailable"，报告中不得给出确定性根因结论

    你必须以 JSON 格式输出：
    {"response": "你的 Markdown 格式完整响应"}

    只输出 JSON，不要包含其他解释文字。
""").strip()


def _create_llm(temperature: float = 0):
    """统一从工厂取模型，provider 由 .env 决定。"""
    return create_chat_model(temperature=temperature, streaming=False)


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
    needs_replan = bool(state.get("needs_replan", False))
    replan_count = int(state.get("replan_count", 0) or 0)
    replan_hint = (
        f"上一步没有产生可用证据，允许 replan（已用 {replan_count}/2 次预算）。"
        if needs_replan and replan_count < 2
        else "上一步正常；本轮禁止 replan，只能 continue 或 respond。"
    )

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

        mcp_tools = await get_mcp_tools_cached()

        all_tools = local_tools + mcp_tools
        logger.info(f"可用工具数量: 本地 {len(local_tools)} + MCP {len(mcp_tools)}")

        tools_description = format_tools_description(all_tools)
    except Exception as e:
        logger.warning(f"获取工具列表失败: {e}")
        tools_description = "无法获取工具列表"

    # 创建 LLM
    llm = _create_llm(temperature=0)

    # 格式化已执行的步骤。保留足够的工具原始输出（含 source 字段），
    # 500 字符会把一次真实指标查询的 provenance 截掉。
    steps_summary = "\n".join([
        f"步骤: {step}\n结果: {result[:4000]}{'...' if len(result) > 4000 else ''}"
        for step, result in past_steps
    ])

    # 如果还有剩余计划，进行决策
    if plan or needs_replan:
        logger.info("还有剩余计划或需要补救，评估下一步行动")

        try:
            remaining_text = "; ".join(
                step.get("description", str(step)) if isinstance(step, dict) else str(step)
                for step in plan
            )
            response = await llm.ainvoke([
                ("system", REPLANNER_SYSTEM_TEMPLATE.replace(
                    "{tools_description}", tools_description,
                ).replace("{replan_hint}", replan_hint)),
                ("user", dedent(f"""
                    原始任务: {input_text}

                    已执行的步骤:
                    {steps_summary}

                    剩余计划: {remaining_text}

                    重要提示：已执行 {len(past_steps)} 个步骤，请优先考虑是否信息已足够生成响应（respond）
                """).strip()),
            ])

            response_text = response.content if hasattr(response, 'content') else str(response)
            logger.info(f"Replanner LLM 响应: {response_text[:300]}...")

            act = _parse_json_from_text(response_text)
            action = act.get("action", "continue")
            new_steps = act.get("updated_steps", []) or act.get("new_steps", [])

            usage = getattr(response, "usage_metadata", None) or {}
            trace_event(
                state.get("run_id", ""),
                "decision",
                action=action,
                remaining_steps=len(plan),
                executed_steps=len(past_steps),
                input_tokens=usage.get("input_tokens"),
                output_tokens=usage.get("output_tokens"),
            )

            logger.info(f"Replanner 决策: {action}")

            if action == "respond":
                logger.info("决定生成最终响应")
                return await _generate_response(state, llm)

            elif action == "replan":
                normalized = normalize_plan(new_steps)
                if not needs_replan or replan_count >= 2 or not normalized:
                    logger.warning(
                        "Replanner 请求 replan，但预算/条件不满足，降级为 continue"
                    )
                    return {}
                logger.warning(
                    f"Replanner 重写剩余计划（第 {replan_count + 1}/2 次）: "
                    f"{act.get('reason', '')}"
                )
                trace_event(
                    state.get("run_id", ""),
                    "replan",
                    reason=act.get("reason", ""),
                    evidence_gap=act.get("evidence_gap", ""),
                    steps=normalized,
                    replan_count=replan_count + 1,
                )
                return {
                    "plan": normalized,
                    "needs_replan": False,
                    "replan_count": replan_count + 1,
                }

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


async def _generate_response(state: PlanExecuteState, llm) -> Dict[str, Any]:
    """生成最终响应"""
    logger.info("生成最终响应...")

    input_text = state.get("input", "")
    past_steps = state.get("past_steps", [])

    # 格式化执行历史：保留完整工具输出，避免把 source/provenance 截掉。
    execution_history = "\n\n".join([
        f"### 步骤: {step}\n**结果:**\n{result}"
        for step, result in past_steps
    ])

    try:
        response = await llm.ainvoke([
            ("system", RESPONSE_SYSTEM_TEMPLATE),
            ("user", dedent(f"""
                原始任务: {input_text}

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

        usage = getattr(response, "usage_metadata", None) or {}
        trace_event(
            state.get("run_id", ""),
            "final_response",
            response_chars=len(final_response),
            result_preview=final_response,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
        )
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
