"""
Executor 节点：执行单个步骤
从计划队列中弹出下一步，由 LLM 决策调用哪个 MCP 工具并执行
"""

import json
from typing import Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import ToolNode
from loguru import logger

from app.core.llm_factory import create_chat_model
from app.observability import trace_event
from app.tools import get_current_time, retrieve_knowledge
from app.agent.mcp_client import get_mcp_tools_cached
from .state import PlanExecuteState


def _create_llm(temperature: float = 0):
    """统一从工厂取模型，provider 由 .env 决定。"""
    return create_chat_model(temperature=temperature, streaming=False)


async def executor(state: PlanExecuteState) -> Dict[str, Any]:
    """
    执行节点：执行计划中的下一个步骤

    使用 LangGraph 的 ToolNode 自动处理工具调用
    """
    logger.info("=== Executor：执行步骤 ===")

    plan = state.get("plan", [])

    # 如果计划为空，不执行
    if not plan:
        logger.info("计划为空，跳过执行")
        return {}

    # 取出第一个步骤（Planner 输出应为 PlanStep dict）
    step = plan[0]
    if isinstance(step, dict):
        task_description = step.get("description") or step.get("tool") or str(step)
    else:
        step = {"kind": "reason", "tool": "", "args": {}, "description": str(step)}
        task_description = str(step["description"])
    logger.info(f"当前任务: {task_description}")

    try:
        # 获取本地工具
        local_tools = [get_current_time, retrieve_knowledge]

        # 获取 MCP 工具
        mcp_tools = await get_mcp_tools_cached()
        logger.info(f"可用工具数量: 本地 {len(local_tools)} + MCP {len(mcp_tools)}")

        # 合并所有工具
        all_tools = local_tools + mcp_tools

        # 快路径：Planner 已给出 tool + args，直接执行，不再让 LLM 翻译一次。
        if step.get("kind") == "tool" and step.get("tool"):
            tool = next((t for t in all_tools if t.name == step["tool"]), None)
            args = step.get("args") or {}
            if tool is None:
                result = f"工具不存在: {step['tool']}"
            else:
                schema = getattr(tool, "args_schema", None)
                allowed = (
                    set(schema.get("properties", {})) if isinstance(schema, dict) else set()
                )
                if allowed:
                    filtered = {key: value for key, value in args.items() if key in allowed}
                    if filtered != args:
                        logger.warning(
                            f"丢弃 Planner 编造的参数: {sorted(set(args) - allowed)}"
                        )
                    args = filtered
                try:
                    tool_result = await tool.ainvoke(args)
                    result = (
                        tool_result
                        if isinstance(tool_result, str)
                        else json.dumps(tool_result, ensure_ascii=False, default=str)
                    )
                except Exception as exc:  # noqa: BLE001 - 工具失败交给 Replanner
                    result = f"工具执行失败: {exc}"
            trace_event(
                state.get("run_id", ""),
                "tool_step",
                step=task_description,
                tools=[{"name": step["tool"], "args": args}],
                result_chars=len(result),
                result_preview=result[:2000],
                input_tokens=None,
                output_tokens=None,
            )
            logger.info(f"步骤执行完成（typed tool），结果长度: {len(result)}")
            return {
                "plan": plan[1:],
                "past_steps": state.get("past_steps", [])
                + [(task_description, result)],
            }

        # 创建 LLM（绑定工具）
        llm = _create_llm(temperature=0)
        llm_with_tools = llm.bind_tools(all_tools)

        # 创建工具节点（自动执行工具调用）
        tool_node = ToolNode(all_tools)

        # 构建消息（只包含当前步骤，避免原始任务干扰）
        messages = [
            SystemMessage(content="""你是一个运维诊断执行器，严格按照步骤指令调用工具。

工作方式：
1. 读取步骤描述，从中提取工具名和参数（如 service_name）
2. **必须调用步骤中指定的工具**，禁止调用步骤中未提到的工具
3. 如果步骤中指定了参数值，直接使用；如果未指定，根据已获取的信息推断
4. 返回工具查询的原始结果，不要省略或概括

严格禁止：
- 禁止将步骤转换为 list_all_services 调用，除非步骤明确要求
- 禁止用一个工具替代另一个工具
- 禁止编造数据
- 工具返回什么就报告什么，数据为空就说"未获取到数据"

专注于当前这一步骤，不要考虑其他步骤。"""),
            HumanMessage(content=f"请执行以下任务: {task_description}")
        ]

        # 第一步：LLM 决定是否调用工具
        llm_response = await llm_with_tools.ainvoke(messages)
        final_response = llm_response
        logger.info(f"LLM 响应类型: {type(llm_response)}")

        # 第二步：如果有工具调用，执行工具
        if hasattr(llm_response, "tool_calls") and llm_response.tool_calls:
            logger.info(f"检测到 {len(llm_response.tool_calls)} 个工具调用")

            # 使用 ToolNode 自动执行工具
            messages.append(llm_response)
            tool_messages = await tool_node.ainvoke({"messages": messages})

            # 第三步：将工具结果返回给 LLM 生成最终答案
            messages.extend(tool_messages["messages"])
            final_response = await llm_with_tools.ainvoke(messages)
            result = final_response.content if hasattr(final_response, 'content') else str(final_response)
        else:
            # 没有工具调用，直接使用 LLM 的输出
            logger.info("LLM 未调用工具，直接返回结果")
            result = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)

        usage = getattr(final_response, "usage_metadata", None) or getattr(
            llm_response, "usage_metadata", None
        )
        trace_event(
            state.get("run_id", ""),
            "tool_step",
            step=task_description,
            tools=[
                {"name": call.get("name"), "args": call.get("args")}
                for call in (getattr(llm_response, "tool_calls", None) or [])
            ],
            result_chars=len(result),
            result_preview=result[:2000],
            input_tokens=(usage or {}).get("input_tokens"),
            output_tokens=(usage or {}).get("output_tokens"),
        )
        logger.info(f"步骤执行完成，结果长度: {len(result)}")

        # 返回更新：移除已执行的步骤，追加本次运行的执行历史
        return {
            "plan": plan[1:],
            "past_steps": state.get("past_steps", []) + [(task_description, result)],
        }

    except Exception as e:
        logger.error(f"执行步骤失败: {e}", exc_info=True)
        return {
            "plan": plan[1:],
            "past_steps": state.get("past_steps", []) + [(task_description, f"执行失败: {str(e)}")],
        }
