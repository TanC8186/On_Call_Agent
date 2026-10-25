"""
通用 Plan-Execute 状态定义

past_steps 使用普通 list（而不是 operator.add reducer）：
每次 run 在初始状态里显式传入空列表，LangGraph 会用新值覆盖旧值，
避免同一 thread_id 下多次诊断把历史步骤累积进来。
"""

from typing import List, TypedDict


class PlanExecuteState(TypedDict):
    """Plan-Execute 状态"""

    # 本次运行 ID（用于 trace / 日志关联）
    run_id: str

    # 用户输入（任务描述）
    input: str

    # 执行计划（PlanStep dict 列表，见 plan_schema.py）
    plan: List[dict]

    # 本次运行已执行的步骤历史
    past_steps: List[tuple]

    # 上一步是否失败、需要 replan
    needs_replan: bool

    # 本次运行已 replan 次数（预算上限 2）
    replan_count: int

    # 本地小模型给出的分诊结果（云端 planner 时为 unknown）
    severity: str
    category: str

    # 最终响应/报告
    response: str
