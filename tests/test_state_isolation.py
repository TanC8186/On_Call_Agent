"""回归测试：同一 thread_id 的多次运行不能累积 past_steps。"""

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from app.agent.aiops.state import PlanExecuteState


def _build_graph():
    def executor(state):
        return {
            "plan": state["plan"][1:],
            "past_steps": state.get("past_steps", []) + [("step", "ok")],
        }

    workflow = StateGraph(PlanExecuteState)
    workflow.add_node("executor", executor)
    workflow.set_entry_point("executor")
    workflow.add_edge("executor", END)
    return workflow.compile(checkpointer=MemorySaver())


def _initial(run_id: str):
    return {
        "run_id": run_id,
        "input": "diagnose",
        "plan": ["step"],
        "past_steps": [],
        "response": "",
    }


def test_past_steps_are_reset_per_run():
    graph = _build_graph()
    config = {"configurable": {"thread_id": "same-session"}}

    for index in range(3):
        graph.invoke(_initial(f"run-{index}"), config)
        state = graph.get_state(config).values
        assert len(state["past_steps"]) == 1, (
            f"第 {index + 1} 次运行累积了 {len(state['past_steps'])} 条历史"
        )
        assert state["run_id"] == f"run-{index}"


@pytest.mark.asyncio
async def test_aiops_service_isolates_runs(monkeypatch):
    """用假节点验证 AIOpsService 的同 session 多次执行互不污染。"""
    import app.services.aiops_service as service_module

    async def fake_planner(state):
        return {"plan": ["step-1"]}

    async def fake_executor(state):
        return {
            "plan": [],
            "past_steps": state.get("past_steps", []) + [("step-1", "done")],
        }

    async def fake_replanner(state):
        return {"response": f"done:{state['run_id']}"}

    monkeypatch.setattr(service_module, "planner", fake_planner)
    monkeypatch.setattr(service_module, "executor", fake_executor)
    monkeypatch.setattr(service_module, "replanner", fake_replanner)

    service = service_module.AIOpsService()
    for _ in range(3):
        events = [event async for event in service.execute("diagnose", "session-1")]
        assert events[-1]["type"] == "complete"
        state = service.graph.get_state(
            {"configurable": {"thread_id": "session-1"}}
        ).values
        assert len(state["past_steps"]) == 1
