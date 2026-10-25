"""本地 planner 的工具/参数校验。"""

import pytest
from langchain_core.tools import tool

from app.agent.aiops.local_planner import (
    _plan_is_valid,
    repair_args,
    repair_tool_names,
    try_local_plan,
)


@tool
def list_all_services() -> dict:
    """List services."""
    return {"total": 1}


@tool
def query_cpu_metrics(service_name: str) -> dict:
    """Query CPU."""
    return {"value": 1}


def _steps(tool_name: str, args: dict) -> list[dict]:
    return [
        {
            "id": "s1",
            "kind": "tool",
            "tool": "list_all_services",
            "args": {},
            "description": "list",
        },
        {
            "id": "s2",
            "kind": "tool",
            "tool": tool_name,
            "args": args,
            "description": "query",
        },
    ]


def test_local_plan_accepts_valid_tool_and_args():
    assert _plan_is_valid(
        _steps("query_cpu_metrics", {"service_name": "agent-test-server"}),
        [list_all_services, query_cpu_metrics],
    )


def test_local_plan_rejects_unknown_tool():
    assert not _plan_is_valid(
        _steps("query_disk_metrics", {}), [list_all_services, query_cpu_metrics]
    )


def test_local_plan_rejects_unknown_args():
    assert not _plan_is_valid(
        _steps("query_cpu_metrics", {"query": "cpu"}),
        [list_all_services, query_cpu_metrics],
    )


def test_local_plan_requires_service_discovery():
    steps = _steps("query_cpu_metrics", {"service_name": "x"})[1:]
    assert not _plan_is_valid(steps, [list_all_services, query_cpu_metrics])



@pytest.mark.asyncio
async def test_local_planner_disabled_by_default():
    assert await try_local_plan("cpu high", "agent-test-server", [], "run") is None



def test_repair_maps_near_miss_tool_name():
    steps = _steps("query_service_logs", {"service_name": "x"})
    fixed = repair_tool_names(steps, [list_all_services, query_cpu_metrics])
    # 相似度低于阈值时保持原样，由 _plan_is_valid 拒绝
    assert fixed[1]["tool"] == "query_service_logs"


def test_repair_maps_close_name_when_candidate_exists():
    @tool
    def search_service_logs(service_name: str) -> dict:
        """Search service logs."""
        return {}

    steps = _steps("query_service_logs", {"service_name": "x"})
    fixed = repair_tool_names(steps, [list_all_services, search_service_logs])
    assert fixed[1]["tool"] == "search_service_logs"



@tool
def search_service_logs(service_name: str, log_level: str = "ERROR", keyword: str = "") -> dict:
    """Search service logs."""
    return {}


def test_repair_args_maps_close_parameter_name():
    steps = _steps("search_service_logs", {"service_name": "x", "level": "ERROR"})
    fixed = repair_args(steps, [list_all_services, search_service_logs])
    assert fixed[1]["args"]["log_level"] == "ERROR"
    assert "level" not in fixed[1]["args"]


def test_repair_args_fills_required_service_name():
    steps = _steps("search_service_logs", {"keyword": "timeout"})
    steps[1]["description"] = "query logs for agent-test-server"
    fixed = repair_args(steps, [list_all_services, search_service_logs])
    assert fixed[1]["args"]["service_name"] == "agent-test-server"


def test_repair_args_marks_unrepairable_step():
    steps = _steps("search_service_logs", {"keyword": "timeout"})
    steps[1]["description"] = "query logs"
    fixed = repair_args(steps, [list_all_services, search_service_logs])
    assert fixed[1].get("_invalid") is True
