"""PlanStep 解析与 typed tool 快路径。"""

import importlib
import json

import pytest
from langchain_core.tools import tool

from app.agent.aiops.plan_schema import normalize_plan

executor_module = importlib.import_module("app.agent.aiops.executor")
planner_module = importlib.import_module("app.agent.aiops.planner")


def test_normalize_plan_accepts_legacy_strings():
    steps = normalize_plan(["collect info", "generate report"])
    assert [s["kind"] for s in steps] == ["reason", "reason"]
    assert steps[0]["description"] == "collect info"


def test_normalize_plan_keeps_tool_args():
    steps = normalize_plan(
        [
            {
                "id": "s1",
                "kind": "tool",
                "tool": "query_cpu_metrics",
                "args": {"service_name": "agent-test-server"},
                "description": "query cpu",
            }
        ]
    )
    assert steps[0]["tool"] == "query_cpu_metrics"
    assert steps[0]["args"] == {"service_name": "agent-test-server"}


@pytest.mark.asyncio
async def test_executor_runs_typed_tool_without_llm(monkeypatch):
    @tool
    def fake_tool(value: str) -> dict:
        """Fake tool for testing."""
        return {"echo": value}

    async def fake_tools():
        return [fake_tool]

    def boom(*_args, **_kwargs):
        raise AssertionError("typed tool path must not construct an LLM")

    monkeypatch.setattr(executor_module, "get_mcp_tools_cached", fake_tools)
    monkeypatch.setattr(executor_module, "_create_llm", boom)

    state = {
        "run_id": "test-run",
        "input": "diagnose",
        "plan": [
            {
                "id": "s1",
                "kind": "tool",
                "tool": "fake_tool",
                "args": {"value": "hello"},
                "description": "call fake tool",
                "success_criteria": "",
            }
        ],
        "past_steps": [],
        "response": "",
    }
    result = await executor_module.executor(state)
    assert result["plan"] == []
    step, payload = result["past_steps"][0]
    assert step == "call fake tool"
    assert json.loads(payload) == {"echo": "hello"}


class _FakeLLM:
    async def ainvoke(self, _messages):
        from langchain_core.messages import AIMessage

        return AIMessage(
            content=json.dumps(
                {
                    "steps": [
                        {
                            "id": "s1",
                            "kind": "tool",
                            "tool": "list_all_services",
                            "args": {},
                            "description": "list services",
                            "success_criteria": "at least one service",
                        },
                        {
                            "id": "s2",
                            "kind": "reason",
                            "tool": "",
                            "args": {},
                            "description": "write report",
                            "success_criteria": "report uses evidence",
                        },
                    ]
                }
            ),
            usage_metadata={"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
        )


class _FakeKnowledge:
    async def ainvoke(self, _args):
        return ""


@pytest.mark.asyncio
async def test_planner_emits_typed_steps(monkeypatch):
    @tool
    def list_all_services() -> dict:
        """List services."""
        return {"total": 1}

    async def fake_tools():
        return [list_all_services]

    monkeypatch.setattr(planner_module, "get_mcp_tools_cached", fake_tools)
    monkeypatch.setattr(planner_module, "retrieve_knowledge", _FakeKnowledge())
    monkeypatch.setattr(planner_module, "_create_llm", lambda **_kwargs: _FakeLLM())

    result = await planner_module.planner(
        {
            "run_id": "test-run",
            "input": "diagnose",
            "plan": [],
            "past_steps": [],
            "response": "",
        }
    )
    assert result["plan"][0]["tool"] == "list_all_services"
    assert result["plan"][1]["kind"] == "reason"


class _ReplanLLM:
    async def ainvoke(self, _messages):
        from langchain_core.messages import AIMessage

        return AIMessage(
            content=json.dumps(
                {
                    "action": "replan",
                    "updated_steps": [
                        {
                            "id": "s3",
                            "kind": "tool",
                            "tool": "search_log",
                            "args": {"keyword": "ERROR"},
                            "description": "retry log search with wider window",
                            "success_criteria": "non-empty logs",
                        }
                    ],
                    "reason": "log query returned no rows",
                    "evidence_gap": "empty log window",
                }
            ),
            usage_metadata={"input_tokens": 5, "output_tokens": 6, "total_tokens": 11},
        )


@pytest.mark.asyncio
async def test_replanner_rewrites_plan_when_verifier_fails(monkeypatch):
    replanner_module = importlib.import_module("app.agent.aiops.replanner")

    async def fake_tools():
        return []

    monkeypatch.setattr(replanner_module, "get_mcp_tools_cached", fake_tools)
    monkeypatch.setattr(replanner_module, "_create_llm", lambda **_kwargs: _ReplanLLM())

    state = {
        "run_id": "test-run",
        "input": "diagnose",
        "plan": [
            {
                "id": "s2",
                "kind": "tool",
                "tool": "search_log",
                "args": {"keyword": "ERROR"},
                "description": "search logs",
                "success_criteria": "non-empty logs",
            }
        ],
        "past_steps": [("query logs", "工具执行失败: timeout")],
        "needs_replan": True,
        "replan_count": 0,
        "response": "",
    }
    result = await replanner_module.replanner(state)
    assert result["replan_count"] == 1
    assert result["plan"][0]["id"] == "s3"
    assert result["needs_replan"] is False


def test_verifier_detects_unavailable_and_failure():
    verifier_module = importlib.import_module("app.agent.aiops.verifier")
    assert verifier_module._is_usable('{"ok": false, "source": "unavailable"}') is False
    assert verifier_module._is_usable("工具执行失败: timeout") is False
    assert verifier_module._is_usable("(empty)") is False
    assert verifier_module._is_usable('{"ok": true, "source": "tencent_cloud"}') is True


@pytest.mark.asyncio
async def test_typed_tool_result_preserves_provenance(monkeypatch):
    @tool
    def big_tool() -> dict:
        """Return a realistically large tool payload."""
        return {
            "ok": True,
            "source": "tencent_cloud",
            "data_points": [{"value": i} for i in range(400)],
        }

    async def fake_tools():
        return [big_tool]

    monkeypatch.setattr(executor_module, "get_mcp_tools_cached", fake_tools)
    monkeypatch.setattr(
        executor_module, "_create_llm", lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("must not call LLM")
        )
    )
    state = {
        "run_id": "test-run",
        "input": "diagnose",
        "plan": [
            {
                "id": "s1",
                "kind": "tool",
                "tool": "big_tool",
                "args": {},
                "description": "big query",
                "success_criteria": "",
            }
        ],
        "past_steps": [],
        "response": "",
    }
    result = await executor_module.executor(state)
    _, payload = result["past_steps"][0]
    assert len(payload) > 4000
    assert '"source": "tencent_cloud"' in payload
