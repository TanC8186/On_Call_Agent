"""运行追踪接口：读取 trace 与评测结果。"""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api import runs
from app.main import app


@pytest.fixture()
def trace_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(runs, "TRACE_DIR", tmp_path / "traces")
    monkeypatch.setattr(runs, "RESULT_DIR", tmp_path / "results")
    (tmp_path / "traces").mkdir(exist_ok=True)
    (tmp_path / "results").mkdir(exist_ok=True)
    return tmp_path


def _write_trace(root: Path, run_id: str, events: list[dict]) -> None:
    path = root / "traces" / f"{run_id}.jsonl"
    path.write_text(
        "\n".join(json.dumps(event, ensure_ascii=False) for event in events) + "\n",
        encoding="utf-8",
    )


def test_list_runs_summarizes_trace(trace_dir):
    _write_trace(
        trace_dir,
        "run-1",
        [
            {"event": "run_start", "ts": "2026-10-22T01:00:00", "session_id": "s1"},
            {"event": "planner_route", "backend": "local", "severity": "P2", "category": "cpu"},
            {"event": "plan", "steps": [{"id": "s1"}, {"id": "s2"}]},
            {"event": "tool_step", "step": "a", "input_tokens": 10, "output_tokens": 5},
            {"event": "tool_step", "step": "b", "input_tokens": 3, "output_tokens": 2},
            {"event": "run_complete", "duration_ms": 1234},
        ],
    )

    with TestClient(app) as client:
        response = client.get("/api/runs")

    assert response.status_code == 200
    run = response.json()["runs"][0]
    assert run["run_id"] == "run-1"
    assert run["session_id"] == "s1"
    assert run["steps"] == 2
    assert run["plan_steps"] == 2
    assert run["backend"] == "local"
    assert run["status"] == "ok"
    assert run["input_tokens"] == 13
    assert run["output_tokens"] == 7


def test_get_run_returns_events(trace_dir):
    _write_trace(trace_dir, "run-2", [{"event": "run_start", "ts": "t"}])

    with TestClient(app) as client:
        response = client.get("/api/runs/run-2")

    assert response.status_code == 200
    assert response.json()["events"][0]["event"] == "run_start"


def test_get_unknown_run_returns_404(trace_dir):
    with TestClient(app) as client:
        assert client.get("/api/runs/nope").status_code == 404


def test_eval_results_aggregate(trace_dir):
    rows = [
        {"id": "a", "tool_f1": 0.5, "task_success": True, "input_tokens": 10, "output_tokens": 5},
        {"id": "b", "tool_f1": 1.0, "task_success": False, "input_tokens": 2, "output_tokens": 1},
    ]
    (trace_dir / "results" / "2026-10-22.json").write_text(
        json.dumps(rows), encoding="utf-8"
    )

    with TestClient(app) as client:
        response = client.get("/api/eval/results")

    result = response.json()["results"][0]
    assert result["mean_tool_f1"] == 0.75
    assert result["success_rate"] == 0.5
    assert result["total_tokens"] == 18
