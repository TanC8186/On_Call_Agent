"""测试公共夹具。

测试会调用真实的 graph 节点，这些节点会写 trace。这里把 trace 目录指到临时目录，
避免污染 `logs/traces/`（否则前端会把测试用的假 run 当成真实诊断展示）。
"""

import pytest

from app import observability


@pytest.fixture(autouse=True)
def isolated_trace_dir(tmp_path, monkeypatch):
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir(exist_ok=True)
    monkeypatch.setattr(observability, "TRACE_DIR", trace_dir)
    return trace_dir
