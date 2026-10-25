"""训练数据契约测试：保证合成数据能被 pydantic + 工具目录接受。"""

import json
from pathlib import Path

from training.schema import TriageOutput, validate_against_catalog

ROOT = Path(__file__).resolve().parents[1]
CATALOG = json.loads(
    (ROOT / "training" / "data" / "tool_catalog.json").read_text(encoding="utf-8")
)


def test_generated_examples_validate_against_catalog():
    path = ROOT / "training" / "data" / "train.jsonl"
    if not path.exists():
        return
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows, "训练集为空"
    for row in rows:
        output = TriageOutput.model_validate(json.loads(row["messages"][-1]["content"]))
        errors = validate_against_catalog(output, CATALOG)
        assert not errors, f"{row['metadata']['template_id']}: {errors}"


def test_unknown_tool_is_rejected():
    output = TriageOutput.model_validate(
        {
            "severity": "P2",
            "category": "cpu",
            "plan": [
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
                    "tool": "not_a_tool",
                    "args": {},
                    "description": "bad",
                },
            ],
        }
    )
    errors = validate_against_catalog(output, CATALOG)
    assert any("未知工具" in error for error in errors)


def test_truncation_keeps_answer_and_prompt_tail():
    """超长样本只截 prompt 左侧，答案必须完整。"""
    from training.dataset_utils import build_example

    prompt_ids = list(range(1000))
    answer_ids = list(range(500, 600))
    input_ids, labels = build_example(prompt_ids, answer_ids, max_length=800)

    assert len(input_ids) == 800
    assert labels[-100:] == answer_ids, "答案被截断了"
    # prompt 保留的是尾部（system prompt 的开头在左侧，因此允许丢掉最早的部分）
    assert labels[:700] == [-100] * 700


def test_truncation_when_answer_alone_is_too_long():
    from training.dataset_utils import build_example

    input_ids, labels = build_example([1, 2, 3], list(range(900)), max_length=800)
    assert labels == list(range(800))
    assert len(input_ids) == 800


def test_critical_tools_derived_from_train_split():
    """关键工具必须来自训练集统计，且只保留高频项。"""
    from training.scoring import CRITICAL_THRESHOLD, critical_tools

    tools = critical_tools()
    assert tools, "训练集为空或解析失败"
    # cpu 场景每个 gold 计划都有 query_cpu_metrics
    assert "query_cpu_metrics" in tools["cpu"]
    # 低频工具不该被当成关键工具
    assert "analyze_log_pattern" not in tools["cpu"]
    assert 0.5 < CRITICAL_THRESHOLD <= 1.0


def test_plan_ok_uses_critical_tools_not_full_gold():
    """比 gold 更短、但覆盖关键工具的计划应判为可用。"""
    import json

    from training.scoring import score

    catalog = json.loads(
        (ROOT / "training" / "data" / "tool_catalog.json").read_text(encoding="utf-8")
    )
    gold = TriageOutput.model_validate(
        {
            "severity": "P2",
            "category": "cpu",
            "plan": [
                {"id": "s1", "kind": "tool", "tool": "list_all_services", "args": {}, "description": "list"},
                {"id": "s2", "kind": "tool", "tool": "query_cpu_metrics",
                 "args": {"service_name": "agent-test-server"}, "description": "cpu"},
                {"id": "s3", "kind": "tool", "tool": "analyze_log_pattern",
                 "args": {"service_name": "agent-test-server"}, "description": "logs"},
                {"id": "s4", "kind": "reason", "tool": "", "args": {}, "description": "report"},
            ],
        }
    )
    short_prediction = TriageOutput.model_validate(
        {
            "severity": "P2",
            "category": "cpu",
            "plan": [
                {"id": "s1", "kind": "tool", "tool": "list_all_services", "args": {}, "description": "list"},
                {"id": "s2", "kind": "tool", "tool": "query_cpu_metrics",
                 "args": {"service_name": "agent-test-server"}, "description": "cpu"},
                {"id": "s3", "kind": "reason", "tool": "", "args": {}, "description": "report"},
            ],
        }
    )

    result = score(short_prediction, gold, short_prediction.model_dump(), catalog,
                   default_service="agent-test-server")
    assert result["plan_ok"] is True, result
    # 旧口径（全 gold 覆盖）仍然是 0.67 左右，说明 plan_ok 不再被长度惩罚
    assert result["required_tool_recall"] < 1.0


def test_plan_ok_fails_when_critical_tool_missing():
    import json

    from training.scoring import score

    catalog = json.loads(
        (ROOT / "training" / "data" / "tool_catalog.json").read_text(encoding="utf-8")
    )
    gold = TriageOutput.model_validate(
        {
            "severity": "P2",
            "category": "cpu",
            "plan": [
                {"id": "s1", "kind": "tool", "tool": "list_all_services", "args": {}, "description": "list"},
                {"id": "s2", "kind": "tool", "tool": "query_cpu_metrics",
                 "args": {"service_name": "svc"}, "description": "cpu"},
            ],
        }
    )
    no_cpu = TriageOutput.model_validate(
        {
            "severity": "P2",
            "category": "cpu",
            "plan": [
                {"id": "s1", "kind": "tool", "tool": "list_all_services", "args": {}, "description": "list"},
                {"id": "s2", "kind": "tool", "tool": "get_service_info",
                 "args": {"service_name": "svc"}, "description": "status"},
            ],
        }
    )
    result = score(no_cpu, gold, no_cpu.model_dump(), catalog, default_service="svc")
    assert result["plan_ok"] is False
    assert "query_cpu_metrics" in result["missing_critical"]


def test_tool_alias_repair_maps_common_hallucinations():
    import json

    from training.scoring import repair_plan

    catalog = json.loads(
        (ROOT / "training" / "data" / "tool_catalog.json").read_text(encoding="utf-8")
    )
    bad = TriageOutput.model_validate(
        {
            "severity": "P1",
            "category": "service",
            "plan": [
                {"id": "s1", "kind": "tool", "tool": "list_all_services", "args": {}, "description": "list"},
                {"id": "s2", "kind": "tool", "tool": "query_service_logs",
                 "args": {"service_name": "svc"}, "description": "logs"},
                {"id": "s3", "kind": "tool", "tool": "query_service_status",
                 "args": {"service_name": "svc"}, "description": "status"},
            ],
        }
    )
    fixed = repair_plan(bad, catalog, default_service="svc")
    tools = [s.tool for s in fixed.plan]
    assert "search_service_logs" in tools
    assert "get_service_info" in tools
    assert "query_service_logs" not in tools
    assert "query_service_status" not in tools


def test_labels_match_dataset():
    """数据集的 severity 必须与当前规则一致。

    这条测试是为了防住 10-23 那次事故：数据 00:06 生成，`labels.py` 15:54 改过，
    于是 test 里 19 条 `disk-2`（"write failures" 旧规则没匹配上）一直带着 P3，
    把**所有**模型的 severity 一起压低。规则和数据必须一起提交。
    """
    from training.labels import severity_of

    mismatches = []
    for split in ("train", "val", "test"):
        path = ROOT / "training" / "data" / f"{split}.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            gold = json.loads(row["messages"][-1]["content"])
            user = next(m["content"] for m in row["messages"] if m["role"] == "user")
            category = "healthy" if row["metadata"]["family"] == "healthy" else gold["category"]
            expected = severity_of(user, category)
            if expected != gold["severity"]:
                mismatches.append(
                    f"{split}/{row['metadata']['template_id']}: gold={gold['severity']} rule={expected}"
                )
    assert not mismatches, "标签与规则不一致，跑 training/fix_labels.py --write:\n" + "\n".join(
        mismatches[:10]
    )

def test_prompt_matches_runtime_prompt():
    """训练数据里的 system prompt 必须和线上推理用的完全一致。

    `app/agent/aiops/local_prompt.py` 改一句规则，训练集就要同步；
    否则 SFT 学的是旧规则说明，线上拿到新 prompt，行为对不上。
    """
    import sys

    sys.path.insert(0, str(ROOT))
    from app.agent.aiops.local_prompt import build_system_prompt, compact_catalog

    catalog = json.loads(
        (ROOT / "training" / "data" / "tool_catalog.json").read_text(encoding="utf-8")
    )
    expected = build_system_prompt(compact_catalog(catalog, with_params=False), tool_names=None)
    for split in ("train", "val", "test"):
        path = ROOT / "training" / "data" / f"{split}.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["messages"][0]["content"] != expected:
                raise AssertionError(
                    f"{split}/{row['metadata']['template_id']} 的 system prompt 与"
                    " app/agent/aiops/local_prompt.py 不一致；跑 training/sync_prompt.py --write"
                )


def test_negation_does_not_count_as_user_impact():
    """否定句不是用户影响。

    2026-10-24 之前 `severity_of` 看到"没有用户请求超时"里的"超时"就判 P2，
    把 test 的 `cpu-3` 整类（30 条）标错，也让模型学不到否定语义。
    """
    from training.labels import has_oom, has_user_impact, severity_of

    assert not has_user_impact("凌晨低峰，只有批量任务在跑，没有用户请求超时。")
    assert not has_user_impact("off-peak batch job; no user-facing timeouts")
    assert has_user_impact("接口超时率上升，用户投诉增加")
    # 否定的是别的东西，影响词仍然成立
    assert has_user_impact("下游调用批量超时，没有扩容计划")
    # 转折句：逗号是硬边界，否定不能跨过它吃掉后半句
    assert has_user_impact("没有 OOM 记录，但接口超时率升到 5%")

    assert not has_oom("内存 88%，尚未出现 OOM")
    assert has_oom("内存 99%，OOM killer 记录")

    assert severity_of("凌晨低峰 CPU 升到 95%，没有用户请求超时。", "cpu") == "P3"
    assert severity_of("内存 92%，OOM killer 记录。", "memory") == "P2"
    assert severity_of("内存 99%，OOM killer 记录。", "memory") == "P1"
    assert severity_of("内存 99%，尚未出现 OOM。", "memory") == "P3"
