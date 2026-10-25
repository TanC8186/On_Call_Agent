# Eval: On-Call Agent

评测是这个仓库的第一优先级：**没有 baseline 和可复算的数字，任何“优化”都只是形容词。**

## 当前状态（2026-10）

已完成：

- `scenarios/*.yaml`：5 条种子场景（CPU / 内存 / 磁盘 / 服务不可用 / 健康对照），覆盖工具选择和数值引用。
- `scripts/validate_eval.py`：校验场景 schema、`gold_tools` 是否存在于 MCP 注册表、`aiops-docs` 是否引用了不存在的工具。
- `tests/`：状态隔离、分块窗口、SSH 注入、RAG 未命中、typed plan、verifier/replan、provenance 保留等回归测试。
- `scripts/run_eval.py`：跑场景并输出 tool recall/precision/F1、steps、tokens、task_success。
- `app/observability.py`：每 run 一份 JSONL trace（节点、工具、决策、token）。
- **typed PlanStep**：Planner 输出经 schema 校验的 `{kind, tool, args, description}`；tool step 直接执行，不再让 LLM 翻译一次。
- **verifier + replan**：上一步没有可用证据时触发重写剩余计划，预算 2 次。
- GitHub Actions：跑 lint（tests）+ pytest。

尚未完成（Roadmap）：

- 进程内假 MCP 平面（空结果 / 超时 / 429 / 错实例 / healthy 五种故障注入）。
- 自动判分的数值 groundedness、p95、RMB/task（当前 run_eval 已输出工具指标/token）。
- 改代码前的 baseline（3 次重复，mean ± range）与 CI 回归门禁。
- 进程内 fake LLM（当前 live 模式需要有效的 LLM_API_KEY）。
- 真实的告警摄入（Alert/Incident webhook）。

## 场景格式

```yaml
id: cpu-high-001              # 唯一 ID
title: CPU 使用率持续过高
symptom: ...                  # 喂给 Agent 的症状描述
service: agent-test-server
expect_no_incident: false     # true = 阴性对照
gold_tools:                   # 期望至少被调用一次的工具
  - query_cpu_metrics
  - search_log
gold_facts:                   # 报告中应出现的事实
  - metric: cpu_usage_percent
    op: ">"
    value: 90
forbidden_claims:             # 报告中不允许出现的结论
  - 磁盘已满
max_steps: 8
notes: 合成场景，不是真实生产事故。
```

## 原则

1. **场景和标签是合成的，必须显式标注**。不要把它包装成真实工单。
2. **先冻结场景，再改代码**。任何指标都要能指向一个 commit 和一条 `make eval` 命令。
3. **判分先规则、后 LLM**。LLM judge 必须报告人工一致性，否则不可证伪。
4. **阴性对照和故障注入同等重要**。只会测 happy path 的评测没有价值。
