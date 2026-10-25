# On-Call Agent

一个**可评测**的 AIOps 诊断 Agent：LangGraph 编排 + MCP 工具平面 + 本地小模型快路径。
重点不是"接了多少框架"，而是每个结论都能被 `make eval` 复算、每次运行都能按 `run_id` 回放。

```
web (React + Vite)  ──SSE──▶  FastAPI (:9900)
                                  │
                     ┌────────────┼─────────────┐
                     ▼            ▼             ▼
              LangGraph 图    Milvus RAG    traces/*.jsonl
       planner→executor→verifier→replanner
                     │
        ┌────────────┼──────────────┐
        ▼            ▼              ▼
   Monitor MCP   CLS MCP        SSH MCP        (FastMCP, 20 tools)
   腾讯云监控     腾讯云日志      远程只读巡检
```

## 为什么这个项目值得看

| 能力 | 实现 | 怎么验证 |
|---|---|---|
| **结构化计划** | Planner 输出 typed `PlanStep {kind, tool, args}`，对上 MCP 注册表校验 | `tests/test_typed_plan.py` |
| **真 replan** | Verifier 做确定性校验（空结果 / `ok=false` / `unavailable`），失败才触发重写剩余计划，预算 2 次 | `tests/test_typed_plan.py::test_replanner_*` |
| **工具快路径** | tool 步骤直接执行，不再让 LLM 二次翻译成工具调用 | 单次诊断 LLM 调用 ~12 → ≤5 |
| **可回放** | 每 run 一份 `logs/traces/<run_id>.jsonl`：节点、工具、决策、token、报告 | 前端"最近运行"点开即回放 |
| **评测** | 5 个场景 + 自动判分（tool P/R/F1、facts、groundedness、tokens） | `python scripts/run_eval.py --live` |
| **溯源** | 所有 MCP 结果带 `source: tencent_cloud / demo / unavailable`，没有数据就报 `unavailable` | `tests/test_knowledge_tool.py` |
| **安全** | SSH 参数白名单 + 路径规范化 + 10 条注入 payload 测试；可选 `X-API-Key` | `tests/test_ssh_safety.py` |

## 快速开始

```bash
# 1. 依赖
uv venv && uv pip install -e ".[dev]"

# 2. 配置
cp .env.example .env    # 填 LLM_API_KEY；没有云凭证时设 DEMO_MODE=1

# 3. 起 MCP 工具（三个终端）
.venv/Scripts/python.exe servers/monitor_server.py   # :8004
.venv/Scripts/python.exe servers/cls_server.py       # :8003
.venv/Scripts/python.exe servers/ssh_server.py       # :8005

# 4. 起后端
.venv/Scripts/python.exe -m uvicorn app.main:app --port 9900

# 5. 构建前端（产物由后端直接托管）
cd web && npm install && npm run build && cd ..
```

开发前端时用 `cd web && npm run dev`（:5173，已配置 `/api` 代理到 :9900）。

## 验证

```bash
.venv/Scripts/python.exe -m pytest tests/ -q          # 回归测试
.venv/Scripts/python.exe scripts/validate_eval.py     # 场景 ↔ 工具注册表 ↔ 文档
.venv/Scripts/python.exe scripts/run_eval.py --live   # 真跑评测，输出 eval/results/*.md
```

## 模型后端

所有 LLM 调用都走 `app/core/llm_factory.py` 一个入口，provider 由 `.env` 决定：

| provider | 适用 | 关键配置 |
|---|---|---|
| `anthropic`（默认） | Messages API，含第三方 Anthropic 兼容网关 | `LLM_BASE_URL=https://www.autodl.art/api/v1/anthropic` |
| `openai` | DeepSeek / DashScope / 本地 vLLM / Ollama | `LLM_PROVIDER=openai` + `LLM_BASE_URL` |

```bash
LLM_PROVIDER=anthropic
LLM_BASE_URL=https://www.autodl.art/api/v1/anthropic
LLM_API_KEY=...
LLM_MODEL=DeepSeek-V4.1-Flash-1017
```

> 网关兼容性：第三方 Anthropic 网关会在 `message_delta` 里返回 dict 形式的
> `context_management`，而 `langchain-anthropic` 1.7.6 假设它是 pydantic 对象，
> 直接 `.model_dump()` 会抛 `AttributeError`。`llm_factory` 里的
> `_CompatChatAnthropic` 在解析前摘掉该字段，流式与 tool calling 都恢复正常。

## 本地小模型（可选）

`training/` 是一个独立的子项目：Qwen2.5-1.5B-Instruct 做 LoRA SFT，输出
`{severity, category, plan[]}`，通过 vLLM 提供 OpenAI 兼容接口，作为 Planner 的本地快路径；
工具/参数校验失败时自动回退云端。

| 指标（v9，held-out 178 条） | fp16 (HF) | Q4_K_M (AutoRound→GGUF, CPU) | AWQ/SignRound (vLLM) | 说明 |
|---|---:|---:|---:|---|
| Schema 合法 | **1.000** | **1.000** | 0.989 | |
| **关键工具召回** | **1.000** | **1.000** | 0.989 | 训练集里 ≥90% 出现的工具 |
| **Plan OK** | **1.000** | **1.000** | 0.989 | 关键工具全覆盖 + 多余 ≤2 |
| Arg exact | 0.843 | **0.858** | 0.778 | |
| **Severity** | **0.989** | 0.787 | **0.989** | Q4 在 OOM 边界（92% vs 95%）上掉 |
| Category | 1.000 | 1.000 | 0.989 | |

三个后端是同一份 v9 权重、同一个 test split、同一套判分。

**severity 这一行要打折看。** test 只有 6 种句式、每种 30 条。把题干换一种说法
（`training/paraphrase_rules.py`，规则改写、可复现），severity 从 0.989 掉到 0.865；
schema / 关键工具 / category 完全不变。受控探针（只把分句顺序换一下）显示：
数字在前的语序准确率 **0.966**，线索在前的语序 **0.534** —— 模型学的是句式，
不是"OOM 且 ≥95%"这条逻辑。原因是 train 里 90 条同类样本**全是**一种语序。

所以对外只讲两件事：plan 类指标（稳，改写前后都是 1.000），和"4-bit 无损"
（三个后端互相印证）。severity 的绝对值不要单独引用。
**4-bit 不再有系统性损失**：AWQ/SignRound 在共同 149 条上 schema / 关键工具 / plan_ok /
severity 全是 **1.000**，只比 fp16 低 arg exact。v8 时代"4-bit 掉 13 点 severity"的结论
**是训练数据没覆盖到边界，不是比特数**——v9 补了否定句与 OOM 边界样本后差距消失。

CPU 端要用 AutoRound 产的 GGUF（`training/quantize_gguf.py`），不要用 `llama-quantize`：
同样是 Q4_K_M，前者 schema 1.000，后者 0.664。

推理性能（同一台机、同一轮测量，`max_tokens=128`、每档 16 请求）：

| 推理（RTX 4060 8GB / CPU） | vLLM fp16 | vLLM AWQ W4A16 | llama.cpp Q4 (CPU) |
|---|---:|---:|---:|
| 模型大小 | 2.9 GB | **1.15 GB** | **941 MB** |
| TTFT p50 @1 | 71.1 ms | 77.3 ms | 34.7 ms |
| TTFT p50 @8 | 73.6 ms | **69.7 ms** | 2384 ms |
| TPOT p50 @8 | 16.7 ms | **15.1 ms** | 38.9 ms |
| 吞吐 @8 | 464 tok/s | **514 tok/s** | 102 tok/s |

（性能表是 v8 权重测的；v9 的模型结构与量化配置相同，未重测。）

### 接入效果（5 个场景，本地 = Q4_K_M GGUF on CPU）

| | 云端 Planner | 本地 Planner |
|---|---:|---:|
| 本地命中 | — | 5/5（无回退） |
| 平均工具 F1 | 0.39 | **0.71** |
| 总 tokens | 131,263 | **47,020（−64%）** |
| 平均步骤 | 8.0 | 2.6 |

失败场景两边一样多，原因是评测集的 gold 用了 `search_log`，而 690 条训练数据里
没有这个工具的样本（模型学的是 `search_service_logs` / `search_system_logs`）。
详情见 `training/RESULTS.md`。

接入方式：

```bash
# GPU：vLLM 加载 AWQ 产物
wsl ~/vllm-venv/bin/vllm serve training/out/qwen15-v9-signround \
    --served-model-name qwen-ops-v9-awq --quantization awq \
    --max-model-len 2048 --enforce-eager --port 8000

# CPU：llama.cpp 加载 AutoRound 产的 GGUF
llama-server.exe -m training/out/qwen15-v9-ops-Q4_K_M-autoround.gguf -t 6 -c 4096 --alias qwen-ops-v9

# env 必须加在跑评测的那个进程上
LOCAL_LLM_BASE_URL=http://127.0.0.1:8000/v1 LOCAL_LLM_MODEL=qwen-ops-v9-awq \
  .venv/Scripts/python.exe -m uvicorn app.main:app --port 9900
```

## 目录

```
app/
  agent/aiops/       planner · executor · verifier · replanner · plan_schema
  agent/             MCP 客户端 + 进程级工具缓存
  api/               chat · aiops · file · health · runs
  services/          AIOps 编排 · RAG · 向量索引
  observability.py   per-run JSONL trace
servers/             3 个 FastMCP server（20 个工具）
aiops-docs/          运维知识库（6 篇）
eval/scenarios/      YAML 故障场景 + 阴性对照
scripts/             validate_eval.py · run_eval.py
training/            LoRA 数据管线 · 训练 · 评测 · 服务基准
tests/               回归测试
web/                 React + Vite 前端
```

## 当前限制

- **记忆在进程内**：`MemorySaver`，重启丢会话，只能单 worker。
- **鉴权可选**：设置 `API_KEY` 后才要求 `X-API-Key`；没有限流。
- **评测集是合成的**：5 条场景 + 240 条训练数据均为模板生成并校验，不是真实工单。
- **RAG 未做混合检索**：只有 dense top-k，没有 rerank、没有 Recall@k 基线。
- **0.5B 尚未跑通**：对学习率敏感，两次训练分别发散/停滞；当前用 1.5B + 4-bit 作为端侧方案。
- **本地 Planner 的评测口径与训练集不一致**：5 个场景里 4 个的 gold 用了 `search_log`，
  而训练数据（690 条）里它出现 0 次；这会让 `task_success` 在本地/云端都偏低。

完整路线图见 [BACKLOG.md](BACKLOG.md)，环境与算力限制见 [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md)。

## License

MIT © TanC
