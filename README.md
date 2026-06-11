# On-Call Agent — 企业级智能运维助手

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Plan--Execute-orange?logo=langchain)](https://langchain-ai.github.io/langgraph/)
[![DeepSeek](https://img.shields.io/badge/LLM-DeepSeek%20V4-purple)](https://platform.deepseek.com/)
[![Milvus](https://img.shields.io/badge/VectorDB-Milvus-00D2B0)](https://milvus.io/)
[![MCP](https://img.shields.io/badge/Protocol-MCP-red)](https://modelcontextprotocol.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](./LICENSE)

**LangGraph Plan-Execute-Replan 工作流 · MCP 多服务架构 · 腾讯云真实 API 集成**

</div>

---

## 📖 项目简介

On-Call Agent 是一个**面向生产环境的智能运维助手**，通过 AI Agent 自动完成服务器健康巡检、故障根因分析和处理建议生成。项目核心采用 **LangGraph 状态机**驱动多步推理过程，通过 **MCP (Model Context Protocol)** 协议接入腾讯云真实 API 和远程服务器 SSH，结合 **RAG 知识库**提供历史经验参考。

> 🎯 **设计目标**：让 AI 像真正的 On-Call 工程师一样工作 — 发现告警 → 查询指标 → 检索日志 → 分析根因 → 给出处理方案。

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        🎨 Web 前端 (SSE 流式)                     │
│                   static/index.html + app.js                     │
└─────────────────────────────┬───────────────────────────────────┘
                              │ HTTP/SSE
┌─────────────────────────────▼───────────────────────────────────┐
│                   🚀 FastAPI 主服务 (:9900)                        │
│  ┌──────────────┐  ┌────────────────┐  ┌────────────────────┐   │
│  │  RAG 对话     │  │  AIOps 诊断     │  │  文档管理 / 健康检查 │   │
│  │  /api/chat   │  │  /api/aiops    │  │  /api/upload|health│   │
│  └──────┬───────┘  └───────┬────────┘  └────────────────────┘   │
│         │                  │                                      │
│         ▼                  ▼                                      │
│  ┌──────────────┐  ┌──────────────────────────────────────────┐ │
│  │ LangChain    │  │        LangGraph Plan-Execute-Replan      │ │
│  │ RAG Agent    │  │  Planner ──▶ Executor ──▶ Replanner       │ │
│  └──────┬───────┘  │      │            │            │          │ │
│         │          │  制定计划      调用工具      评估/继续      │ │
│         │          └──────┼────────────┼────────────┼──────────┘ │
│         │                 │            │            │             │
├─────────┼─────────────────┼────────────┼────────────┼────────────┤
│         ▼                 │            │            │             │
│  ┌──────────┐             ▼            ▼            ▼             │
│  │ Milvus   │    ┌───────────────────────────────────────────┐   │
│  │ 向量数据库 │    │         MCP Client (MultiServerMCP)        │   │
│  │ :19530   │    │  重试拦截器 · 指数退避 · 自动故障转移          │   │
│  └──────────┘    └──┬──────────┬──────────┬──────────────────┘   │
└─────────────────────┼──────────┼──────────┼──────────────────────┘
                      │          │          │
         ┌────────────▼──┐  ┌───▼──────┐  ┌▼──────────────┐
         │  Monitor MCP  │  │  CLS MCP │  │   SSH MCP     │
         │    :8004      │  │  :8003   │  │    :8005      │
         │  CPU/内存/磁盘  │  │  日志搜索 │  │ 系统日志/进程  │
         │ 腾讯云云监控API │  │ 腾讯云CLS │  │ Paramiko SSH │
         └───────────────┘  └──────────┘  └───────────────┘
```

### 核心技术选型

| 层级 | 技术 | 选型理由 |
|------|------|---------|
| **Agent 框架** | LangGraph | 原生支持 Plan-Execute-Replan 状态机，比 AutoGPT 类方案更可控 |
| **工具协议** | MCP (Model Context Protocol) | 将外部能力标准化为 LLM 可调用的工具，解耦 Agent 与数据源 |
| **LLM** | DeepSeek V4 Pro | 国内可用、性价比高、支持 OpenAI 兼容 API |
| **向量库** | Milvus | 云原生向量数据库，支持十亿级检索，社区活跃 |
| **Web 框架** | FastAPI | 原生异步 + SSE 流式输出，性能优于 Flask/Django |
| **SSH** | Paramiko | 纯 Python 实现，无需系统 SSH 客户端 |

---

## 🔄 AIOps 工作流详解

```
用户发起诊断
     │
     ▼
┌──────────┐
│  PLANNER │  ① 查询 RAG 知识库获取历史经验
│  制定计划 │  ② 枚举可用 MCP 工具
└────┬─────┘  ③ LLM 生成分步执行计划 (JSON)
     │          "每步一个工具调用，至少 5 步"
     ▼
┌──────────┐
│ EXECUTOR │  ④ 从 plan 中弹出下一步
│  执行步骤 │  ⑤ LLM 决策调用哪个工具 + 参数
└────┬─────┘  ⑥ 实际执行工具调用 → 记录到 past_steps
     │
     ▼
┌──────────┐  ⑦ LLM 评估已收集的数据
│REPLANNER│  → 数据不足?  继续执行 (continue)
│  评估决策 │  → 计划需改?  重新规划 (replan)
└────┬─────┘  → 信息充分?  生成最终报告 (respond)
     │
     ▼
  最终诊断报告
  (根因分析 + 日志证据 + 处理建议)
```

**关键设计决策**：
- **DeepSeek 不支持 `with_structured_output`** → 自研三层 JSON 容错解析（直接解析 → Markdown 提取 → 花括号匹配 → 逐行回退）
- **MCP 调用不稳定** → 指数退避重试拦截器（最多 3 次），失败返回友好错误而非崩溃
- **历史步骤累积** → 使用 LangGraph `operator.add` reducer，每步结果追加而非覆盖

---

## 📊 MCP 工具矩阵

### Monitor Server → 腾讯云云监控 (QCE/LIGHTHOUSE + QCE/CVM)

| 工具 | 数据来源 | 说明 |
|------|----------|------|
| `list_all_services` | 本地注册表 | 服务名 → 实例 ID 映射 |
| `query_cpu_metrics` | 腾讯云 GetMonitorData | 自动识别 Lighthouse/CVM namespace |
| `query_memory_metrics` | 腾讯云 GetMonitorData | 支持自定义 period 和统计方式 |
| `query_disk_metrics` | 腾讯云 GetMonitorData | Lighthouse 用 DiskUsage，CVM 用 CvmDiskUsage |
| `query_network_metrics` | 腾讯云 GetMonitorData | 仅 CVM 支持，Lighthouse 返回友好提示 |

### SSH Server → Windows/Linux 远程服务器

| 工具 | 实现 | 安全约束 |
|------|------|---------|
| `get_system_info` | `systeminfo` + `wmic` / `uname` + `/proc` | 只读命令 |
| `get_process_list` | `tasklist` / `ps` | 只读命令 |
| `search_system_logs` | `wevtutil` / `tail` + `grep` | 按级别/关键词过滤 |
| `check_disk_usage` | `wmic logicaldisk` / `df -h` | 只读命令 |
| `read_log_file` | PowerShell `Get-Content -Tail` | **路径白名单**，仅限 `/var/log/` 等 |

### CLS Server → 腾讯云日志服务

| 工具 | 说明 |
|------|------|
| `search_log` | 全文检索 + 时间范围 + CQL 语法 |
| `get_topic_info_by_name` | 日志主题发现 |
| `search_topic_by_service_name` | 按服务名关联日志主题 |

---

## 🚀 快速开始

### 环境要求

- Python 3.11+
- Docker Desktop（Milvus 向量库）
- 腾讯云 API 密钥（Monitor/CLS 服务）
- DeepSeek API Key（或任意 OpenAI 兼容 API）

### 1. 克隆项目

```bash
git clone https://github.com/<your-username>/on-call-agent.git
cd on-call-agent
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入你的 API 密钥
```

### 3. 安装依赖

```bash
# 使用 uv（推荐）
pip install uv
uv venv
.venv\Scripts\activate  # Windows
uv pip install -e .

# 使用 pip
pip install -e .
```

### 4. 启动服务

```powershell
# 一键启动（Windows）
.\start-windows.bat

# 或手动启动三个 MCP 服务 + Web 服务
python servers/cls_server.py      # 终端1: CLS MCP → :8003
python servers/monitor_server.py  # 终端2: Monitor MCP → :8004
python servers/ssh_server.py      # 终端3: SSH MCP → :8005
python servers/web_server.py      # 终端4: FastAPI → :9900
```

### 5. 访问

| 地址 | 功能 |
|------|------|
| `http://localhost:9900` | Web 管理界面 |
| `http://localhost:9900/docs` | Swagger API 文档 |
| `http://localhost:8000` | Attu (Milvus 可视化管理) |

---

## 🧪 API 示例

```bash
# AIOps 智能诊断（SSE 流式输出）
curl -X POST "http://localhost:9900/api/aiops" \
  -H "Content-Type: application/json" \
  -d '{"session_id":"demo"}' \
  --no-buffer

# 流式对话
curl -X POST "http://localhost:9900/api/chat_stream" \
  -H "Content-Type: application/json" \
  -d '{"Id":"demo","Question":"agent-test-server 的 CPU 使用率是多少？"}' \
  --no-buffer

# 知识库文档上传
curl -X POST "http://localhost:9900/api/upload" \
  -F "file=@aiops-docs/cpu_high_usage.md"
```

---

## 📁 项目结构

```
on_call_agent/
├── app/                          # 应用核心
│   ├── agent/                    # AI Agent 层
│   │   ├── aiops/                # AIOps 三节点工作流
│   │   │   ├── planner.py        #   Planner — LLM 生成分步计划
│   │   │   ├── executor.py       #   Executor — 逐步调用 MCP 工具
│   │   │   ├── replanner.py      #   Replanner — 评估数据决定继续/完成
│   │   │   ├── state.py          #   LangGraph TypedDict 状态定义
│   │   │   └── utils.py          #   工具描述格式化
│   │   └── mcp_client.py         # MCP 客户端单例 + 重试拦截器
│   ├── services/                 # 业务服务层
│   │   ├── aiops_service.py      #   AIOps 流程编排（状态图编译）
│   │   ├── rag_agent_service.py  #   RAG 对话服务
│   │   ├── vector_*.py           #   向量嵌入/索引/检索/存储
│   │   └── document_splitter.py  #   三级文档分割器
│   ├── api/                      # FastAPI 路由
│   │   ├── aiops.py              #   POST /api/aiops (SSE)
│   │   ├── chat.py               #   POST /api/chat[_stream]
│   │   ├── file.py               #   POST /api/upload
│   │   └── health.py             #   GET /api/health
│   ├── models/                   # Pydantic v2 数据模型
│   ├── tools/                    # 本地 Agent 工具
│   └── core/                     # LLM 工厂 + Milvus 客户端
├── servers/                      # MCP 服务进程
│   ├── monitor_server.py         #   腾讯云监控 (:8004)
│   ├── cls_server.py             #   腾讯云日志 (:8003)
│   ├── ssh_server.py             #   远程 SSH 巡检 (:8005)
│   └── web_server.py             #   FastAPI 入口
├── static/                       # 纯静态前端
│   ├── index.html                #   SSE 流式渲染
│   ├── app.js                    #   对话/AIOps 交互逻辑
│   └── styles.css                #   响应式布局
├── aiops-docs/                   # 运维知识库（Markdown）
│   ├── cpu_high_usage.md
│   ├── memory_high_usage.md
│   ├── disk_high_usage.md
│   └── service_unavailable.md
├── .env.example                  # 环境变量模板（可提交）
├── pyproject.toml                # 项目元数据 + 工具链配置
├── vector-database.yml           # Milvus Docker Compose
└── start-windows.bat             # 一键启动脚本
```

---

## 🔧 工程亮点

### 1. 健壮性设计

```python
# MCP 工具调用自动重试（指数退避）
async def retry_interceptor(request, handler, max_retries=3, delay=1.0):
    for attempt in range(max_retries):
        try:
            return await handler(request)
        except Exception as e:
            wait_time = delay * (2 ** attempt)  # 1s → 2s → 4s
            await asyncio.sleep(wait_time)
    return CallToolResult(isError=True, ...)  # 优雅降级，不抛异常
```

### 2. DeepSeek 兼容性适配

DeepSeek API 不支持 OpenAI 的 `response_format` / `with_structured_output`，自研三层容错 JSON 解析器：
- 第一层：直接 `json.loads()` 
- 第二层：正则提取 Markdown 代码块中的 JSON
- 第三层：正则匹配花括号内容
- 回退：逐行解析为步骤列表

### 3. 云 API 命名空间自动识别

```python
def _get_namespace_for_instance(instance_id: str) -> str:
    """自动区分 Lighthouse 轻量服务器 vs CVM 云服务器"""
    if instance_id.startswith("lhins-"):
        return "QCE/LIGHTHOUSE"  # 不同的 metric 名称和维度
    return "QCE/CVM"
```

### 4. 安全约束

- SSH 工具**只执行只读命令**，硬编码命令模板，不接受任意命令
- 文件读取**路径白名单**，仅限 `/var/log/`、`C:/logs/` 等
- API 密钥通过 `.env` 管理，`.gitignore` 排除，提供 `.env.example` 模板

---

## 📝 License

MIT © TanC

---

<div align="center">
  <sub>Built with ❤️ using FastAPI + LangGraph + DeepSeek + Milvus</sub>
</div>
