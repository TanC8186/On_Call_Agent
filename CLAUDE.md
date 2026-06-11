# On-Call Agent - 项目状态

## 项目概述

企业级智能 On-Call 运维助手，基于 FastAPI + LangChain + LangGraph + DeepSeek。

## 当前运行状态

| 组件 | 状态 | 端口 |
|------|------|------|
| 🤖 AI 对话（DeepSeek） | ✅ 正常 | - |
| 📚 RAG 知识库 | ✅ 正常 | - |
| 🗄️ Milvus 向量库 | ✅ 正常（Docker） | 19530 |
| 🎨 Web 界面 | ✅ 正常 | 9900 |
| 📊 Attu（Milvus 管理） | ✅ 正常 | 8000 |
| 📋 CLS MCP 服务 | ⚠️ 部分实现 | 8003 |
| 📊 Monitor MCP 服务 | ⚠️ 部分实现 | 8004 |

## LLM 配置

- **提供商**: DeepSeek (`https://api.deepseek.com`)
- **模型**: `deepseek-v4-pro`
- **Embedding**: 本地 `BAAI/bge-small-zh-v1.5`（HF 镜像 `hf-mirror.com`）
- **API Key**: 已配置在 `.env`

## 已修改的文件

### LLM 切换（ChatQwen → ChatOpenAI/DeepSeek）
- `app/config.py` — 通用 LLM 配置，`.env` 路径改为绝对路径
- `app/core/llm_factory.py` — ChatOpenAI 通用工厂
- `app/services/rag_agent_service.py` — MCP 不可用时自动降级
- `app/services/vector_embedding_service.py` — 支持 local/dashscope/openai
- `app/services/vector_store_manager.py` — 懒加载，Milvus 不可用不影响对话
- `app/agent/aiops/planner.py` — ChatOpenAI
- `app/agent/aiops/executor.py` — ChatOpenAI
- `app/agent/aiops/replanner.py` — ChatOpenAI
- `app/models/request.py` — Pydantic v2 model_config 修复

### 启动修复
- `app/main.py` — Milvus 连接非致命，static 目录绝对路径
- `app/utils/logger.py` — Windows GBK 编码修复（禁用彩色输出，utf-8 编码）

### 配置
- `.env` — DeepSeek API + 本地 Embedding

## MCP 工具实现进度

### CLS Server（日志服务）
- [x] `get_current_timestamp`
- [x] `get_region_code_by_name`
- [x] `get_topic_info_by_name`
- [x] `search_topic_by_service_name`
- [x] `search_log`
- [ ] `search_service_logs`
- [ ] `analyze_log_pattern`

### Monitor Server（监控服务）
- [x] `query_cpu_metrics`
- [x] `query_memory_metrics`
- [ ] `query_disk_metrics`
- [ ] `query_network_metrics`
- [ ] `query_process_list`
- [ ] `search_historical_tickets`
- [ ] `get_service_info`
- [ ] `list_all_services`

## 启动方式

### 前置条件
1. Docker Desktop 运行（Milvus 容器）
2. `.env` 中 DeepSeek API Key 已配置

### 手动启动（所有启动脚本已统一到 `servers/` 目录）
```powershell
# 终端 1 — CLS MCP 服务（端口 8003）
cd E:\On_Call_Agent\on_call_agent
.venv\Scripts\activate
python servers/cls_server.py

# 终端 2 — Monitor MCP 服务（端口 8004）
cd E:\On_Call_Agent\on_call_agent
.venv\Scripts\activate
python servers/monitor_server.py

# 终端 3 — FastAPI Web 服务（端口 9900）
cd E:\On_Call_Agent\on_call_agent
.venv\Scripts\activate
$env:HF_ENDPOINT="https://hf-mirror.com"
python servers/web_server.py
```

### 一键启动
```powershell
.\start-windows.bat
```

### Docker
```powershell
docker compose -f vector-database.yml up -d   # 启动 Milvus
docker compose -f vector-database.yml down     # 停止
```

## 已知问题
1. Pydantic v2 `class Config` 语法不兼容，已改为 `model_config`
2. Windows 控制台 GBK 编码导致 emoji 崩溃，已禁用 loguru 彩色输出
3. HuggingFace 被墙，需设置 `HF_ENDPOINT=https://hf-mirror.com`
4. Docker 镜像拉取：国内需用 `docker.1ms.run` 中转，etcd 用 `rancher/mirrored-coreos-etcd` 替代 `quay.io/coreos/etcd`
