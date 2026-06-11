# MCP 服务说明

三个 MCP Server 提供标准化工具接口，供 AIOps Agent 调用。

## 架构

```
FastAPI 主服务 (:9900)
    │
    ▼
MultiServerMCPClient
    ├── CLS Server (:8003)     → 腾讯云日志服务 API
    ├── Monitor Server (:8004) → 腾讯云云监控 API
    └── SSH Server (:8005)     → 远程服务器 SSH 巡检
```

## CLS Server — 腾讯云日志服务

| 工具 | 说明 |
|------|------|
| `get_current_timestamp` | 获取当前时间戳（毫秒） |
| `get_region_code_by_name` | 区域名称 → 区域代码 |
| `get_topic_info_by_name` | 根据名称查询日志主题 |
| `search_topic_by_service_name` | 按服务名关联日志主题 |
| `search_log` | 全文检索日志（支持 CQL 语法） |

## Monitor Server — 腾讯云监控

| 工具 | 说明 |
|------|------|
| `list_all_services` | 列出所有注册服务及其实例映射 |
| `query_cpu_metrics` | CPU 使用率查询 |
| `query_memory_metrics` | 内存使用率查询 |
| `query_disk_metrics` | 磁盘使用率查询 |
| `query_network_metrics` | 网络流量查询（仅 CVM） |
| `get_service_info` | 获取单个服务详细信息 |
| `search_historical_tickets` | 查询历史工单记录 |

## SSH Server — 远程服务器巡检

| 工具 | 说明 |
|------|------|
| `get_system_info` | 系统基本信息（OS/CPU/内存/磁盘） |
| `get_process_list` | 当前进程列表 |
| `search_system_logs` | Windows Event Log 搜索 |
| `check_disk_usage` | 指定磁盘使用率 |
| `read_log_file` | 读取日志文件（路径白名单） |

## 启动

```powershell
python servers/cls_server.py      # CLS MCP → :8003
python servers/monitor_server.py  # Monitor MCP → :8004
python servers/ssh_server.py      # SSH MCP → :8005
```
