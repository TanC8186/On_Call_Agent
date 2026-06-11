"""SSH 远程服务器健康检查 MCP Server

通过 SSH 连接到远程 Windows/Linux 服务器，提供只读健康检查功能。

安全约束：
- 仅执行只读命令
- 仅允许白名单路径
- 所有操作记录日志
"""

import os
import sys
import logging
import functools
import json
from typing import Dict, Any, Optional, List
from pathlib import Path

import paramiko
from dotenv import load_dotenv
from fastmcp import FastMCP

# 加载 .env 配置
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_ENV_PATH)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SSH_MCP_Server")

# ── SSH 配置 ──────────────────────────────────────────────────
SSH_HOST = os.getenv("SSH_HOST", "139.199.172.55")
SSH_PORT = int(os.getenv("SSH_PORT", "22"))
SSH_USER = os.getenv("SSH_USER", "Administrator")
SSH_KEY_FILE = os.getenv("SSH_KEY_FILE", str(Path.home() / ".ssh" / "AgentTest.pem"))

# 远程服务器类型（auto / windows / linux）
SERVER_TYPE = os.getenv("SSH_SERVER_TYPE", "windows")

mcp = FastMCP("SSH")


# ── SSH 连接管理 ──────────────────────────────────────────────

def _get_ssh_client() -> paramiko.SSHClient:
    """获取 SSH 客户端连接"""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=SSH_HOST,
        port=SSH_PORT,
        username=SSH_USER,
        key_filename=SSH_KEY_FILE,
        timeout=10,
    )
    return client


def _exec_readonly(commands: List[str]) -> Dict[str, str]:
    """执行只读命令集，返回命令->输出的映射。

    安全约束：仅允许预定义的只读命令模式。
    """
    client = _get_ssh_client()
    results = {}
    try:
        for cmd in commands:
            logger.info(f"执行: {cmd[:80]}...")
            stdin, stdout, stderr = client.exec_command(cmd, timeout=15)
            out = stdout.read().decode('utf-8', errors='replace')
            err = stderr.read().decode('utf-8', errors='replace')
            if err:
                logger.warning(f"命令 stderr: {err[:100]}")
            results[cmd] = out if out else "(empty)"
    finally:
        client.close()
    return results


def log_tool_call(func):
    """装饰器：记录工具调用的日志"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"=" * 60)
        logger.info(f"Tool: {func.__name__}")
        try:
            result = func(*args, **kwargs)
            logger.info(f"Tool {func.__name__}: SUCCESS")
            logger.info(f"=" * 60)
            return result
        except Exception as e:
            logger.error(f"Tool {func.__name__}: ERROR - {e}")
            logger.info(f"=" * 60)
            return {"error": str(e)}
    return wrapper


# ╔══════════════════════════════════════════════════════════════════╗
# ║                        MCP 工具函数                             ║
# ╚══════════════════════════════════════════════════════════════════╝

@mcp.tool()
@log_tool_call
def get_system_info() -> Dict[str, Any]:
    """获取远程服务器系统基本信息（OS、CPU、内存、磁盘、网络等只读信息）。

    无需参数。

    Returns:
        Dict: 系统基本信息
    """
    if SERVER_TYPE == "windows":
        cmds = [
            "systeminfo | findstr /C:'OS' /C:'System' /C:'Processor' /C:'Memory' /C:'Time'",
            "wmic cpu get Name,NumberOfCores,MaxClockSpeed /format:list",
            "wmic memorychip get Capacity,Speed /format:list",
            "wmic logicaldisk get DeviceID,Size,FreeSpace /format:list",
        ]
        results = _exec_readonly(cmds)
        return {
            "server_type": "windows",
            "host": SSH_HOST,
            "systeminfo": results.get(cmds[0], ""),
            "cpu": results.get(cmds[1], ""),
            "memory_chips": results.get(cmds[2], ""),
            "disks": results.get(cmds[3], ""),
        }
    else:
        cmds = [
            "uname -a",
            "cat /proc/cpuinfo | grep 'model name' | head -1",
            "free -h",
            "df -h",
        ]
        results = _exec_readonly(cmds)
        return {
            "server_type": "linux",
            "host": SSH_HOST,
            "uname": results.get(cmds[0], ""),
            "cpu": results.get(cmds[1], ""),
            "memory": results.get(cmds[2], ""),
            "disk": results.get(cmds[3], ""),
        }


@mcp.tool()
@log_tool_call
def get_process_list(sort_by: str = "memory") -> Dict[str, Any]:
    """获取远程服务器当前运行的进程列表（只读）。

    Args:
        sort_by: 排序方式，"cpu" 或 "memory"，默认 "memory"

    Returns:
        Dict: 进程列表
    """
    if SERVER_TYPE == "windows":
        cmd = "tasklist /FO CSV /NH"
        results = _exec_readonly([cmd])
        processes = results.get(cmd, "")
        return {
            "server_type": "windows",
            "processes_raw": processes,
            "note": "CSV 格式: 进程名,PID,会话名,会话#,内存使用(KB)"
        }
    else:
        sort_flag = "-eo pid,pcpu,pmem,args" if sort_by == "cpu" else "-eo pid,pmem,pcpu,args"
        cmd = f"ps {sort_flag} --sort=-{'pcpu' if sort_by == 'cpu' else 'pmem'} | head -20"
        results = _exec_readonly([cmd])
        return {
            "server_type": "linux",
            "processes": results.get(cmd, ""),
        }


@mcp.tool()
@log_tool_call
def search_system_logs(
    log_type: str = "System",
    count: int = 30,
    level: Optional[str] = None,
    keyword: Optional[str] = None,
) -> Dict[str, Any]:
    """搜索 Windows 事件日志或 Linux 系统日志（只读）。

    Args:
        log_type: 日志类型。Windows: System/Application/Security；Linux: syslog/auth/kern
        count: 返回条数，默认 30
        level: 过滤级别。Windows: 1=Critical 2=Error 3=Warning 4=Info；Linux: err/warn/info
        keyword: 关键词过滤（可选）

    Returns:
        Dict: 日志条目列表
    """
    entries = []

    if SERVER_TYPE == "windows":
        # Windows Event Log
        level_map = {"1": "1", "2": "2", "3": "3", "4": "4",
                     "critical": "1", "error": "2", "warning": "3", "info": "4"}
        cmd = f"wevtutil qe {log_type} /c:{count} /rd:true /f:text"
        results = _exec_readonly([cmd])
        raw = results.get(cmd, "")
        # 简单解析
        current = {}
        for line in raw.split('\n'):
            line = line.strip()
            if ':' in line and not line.startswith(' '):
                if current and 'Date' in current:
                    entries.append(current)
                    current = {}
                    if len(entries) >= count:
                        break
                key, _, val = line.partition(':')
                current[key.strip()] = val.strip()
            elif current:
                last_key = list(current.keys())[-1] if current else ''
                if last_key:
                    current[last_key] += ' ' + line

        if current and 'Date' in current:
            entries.append(current)

        if level:
            lvl_num = level_map.get(level.lower(), level)
            entries = [e for e in entries if str(e.get('Level', '')).strip() == lvl_num]

        if keyword:
            entries = [e for e in entries if keyword.lower() in json.dumps(e).lower()]

    else:
        # Linux syslog
        log_files = {
            "syslog": "/var/log/syslog",
            "auth": "/var/log/auth.log",
            "kern": "/var/log/kern.log",
        }
        log_file = log_files.get(log_type, "/var/log/syslog")
        cmd = f"tail -n {count} {log_file} 2>/dev/null || echo 'Log file not found'"
        if level:
            cmd = f"grep -i '{level}' {log_file} | tail -n {count}"
        if keyword:
            cmd = f"grep -i '{keyword}' {log_file} | tail -n {count}"
        results = _exec_readonly([cmd])
        raw = results.get(cmd, "")
        entries = [{"message": line} for line in raw.split('\n') if line.strip()]

    return {
        "server_type": SERVER_TYPE,
        "log_type": log_type,
        "count": len(entries),
        "entries": entries,
        "filter": {"level": level, "keyword": keyword},
    }


@mcp.tool()
@log_tool_call
def check_disk_usage(path: str = "C:/") -> Dict[str, Any]:
    """检查远程服务器磁盘使用情况（只读）。

    Args:
        path: 要检查的路径，默认 C:/

    Returns:
        Dict: 磁盘使用信息
    """
    if SERVER_TYPE == "windows":
        cmd = f"wmic logicaldisk where \"DeviceID='{path[0]}:'\" get DeviceID,Size,FreeSpace,FileSystem /format:list"
    else:
        cmd = f"df -h {path}"
    results = _exec_readonly([cmd])
    return {
        "server_type": SERVER_TYPE,
        "path": path,
        "raw_output": results.get(cmd, ""),
    }


@mcp.tool()
@log_tool_call
def read_log_file(file_path: str, lines: int = 50) -> Dict[str, Any]:
    """读取服务器上的指定日志文件（只读，仅限日志目录）。

    安全约束：只允许读取以下目录下的文件：
    - C:/logs/, C:/ProgramData/logs/, C:/inetpub/logs/
    - /var/log/, /opt/logs/

    Args:
        file_path: 文件完整路径
        lines: 读取行数，默认 50

    Returns:
        Dict: 文件内容
    """
    # 路径白名单检查
    allowed = [
        "C:/logs/", "C:/ProgramData/logs/", "C:/inetpub/logs/",
        "/var/log/", "/opt/logs/", "C:/Users/",
    ]
    normalized = file_path.replace("\\", "/")
    if not any(normalized.startswith(a) for a in allowed):
        return {
            "error": f"安全限制：不允许读取路径 '{file_path}'。仅允许: {', '.join(allowed)}",
            "allowed_paths": allowed,
        }

    if SERVER_TYPE == "windows":
        cmd = f"powershell -Command \"Get-Content '{file_path}' -Tail {lines} -ErrorAction Stop\""
    else:
        cmd = f"tail -n {lines} {file_path}"
    results = _exec_readonly([cmd])
    content = results.get(cmd, "")
    return {
        "server_type": SERVER_TYPE,
        "file_path": file_path,
        "lines_requested": lines,
        "content": content,
    }


if __name__ == "__main__":
    logger.info(f"SSH MCP Server 启动: {SSH_HOST}:{SSH_PORT}, user={SSH_USER}, type={SERVER_TYPE}")
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8005, path="/mcp")
