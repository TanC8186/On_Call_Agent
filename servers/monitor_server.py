"""智能运维监控 MCP Server

接入真实腾讯云监控 (Cloud Monitor) API，提供：
- 监控数据查询（CPU、内存、磁盘、网络等）
- 进程信息查询
- 历史工单查询
- 服务信息查询

支持 CVM 实例监控和自定义指标上报。
"""

import os
import sys
import logging
import functools
import json
import random
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv
from fastmcp import FastMCP

# 加载 .env 配置
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_ENV_PATH)

from tencentcloud.common import credential
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.monitor.v20180724 import monitor_client, models as monitor_models

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Monitor_MCP_Server")

# ── 腾讯云凭证 ──────────────────────────────────────────────────
SECRET_ID = os.getenv("TENCENTCLOUD_SECRET_ID", "")
SECRET_KEY = os.getenv("TENCENTCLOUD_SECRET_KEY", "")
DEFAULT_REGION = os.getenv("TENCENTCLOUD_REGION", "ap-beijing")
USE_REAL_API = bool(SECRET_ID and SECRET_KEY)

if USE_REAL_API:
    _cred = credential.Credential(SECRET_ID, SECRET_KEY)
    logger.info(f"✅ Monitor Server: 已加载真实腾讯云凭证，默认区域={DEFAULT_REGION}")
else:
    logger.warning("⚠️  Monitor Server: 未配置腾讯云凭证，将返回模拟数据")

mcp = FastMCP("Monitor")


# ── 辅助函数 ────────────────────────────────────────────────────

def _get_monitor_client(region: str = None):
    """获取 Monitor 客户端实例"""
    return monitor_client.MonitorClient(_cred, region or DEFAULT_REGION)


def _resolve_instance(service_name: str, instance_id: str = None) -> tuple:
    """根据服务名解析实例 ID 和区域。

    如果传入的是真实实例 ID（ins-* / lhins-*），直接使用；
    如果传入的是服务名，从服务列表中查找对应的实例 ID。

    Returns:
        (instance_id, region_code) 元组
    """
    # 如果传入了真实实例 ID，直接使用
    if instance_id and (instance_id.startswith("ins-") or instance_id.startswith("lhins-")):
        return instance_id, None

    # 从服务列表中查找
    for svc in _get_service_list():
        if svc["service_name"] == service_name:
            return svc["instance_id"], svc.get("region")

    # 回退：用 service_name 作为 instance_id
    return (instance_id or service_name), None


def _get_service_list() -> list:
    """获取服务列表（供内部使用，与 list_all_services 共享数据源）"""
    return [
        {"service_name": "agent-test-server", "instance_id": "lhins-nst2h9mv", "region": "ap-guangzhou", "status": "active", "type": "lighthouse"},
    ]


def log_tool_call(func):
    """装饰器：记录工具调用的日志"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        method_name = func.__name__
        logger.info(f"=" * 80)
        logger.info(f"调用方法: {method_name}")
        if kwargs:
            try:
                params_str = json.dumps(kwargs, ensure_ascii=False, indent=2)
            except (TypeError, ValueError):
                params_str = str(kwargs)
            logger.info(f"参数信息:\n{params_str}")
        else:
            logger.info("参数信息: 无")
        try:
            result = func(*args, **kwargs)
            logger.info(f"返回状态: SUCCESS")
            if isinstance(result, dict):
                summary = {k: v if not isinstance(v, (list, dict)) else f"<{type(v).__name__} with {len(v)} items>"
                          for k, v in list(result.items())[:5]}
                logger.info(f"返回结果摘要: {json.dumps(summary, ensure_ascii=False)}")
            logger.info(f"=" * 80)
            return result
        except Exception as e:
            logger.error(f"返回状态: ERROR")
            logger.error(f"错误信息: {str(e)}")
            logger.error(f"=" * 80)
            raise
    return wrapper


def parse_time_or_default(time_str: Optional[str], default_offset_hours: int = 0) -> datetime:
    """解析时间字符串或返回默认时间"""
    if time_str:
        try:
            return datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            pass
    return datetime.now() + timedelta(hours=default_offset_hours)


def _datetime_to_iso(dt: datetime) -> str:
    """将 datetime 转换为腾讯云 API 所需的 ISO 8601 格式"""
    return dt.strftime("%Y-%m-%dT%H:%M:%S+08:00")


def _calc_statistics(values: List[float]) -> Dict[str, Any]:
    """计算数据统计信息"""
    if not values:
        return {"avg": 0, "max": 0, "min": 0, "count": 0}
    return {
        "avg": round(sum(values) / len(values), 2),
        "max": round(max(values), 2),
        "min": round(min(values), 2),
        "p95": round(sorted(values)[int(len(values) * 0.95)] if len(values) > 1 else max(values), 2),
        "count": len(values),
    }


# ── 模拟数据生成（真实 API 不可用时的回退） ─────────────────────

def _mock_metric_data(
    service_name: str,
    metric_name: str,
    start_dt: datetime,
    end_dt: datetime,
    interval_minutes: int,
    base_value: float,
    max_value: float,
    unit: str = "%"
) -> Dict[str, Any]:
    """生成模拟监控数据"""
    data_points = []
    current_time = start_dt
    time_index = 0

    while current_time <= end_dt:
        if time_index < 3:
            value = base_value + (time_index * 1.0)
        else:
            growth = (time_index - 2) * ((max_value - base_value) / 10)
            value = min(base_value + growth, max_value)
        value = round(value + random.uniform(-2, 2), 1)
        value = max(0, min(100, value))

        data_points.append({
            "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "value": value,
            "unit": unit,
        })
        current_time += timedelta(minutes=interval_minutes)
        time_index += 1

    values = [d["value"] for d in data_points]
    stats = _calc_statistics(values)

    threshold = 80.0 if metric_name == "cpu_usage_percent" else 70.0
    alert_triggered = stats["max"] > threshold

    return {
        "service_name": service_name,
        "metric_name": metric_name,
        "interval": f"{interval_minutes}m",
        "data_points": data_points,
        "statistics": stats,
        "alert_info": {
            "triggered": alert_triggered,
            "threshold": threshold,
            "message": f"{metric_name} 超过 {threshold}% 阈值" if alert_triggered else f"{metric_name} 正常"
        },
        "source": "mock",
    }


# ── 真实腾讯云 Monitor API 查询 ─────────────────────────────────

def _get_namespace_for_instance(instance_id: str) -> str:
    """根据实例 ID 前缀自动判断监控命名空间。

    - lhins-* → QCE/LIGHTHOUSE（轻量应用服务器）
    - ins-*   → QCE/CVM（云服务器）
    """
    if instance_id.startswith("lhins-"):
        return "QCE/LIGHTHOUSE"
    return "QCE/CVM"


def _query_tencent_metric(
    namespace: str,
    metric_name: str,
    instance_ids: List[str],
    start_dt: datetime,
    end_dt: datetime,
    period: int = 60,
    region: str = None,
) -> Dict[str, Any]:
    """调用腾讯云 GetMonitorData API 查询监控指标。

    Args:
        namespace: 监控命名空间（如 QCE/CVM）
        metric_name: 指标名称（如 CpuUsage）
        instance_ids: 实例 ID 列表
        start_dt: 开始时间
        end_dt: 结束时间
        period: 数据粒度（秒），默认 60
        region: 地区

    Returns:
        Dict: API 原始响应
    """
    client = _get_monitor_client(region)
    req = monitor_models.GetMonitorDataRequest()
    req.Namespace = namespace
    req.MetricName = metric_name

    # 构造实例维度
    req.Instances = []
    for inst_id in instance_ids:
        instance = monitor_models.Instance()
        dimension = monitor_models.Dimension()
        dimension.Name = "InstanceId"
        dimension.Value = inst_id
        instance.Dimensions = [dimension]
        req.Instances.append(instance)

    req.StartTime = _datetime_to_iso(start_dt)
    req.EndTime = _datetime_to_iso(end_dt)
    req.Period = period

    resp = client.GetMonitorData(req)
    return {
        "data_points": resp.DataPoints or [],
        "period": resp.Period,
        "metric_name": resp.MetricName,
        "start_time": resp.StartTime,
        "end_time": resp.EndTime,
        "request_id": resp.RequestId,
    }


# ╔══════════════════════════════════════════════════════════════════╗
# ║                        MCP 工具函数                             ║
# ╚══════════════════════════════════════════════════════════════════╝

@mcp.tool()
@log_tool_call
def query_cpu_metrics(
    service_name: str,
    instance_id: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    interval: str = "1m",
    region_code: Optional[str] = None
) -> Dict[str, Any]:
    """查询服务的 CPU 使用率监控数据。

    优先使用真实腾讯云 Monitor API；未配置凭证时使用模拟数据。

    Args:
        service_name: 服务名称（必填）
        instance_id: CVM 实例 ID（可选，如 ins-xxxxxxxx）
                     腾讯云 API 需要实例 ID 来查询 CVM 指标。
                     如果不提供，将使用 service_name 作为维度值查询自定义指标。
        start_time: 开始时间（可选，格式 "YYYY-MM-DD HH:MM:SS"，默认1小时前）
        end_time: 结束时间（可选，格式 "YYYY-MM-DD HH:MM:SS"，默认当前时间）
        interval: 数据聚合间隔（可选，"1m"/"5m"/"1h"，默认 "1m"）
        region_code: 地区代码（可选，默认 ap-beijing）

    Returns:
        Dict: CPU 监控数据
    """
    start_dt = parse_time_or_default(start_time, default_offset_hours=-1)
    end_dt = parse_time_or_default(end_time, default_offset_hours=0)
    region = region_code or DEFAULT_REGION

    # 解析 interval
    interval_minutes = 1
    if interval.endswith('m'):
        interval_minutes = int(interval[:-1])
    elif interval.endswith('h'):
        interval_minutes = int(interval[:-1]) * 60

    if not USE_REAL_API:
        return _mock_metric_data(service_name, "cpu_usage_percent", start_dt, end_dt, interval_minutes, 10.0, 96.0)

    # ── 真实 API ──
    # 优先使用 CVM 实例 ID 查询；否则用自定义指标命名空间
    inst_id, resolved_region = _resolve_instance(service_name, instance_id)
    region = resolved_region or region
    namespace = _get_namespace_for_instance(inst_id)

    try:
        result = _query_tencent_metric(
            namespace=namespace,
            metric_name="CpuUsage",
            instance_ids=[inst_id],
            start_dt=start_dt,
            end_dt=end_dt,
            period=interval_minutes * 60,
            region=region,
        )

        data_points = []
        raw_points = result.get("data_points", [])
        if raw_points:
            for dp in raw_points[0].Values if raw_points else []:
                data_points.append({"value": dp, "unit": "%"})
        elif raw_points and hasattr(raw_points[0], 'Timestamps'):
            for ts, val in zip(raw_points[0].Timestamps, raw_points[0].Values):
                data_points.append({"timestamp": ts, "value": val, "unit": "%"})

        values = [d["value"] for d in data_points] if data_points else []
        stats = _calc_statistics(values)
        alert_triggered = stats.get("max", 0) > 80.0

        return {
            "service_name": service_name,
            "instance_id": inst_id,
            "metric_name": "cpu_usage_percent",
            "interval": interval,
            "data_points": data_points,
            "statistics": stats,
            "alert_info": {
                "triggered": alert_triggered,
                "threshold": 80.0,
                "message": "CPU 使用率超过 80%" if alert_triggered else "CPU 使用率正常"
            },
            "source": "tencent_cloud",
            "request_id": result.get("request_id"),
        }

    except TencentCloudSDKException as e:
        logger.warning(f"{namespace} 命名空间查询失败，尝试自定义指标: {e}")
        # 回退到自定义指标命名空间
        try:
            result = _query_tencent_metric(
                namespace="QCE/APP",
                metric_name="cpu_usage",
                instance_ids=[service_name],
                start_dt=start_dt,
                end_dt=end_dt,
                period=interval_minutes * 60,
                region=region,
            )
            data_points = []
            for dp in result.get("data_points", []):
                if hasattr(dp, 'Values'):
                    for v in dp.Values:
                        data_points.append({"value": v, "unit": "%"})
            values = [d["value"] for d in data_points] if data_points else []
            stats = _calc_statistics(values)
            return {
                "service_name": service_name,
                "metric_name": "cpu_usage_percent",
                "interval": interval,
                "data_points": data_points,
                "statistics": stats,
                "source": "tencent_cloud_custom",
                "request_id": result.get("request_id"),
            }
        except TencentCloudSDKException as e2:
            logger.error(f"自定义指标查询也失败: {e2}")
            return {
                "service_name": service_name,
                "metric_name": "cpu_usage_percent",
                "interval": interval,
                "data_points": [],
                "statistics": {},
                "error": f"腾讯云 API 查询失败: {e2.message}",
                "message": f"查询失败: 既未找到实例 '{inst_id}'（命名空间: {namespace}），也无自定义指标数据"
            }


@mcp.tool()
@log_tool_call
def query_memory_metrics(
    service_name: str,
    instance_id: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    interval: str = "1m",
    region_code: Optional[str] = None
) -> Dict[str, Any]:
    """查询服务的内存使用监控数据。

    优先使用真实腾讯云 Monitor API；未配置凭证时使用模拟数据。

    Args:
        service_name: 服务名称（必填）
        instance_id: CVM 实例 ID（可选，如 ins-xxxxxxxx）
        start_time: 开始时间（可选，格式 "YYYY-MM-DD HH:MM:SS"，默认1小时前）
        end_time: 结束时间（可选，格式 "YYYY-MM-DD HH:MM:SS"，默认当前时间）
        interval: 数据聚合间隔（可选，"1m"/"5m"/"1h"，默认 "1m"）
        region_code: 地区代码（可选）

    Returns:
        Dict: 内存监控数据
    """
    start_dt = parse_time_or_default(start_time, default_offset_hours=-1)
    end_dt = parse_time_or_default(end_time, default_offset_hours=0)
    region = region_code or DEFAULT_REGION

    interval_minutes = 1
    if interval.endswith('m'):
        interval_minutes = int(interval[:-1])
    elif interval.endswith('h'):
        interval_minutes = int(interval[:-1]) * 60

    if not USE_REAL_API:
        return _mock_metric_data(service_name, "memory_usage_percent", start_dt, end_dt, interval_minutes, 30.0, 85.0)

    # ── 真实 API ──
    inst_id, resolved_region = _resolve_instance(service_name, instance_id)
    region = resolved_region or region
    namespace = _get_namespace_for_instance(inst_id)

    try:
        result = _query_tencent_metric(
            namespace=namespace,
            metric_name="MemUsage",
            instance_ids=[inst_id],
            start_dt=start_dt,
            end_dt=end_dt,
            period=interval_minutes * 60,
            region=region,
        )

        data_points = []
        raw_points = result.get("data_points", [])
        if raw_points:
            for dp in raw_points[0].Values if raw_points else []:
                data_points.append({"value": dp, "unit": "%"})

        values = [d["value"] for d in data_points] if data_points else []
        stats = _calc_statistics(values)
        alert_triggered = stats.get("max", 0) > 70.0

        return {
            "service_name": service_name,
            "instance_id": inst_id,
            "metric_name": "memory_usage_percent",
            "interval": interval,
            "data_points": data_points,
            "statistics": stats,
            "alert_info": {
                "triggered": alert_triggered,
                "threshold": 70.0,
                "message": "内存使用率超过 70%" if alert_triggered else "内存使用率正常"
            },
            "source": "tencent_cloud",
            "request_id": result.get("request_id"),
        }

    except TencentCloudSDKException as e:
        logger.warning(f"{namespace} 命名空间查询失败: {e}")
        return {
            "service_name": service_name,
            "metric_name": "memory_usage_percent",
            "interval": interval,
            "data_points": [],
            "statistics": {},
            "error": f"腾讯云 API 查询失败: {e.message}",
            "message": f"查询失败: 未找到实例 '{inst_id}' 的内存监控数据"
        }


@mcp.tool()
@log_tool_call
def query_disk_metrics(
    service_name: str,
    instance_id: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    interval: str = "5m",
    region_code: Optional[str] = None
) -> Dict[str, Any]:
    """查询服务的磁盘使用监控数据。

    Args:
        service_name: 服务名称（必填）
        instance_id: CVM 实例 ID（可选）
        start_time: 开始时间（可选）
        end_time: 结束时间（可选）
        interval: 数据聚合间隔（可选，默认 "5m"）
        region_code: 地区代码（可选）

    Returns:
        Dict: 磁盘监控数据
    """
    start_dt = parse_time_or_default(start_time, default_offset_hours=-1)
    end_dt = parse_time_or_default(end_time, default_offset_hours=0)
    region = region_code or DEFAULT_REGION

    interval_minutes = 5
    if interval.endswith('m'):
        interval_minutes = int(interval[:-1])
    elif interval.endswith('h'):
        interval_minutes = int(interval[:-1]) * 60

    if not USE_REAL_API:
        return _mock_metric_data(service_name, "disk_usage_percent", start_dt, end_dt, interval_minutes, 40.0, 92.0)

    inst_id, resolved_region = _resolve_instance(service_name, instance_id)
    region = resolved_region or region
    namespace = _get_namespace_for_instance(inst_id)

    # Lighthouse 用 DiskUsage，CVM 用 CvmDiskUsage
    disk_metric = "DiskUsage" if namespace == "QCE/LIGHTHOUSE" else "CvmDiskUsage"

    try:
        result = _query_tencent_metric(
            namespace=namespace,
            metric_name=disk_metric,
            instance_ids=[inst_id],
            start_dt=start_dt,
            end_dt=end_dt,
            period=interval_minutes * 60,
            region=region,
        )

        data_points = []
        for dp in result.get("data_points", []):
            if hasattr(dp, 'Values'):
                for v in dp.Values:
                    data_points.append({"value": v, "unit": "%"})

        values = [d["value"] for d in data_points] if data_points else []
        stats = _calc_statistics(values)

        return {
            "service_name": service_name,
            "instance_id": inst_id,
            "metric_name": "disk_usage_percent",
            "interval": interval,
            "data_points": data_points,
            "statistics": stats,
            "alert_info": {
                "triggered": stats.get("max", 0) > 85.0,
                "threshold": 85.0,
                "message": "磁盘使用率超过 85%" if stats.get("max", 0) > 85.0 else "磁盘使用率正常"
            },
            "source": "tencent_cloud",
            "request_id": result.get("request_id"),
        }

    except TencentCloudSDKException as e:
        logger.error(f"磁盘指标查询失败: {e}")
        return {
            "service_name": service_name,
            "metric_name": "disk_usage_percent",
            "interval": interval,
            "data_points": [],
            "statistics": {},
            "error": f"API 错误: {e.message}",
        }


@mcp.tool()
@log_tool_call
def query_network_metrics(
    service_name: str,
    instance_id: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    interval: str = "5m",
    region_code: Optional[str] = None
) -> Dict[str, Any]:
    """查询服务的网络流量监控数据。

    Args:
        service_name: 服务名称（必填）
        instance_id: CVM 实例 ID（可选）
        start_time: 开始时间（可选）
        end_time: 结束时间（可选）
        interval: 数据聚合间隔（可选，默认 "5m"）
        region_code: 地区代码（可选）

    Returns:
        Dict: 网络监控数据（入/出流量 Mbps）
    """
    start_dt = parse_time_or_default(start_time, default_offset_hours=-1)
    end_dt = parse_time_or_default(end_time, default_offset_hours=0)
    region = region_code or DEFAULT_REGION

    interval_minutes = 5
    if interval.endswith('m'):
        interval_minutes = int(interval[:-1])
    elif interval.endswith('h'):
        interval_minutes = int(interval[:-1]) * 60

    if not USE_REAL_API:
        # 网络流量模拟
        data_points = []
        current_time = start_dt
        ti = 0
        while current_time <= end_dt:
            rx = round(50 + ti * 5 + random.uniform(-10, 10), 1)
            tx = round(30 + ti * 3 + random.uniform(-5, 5), 1)
            data_points.append({
                "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                "rx_mbps": max(0, rx),
                "tx_mbps": max(0, tx),
            })
            current_time += timedelta(minutes=interval_minutes)
            ti += 1
        return {
            "service_name": service_name,
            "metric_name": "network_traffic_mbps",
            "interval": interval,
            "data_points": data_points,
            "source": "mock",
        }

    inst_id, resolved_region = _resolve_instance(service_name, instance_id)
    region = resolved_region or region
    namespace = _get_namespace_for_instance(inst_id)

    try:
        # Lighthouse 轻量服务器不提供 Cloud Monitor 网络指标
        if namespace == "QCE/LIGHTHOUSE":
            return {
                "service_name": service_name,
                "instance_id": inst_id,
                "metric_name": "network_traffic_mbps",
                "interval": interval,
                "data_points": [],
                "statistics": {},
                "message": "Lighthouse 轻量应用服务器不提供网络流量监控指标。建议通过 SSH 登录实例使用 iftop / nethogs 查看。",
                "source": "unsupported",
            }

        # 查询外网出带宽
        result_out = _query_tencent_metric(
            namespace=namespace,
            metric_name="WanOuttraffic",
            instance_ids=[inst_id],
            start_dt=start_dt,
            end_dt=end_dt,
            period=interval_minutes * 60,
            region=region,
        )
        # 查询外网入带宽
        result_in = _query_tencent_metric(
            namespace=namespace,
            metric_name="WanIntraffic",
            instance_ids=[inst_id],
            start_dt=start_dt,
            end_dt=end_dt,
            period=interval_minutes * 60,
            region=region,
        )

        out_vals = []
        in_vals = []
        for dp in result_out.get("data_points", []):
            if hasattr(dp, 'Values'):
                out_vals = dp.Values
        for dp in result_in.get("data_points", []):
            if hasattr(dp, 'Values'):
                in_vals = dp.Values

        data_points = []
        max_len = max(len(out_vals), len(in_vals))
        for i in range(max_len):
            data_points.append({
                "rx_mbps": round(in_vals[i] / 1024 / 1024, 2) if i < len(in_vals) else 0,
                "tx_mbps": round(out_vals[i] / 1024 / 1024, 2) if i < len(out_vals) else 0,
            })

        return {
            "service_name": service_name,
            "instance_id": inst_id,
            "metric_name": "network_traffic_mbps",
            "interval": interval,
            "data_points": data_points,
            "source": "tencent_cloud",
            "request_id": result_out.get("request_id"),
        }

    except TencentCloudSDKException as e:
        logger.error(f"网络指标查询失败: {e}")
        return {
            "service_name": service_name,
            "metric_name": "network_traffic_mbps",
            "interval": interval,
            "data_points": [],
            "error": f"API 错误: {e.message}",
        }


@mcp.tool()
@log_tool_call
def query_process_list(
    service_name: str,
    instance_id: Optional[str] = None,
    region_code: Optional[str] = None
) -> Dict[str, Any]:
    """查询服务所在实例的进程列表。

    注意：腾讯云默认监控不提供进程列表功能，需要通过 TAT（自动化助手）
    或自建 Agent 获取。此工具当前返回说明信息。

    Args:
        service_name: 服务名称
        instance_id: CVM 实例 ID（可选）
        region_code: 地区代码（可选）

    Returns:
        Dict: 进程列表信息
    """
    return {
        "service_name": service_name,
        "instance_id": instance_id,
        "message": (
            "腾讯云 Monitor API 不直接提供进程列表。"
            "请通过以下方式获取：\n"
            "1. 腾讯云 TAT（自动化助手）：使用 InvokeCommand 远程执行 ps/top\n"
            "2. 自建监控 Agent：通过自定义指标上报进程信息\n"
            "3. 腾讯云 CVM 控制台 → 实例详情 → 进程监控（需安装云监控 Agent）"
        ),
        "suggested_tools": ["TAT InvokeCommand", "自定义监控 Agent"],
    }


@mcp.tool()
@log_tool_call
def search_historical_tickets(
    service_name: Optional[str] = None,
    issue_type: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    limit: int = 10
) -> Dict[str, Any]:
    """搜索历史工单。

    注意：腾讯云无统一工单 API。此工具返回模拟数据作为示例。
    实际使用时建议对接自有工单系统（如 Jira、禅道、企业微信审批）。

    Args:
        service_name: 服务名称（可选）
        issue_type: 问题类型（可选，如 cpu/memory/disk/network）
        start_time: 开始时间（可选）
        end_time: 结束时间（可选）
        limit: 返回条数（默认 10）

    Returns:
        Dict: 工单列表
    """
    mock_tickets = [
        {
            "ticket_id": "TK-20260214-001",
            "service_name": "data-sync-service",
            "issue_type": "cpu",
            "title": "数据同步服务 CPU 使用率突增",
            "severity": "high",
            "status": "resolved",
            "created_at": "2026-02-14 10:30:00",
            "resolved_at": "2026-02-14 11:45:00",
            "description": "凌晨2点 CPU 从 10% 升至 95%，持续约30分钟",
            "root_cause": "数据同步任务堆积导致 CPU 满载",
            "resolution": "增加消费者实例数，优化数据分区策略",
            "tags": ["cpu", "data-sync", "性能"],
        },
        {
            "ticket_id": "TK-20260213-002",
            "service_name": "api-gateway-service",
            "issue_type": "memory",
            "title": "API 网关内存持续增长",
            "severity": "medium",
            "status": "resolved",
            "created_at": "2026-02-13 14:20:00",
            "resolved_at": "2026-02-13 18:00:00",
            "description": "内存从 30% 增长至 85%，疑似内存泄漏",
            "root_cause": "HTTP 连接池未正确回收，导致连接对象堆积",
            "resolution": "修复连接池配置，增加 max_keepalive 限制",
            "tags": ["memory", "api-gateway", "内存泄漏"],
        },
        {
            "ticket_id": "TK-20260210-003",
            "service_name": "data-sync-service",
            "issue_type": "disk",
            "title": "数据同步服务磁盘使用率告警",
            "severity": "high",
            "status": "resolved",
            "created_at": "2026-02-10 08:00:00",
            "resolved_at": "2026-02-10 09:30:00",
            "description": "磁盘使用率超过 90%，日志文件激增",
            "root_cause": "日志轮转配置错误，debug 日志未自动清理",
            "resolution": "修复 logrotate 配置，清理历史日志",
            "tags": ["disk", "日志", "存储"],
        },
    ]

    # 筛选
    filtered = mock_tickets
    if service_name:
        filtered = [t for t in filtered if service_name.lower() in t["service_name"].lower()]
    if issue_type:
        filtered = [t for t in filtered if issue_type.lower() == t["issue_type"].lower()]
    filtered = filtered[:limit]

    return {
        "total": len(filtered),
        "tickets": filtered,
        "query": {
            "service_name": service_name,
            "issue_type": issue_type,
            "limit": limit,
        },
        "message": (
            f"找到 {len(filtered)} 条工单（模拟数据）。"
            "生产环境请对接自有工单系统 API。"
        ),
    }


@mcp.tool()
@log_tool_call
def get_service_info(
    service_name: str,
    region_code: Optional[str] = None
) -> Dict[str, Any]:
    """获取服务的基本信息和当前状态。

    整合了多种数据源（监控指标 + 日志），提供服务的全景视图。

    Args:
        service_name: 服务名称
        region_code: 地区代码（可选）

    Returns:
        Dict: 服务综合信息
    """
    # 汇总服务信息：同时查询 CPU 和 Memory 指标
    cpu_data = query_cpu_metrics(service_name, region_code=region_code)
    mem_data = query_memory_metrics(service_name, region_code=region_code)

    return {
        "service_name": service_name,
        "region": region_code or DEFAULT_REGION,
        "status": {
            "cpu": {
                "current_avg": cpu_data.get("statistics", {}).get("avg", "N/A"),
                "current_max": cpu_data.get("statistics", {}).get("max", "N/A"),
                "alert": cpu_data.get("alert_info", {}).get("triggered", False),
                "source": cpu_data.get("source", "unknown"),
            },
            "memory": {
                "current_avg": mem_data.get("statistics", {}).get("avg", "N/A"),
                "current_max": mem_data.get("statistics", {}).get("max", "N/A"),
                "alert": mem_data.get("alert_info", {}).get("triggered", False),
                "source": mem_data.get("source", "unknown"),
            },
        },
        "health": "healthy" if not (
            cpu_data.get("alert_info", {}).get("triggered", False) or
            mem_data.get("alert_info", {}).get("triggered", False)
        ) else "warning",
        "query_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


@mcp.tool()
@log_tool_call
def list_all_services(region_code: Optional[str] = None) -> Dict[str, Any]:
    """列出所有已知服务和其实例映射。

    返回服务名与 CVM 实例 ID 的对照表，方便后续查询具体指标时使用。

    Args:
        region_code: 地区代码（可选）

    Returns:
        Dict: 服务列表
    """
    services = _get_service_list()

    if region_code:
        services = [s for s in services if s["region"] == region_code]

    return {
        "total": len(services),
        "services": services,
        "region": region_code or "all",
        "message": (
            "服务列表为预配置的映射表。"
            "生产环境建议通过 CVM 标签（Tag）或命名规范自动发现服务实例。"
        ),
    }


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8004, path="/mcp")
