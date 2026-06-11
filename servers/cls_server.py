"""腾讯云 CLS (Cloud Log Service) MCP Server

接入真实腾讯云 CLS API，提供日志查询、检索和分析功能。
"""

import os
import sys
import logging
import functools
import json
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv
from fastmcp import FastMCP

# 加载 .env 配置
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_ENV_PATH)

from tencentcloud.common import credential
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.cls.v20201016 import cls_client, models as cls_models

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CLS_MCP_Server")

# ── 腾讯云凭证 ──────────────────────────────────────────────────
SECRET_ID = os.getenv("TENCENTCLOUD_SECRET_ID", "")
SECRET_KEY = os.getenv("TENCENTCLOUD_SECRET_KEY", "")
DEFAULT_REGION = os.getenv("TENCENTCLOUD_REGION", "ap-beijing")
USE_REAL_API = bool(SECRET_ID and SECRET_KEY)

if USE_REAL_API:
    _cred = credential.Credential(SECRET_ID, SECRET_KEY)
    logger.info(f"✅ CLS Server: 已加载真实腾讯云凭证，默认区域={DEFAULT_REGION}")
else:
    logger.warning("⚠️  CLS Server: 未配置腾讯云凭证，部分工具将返回模拟数据")

mcp = FastMCP("CLS")


# ── 辅助函数 ────────────────────────────────────────────────────

def _get_cls_client(region: str = None):
    """获取 CLS 客户端实例"""
    return cls_client.ClsClient(_cred, region or DEFAULT_REGION)


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


# ── 地区映射表（静态） ──────────────────────────────────────────

_REGION_MAP = {
    "北京":       {"region_code": "ap-beijing",       "region_name": "北京"},
    "上海":       {"region_code": "ap-shanghai",       "region_name": "上海"},
    "广州":       {"region_code": "ap-guangzhou",      "region_name": "广州"},
    "成都":       {"region_code": "ap-chengdu",        "region_name": "成都"},
    "重庆":       {"region_code": "ap-chongqing",      "region_name": "重庆"},
    "南京":       {"region_code": "ap-nanjing",        "region_name": "南京"},
    "深圳":       {"region_code": "ap-shenzhen",       "region_name": "深圳"},
    "香港":       {"region_code": "ap-hongkong",       "region_name": "香港"},
    "新加坡":     {"region_code": "ap-singapore",      "region_name": "新加坡"},
    "东京":       {"region_code": "ap-tokyo",          "region_name": "东京"},
    "硅谷":       {"region_code": "na-siliconvalley",  "region_name": "硅谷"},
    "弗吉尼亚":   {"region_code": "na-ashburn",        "region_name": "弗吉尼亚"},
    "法兰克福":   {"region_code": "eu-frankfurt",      "region_name": "法兰克福"},
}


# ╔══════════════════════════════════════════════════════════════════╗
# ║                        MCP 工具函数                             ║
# ╚══════════════════════════════════════════════════════════════════╝

@mcp.tool()
@log_tool_call
def get_current_timestamp() -> int:
    """获取当前时间戳（以毫秒为单位）。

    此工具用于获取标准的毫秒时间戳，可用于：
    1. 作为 search_log 的 end_time 参数（查询到现在）
    2. 计算历史时间点作为 start_time 参数

    Returns:
        int: 当前时间戳（毫秒），例如: 1708012345000

    使用示例:
        current = get_current_timestamp()
        fifteen_min_ago = current - (15 * 60 * 1000)
    """
    return int(datetime.now().timestamp() * 1000)


@mcp.tool()
@log_tool_call
def get_region_code_by_name(region_name: str) -> Dict[str, Any]:
    """根据地区名称搜索对应的地区参数。

    Args:
        region_name: 地区名称（如：北京、上海、广州等）

    Returns:
        Dict: 包含地区代码和相关信息的字典
    """
    result = _REGION_MAP.get(region_name)
    if result:
        return {**result, "available": True}
    # 也支持直接用 region_code 查找
    for v in _REGION_MAP.values():
        if v["region_code"] == region_name:
            return {**v, "available": True}
    return {
        "region_code": None,
        "region_name": region_name,
        "available": False,
        "error": f"未找到地区: {region_name}"
    }


@mcp.tool()
@log_tool_call
def get_topic_info_by_name(
    topic_name: str,
    region_code: Optional[str] = None
) -> Dict[str, Any]:
    """根据主题名称搜索相关的日志主题信息。

    优先使用真实腾讯云 CLS API；未配置凭证时回退到模拟数据。

    Args:
        topic_name: 主题名称（支持模糊搜索）
        region_code: 地区代码（可选）

    Returns:
        Dict: 包含主题信息的字典
    """
    region = region_code or DEFAULT_REGION

    if not USE_REAL_API:
        # ── 模拟数据回退 ──
        mock_topics = [
            {
                "topic_id": "topic-001",
                "topic_name": "数据同步服务日志",
                "service_name": "data-sync-service",
                "region_code": "ap-beijing",
                "create_time": "2024-01-01 10:00:00",
                "log_count": 0,
                "description": "服务应用日志"
            }
        ]
        for t in mock_topics:
            if topic_name in t["topic_name"]:
                return t
        return {"topic_id": None, "topic_name": topic_name, "error": f"未找到主题: {topic_name}"}

    # ── 真实 API ──
    try:
        client = _get_cls_client(region)
        req = cls_models.DescribeTopicsRequest()
        req.Filters = [
            cls_models.Filter()
        ]
        req.Filters[0].Key = "topicName"
        req.Filters[0].Values = [topic_name]
        resp = client.DescribeTopics(req)

        if resp.Topics:
            t = resp.Topics[0]
            return {
                "topic_id": t.TopicId,
                "topic_name": t.TopicName,
                "region_code": region,
                "create_time": t.CreateTime,
                "log_count": getattr(t, 'LogCount', 0),
                "description": getattr(t, 'Describes', ''),
            }
        return {"topic_id": None, "topic_name": topic_name, "error": f"未找到主题: {topic_name}"}

    except TencentCloudSDKException as e:
        logger.error(f"CLS API 错误: {e}")
        return {"topic_id": None, "topic_name": topic_name, "error": f"API 错误: {e.message}"}


@mcp.tool()
@log_tool_call
def search_topic_by_service_name(
    service_name: str,
    region_code: Optional[str] = None,
    fuzzy: bool = True
) -> Dict[str, Any]:
    """根据服务名称搜索相关的日志主题信息，支持模糊搜索。

    Args:
        service_name: 服务名称（必填）
        region_code: 地区代码（可选）
        fuzzy: 是否启用模糊搜索（默认 True）

    Returns:
        Dict: 搜索结果，包含 topics 列表
    """
    region = region_code or DEFAULT_REGION

    if not USE_REAL_API:
        # ── 模拟数据回退 ──
        mock_topics = [
            {
                "topic_id": "topic-001",
                "topic_name": "数据同步服务日志",
                "service_name": "data-sync-service",
                "region_code": "ap-beijing",
                "create_time": "2024-01-01 10:00:00",
                "log_count": 0,
                "description": "数据同步服务的应用日志"
            },
            {
                "topic_id": "topic-002",
                "topic_name": "数据同步服务错误日志",
                "service_name": "data-sync-service",
                "region_code": "ap-beijing",
                "create_time": "2024-01-01 10:00:00",
                "log_count": 0,
                "description": "数据同步服务的错误日志"
            },
            {
                "topic_id": "topic-003",
                "topic_name": "API网关服务日志",
                "service_name": "api-gateway-service",
                "region_code": "ap-shanghai",
                "create_time": "2024-01-01 10:00:00",
                "log_count": 0,
                "description": "API网关服务日志"
            }
        ]
        matched = []
        for t in mock_topics:
            if region_code and t["region_code"] != region_code:
                continue
            sn = t.get("service_name", "")
            if fuzzy:
                if service_name.lower() in sn.lower() or sn.lower() in service_name.lower():
                    matched.append(t)
            else:
                if sn == service_name:
                    matched.append(t)
        return {
            "total": len(matched),
            "topics": matched,
            "query": {"service_name": service_name, "region_code": region_code, "fuzzy": fuzzy},
            "message": f"找到 {len(matched)} 个匹配的日志主题" if matched else f"未找到服务 '{service_name}' 的日志主题"
        }

    # ── 真实 API ──
    try:
        client = _get_cls_client(region)
        req = cls_models.DescribeTopicsRequest()

        # 使用 topicName 过滤（腾讯云 CLS 没有 service_name 字段，
        # 通常通过 topic 命名规范来区分服务，如 "{service_name}-log"）
        if fuzzy:
            req.Filters = [
                cls_models.Filter()
            ]
            req.Filters[0].Key = "topicName"
            req.Filters[0].Values = [service_name]

        resp = client.DescribeTopics(req)
        topics = []
        for t in resp.Topics:
            topic_name_lower = (t.TopicName or "").lower()
            svc_lower = service_name.lower()

            # 软件层面再做一次模糊/精确匹配
            if fuzzy:
                if svc_lower not in topic_name_lower and topic_name_lower not in svc_lower:
                    continue
            else:
                if topic_name_lower != svc_lower:
                    continue

            topics.append({
                "topic_id": t.TopicId,
                "topic_name": t.TopicName,
                "service_name": service_name,
                "region_code": region,
                "create_time": t.CreateTime,
                "log_count": getattr(t, 'LogCount', 0),
                "description": getattr(t, 'Describes', ''),
            })

        return {
            "total": len(topics),
            "topics": topics,
            "query": {"service_name": service_name, "region_code": region, "fuzzy": fuzzy},
            "message": f"找到 {len(topics)} 个匹配的日志主题" if topics else f"未找到服务 '{service_name}' 的日志主题"
        }

    except TencentCloudSDKException as e:
        logger.error(f"CLS API 错误: {e}")
        return {"total": 0, "topics": [], "error": f"API 错误: {e.message}"}


@mcp.tool()
@log_tool_call
def search_log(
    topic_id: str,
    start_time: int,
    end_time: int,
    query: Optional[str] = None,
    limit: int = 100,
    region_code: Optional[str] = None
) -> Dict[str, Any]:
    """基于提供的查询参数搜索日志。

    优先使用真实腾讯云 CLS SearchLog API；未配置凭证时回退到模拟数据。

    Args:
        topic_id: 主题ID（必填）
        start_time: 开始时间戳，单位为毫秒（必填，int类型）
        end_time: 结束时间戳，单位为毫秒（必填，int类型）
        query: 查询语句（可选，CLS 检索语法）
            示例: "level:ERROR" 或 "message:timeout"
        limit: 返回结果数量限制（默认100）
        region_code: 地区代码（可选）

    Returns:
        Dict: 搜索结果，包含 logs 列表
    """
    region = region_code or DEFAULT_REGION

    if not USE_REAL_API:
        # ── 模拟数据回退 ──
        if topic_id == "topic-001":
            logs = []
            current_time_ms = start_time
            count = 0
            max_logs_by_time = int((end_time - start_time) / (60 * 1000)) + 1
            actual_limit = min(limit, max_logs_by_time)
            while current_time_ms <= end_time and count < actual_limit:
                log_time = datetime.fromtimestamp(current_time_ms / 1000)
                time_str = log_time.strftime("%Y-%m-%d %H:%M:%S")
                logs.append({
                    "timestamp": time_str,
                    "level": "INFO",
                    "message": "正在同步元数据……"
                })
                count += 1
                current_time_ms += 60 * 1000
            return {
                "topic_id": topic_id,
                "start_time": start_time,
                "end_time": end_time,
                "query": query,
                "limit": limit,
                "total": len(logs),
                "logs": logs,
                "took_ms": 50,
                "message": f"成功查询 {len(logs)} 条应用日志（模拟数据）"
            }
        else:
            return {
                "topic_id": topic_id,
                "start_time": start_time,
                "end_time": end_time,
                "total": 0,
                "logs": [],
                "error": f"主题不存在: {topic_id}",
                "message": f"错误: 未找到主题 {topic_id}"
            }

    # ── 真实 API ──
    try:
        client = _get_cls_client(region)
        req = cls_models.SearchLogRequest()
        req.TopicId = topic_id
        req.From = start_time
        req.To = end_time
        req.Limit = limit
        if query:
            req.Query = query

        resp = client.SearchLog(req)

        logs = []
        if resp.Results:
            for r in resp.Results:
                try:
                    log_entry = json.loads(r.LogJson) if r.LogJson else {}
                except json.JSONDecodeError:
                    log_entry = {"raw": r.LogJson or ""}
                log_entry.setdefault("topic_id", r.TopicId)
                log_entry.setdefault("timestamp",
                    datetime.fromtimestamp(r.Time / 1000).strftime("%Y-%m-%d %H:%M:%S") if r.Time else "")
                logs.append(log_entry)

        return {
            "topic_id": topic_id,
            "start_time": start_time,
            "end_time": end_time,
            "query": query or "",
            "limit": limit,
            "total": len(logs),
            "logs": logs,
            "took_ms": getattr(resp, 'CostTime', 0),
            "list_over": resp.ListOver if hasattr(resp, 'ListOver') else True,
            "message": f"成功查询 {len(logs)} 条日志"
        }

    except TencentCloudSDKException as e:
        logger.error(f"CLS SearchLog API 错误: {e}")
        return {
            "topic_id": topic_id,
            "start_time": start_time,
            "end_time": end_time,
            "query": query,
            "total": 0,
            "logs": [],
            "error": f"API 错误: {e.message}",
            "message": f"查询失败: {e.message}"
        }


@mcp.tool()
@log_tool_call
def search_service_logs(
    service_name: str,
    log_level: Optional[str] = None,
    keyword: Optional[str] = None,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
    limit: int = 100,
    region_code: Optional[str] = None
) -> Dict[str, Any]:
    """根据服务名称搜索日志，支持级别筛选和关键词搜索。

    这是一个高级封装：自动根据 service_name 查找 topic，再执行日志搜索。

    Args:
        service_name: 服务名称（必填）
        log_level: 日志级别过滤（可选）如 ERROR, WARN, INFO
        keyword: 关键词搜索（可选）
        start_time: 开始时间戳毫秒（可选，默认15分钟前）
        end_time: 结束时间戳毫秒（可选，默认当前时间）
        limit: 返回条数（默认100）
        region_code: 地区代码（可选）

    Returns:
        Dict: 搜索结果
    """
    # 自动计算时间范围
    now_ms = int(datetime.now().timestamp() * 1000)
    if end_time is None:
        end_time = now_ms
    if start_time is None:
        start_time = now_ms - 15 * 60 * 1000  # 默认 15 分钟

    # 步骤1：根据服务名查找 topic
    topic_result = search_topic_by_service_name(service_name, region_code=region_code)
    topics = topic_result.get("topics", [])
    if not topics:
        return {
            "total": 0,
            "logs": [],
            "error": f"未找到服务 '{service_name}' 的日志主题",
            "message": f"错误: 未找到服务 '{service_name}' 的日志主题"
        }

    # 步骤2：构建查询语句
    query_parts = []
    if log_level:
        query_parts.append(f"level:{log_level}")
    if keyword:
        query_parts.append(keyword)
    query_str = " AND ".join(query_parts) if query_parts else None

    # 步骤3：对每个匹配的 topic 执行搜索
    all_logs = []
    for topic in topics:
        log_result = search_log(
            topic_id=topic["topic_id"],
            start_time=start_time,
            end_time=end_time,
            query=query_str,
            limit=limit,
            region_code=region_code or topic.get("region_code")
        )
        if log_result.get("logs"):
            all_logs.extend(log_result["logs"])

    # 截断到 limit
    all_logs = all_logs[:limit]

    return {
        "total": len(all_logs),
        "logs": all_logs,
        "query": {
            "service_name": service_name,
            "log_level": log_level,
            "keyword": keyword,
            "query_string": query_str,
        },
        "time_range": {
            "start_time": start_time,
            "end_time": end_time,
        },
        "message": f"成功查询 {len(all_logs)} 条日志"
    }


@mcp.tool()
@log_tool_call
def analyze_log_pattern(
    service_name: str,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
    region_code: Optional[str] = None
) -> Dict[str, Any]:
    """分析服务的日志模式，统计各级别日志数量和常见错误。

    Args:
        service_name: 服务名称（必填）
        start_time: 开始时间戳毫秒（可选，默认30分钟前）
        end_time: 结束时间戳毫秒（可选，默认当前时间）
        region_code: 地区代码（可选）

    Returns:
        Dict: 日志分析结果
    """
    now_ms = int(datetime.now().timestamp() * 1000)
    if end_time is None:
        end_time = now_ms
    if start_time is None:
        start_time = now_ms - 30 * 60 * 1000

    # 分别查询各级别日志
    analysis = {
        "service_name": service_name,
        "time_range": {"start_time": start_time, "end_time": end_time},
        "level_summary": {},
        "top_errors": [],
        "total_count": 0,
    }

    for level in ["ERROR", "WARN", "INFO"]:
        result = search_service_logs(
            service_name=service_name,
            log_level=level,
            start_time=start_time,
            end_time=end_time,
            limit=50,
            region_code=region_code
        )
        count = result.get("total", 0)
        analysis["level_summary"][level] = count
        analysis["total_count"] += count

        if level == "ERROR" and result.get("logs"):
            # 提取错误消息模式
            error_msgs = {}
            for log in result["logs"]:
                msg = log.get("message", str(log))
                # 简化错误消息用于分组
                simplified = msg[:120]
                error_msgs[simplified] = error_msgs.get(simplified, 0) + 1
            analysis["top_errors"] = sorted(
                [{"pattern": k, "count": v} for k, v in error_msgs.items()],
                key=lambda x: x["count"], reverse=True
            )[:10]

    # 计算占比
    total = analysis["total_count"]
    if total > 0:
        analysis["level_percent"] = {
            k: round(v / total * 100, 1) for k, v in analysis["level_summary"].items()
        }

    analysis["message"] = (
        f"共分析 {total} 条日志: "
        + ", ".join(f"{k}={v}" for k, v in analysis["level_summary"].items())
    )
    return analysis


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8003, path="/mcp")
