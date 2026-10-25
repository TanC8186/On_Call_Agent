"""确定性 severity 规则。

教师模型给 severity 打标签时前后不一致：同一句"磁盘 92%，写入失败"有时 P1 有时 P2，
同一句"内存 92%，OOM"有时 P2 有时 P3。这让模型无法从文本中学到边界。

这里改成**用规则生成标签**，模型只学"从文本映射到这个规则"，标签噪声为零：

    P1  服务不可用 / 5xx / 健康检查失败；磁盘 ≥95%；内存 ≥95% 且有 OOM
    P2  有用户影响（延迟、超时、错误率、投诉）但服务仍可用

**否定句不算影响**："没有用户请求超时""无错误率上升""no user-facing timeouts"
指的是*没有*影响，不能因为句子里出现"超时"二字就判 P2。
2026-10-24 之前这条规则没有处理否定，把 test 的 `cpu-3` 整类标成了 P2。
    P3  只有指标异常，没有用户影响；或明确处于低峰/夜间
    P4  明确正常（巡检 / 无告警）

规则必须和 `app/agent/aiops/local_prompt.py` 里的提示词逐条对应，改一处要同步另一处。
"""

from __future__ import annotations

import re

_CATEGORY_KEYS = {
    "cpu": ["cpu", "load average"],
    "memory": ["内存", "memory", "oom"],
    "disk": ["磁盘", "disk", "no space left", "分区"],
    "network": ["网络", "丢包", "带宽", "network", "packet loss"],
    "service": ["不可用", "502", "503", "5xx", "健康检查", "unavailable", "无响应"],
    "healthy": ["正常", "无告警", "巡检", "healthy"],
}

_UNAVAILABLE = ("不可用", "502", "503", "5xx", "健康检查失败", "无法访问", "unavailable")
_IMPACT = (
    "延迟", "超时", "错误率", "投诉", "timeout", "latency", "用户受影响", "写入失败",
    "write failures", "write failure", "error rate",
)
_LOW_TRAFFIC = ("低峰", "凌晨", "夜间", "批量任务")
_HEALTHY = ("各项指标正常", "指标正常", "巡检正常", "无告警", "均正常", "无 error", "无error")

_IMPACT_RE = "|".join(re.escape(key) for key in _IMPACT)

# 否定前缀 + 最多 10 个字内出现影响词 -> 整段视为"没有影响"。
# 用 16 是量出来的：英文 "no user-facing timeouts" 里否定词到影响词隔 13 个字符，
# 再宽就会把"没有 OOM 记录，但接口超时"这类转折句一起吃掉（逗号/句号是硬边界）。
_NEGATED_IMPACT = re.compile(
    r"(?:没有|并未|未出现|未见|不存在|无任何|无|no|not|without)[^。；;，,\n]{0,16}?(?:"
    + _IMPACT_RE
    + r")",
    re.IGNORECASE,
)


_NEGATED_OOM = re.compile(
    r"(?:没有|并未|未出现|未见|不存在|尚未出现|尚未|无|no|without)[^。；;，,\n]{0,12}?oom",
    re.IGNORECASE,
)


def has_oom(text: str) -> bool:
    """文本里是否有**未被否定**的 OOM。

    "尚未出现 OOM" -> False；"OOM killer 记录" -> True。
    """
    if "oom" not in text.lower():
        return False
    return _NEGATED_OOM.search(text) is None


def has_user_impact(text: str) -> bool:
    """文本里是否存在**未被否定**的用户影响。

    "没有用户请求超时" -> False；"部分请求超时" -> True。
    """
    stripped = _NEGATED_IMPACT.sub(" ", text)
    return any(key in stripped.lower() for key in _IMPACT)


def _percent(text: str) -> float | None:
    numbers = [float(m) for m in re.findall(r"(\d+(?:\.\d+)?)\s*%", text)]
    return max(numbers) if numbers else None


def severity_of(text: str, category: str) -> str:
    """按文本和类别给出确定性 severity。"""
    lowered = text.lower()
    pct = _percent(text)

    if any(key in lowered for key in _UNAVAILABLE):
        return "P1"
    if category == "disk" and pct is not None and pct >= 95:
        return "P1"
    if category == "memory" and pct is not None and pct >= 95 and has_oom(text):
        return "P1"
    # 先判“正常”，否则 healthy 模板里的“低峰期巡检”会被下面的 P3 规则抢走
    if any(key in lowered for key in _HEALTHY):
        return "P4"
    if category == "healthy":
        return "P4"
    # OOM 本身就是用户影响，不能因为它出现在低峰就降级
    if has_user_impact(text) or has_oom(text):
        return "P2"
    if any(key in lowered for key in _LOW_TRAFFIC):
        return "P3"
    return "P3"


def category_of(text: str) -> str:
    """从文本推断类别。

    先判 healthy：巡检文本里通常也会出现 CPU/内存/磁盘字样，
    按字典顺序会被误判成 cpu。
    """
    lowered = text.lower()
    if any(key in lowered for key in _HEALTHY):
        return "healthy"
    for category, keys in _CATEGORY_KEYS.items():
        if category == "healthy":
            continue
        if any(key in lowered for key in keys):
            return category
    return "unknown"
