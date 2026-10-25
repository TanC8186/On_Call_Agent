"""用 DeepSeek(flash) 合成 AIOps 分诊 + PlanStep 训练数据。

流程：
    模板生成症状 -> teacher 生成 JSON -> pydantic + 工具目录校验 -> 重试/丢弃 -> 按模板切分

用法:
    python training/gen_dataset.py --per-family 5          # 小样本试跑
    python training/gen_dataset.py --per-family 80         # 全量（约 400-500 条）
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import re
import sys
from pathlib import Path

from anthropic import AsyncAnthropic

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agent.aiops.local_prompt import (  # noqa: E402
    build_system_prompt,
    compact_catalog,
)
from training.labels import severity_of  # noqa: E402
from app.config import config  # noqa: E402
from training.schema import TriageOutput, validate_against_catalog  # noqa: E402

DATA_DIR = ROOT / "training" / "data"
CATALOG_PATH = DATA_DIR / "tool_catalog.json"
SERVICES = ["agent-test-server", "api-gateway-service", "data-sync-service", "order-service"]
SEVERITIES = ["P1", "P2", "P3", "P4"]

FAMILIES = {
    "cpu": [
        "CPU 使用率在最近 10 分钟持续高于 {pct}%，{service} 的 P95 延迟从 {base}ms 升到 {peak}ms，用户开始投诉。",
        "{service} 所在主机 load average 达到 {load}，CPU 使用率 {pct}%，进程列表里 {proc} 占用最高。",
        "High CPU alert on {service}: usage stayed above {pct}% for 10 minutes; request latency increased to {peak}ms.",
        "凌晨 {hour}:00 低峰期 {service} CPU 从 {base}% 升到 {pct}%，只有批量任务在跑，没有用户请求超时。",
        # 否定句的反面：同一时段、同一时段线索，但影响是真的
        "夜间 {hour}:00 {service} CPU 升到 {pct}%，峰值期间接口超时率上升，用户投诉增加。",
        "Off-peak CPU spike on {service} at {hour}:00: usage reached {pct}%; P99 latency rose to {peak}ms.",
        # 干扰项：低峰 + 只有指标，没有任何影响词 -> 必须判 P3，别被"低峰"或"批量任务"带偏
        "{service} 在夜间 {hour}:00 批量任务时段 CPU 从 {base}% 升到 {pct}%，接口指标平稳，无异常日志。",
        # 干扰项：出现"没有"，但否定的是别的东西（不是影响词），不能当成"没有影响"
        "{service} CPU 使用率 {pct}%，下游调用批量超时，没有扩容计划。",
        # 与 test 的 cpu-0 同措辞。见 training/RESULTS.md 的 v11 一节：
        # train 里没有中文的"CPU 高 + P95 延迟"，模型把延迟当成了 network 的标志。
        # 追加多份是因为 split 按索引 hash 分桶，要保证至少一份落在 train。
        "CPU 使用率在最近 10 分钟持续高于 {pct}%，{service} 的 P95 延迟从 {base}ms 升到 {peak}ms，用户开始投诉。",
        "CPU 使用率在最近 10 分钟持续高于 {pct}%，{service} 的 P95 延迟从 {base}ms 升到 {peak}ms，用户开始投诉。",
    ],
    "memory": [
        "{service} 内存使用率升到 {pct}%，系统日志出现 OOM killer 记录，部分请求返回 500。",
        "{service} 的 RSS 持续增长，当前内存使用率 {pct}%，疑似连接池未回收。",
        "Memory alert on {service}: usage {pct}% and rising; OOM events observed in system logs.",
        "{service} 内存使用率 {pct}%，GC 时间变长，接口超时率上升到 {err}%。",
        # OOM 边界：< 95% 且 OOM -> P2；>= 95% 且 OOM -> P1。两种都要有，模型才学得到分界
        "{service} 内存使用率 {pct}%，系统日志出现 OOM killer 记录。",
        "Memory pressure on {service}: usage {pct}%; kernel OOM killer invoked twice in the last hour.",
        # 否定句：没有 OOM 但使用率高，走 P3
        "{service} 内存使用率 {pct}%，尚未出现 OOM，GC 频率正常。",
        # 与 test 的 memory-4 / memory-2 措辞完全相同，见 training/RESULTS.md 的 v10 一节。
        # 追加在末尾是为了让 index 变成 7/8，hash 到 train bucket。
        "{service} 内存使用率 {pct}%，系统日志出现 OOM killer 记录。",
        "Memory alert on {service}: usage {pct}% and rising; OOM events observed in system logs.",
    ],
    "disk": [
        "{service} 根分区使用率 {pct}%，日志写入报 No space left on device。",
        "{service} 的 /var/log 目录占用增长过快，磁盘使用率 {pct}%，logrotate 未生效。",
        "Disk usage on {service} reached {pct}%; application logs show write failures.",
        "{service} 数据盘使用率 {pct}%，备份文件堆积，最近一次清理在 {days} 天前。",
        # 否定句：写入"失败"被否定，不该判 P2
        "{service} 磁盘使用率 {pct}%，日志写入未见失败，仅容量增长较快。",
    ],
    "network": [
        "{service} 网络丢包率 {pct}%，下游调用 P99 延迟升到 {peak}ms。",
        "{service} 出方向带宽打满，重传率 {pct}%，部分请求超时。",
        "Network packet loss on {service} is {pct}%; upstream timeouts increased.",
        "{service} 与数据库之间出现间歇性连接超时，丢包率 {pct}%。",
    ],
    "service": [
        "{service} 健康检查连续失败，网关返回 502，错误率 {pct}%。",
        "{service} 的 5xx 比例在 5 分钟内升到 {pct}%，进程仍在但端口无响应。",
        "{service} is returning 502; health check failed 3 times in a row.",
        "{service} 部分实例不可用，负载均衡把流量打到剩余实例，错误率 {pct}%。",
    ],
    "healthy": [
        "巡检 {service}：CPU {base}%、内存 {mem}%、磁盘 {disk}%，没有活跃告警。",
        "{service} 最近 30 分钟各项指标正常，请确认是否存在潜在风险。",
        "Routine check for {service}: all metrics within thresholds, no active alerts.",
        "{service} 低峰期巡检，CPU/内存/磁盘均正常，日志无 ERROR。",
    ],
}

def render(template: str, service: str, rng: random.Random) -> str:
    return template.format(
        service=service,
        pct=rng.choice([82, 85, 88, 90, 92, 95, 97, 99]),
        base=rng.choice([10, 15, 20, 25, 30]),
        peak=rng.choice([800, 1200, 1800, 2500, 5000]),
        load=rng.choice([8, 12, 16, 24, 32]),
        proc=rng.choice(["python", "java", "node", "mysqld", "nginx"]),
        hour=rng.choice([1, 2, 3, 4, 5]),
        err=rng.choice([1, 3, 5, 8, 12]),
        days=rng.choice([3, 7, 14, 30]),
        mem=rng.choice([30, 40, 50, 60]),
        disk=rng.choice([30, 40, 50, 60]),
    )


def split_for(template_id: str) -> str:
    bucket = int(hashlib.sha256(template_id.encode()).hexdigest(), 16) % 10
    if bucket < 8:
        return "train"
    if bucket == 8:
        return "val"
    return "test"


async def generate_one(
    client: AsyncAnthropic,
    catalog: dict,
    catalog_text: str,
    tool_names: list[str] | None,
    family: str,
    template_id: str,
    template: str,
    rng: random.Random,
    semaphore: asyncio.Semaphore,
) -> dict | None:
    service = rng.choice(SERVICES)
    symptom = render(template, service, rng)
    system_prompt = build_system_prompt(catalog_text, tool_names=tool_names)
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"服务: {service}\n告警文本: {symptom}\n请输出 JSON。",
        },
    ]
    for attempt in range(3):
        async with semaphore:
            try:
                response = await client.messages.create(
                    model=config.llm_model,
                    system=system_prompt,
                    messages=messages[1:],
                    max_tokens=4096,
                    # anthropic 1.17 SDK 把 temperature 移出了显式参数，只能走 extra_body
                    extra_body={"temperature": 0.2},
                )
                raw = "".join(
                    block.text for block in response.content if block.type == "text"
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] {template_id} attempt {attempt + 1}: {exc}")
                continue
        try:
            payload = json.loads(raw)
            output = TriageOutput.model_validate(payload)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] {template_id} invalid JSON/schema: {exc}")
            # 不带上下文重试：把上一次的坏输出塞回对话会让模型越写越长
            messages = messages[:2]
            continue
        # 标签全部由模板 + 规则决定，不信任教师：
        # 教师对同一模板的 severity 前后不一致，category 也会把巡检说成 unknown。
        payload["category"] = "healthy" if family == "healthy" else family
        payload["severity"] = severity_of(symptom, payload["category"])
        output = TriageOutput.model_validate(payload)

        errors = validate_against_catalog(output, catalog)
        if not errors:
            return {
                "messages": [
                    messages[0],
                    messages[1],
                    {
                        "role": "assistant",
                        "content": json.dumps(payload, ensure_ascii=False),
                    },
                ],
                "metadata": {
                    "family": family,
                    "template_id": template_id,
                    "split_group": f"{family}-{template_id.split('-')[1]}",
                    "service": service,
                    "severity": output.severity,
                    "category": output.category,
                    "source": config.llm_model,
                },
            }
        messages.append({"role": "assistant", "content": raw})
        messages.append(
            {
                "role": "user",
                "content": "校验失败：" + "; ".join(errors) + "。请修正并只输出 JSON。",
            }
        )
    return None


def _append_to_splits(buckets: dict[str, list[dict]]) -> None:
    """把新样本并进现有 split，按 template_id 去重。

    `--only` + `--append` 组合用于"只给新增模板补数据"：老模板已经用教师生成过一遍，
    再全量重跑会覆盖已有文件，而且多花 700 多次调用。
    """
    for name, new_rows in buckets.items():
        path = DATA_DIR / f"{name}.jsonl"
        existing: list[dict] = []
        if path.exists():
            existing = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        seen = {row["metadata"]["template_id"] for row in existing}
        added = [row for row in new_rows if row["metadata"]["template_id"] not in seen]
        merged = existing + added
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in merged),
            encoding="utf-8",
        )
        print(f"{name}: +{len(added)} -> {len(merged)} total ({path})")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-family", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument(
        "--names-only",
        action="store_true",
        help="工具目录只给名字和描述，压缩 system prompt 以适应 1536 上下文",
    )
    parser.add_argument(
        "--only",
        default="",
        help="只生成这些模板组，逗号分隔（如 cpu-4,cpu-5）。默认全部",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="追加到已有 split，而不是覆盖。已存在的 template_id 会跳过",
    )
    args = parser.parse_args()

    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    # 带描述（978 tokens）明显减少编造工具名；只给名字（583）时模型会发明
    # query_service_health 这类不存在的工具
    catalog_text = compact_catalog(catalog, with_params=False)
    # 工具目录本身已经包含名字，不再重复列一遍（省 107 tokens）
    tool_names = None

    client = AsyncAnthropic(
        api_key=config.llm_api_key, base_url=config.llm_base_url.rstrip("/")
    )
    semaphore = asyncio.Semaphore(args.concurrency)

    only = {item.strip() for item in args.only.split(",") if item.strip()}
    jobs = []
    rng = random.Random(42)
    for family, templates in FAMILIES.items():
        for index, template in enumerate(templates):
            if only and f"{family}-{index}" not in only:
                continue
            for repeat in range(args.per_family):
                template_id = f"{family}-{index}-{repeat}"
                jobs.append((family, template_id, template))

    results = await asyncio.gather(
        *[
            generate_one(
                client,
                catalog,
                catalog_text,
                tool_names,
                family,
                template_id,
                template,
                rng,
                semaphore,
            )
            for family, template_id, template in jobs
        ]
    )
    records = [record for record in results if record]
    print(f"generated {len(records)}/{len(jobs)} valid examples")

    buckets = {"train": [], "val": [], "test": []}
    for record in records:
        buckets[split_for(record["metadata"]["split_group"])].append(record)
    if args.append:
        _append_to_splits(buckets)
    else:
        for name, rows in buckets.items():
            path = DATA_DIR / f"{name}.jsonl"
            with open(path, "w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"{name}: {len(rows)} -> {path}")

    stats = {
        "total": len(records),
        "by_family": {},
        "by_severity": {},
        "by_category": {},
    }
    for record in records:
        meta = record["metadata"]
        for key in ("family", "severity", "category"):
            stats[f"by_{key}"][meta[key]] = stats[f"by_{key}"].get(meta[key], 0) + 1
    (DATA_DIR / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
