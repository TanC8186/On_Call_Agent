"""确定性改写：把 test 集题干换一种说法，检验模型是不是在背模板。

## 为什么不用 LLM 改写

先试过让教师模型改写，两个问题：
1. 网关的内容过滤按词命中，"保留 OOM / 超时 / 低峰"这类带术语清单的指令直接 400；
   部分指令返回 200 但正文为空。178 条只成功 127 条，且失败的集中在特定模板，
   等于只在容易的样本上做了测试。
2. 不可复现：换一次温度、换一次网关版本，改写集就变了，分数无法对照。

所以改成**规则改写**：一组显式替换 + 句子重排，全部确定性（固定 seed）。
每条改写都过三道闸，任何一道不过就丢弃、换下一个组合：

    severity_of(改写后, category) == 原 gold severity     # 结论没变
    category_of(改写后) == 原 category                    # 类别没变
    数字和服务名集合是原文的超集                        # 事实没变

外加相似度上下限：太像（>0.90）测不出鲁棒性，太不像（<0.55）可能已经换题。

## 改写覆盖什么

- 语序重排（两个分句互换、主从句换序）
- 同义动词（升到/上涨至、持续高于/一直高于、开始投诉/出现投诉）
- 中英模板各自的自然变体（"Disk usage on X reached Y%" → "X disk usage stood at Y%"）
- 连接词与语气（书面 / 工单摘要 / 口语）

**不改**：规则依赖的关键词（超时、延迟、错误率、投诉、低峰、凌晨、批量任务、OOM、
不可用、5xx、正常、巡检）、数字、服务名。这些一改就是换题，不是换说法。

用法::

    .venv-train/Scripts/python.exe training/paraphrase_rules.py            # 生成
    .venv-train/Scripts/python.exe training/paraphrase_rules.py --dry-run  # 看样例
"""

from __future__ import annotations

import argparse
import difflib
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.labels import category_of, severity_of  # noqa: E402

DATA_DIR = ROOT / "training" / "data"

# 这些词一旦被换掉，规则就不再落回同一条分支 —— 它们不是"说法"，是标签的依据
# 只列"换掉就一定改变题意"的词。像 延迟→响应时间、错误率→报错比例 是同义，
# 不该拦；真正要防的是 OOM、超时、低峰这些直接决定分支的线索被删掉。
PROTECTED = (
    # 凌晨/夜间 不列进来：具体点数由"数字必须保留"兜底，
    # 而且 低峰 / 批量任务 已经足以把规则落在 P3 分支。
    "超时", "投诉", "低峰", "批量任务",
    "OOM", "oom", "不可用", "502", "503", "5xx", "健康检查",
    "无告警", "巡检", "内存", "磁盘", "CPU", "cpu", "丢包", "带宽",
)

# 通用同义替换：只动词/副词/连接，不碰上面那些词
GENERIC = [
    ("持续高于", "一直高于"),
    ("持续高于", "连续高于"),
    ("升到", "上升至"),
    ("升到", "涨至"),
    ("上升到", "升至"),
    ("涨到", "涨至"),
    ("用户开始投诉", "用户出现投诉"),
    ("用户开始投诉", "用户反馈变差并投诉"),
    ("用户投诉增加", "投诉量明显增加"),
    ("错误率", "报错比例"),
    ("P95 延迟", "P95 响应时间"),
    ("系统日志出现", "系统日志里有"),
    ("系统日志中出现", "系统日志里有"),
    ("写入失败", "写盘失败"),
    ("疑似", "可能是"),
    ("部分实例", "一部分实例"),
    ("进程仍在但端口无响应", "进程还在，但端口已经不应答"),
    ("连续失败", "接连失败"),
    ("占用最高", "占用最大"),
    ("增长过快", "增长偏快"),
    ("未生效", "没能生效"),
    ("堆积", "积压"),
    ("重传率", "重发比例"),
    ("打满", "跑满"),
    ("间歇性", "时断时续的"),
    ("负载均衡把流量打到剩余实例", "负载均衡把流量切到其余实例"),
    ("CPU 使用率在最近 10 分钟", "最近 10 分钟里 CPU 使用率"),
    ("内存使用率升到", "内存占用涨到"),
    ("内存使用率", "内存占用"),
    ("根分区使用率", "根分区占用"),
    ("磁盘使用率", "磁盘占用"),
    ("所在主机 load average 达到", "所在机器的 load average 到了"),
    ("进程列表里", "进程列表中是"),
    ("的 RSS 持续增长", "的常驻内存一直在涨"),
    ("当前内存占用", "目前的内存占用"),
    ("GC 时间变长", "GC 耗时变长"),
    ("接口超时率上升到", "接口超时比例涨到"),
    ("网络丢包率", "网络丢包比例"),
    ("下游调用 P99 延迟升到", "调用下游的 P99 延迟涨到"),
    ("出方向带宽打满", "出口带宽已跑满"),
    ("部分请求超时", "有一批请求超时"),
    ("与数据库之间出现间歇性连接超时", "连数据库时出现时断时续的超时"),
    ("健康检查连续失败", "健康检查接连失败"),
    ("网关返回 502", "网关回 502"),
    ("的 5xx 比例在 5 分钟内升到", "的 5xx 占比在 5 分钟内涨到"),
    ("巡检", "例行检查"),
    ("最近 30 分钟各项指标正常", "过去 30 分钟各项指标都正常"),
    ("请确认是否存在潜在风险", "需要确认有没有潜在风险"),
    ("低峰期巡检", "低峰时段例行检查"),
    ("日志无 ERROR", "日志里没有 ERROR"),
    ("没有活跃告警", "当前没有活跃告警"),
    ("备份文件堆积", "备份文件已经积压"),
    ("最近一次清理在", "上次清理是在"),
    ("低峰期", "低峰时段"),
    ("只有批量任务在跑", "当时正在跑的只有批量任务"),
    ("没有用户请求超时", "用户请求没有超时"),
    ("CPU 从", "CPU 由"),
    ("从 {base}% 升到", "由 {base}% 涨至"),
    ("只有指标异常", "仅有指标异常"),
]

# 中文模板的整句变体。字符级相似度对中文很敏感——只换几个词，ratio 还在 0.9 以上，
# 测不出"换说法"；所以必须有整句重写。
ZH_SENTENCE = [
    (
        r"CPU 使用率在最近 (\d+) 分钟持续高于 (\d+)%，(.+?) 的 P95 延迟从 (\d+)ms 升到 (\d+)ms，用户开始投诉。",
        "最近 {0} 分钟里 {2} 的 CPU 一直高于 {1}%，P95 响应时间也由 {3}ms 涨到 {4}ms，"
        "已经收到用户投诉。",
    ),
    (
        r"凌晨 (\d+):00 低峰期 (.+?) CPU 从 (\d+)% 升到 (\d+)%，只有批量任务在跑，没有用户请求超时。",
        "{1} 在低峰时段（{0}:00 左右）CPU 由 {2}% 涨至 {3}%；"
        "当时只有批量任务在运行，用户请求并没有超时。",
    ),
    (
        r"(.+?) 内存使用率 (\d+)%，系统日志出现 OOM killer 记录。",
        "{0} 的内存占用达到 {1}%，系统日志里能看到 OOM killer 的记录。",
    ),
    (
        r"(.+?) 所在主机 load average 达到 (\d+)，CPU 使用率 (\d+)%，进程列表里 (\w+) 占用最高。",
        "{0} 那台机器的 load average 到了 {1}，CPU 使用率是 {2}%，"
        "进程列表里占得最多的是 {3}。",
    ),
    (
        r"(.+?) 的 /var/log 目录占用增长过快，磁盘使用率 (\d+)%，logrotate 未生效。",
        "{0} 的 /var/log 目录涨得太快，磁盘使用率已经 {1}%，logrotate 没能生效。",
    ),
]


def apply_zh_sentence(text: str, rng: random.Random) -> str | None:
    candidates = []
    for pattern, template in ZH_SENTENCE:
        match = re.fullmatch(pattern, text.strip())
        if not match:
            continue
        try:
            candidates.append(template.format(*match.groups()))
        except IndexError:
            continue
    if not candidates:
        return None
    return rng.choice(candidates)


# 英文模板的变体（按整句替换，保证语法自然）
ENGLISH = [
    ("Disk usage on {svc} reached {pct}%; application logs show write failures.",
     "{svc} disk usage stood at {pct}%; write failures appear in the application logs."),
    ("Disk usage on {svc} reached {pct}%; application logs show write failures.",
     "Application logs for {svc} report write failures as disk usage hits {pct}%."),
    ("Memory alert on {svc}: usage {pct}% and rising; OOM events observed in system logs.",
     "{svc} memory alert: usage at {pct}% and climbing; system logs record OOM events."),
    ("Memory alert on {svc}: usage {pct}% and rising; OOM events observed in system logs.",
     "System logs for {svc} show OOM events; memory usage is {pct}% and still climbing."),
    ("{svc} is returning 502; health check failed 3 times in a row.",
     "502 responses are coming from {svc}; its health check failed 3 times in a row."),
    ("{svc} is returning 502; health check failed 3 times in a row.",
     "Health checks on {svc} failed 3 consecutive times, and it now returns 502."),
    ("High CPU alert on {svc}: usage stayed above {pct}% for 10 minutes; request latency increased to {peak}ms.",
     "For 10 minutes {svc} CPU stayed above {pct}%; request latency climbed to {peak}ms."),
    ("Off-peak CPU spike on {svc} at {hour}:00: usage reached {pct}%; P99 latency rose to {peak}ms.",
     "At {hour}:00 off-peak, {svc} CPU jumped to {pct}%; P99 latency climbed to {peak}ms."),
]


def split_symptom(text: str) -> tuple[str, str, str]:
    marker = "告警文本:"
    if marker not in text:
        return "", text, ""
    head, rest = text.split(marker, 1)
    if "\n" in rest:
        symptom, tail = rest.split("\n", 1)
        return f"{head}{marker}", symptom.strip(), "\n" + tail
    return f"{head}{marker}", rest.strip(), ""


def numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?%?", text))


def services(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)+", text))


def facts_preserved(original: str, rewritten: str) -> bool:
    return numbers(original) <= numbers(rewritten) and services(original) <= services(rewritten)


def protected_intact(original: str, rewritten: str) -> bool:
    """规则依赖的关键词不能消失。"""
    return all(word in rewritten for word in PROTECTED if word in original)


def swap_clauses(text: str) -> str | None:
    """把 "A，B。" 换序成 "B，A。"，只处理恰好两个分句且以中文分号/句号结尾的情况。"""
    body = text.rstrip("。")
    parts = [part for part in body.split("，") if part.strip()]
    if len(parts) != 2:
        return None
    first, second = parts
    if second.startswith("用户") or second.startswith("接口"):
        # 结论性分句前置读起来更自然，也更接近真实的另一种说法
        return f"{second}，{first}。"
    return f"{second}，{first}。"


def apply_generic(text: str, rng: random.Random, count: int = 3) -> str:
    pool = [pair for pair in GENERIC if pair[0] in text]
    rng.shuffle(pool)
    out = text
    applied = 0
    for old, new in pool:
        if applied >= count:
            break
        if old in out and new not in out:
            out = out.replace(old, new, 1)
            applied += 1
    return out


def apply_english(text: str, rng: random.Random) -> str | None:
    service_match = re.search(r"[a-z0-9]+(?:-[a-z0-9]+)+", text)
    if not service_match:
        return None
    svc = service_match.group(0)
    pct_match = re.search(r"(\d+(?:\.\d+)?)%", text)
    peak_match = re.search(r"(\d+)\s*ms", text)
    hour_match = re.search(r"(\d{1,2}):00", text)
    candidates = []
    for old, new in ENGLISH:
        try:
            rendered_old = old.format(
                svc=svc,
                pct=pct_match.group(1) if pct_match else "",
                peak=peak_match.group(1) if peak_match else "",
                hour=hour_match.group(1) if hour_match else "",
            )
        except (KeyError, IndexError):
            continue
        if rendered_old != text.strip():
            continue
        try:
            candidates.append(
                new.format(
                    svc=svc,
                    pct=pct_match.group(1) if pct_match else "",
                    peak=peak_match.group(1) if peak_match else "",
                    hour=hour_match.group(1) if hour_match else "",
                )
            )
        except (KeyError, IndexError):
            continue
    if not candidates:
        return None
    return rng.choice(candidates)


def rewrite(symptom: str, rng: random.Random) -> str | None:
    """返回一个通过全部校验的改写；都得不出就返回 None。"""
    attempts: list[str] = []

    english = apply_english(symptom, rng)
    if english:
        attempts.append(english)

    chinese = apply_zh_sentence(symptom, rng)
    if chinese:
        attempts.append(chinese)

    swapped = swap_clauses(symptom)
    if swapped:
        attempts.append(swapped)

    generic = apply_generic(symptom, rng, count=3)
    attempts.append(generic)

    if swapped:
        attempts.append(apply_generic(swapped, rng, count=1))

    # 只做单句同义替换、不动语序的版本，作为最保守的对照
    attempts.append(apply_generic(symptom, rng, count=1))

    rng.shuffle(attempts)
    for candidate in attempts:
        if candidate == symptom:
            continue
        if not facts_preserved(symptom, candidate):
            continue
        if not protected_intact(symptom, candidate):
            continue
        ratio = difflib.SequenceMatcher(None, symptom, candidate).ratio()
        if ratio > 0.90 or ratio < 0.42:
            continue
        yield_candidate = candidate
        return yield_candidate
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in (DATA_DIR / f"{args.split}.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rng = random.Random(args.seed)

    kept: list[dict] = []
    skipped: dict[str, int] = {}
    for row in rows:
        gold = json.loads(row["messages"][-1]["content"])
        prefix, symptom, suffix = split_symptom(row["messages"][1]["content"])
        new_symptom = rewrite(symptom, rng)
        if not new_symptom:
            group = row["metadata"]["template_id"].rsplit("-", 1)[0]
            skipped[group] = skipped.get(group, 0) + 1
            continue
        # 最后一道闸：改写必须落在同一条规则上
        if severity_of(new_symptom, gold["category"]) != gold["severity"]:
            group = row["metadata"]["template_id"].rsplit("-", 1)[0]
            skipped[group] = skipped.get(group, 0) + 1
            continue
        rule_category = category_of(new_symptom)
        if rule_category not in ("unknown", gold["category"]):
            group = row["metadata"]["template_id"].rsplit("-", 1)[0]
            skipped[group] = skipped.get(group, 0) + 1
            continue

        new_row = json.loads(json.dumps(row))
        new_row["messages"][1]["content"] = f"{prefix} {new_symptom}{suffix}"
        new_row["metadata"]["paraphrase_of"] = row["metadata"]["template_id"]
        new_row["metadata"]["original_symptom"] = symptom
        new_row["metadata"]["similarity"] = round(
            difflib.SequenceMatcher(None, symptom, new_symptom).ratio(), 3
        )
        kept.append(new_row)

    if args.dry_run:
        for row in kept[:6]:
            meta = row["metadata"]
            print(f"--- {meta['template_id']} (sim {meta['similarity']})")
            print("  原:", meta["original_symptom"])
            print("  改:", split_symptom(row["messages"][1]["content"])[1])
        return

    out_path = Path(args.out) if args.out else DATA_DIR / f"{args.split}_paraphrased.jsonl"
    out_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in kept),
        encoding="utf-8",
    )
    sims = sorted(row["metadata"]["similarity"] for row in kept)
    print(
        json.dumps(
            {
                "out": str(out_path),
                "n": len(kept),
                "skipped": skipped,
                "similarity_p50": sims[len(sims) // 2] if sims else None,
                "similarity_min": sims[0] if sims else None,
                "similarity_max": sims[-1] if sims else None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
