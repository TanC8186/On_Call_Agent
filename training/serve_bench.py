"""OpenAI-compatible 推理服务基准：TTFT / TPOT / 吞吐 / p95。

适用于 vLLM、llama.cpp server、Ollama、SGLang 的 /v1 接口。

用法:
    python training/serve_bench.py --base-url http://127.0.0.1:8000/v1 --model qwen05
    python training/serve_bench.py --base-url http://127.0.0.1:11434/v1 --model qwen2.5:0.5b
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

from openai import AsyncOpenAI

PROMPT = (
    "你是运维告警分诊模型。CPU 使用率 95%，服务 agent-test-server 延迟升高。"
    "请输出 severity 和排查计划。"
)


async def one_request(client: AsyncOpenAI, model: str, max_tokens: int) -> dict:
    start = time.perf_counter()
    first = None
    chunks = 0
    async for chunk in await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": PROMPT}],
        max_tokens=max_tokens,
        temperature=0.0,
        stream=True,
    ):
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content or ""
        if delta:
            chunks += 1
            if first is None:
                first = time.perf_counter()
    end = time.perf_counter()
    ttft = (first - start) if first else (end - start)
    total = end - start
    tpot = (total - ttft) / max(chunks - 1, 1)
    return {
        "ttft": ttft,
        "total": total,
        "tokens": chunks,
        "tpot": tpot,
        "tokens_per_s": chunks / total if total else 0,
    }


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((p / 100) * (len(ordered) - 1))))
    return ordered[index]


async def bench(base_url: str, model: str, concurrency: int, requests: int, max_tokens: int):
    client = AsyncOpenAI(base_url=base_url, api_key="EMPTY")
    semaphore = asyncio.Semaphore(concurrency)
    results = []

    async def worker():
        async with semaphore:
            results.append(await one_request(client, model, max_tokens))

    start = time.perf_counter()
    await asyncio.gather(*[worker() for _ in range(requests)])
    wall = time.perf_counter() - start

    ttfts = [r["ttft"] * 1000 for r in results]
    tpots = [r["tpot"] * 1000 for r in results]
    totals = [r["total"] for r in results]
    tokens = sum(r["tokens"] for r in results)
    return {
        "concurrency": concurrency,
        "requests": requests,
        "wall_s": round(wall, 2),
        "throughput_tokens_s": round(tokens / wall, 1),
        "ttft_p50_ms": round(statistics.median(ttfts), 1),
        "ttft_p95_ms": round(percentile(ttfts, 95), 1),
        "tpot_p50_ms": round(statistics.median(tpots), 1),
        "tpot_p95_ms": round(percentile(tpots, 95), 1),
        "latency_p50_s": round(statistics.median(totals), 2),
        "latency_p95_s": round(percentile(totals, 95), 2),
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--concurrency", default="1,4,8")
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--out", default="training/out/serve_bench.json")
    args = parser.parse_args()

    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY")
    for _ in range(args.warmup):
        try:
            await one_request(client, args.model, args.max_tokens)
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"warmup failed: {exc}") from exc

    rows = []
    for concurrency in [int(item) for item in args.concurrency.split(",")]:
        row = await bench(
            args.base_url, args.model, concurrency, args.requests, args.max_tokens
        )
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
