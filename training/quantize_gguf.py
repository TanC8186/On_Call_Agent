"""用 AutoRound 直接产出 GGUF（经过 SignRound，而不是 llama-quantize 的纯 RTN）。

## 为什么

老的 GGUF 是 `llama-quantize Q4_K_M` 产出的，本质是逐 block 的 RTN：
没有校准数据、没有误差补偿。同一条 test 上它的 schema 只有 0.664
（反复编造 `query_service_logs`），而同样 4-bit 的 SignRound AWQ 是 1.000。

AutoRound 支持 `--format gguf:q4_k_m` 直接导出 GGUF：先做 SignRound 的逐 block
误差补偿，再按 GGUF 的 k-quant 打包。这样 CPU 端也能拿到同样的权重质量。

## 用法（WSL）

    ~/vllm-venv/bin/python training/quantize_gguf.py \\
        --model ~/work/qwen15-v9-merged \\
        --dataset ~/work/calib_awq_128.jsonl \\
        --output-dir ~/work/qwen15-v9-gguf --nsamples 128 --iters 200
"""

from __future__ import annotations

import argparse
import os
import sys

import datasets.config as _datasets_config

_datasets_config.TORCHVISION_AVAILABLE = False

import torch  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.path.expanduser("~/work/qwen15-v9-merged"))
    parser.add_argument("--dataset", default=os.path.expanduser("~/work/calib_awq_128.jsonl"))
    parser.add_argument("--output-dir", default=os.path.expanduser("~/work/qwen15-v9-gguf"))
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--algorithm", default="auto_round")
    parser.add_argument("--format", default="gguf:q4_k_m", help="gguf:q4_k_m / gguf:q5_k_m / gguf:q8_0")
    args = parser.parse_args()

    from auto_round import AutoRound

    print(
        {
            "torch": torch.__version__,
            "cuda": torch.cuda.is_available(),
            "model": args.model,
            "dataset": args.dataset,
            "nsamples": args.nsamples,
            "seqlen": args.seqlen,
            "iters": args.iters,
            "format": args.format,
        },
        flush=True,
    )

    quantizer = AutoRound(
        model=args.model,
        scheme="W4A16" if "q4" in args.format else "W5A16",
        dataset=args.dataset,
        seqlen=args.seqlen,
        nsamples=args.nsamples,
        batch_size=args.batch_size,
        iters=args.iters,
        algorithm=args.algorithm,
        low_gpu_mem_usage=True,
        device_map="0" if torch.cuda.is_available() else "cpu",
        enable_torch_compile=False,
    )
    quantizer.quantize_and_save(output_dir=args.output_dir, format=args.format)
    print(f"saved -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
