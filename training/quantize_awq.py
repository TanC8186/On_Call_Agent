"""AWQ 量化驱动（在 WSL 的 vLLM 环境里跑，不在 .venv-train）。

为什么不用 `auto_round` 命令行：这个环境里 `datasets 4.0.0` 会在张量化时
`from torchvision.io import VideoReader`，而 torchvision 0.28 已经删掉了这个符号，
CLI 又没有地方关掉它。驱动脚本在导入 datasets 之前把 `TORCHVISION_AVAILABLE`
置为 False，绕开这条根本没被用到的分支（我们只喂 token id，没有视频）。

用法（WSL）::

    ~/vllm-venv/bin/python training/quantize_awq.py \
        --model ~/work/qwen15-v8-merged \
        --dataset ~/work/calib_awq_128.jsonl \
        --output-dir ~/work/qwen15-v8-awq --nsamples 8     # 小样本验证
    ~/vllm-venv/bin/python training/quantize_awq.py --nsamples 128   # 全量

产物是 AutoAWQ 格式（`quant_method: awq`, `version: gemm`），
`vllm serve <dir> --quantization awq` 直接加载。
"""

from __future__ import annotations

import argparse
import os
import sys

# 必须在 import datasets 之前生效：datasets 在 import 时读这两个开关。
import datasets.config as _datasets_config  # noqa: E402

_datasets_config.TORCHVISION_AVAILABLE = False

import torch  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.path.expanduser("~/work/qwen15-v8-merged"))
    parser.add_argument("--dataset", default=os.path.expanduser("~/work/calib_awq_128.jsonl"))
    parser.add_argument("--output-dir", default=os.path.expanduser("~/work/qwen15-v8-awq"))
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--iters", type=int, default=0, help="0 = 纯 AWQ（RTN 收尾，不做 SignRound 迭代）")
    parser.add_argument(
        "--algorithm",
        default="awq",
        help="awq = AWQ 平滑 + RTN；auto_round = SignRound 迭代（对照：同样是 4-bit，但算法不同）",
    )
    parser.add_argument("--num-hidden-layers", type=int, default=-1)
    parser.add_argument("--format", default="auto_awq")
    args = parser.parse_args()

    if args.num_hidden_layers > 0:
        from auto_round import envs

        envs.set_config(AR_DEBUG_LAYER_NUM=args.num_hidden_layers)

    from auto_round import AutoRound

    print(
        {
            "torch": torch.__version__,
            "cuda": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "model": args.model,
            "dataset": args.dataset,
            "nsamples": args.nsamples,
            "seqlen": args.seqlen,
            "bits": args.bits,
            "group_size": args.group_size,
            "iters": args.iters,
            "algorithm": args.algorithm,
            "format": args.format,
        },
        flush=True,
    )

    quantizer = AutoRound(
        model=args.model,
        scheme="W4A16" if args.bits == 4 else f"W{args.bits}A16",
        bits=args.bits,
        group_size=args.group_size,
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
