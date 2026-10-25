# 实验环境

## 本机（Windows 11 + RTX 4060 Laptop 8GB）

| 组件 | 版本 | 用途 |
|---|---|---|
| Python | 3.13（应用） / 3.11 兼容声明 | FastAPI / LangGraph |
| torch | 2.6.0+cu124（`.venv-train`） | LoRA 训练 |
| transformers | 5.20.2 | 模型加载 |
| peft | 0.21.2 | LoRA |
| llama.cpp | 自建 `build-cpu`（MSVC 2022） | GGUF 量化 + CPU 推理 |
| vLLM | 0.32.0（WSL2 Ubuntu） | GPU 推理基准 |

```powershell
# 应用依赖
uv venv; uv pip install -e ".[dev]"

# 训练依赖（独立环境，避免和应用的 torch 冲突）
python -m venv .venv-train
.venv-train\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu124
.venv-train\Scripts\python.exe -m pip install transformers peft accelerate pydantic openai gguf sentencepiece
```

## AutoDL（备用算力）

实例 `connect.sha1.seetacloud.com:12792`，Ubuntu 22.04 + MUSA 3.1（摩尔线程 GPU），Python 3.10.8。

**当前不可用**：容器内 `/usr/lib/x86_64-linux-gnu/musa/libsrv_um_MUSA.so` 是 0 字节，`musaInfo`
和 `import torch` 都失败；`/dev` 下也没有 GPU 设备节点。需要先在 AutoDL 控制台重新挂载
GPU 驱动（或换一个 GPU 型号的实例）才能跑训练。

恢复后可复用 `training/` 下全部脚本：

```bash
scp -P 12792 -r training/ root@connect.sha1.seetacloud.com:/root/
ssh -p 12792 root@connect.sha1.seetacloud.com
cd /root && pip install -r <(echo "torch transformers peft accelerate pydantic openai")
python training/gen_dataset.py --per-family 30 --concurrency 8
python training/train_lora.py --model /root/models/Qwen2.5-1.5B-Instruct --epochs 2 --attn-only
```

## 为什么 GGUF 基准跑在 CPU

**结论：vLLM 0.32 不认 GGUF 格式，llama.cpp 的 CUDA 后端在本机编不出来。**
但 **4-bit 不必走 GGUF** —— 走 AWQ / AutoRound 就能在 GPU 上跑，这条路线已经打通（见下）。

| 路径 | 结果 | 原因 |
|---|---|---|
| `vllm serve *.gguf` | `Unknown quantization method: gguf` | vLLM 0.32 的量化白名单里没有 `gguf`，需要 `GGUF` 插件或更早版本 |
| `vllm serve` + 同名 config.json | `config file ... is not a valid JSON file` | vLLM 把 `.gguf` 当成 HF 目录解析 |
| Windows 下 CMake + CUDA 11.8 | `unsupported Microsoft Visual Studio version` | CUDA 11.8 只支持到 MSVC 2022（14.3x），本机是 14.44；`-allow-unsupported-compiler` 在 CMake 探测阶段不生效 |
| WSL 下 CMake + CUDA 13.4 | `ptxas: Unsupported .version 9.4; current version is '9.0'` | pip wheel 里的 nvcc 13.4 与 ptxas 13.0 版本错配 |
| WSL 下 Vulkan 后端 | `find_package(Vulkan)` 失败 | 缺 Vulkan SDK |

### 打通的路：AWQ / SignRound → vLLM

`AutoAWQ` 已归档且在 Windows 缺 CUDA 编译链，改用 Intel 的 `auto-round`（纯 Python wheel）：

```bash
# WSL 里装（WSL 的 pypi 直连超时，wheel 在 Windows 侧下好再拷进去）
pip install --no-deps ~/awqdl/ar/auto_round-0.16.1-py3-none-any.whl
pip install --no-deps --no-index --find-links ~/wheels_linux accelerate datasets pyarrow pandas multiprocess xxhash ...
```

踩到的三个坑，都写进 `training/quantize_awq.py` 了：

| 症状 | 原因 | 处理 |
|---|---|---|
| `cannot import name 'VideoReader' from 'torchvision.io'` | `datasets` 4.0 在 `Dataset.map` 张量化时无条件 `import` 该符号，torchvision 0.28 已删除；`auto_round` CLI 没有开关 | 驱动脚本在 `import datasets` 前把 `datasets.config.TORCHVISION_AVAILABLE = False` |
| `Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist` | `VLLM_USE_FLASHINFER_SAMPLER` 默认 1，FlashInfer 要现场 JIT 编译采样 kernel | serve 时 `VLLM_USE_FLASHINFER_SAMPLER=0` |
| CUDA graph 捕获完成后 `ptxas ... died with <Signals.SIGHUP>`（kernel `_apply_write_kernel`，`.version 9.4` vs `9.0`） | triton 3.7.1 生成的 PTX 版本高于握手时探测到的 ptxas 支持版本 | serve 加 `--enforce-eager`（本机 8GB 上本来也不该开 CUDA graph） |

量化耗时：`iters=0` 的纯 AWQ 约 6 分钟（28 层 × ~1.8 s），`iters=200` 的 SignRound 约 12 分钟（28 层 × ~23 s）。
峰值显存 1.3 GB，内存 6.2 GB —— 8GB 卡跑得动，但要先关掉别的 GPU 进程。

因此 GGUF 的数字都是 **CPU（6 线程）** 的；GPU 侧的 4-bit 对照用 AWQ / SignRound，
**同一条 test split、同一套判分脚本**，可以直接对比（数字见 `training/RESULTS.md`）。

## 已知的硬件限制

- **8GB 显存**：1.5B + 2048 上下文的 LoRA 会触发 `CUBLAS_STATUS_EXECUTION_FAILED`
  （数据 p50=1884 tokens）。当前配置降到 `--max-length 1536 --attn-only --lora-r 8` 才稳定，
  但吞吐只有约 12 s/step。
- **CUDA 11.8 + MSVC 14.44**：nvcc 拒绝编译，`-allow-unsupported-compiler` 在 CMake
  探测阶段不生效，所以 llama.cpp 只有 CPU 后端。
- **没有 CUDA Toolkit 的 WSL**：vLLM 只能用预编译 wheel，不能自建算子。
