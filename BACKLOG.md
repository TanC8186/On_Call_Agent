# BACKLOG

按"为什么现在不做"排序。每一条都写清楚：做了什么判断、卡在哪里、下一步的条件。

## 训练 / 端侧

- **720 条数据的重训已完成（v4）**：工具目录压成"名字 + 一句描述"，并把显式工具名列表写进 system prompt，
  序列长度 p50 从 1884 降到 1326。带参数签名的版本在 8GB 上仍会 OOM。
- **v5/v6/v7 都没跑完，已由 v8 取代**：v5 在 47/120 步 OOM；v6 撞上
  `CUDA error: misaligned address`；v7 的 139 条评测里 30 条编造了 `query_service_health`。
  根因是 bf16 matmul 在本机驱动上不稳定（Windows 事件日志有 nvlddmkm 153）。
  v8 改用 fp16 后训练正常完成。
- **Qwen2.5-0.5B-Instruct 没跑通**：lr=2e-4 发散（loss 79k 后归零，输出 `!!!!`），
  lr=1e-4 停在 9/40 步。需要 LR sweep + 梯度裁剪 + r=8，或从 1.5B 蒸馏。
  当前端侧方案是 **1.5B LoRA v9 + 4-bit**；量化代价见 `training/RESULTS.md`（AWQ/SignRound 无损，CPU 端 GGUF 的 severity 掉到 0.787）。
- **AWQ / SignRound 量化已做（10-24）**：`autoawq` 归档且 Windows 缺编译链，改用 Intel `auto-round`。
  v8 上：纯 AWQ（`iters=0`）不可用，`cpu-3` 一类 17–18/30 条 JSON 崩；SignRound（`iters=200`）可用。
  v9 上：SignRound 的 4-bit 在共同 149 条上与 fp16 持平（schema/关键工具/plan_ok/severity 全 1.000）。
  细节见 `training/RESULTS.md`、`docs/ENVIRONMENT.md`。
- **AutoRound 的 GGUF 已验证可用（10-24）**：`training/quantize_gguf.py` 走
  `--format gguf:q4_k_m`，v9 上 schema 1.000（llama-quantize 的 Q4_K_M 只有 0.664）。
  需要 `LLAMA_CPP_ROOT` 指向本地 llama.cpp checkout，且 gguf 包必须是该 checkout 的
  `gguf-py`（PyPI 的 0.19.0 缺 `MODEL_ARCH.DFLASH`，qwen.py 一 import 就炸）。
- **CUDA 版 llama.cpp 没编成**：本机 CUDA 11.8 不支持 MSVC 14.44，`-allow-unsupported-compiler`
  在 CMake 探测阶段不生效；Vulkan 后端缺 SDK。GGUF 基准因此是 **CPU** 的，GPU 侧用 vLLM fp16 对照。
- **AutoDL 实例当前不可用**：容器内 MUSA 驱动库是 0 字节，`musaInfo` 和 `import torch` 都失败，
  `/dev` 下无 GPU 节点。需要在控制台重挂驱动后才能作为训练算力。

## Agent

- **Checkpointer 仍是 MemorySaver。** 多 worker / 重启恢复需要 Postgres 或 Redis，属于部署项，不是 demo 必需。
- **RAG 只有 dense top-k。** 没有 BM25 混合、没有 rerank、没有 Recall@k 基线。加 reranker 需要先有 golden set，否则无法证明有效。
- **告警摄入是固定 prompt。** `/api/aiops` 不接受真实 alert payload；接 Alertmanager webhook 是下一步（会影响所有评测场景的构造方式）。
- **没有限流。** 单机 demo 场景下优先做对的鉴权，而不是限流。

## 评测

- **test 覆盖不全**：test 只有 cpu-0 / cpu-3 / disk-2 / memory-2 / service-2 五组，
  network 和 healthy 全在 train/val，这两类没有 held-out 数字。
- **标签与规则的一致性靠单测锁住**（`tests/test_training_schema.py::test_labels_match_dataset`）：
  10-23 的数据生成早于规则改动，19 条 `disk-2` 带着旧规则的 P3，把所有权重的 severity 一起压低 8~12 点。
  改规则后跑 `training/fix_labels.py --write`，不要手改 jsonl。

- **5 条场景太少。** 当前是为了跑通链路而不是统计显著性；扩到 30–50 条需要先有 fake MCP 平面（避免每次评测打真实云 API）。
- **groundedness 是字符串匹配。** 报告里的数字只要在工具预览里出现过就算命中，没有做数值容差和归属校验。

## 前端

- **没有做移动端深度适配。** 1180px 以下隐藏右侧检查器，760px 以下单列；够用但没有专门设计。
- **报告渲染用 `dangerouslySetInnerHTML`。** 已在 marked 的 `html` renderer 里转义标签，但没有上 DOMPurify；如果报告可能来自不可信输入，需要补。
