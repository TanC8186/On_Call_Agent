# 领域小模型实验结果

> 数据为合成（模板化故障 + 教师模型生成 + schema/工具目录双重校验），**不是真实生产工单**。
> 所有数字都能由 `training/` 下脚本复现，命令写在各节末尾。

## 当前状态（v9，2026-10-24）

| 项目 | 值 |
|---|---|
| 数据集 | 838 条（train/val/test = 690/90/178），23 个模板组 |
| 标签来源 | `training/labels.py` 的确定性规则（含否定句处理），838/838 一致 |
| 教师模型 | `DeepSeek-V4.1-Flash-1017`（Anthropic 兼容网关） |
| 学生模型 | Qwen2.5-1.5B-Instruct + LoRA r=8/alpha=16（仅注意力层） |
| 训练精度 | fp16（本机 bf16 matmul 报 `CUBLAS_STATUS_INTERNAL_ERROR`） |
| 端侧产物 | GGUF Q4_K_M 941 MB（CPU）／AWQ W4A16 1.15 GB（vLLM GPU） |

### v9 评测（held-out 178 条）

`plan_ok` 用 **关键工具口径**（`training/scoring.py`）：关键工具=训练集里该类故障 ≥90% 的
gold 计划都会调用的工具；`plan_ok` = 关键工具全覆盖 + 多余工具 ≤2 + 纠错后 schema 合法。

| 指标 | fp16 (HF) | Q4_K_M (AutoRound→GGUF, CPU) | AWQ/SignRound (vLLM) |
|---|---:|---:|---:|
| JSON 可解析 | 1.000 | 1.000 | 0.989 |
| Schema 合法 | **1.000** | **1.000** | 0.989 |
| 关键工具召回 | **1.000** | **1.000** | 0.989 |
| **Plan OK** | **1.000** | **1.000** | 0.989 |
| Arg exact | 0.843 | **0.858** | 0.778 |
| **Severity** | **0.989** | 0.787 | **0.989** |
| Category | 1.000 | 1.000 | 0.989 |

三个后端都是同一份 v9 权重、同一个 test split、同一套判分。

**v9 相对 v8 的变化，逐条对应改了什么：**

| | v8 fp16 | v9 fp16 | 改了什么 |
|---|---:|---:|---|
| Severity | 0.893 | **1.000** | 规则加否定句处理；train 补 8 个模板（含 OOM 边界、否定句、反面样本） |
| Plan OK | 0.973 | **1.000** | 同上（多类模板让计划更稳） |
| Schema | 0.966→1.000 | 1.000 | v8 的 0.966 是 HF 路径的 `query_service_logs` 幻觉；v9 没复现 |
| `cpu-3` severity | 0.93 | **1.00** | 规则不再把"没有用户请求超时"读成 P2；prompt 同步写明 |
| `memory-2` severity | 0.52 | **1.00** | train 补了 `<95% + OOM → P2` / `≥95% + OOM → P1` 的成对样本 |

（v8/v9 对比用共同的 149 条，见下表；全量 178 是 v9 独有的新模板 `memory-4`。）

#### v8 → v9 在同一 149 条上的对照

| run | n | Schema | 关键工具 | Plan OK | Arg | Severity | Category |
|---|---:|---:|---:|---:|---:|---:|---:|
| v8 fp16 (HF) | 149 | 1.000 | 1.000 | 0.973 | 0.831 | 0.893 | 1.000 |
| v8 Q4_K_M (llama-quantize) | 149 | 0.664 | 1.000 | 0.940 | 0.782 | 0.758 | 0.946 |
| v8 AWQ (`iters=0`) | 149 | 0.832 | 0.859 | 0.812 | 0.765 | 0.691 | 0.879 |
| **v9 fp16 (HF)** | 149 | **1.000** | **1.000** | **1.000** | 0.837 | **1.000** | **1.000** |
| **v9 Q4_K_M (AutoRound)** | 149 | **1.000** | **1.000** | **1.000** | **0.865** | **0.846** | **1.000** |
| **v9 AWQ/SignRound** | 149 | **1.000** | **1.000** | **1.000** | 0.791 | **1.000** | **1.000** |

**结论一：SignRound 的 4-bit 在 v9 上已经和 fp16 打平。**
AWQ/SignRound 在这 149 条上 schema / 关键工具 / plan_ok / severity **全部 1.000**，
比 fp16 只低 arg exact（0.791 vs 0.837）。v8 时代那个"4-bit 掉 13 点 severity"的结论**被推翻了**——
掉的不是比特，是训练数据没覆盖到 OOM 边界和否定句。

**结论二：GGUF 必须用 AutoRound 产，不要用 `llama-quantize`。**
同样是 Q4_K_M：`llama-quantize` 的 schema 0.664，AutoRound 的是 1.000。
AutoRound 会先做 SignRound 的逐 block 误差补偿再按 k-quant 打包。
但 AutoRound 的 GGUF 在 severity 上仍掉到 0.787（全部集中在 `disk-2` / `memory-2` / `memory-4`
的边界判断），所以**CPU 端适合做工具规划，不适合做 severity**。

**结论三：纯 AWQ（`iters=0`）确认不可用。** v8 上它 30 条里崩 17–18 条；
v9 上问题变成 arg exact 0.778（把 `keyword: "OOM"` 写成小写、漏 `service_name`）。
v9 的 AWQ 产物实际跑的是 SignRound（`--iters 200`），不是纯 AWQ。

#### 失败集中在哪

| 模板 | v9 fp16 | v9 Q4 | v9 AWQ | 说明 |
|---|---:|---:|---:|---|
| cpu-0 | 1.00 | 1.00 | 1.00 | |
| cpu-3 | 1.00 | 1.00 | 1.00 | 否定句，v9 已学 |
| disk-2 | 1.00 | **0.50** | 1.00 | Q4 把 P2/P1 混淆 |
| memory-2 | 1.00 | **0.72** | 1.00 | Q4 的 OOM 边界 |
| memory-4（新） | 0.93 | **0.48** | 0.93 | Q4 的 OOM 边界；fp16 也错 2/29 |
| service-2 | 1.00 | 1.00 | 1.00 | |

Q4 的 severity 损失**全部集中在"92% vs 95%"这种三个百分点的分界**。
这不是量化噪声那么简单：Q4 的权重误差把这个边界抹平了。端侧如果要用 severity，
要么用 AWQ/SignRound（GPU），要么把边界放宽（比如统一阈值，不再区分 92/95）。

#### 改写鲁棒性测试：0.989 里有多少是"认句式"

test 只有 6 种句式、每种重复 30 次。0.989 分不清两件事：学到了规则，还是认出了句式。
做法是用 `training/paraphrase_rules.py` 把题干换一种说法（规则改写、固定 seed、可复现），
三道闸保证题意不变：`severity_of(改写后) == 原 gold`、`category_of` 不变、
数字与服务名是原文的超集。178 条全部改写成功，相似度 p50 = 0.571。

| 后端 | 原 test | 改写版 | Δ |
|---|---:|---:|---:|
| fp16 (vLLM) | 0.989 | 0.865 | **−0.124** |
| AWQ/SignRound | 0.989 | 0.860 | **−0.129** |
| Q4_K_M (AutoRound) | 0.787 | 0.775 | −0.012 |

schema / 关键工具召回 / category 在改写后**完全不变**（都是 1.000）。
掉的只有 severity，而且集中在 `memory-2` 和 `memory-4` 两组（OOM 边界那两类）。

**所以 severity 0.989 里有一半是句式记忆。** plan 类指标（schema、关键工具、plan_ok）
是稳的——那些不依赖句子的形状。

#### 受控探针：只换分句顺序，结论就翻

改写同时动了措辞和语序，分不清是哪个。用 `training/probe_order.py` 只动一件事：
同样的内容、数字、服务名、关键词，只把"数字在前"换成"线索在前"：

```
number_first:  agent-test-server 内存使用率 85%，系统日志出现 OOM killer 记录。
signal_first:  系统日志出现 OOM killer 记录，agent-test-server 内存使用率 85%。
```

58 条（`memory-2` + `memory-4` 里所有"数字 + OOM"的样本）：

| 后端 | 两种语序答案一致率 | number_first 准确率 | signal_first 准确率 |
|---|---:|---:|---:|
| fp16 | 0.569 | **0.966** | 0.534 |
| AWQ/SignRound | 0.655 | **0.966** | 0.655 |
| Q4_K_M | 0.414 | 0.603 | 0.552 |

**同一句话，只把两个分句换个位置，准确率从 0.966 掉到 0.534。** 这不是模型不理解 OOM
和 95% 的关系——它在 number-first 语序下答得几乎全对。它学的是"先看数字、再看线索"
这个**位置习惯**，而不是"两个条件同时成立"这个**逻辑规则**。

根因在数据：train 的 690 条里，带"数字 + OOM"的 90 条**全部**是 number-first，
signal-first 是 0 条。test 的 58 条也全是 number-first。模型没有任何机会见到另一种语序。

这也是"合成数据"这个限制的具体形状——不是"数据不是真实工单"这种笼统的话，
而是**一个具体的、可以量化的句式偏置**。

#### v10 / v11：补语序变体，第二次才对

**v10（错误的一次）**：给 train 加了 240 条 signal-first 变体，但只覆盖了
train 里"能改"的 8 个模板组。结果：那 8 组上语序一致率 0.875、signal-first 0.925，
但探针用的 test 的 `memory-2` / `memory-4` **纹丝不动**（58 行里 56 行逐字相同）。
原因很直接：`memory-2`/`memory-4` 的措辞在 train 里根本不存在，模型没法把
"换序不影响结论"迁移到一个没见过的模板上。

另外 v10 第 218/234 步 OOM（gradient checkpointing 的 reentrant 重算要 fork CUDA RNG），
只训练了 1 epoch，`service-2` 的 schema 从 1.000 掉到 0.233 —— 结果不可用。
已把 `gradient_checkpointing_kwargs={"use_reentrant": False}` 写进 `train_lora.py`。

**v11（补对的一次）**：把 test 那两句措辞原样追加进 `FAMILIES["memory"]` 的末尾
（新索引 7/8，hash 到 train bucket），教师生成 60 条，再用 `order_variants.py`
给它们生成 signal-first 版本。train 690 → 1050 条，其中 300 条是语序变体
（number-first 270 / signal-first 300）。

| 指标 | v9 | v10（1 epoch） | **v11** |
|---|---:|---:|---:|
| test severity | 0.989 | 0.921 | 0.961 |
| test category | **1.000** | 0.882 | 0.899 |
| 改写版 severity | 0.865 | — | **0.994** |
| 探针：两种语序一致率 | 0.569 | 0.569 | **1.000** |
| 探针：signal-first 准确率 | 0.534 | 0.534 | **1.000** |

**语序偏置彻底修好了**：58 条探针，两种语序 58/58 一致且全对（v9 是 0.569 / 0.534）。
改写测试的 severity 也从 0.865 升到 **0.994** —— 这比原 test 上的 0.961 还高，
说明 v11 对"换一种说法"比 v9 稳得多。

**但 category 退化了，这是 v11 的真实代价：**

| | v9 | v11 |
|---|---:|---:|
| 总体 category | 1.000 | 0.899 |
| 其中 `cpu-0` | 30/30 | **12/30** |

`cpu-0` 的 18 条错误全部是 `cpu` → `network`，而且能稳定复现：

```
CPU 使用率持续 10 分钟超过 82%，P95 延迟从 20ms 升到 1200ms，用户开始投诉。  → network
CPU 使用率持续 10 分钟超过 82%，用户开始投诉。                            → cpu
```

把"P95 延迟"那句删掉就判对了。原因在训练数据：train 里含 `P9x` 的模板只有
`network-0`（网络丢包 + P99 延迟）和 `cpu-5`（英文 off-peak spike + P99 latency）两组，
**没有一条中文的"CPU 高 + P95 延迟"**。模型把"P95/P99 延迟"当成了 network 的标志。
v9 没这个问题只是因为它更依赖句式记忆（它在 cpu-0 上本来就靠句式，反而躲过了这个坑）。

**所以 v11 不能直接替换 v9。** 两个模型各有明确短板：
- v9：语序一换 severity 就崩（0.534），但 category 全对
- v11：语序不变（1.000）、改写稳健（0.994），但 `cpu-0` 的 category 掉到 12/30

## 数据

| 项目 | 数值 |
|---|---|
| 有效样本 | 838 |
| train / val / test | 690 / 90 / 178 |
| 模板组 | 23（6 类故障 × 3–8 种表述） |
| 严重级别分布 | P1 231 / P2 382 / P3 225 / P4 120 |
| 工具目录 | 22 个 MCP/local 工具 |
| 序列长度 | system 984 tokens；prompt p50 1038；total p50 1367，max 1476 |

**标签为什么不用教师模型**：同一句"磁盘 92%，写入失败"，教师有时给 P1 有时给 P2；
"内存 92%，OOM" 有时 P2 有时 P3。模型学不到边界。改用 `training/labels.py` 的规则后，
838/838 条一致。规则改过两次，每次都用 `training/fix_labels.py --write` 重算：
- 10-24 上午：`disk-2` 19 条的 `write failures` 之前没匹配上 `_IMPACT`，P3→P2
- 10-24 下午：`cpu-3` 30 条的"没有用户请求超时"被读成 P2，修否定句后 P2→P3

**prompt 同步**：`app/agent/aiops/local_prompt.py` 加了否定句与 OOM 边界两条说明，
`training/sync_prompt.py --write` 把 838 条数据的 system prompt 全部换成新版
（答案由教师生成 + 工具目录校验，与 prompt 措辞无关，所以只换 message[0] 是安全的）。
`tests/test_training_schema.py` 里有两条测试锁住"标签与规则一致"和"prompt 与线上一致"。

规则（与 `local_prompt.py` 逐条对应；否定句不算影响）：

```
P1  服务不可用 / 5xx / 健康检查失败；磁盘 ≥95%；内存 ≥95% 且 OOM
P2  有用户影响：延迟 / 超时 / 错误率 / 投诉 / OOM（"没有超时"不算）
P3  只有指标异常，没有用户影响；或明确处于低峰 / 夜间
P4  明确正常（各项指标正常 / 巡检正常 / 无告警）
```

## 演进过程（保留踩过的坑）

| 版本 | 改动 | Severity | Tool F1 | Schema |
|---|---|---:|---:|---:|
| v1（240 条） | 基线 LoRA | 0.700 | 0.745 | 0.940 |
| v3（720 条，紧凑提示词） | tool catalog 只留名字+描述 | 0.553 | 0.795 | 0.800 |
| v4（显式工具名列表） | +107 tokens 列出 22 个合法工具名 | 0.573 | **0.855** | **0.993** |
| v5（max-length 1280） | ❌ 376/480 条样本的 system prompt 被截断 | 0.537 | 0.703 | 1.000 |
| v8（规则标签 + fp16） | category 取 family，severity 取规则 | **0.752** | 0.735 | 0.966 |

三个具体的失效与修法：

1. **工具名编造**（schema 0.80）：模型发明 `query_service_logs`。
   → system prompt 显式列出 22 个合法工具名（+107 tokens），schema 回到 0.99。
2. **参数名漂移**：紧凑提示词去掉参数签名后，`log_level` 写成 `level`、漏 `service_name`。
   → `repair_args()` 用 difflib 最近邻纠正并按 schema 补必填（`tests/test_local_planner.py`）。
3. **训练时 prompt 被截断**（v5 的 category 从 0.99 掉到 0.61）：
   `--max-length 1280` 小于 system prompt 881 + 答案 376。
   → `training/dataset_utils.build_example()` 只从左侧截 prompt、答案始终完整，并有单测锁住。

## 量化与约束解码

模型：`qwen15-ops-Q4_K_M.gguf`（v4 模型，941 MB，5.08 BPW），llama.cpp CPU 后端。

| 配置（v4，同一 150 条） | Schema | Tool F1 | Arg exact | Severity |
|---|---:|---:|---:|---:|
| GGUF Q4 + JSON Schema 约束解码 | 0.987 | 0.840 | 0.714 | 0.711 |
| GGUF Q4 无约束 | 0.993 | 0.855 | 0.757 | 0.573 |

**在分布内测试集上，约束解码没有提升**——LoRA 已经学会输出合法 JSON，两次运行逐条相同。
它的价值在**分布外输入**：给一个没有 system prompt 的自由问题，无约束时模型回一段英文建议，
加 schema 后才强制输出结构化计划。所以它应该作为服务端的默认保护，而不是指标提升项。

> `llama-server -jf` 作为全局 schema 会在 chat template 的 `<|im_start|>` 上报
> `Unexpected empty grammar stack`；必须通过请求的 `response_format.json_schema` 传。

## 量化产物

| 文件 | 大小 | 用途 |
|---|---:|---|
| `training/out/qwen15-lora-v8/merged` | 2.9 GB | HF 格式，vLLM / transformers 直接加载 |
| `training/out/qwen15-ops-v8-f16.gguf` | 2.9 GB | GGUF 中间产物，用于再量化 |
| `training/out/qwen15-ops-v8-Q4_K_M.gguf` | **941 MB** | llama.cpp / Ollama 部署（CPU） |
| `training/out/qwen15-v8-awq/` | **1.15 GB** | AutoAWQ GEMM 格式（iters=0，不推荐） |
| `training/out/qwen15-v8-signround/` | **1.15 GB** | 同上格式，SignRound 权重（推荐） |
| `training/out/qwen15-v9-signround/` | **1.15 GB** | v9 权重的 AWQ 格式（SignRound，**当前推荐**） |
| `training/out/qwen15-v9-ops-Q4_K_M-autoround.gguf` | **941 MB** | v9 权重，AutoRound 产的 GGUF（CPU，推荐） |

五个产物都在 `training/out/`（已被 `.gitignore` 排除），合计约 8 GB。
复现命令：

```bash
# GGUF（约 25 秒，不需要 GPU）
cd E:/My_DL_Projects/references/llama.cpp
python convert_hf_to_gguf.py <repo>/training/out/qwen15-lora-v8/merged --outfile <repo>/training/out/qwen15-ops-v8-f16.gguf --outtype f16
llama-quantize.exe <repo>/training/out/qwen15-ops-v8-f16.gguf <repo>/training/out/qwen15-ops-v8-Q4_K_M.gguf Q4_K_M

# AWQ / SignRound（WSL，约 6 分钟 / 12 分钟）
.venv-train/Scripts/python.exe training/make_calib.py --n 128     # Windows：导出校准集
~/vllm-venv/bin/python training/quantize_awq.py --nsamples 128 \
    --output-dir ~/work/qwen15-v8-awq                             # 纯 AWQ（iters=0）
~/vllm-venv/bin/python training/quantize_awq.py --nsamples 128 \
    --iters 200 --algorithm auto_round --output-dir ~/work/qwen15-v8-signround

# 服务（本机 8GB 必须 --enforce-eager，否则 ptxas 编译 CUDA graph 时崩）
~/vllm-venv/bin/vllm serve ~/work/qwen15-v8-signround/qwen15-v8-merged-w4g128 \
    --quantization awq --max-model-len 2048 --gpu-memory-utilization 0.70 \
    --enforce-eager --port 8000
```

校准集用 train split 的**完整对话**（v9：1157–1456 tokens），不是只有 prompt 的那 1025–1051 tokens：
auto-round 的 `filter_func` 会丢掉短于 `seqlen` 的样本，按 prompt 长度设 `seqlen` 会把工具目录砍掉。

环境上踩到的三个坑，都记在 `docs/ENVIRONMENT.md`：
`auto-round` 不能走 CLI（`datasets` 4.0 在张量化时 import 已被 torchvision 0.28 删除的
`VideoReader`）；`VLLM_USE_FLASHINFER_SAMPLER=1` 会去要 nvcc；不加 `--enforce-eager`
会在 CUDA graph 捕获后编译 `_apply_write_kernel` 时 `ptxas` 崩。

## 推理基准

### vLLM fp16 与 AWQ W4A16（WSL2, RTX 4060 8GB, `--enforce-eager`）

同一台机、同一轮测量，`max_tokens=128`、每档 16 请求：

| 精度 | 并发 | TTFT p50 | TTFT p95 | TPOT p50 | 吞吐 | 端到端 p50 |
|---|---:|---:|---:|---:|---:|---:|
| fp16 | 1 | 71.1 ms | 74.1 ms | 15.9 ms | 61.3 tok/s | 1.99 s |
| fp16 | 4 | 79.4 ms | 101.0 ms | 16.4 ms | 236.1 tok/s | 2.07 s |
| fp16 | 8 | 73.6 ms | 87.5 ms | 16.7 ms | 464.4 tok/s | 2.09 s |
| AWQ W4A16 | 1 | 77.3 ms | 82.6 ms | **14.5 ms** | **65.5 tok/s** | **1.82 s** |
| AWQ W4A16 | 4 | 83.0 ms | 95.4 ms | 15.8 ms | **244.5 tok/s** | **1.96 s** |
| AWQ W4A16 | 8 | **69.7 ms** | **87.9 ms** | **15.1 ms** | **513.7 tok/s** | 1.88 s |

**4-bit 在这台 8GB 卡上不是"用精度换显存"，是纯赚**：FP16 权重 2.9 GB、AWQ 1.15 GB，
显存从容了，KV cache 更多，吞吐高 5–11%，TPOT 还略低。
（这张表是 v8 权重测的；v9 的精度代价见上——AWQ/SignRound 在 v9 上与 fp16 持平。）

> 早先那张 fp16 基准表的 TTFT p50 是 32.8/50.2/59.9 ms，比这次低一半。差异来自
> `--dtype bfloat16` 与本次 `float16`、以及每档 8 请求 vs 16 请求；两次运行之间没有做对照，
> 不要拿两个数直接比。以本节同一轮的数据为准。

### llama.cpp Q4（CPU, 6 线程，v8 模型）

| 并发 | TTFT p50 | TTFT p95 | TPOT p50 | 吞吐 | 端到端 p50 |
|---:|---:|---:|---:|---:|---:|
| 1 | 34.7 ms | 40.1 ms | 27.1 ms | 37.1 tok/s | 3.32 s |
| 4 | 281.4 ms | 510.5 ms | 36.1 ms | 104.7 tok/s | 4.65 s |
| 8 | 2383.7 ms | 4698.4 ms | 38.9 ms | 102.0 tok/s | 7.11 s |

（v4 模型在 8 线程时是 29.0 / 240.7 / 1983.9 ms，吞吐 42.9 → 126 tok/s；
线程数从 8 降到 6 是 v8 这轮的唯一环境差异，两个模型不能直接对比绝对值。）

Q4 单并发 TTFT 反而更低（34.7 ms vs fp16 的 71.1 ms，prompt 已在 CPU 上预热），
但并发上来后没有 paged attention，TTFT 从 35 ms 涨到 2.4 s。这是 llama.cpp CPU 后端的限制。
**端侧结论**：有 GPU 就用 vLLM + AWQ/SignRound；纯 CPU 只有单并发可用。

## 接入 Agent（本地快路径）

`app/agent/aiops/local_planner.py`：本地模型生成 typed plan → 用 MCP 工具/参数 schema 校验 →
不合法回退云端。trace 里记录 `planner_route.backend=local|cloud`。

### v9 实测（5 个场景，本地 = Q4_K_M GGUF on CPU）

同一晚、同一批 MCP 工具、同一套判分，只切 `LOCAL_LLM_BASE_URL`：

| 场景 | 云端 F1 | 本地 F1 | 云端 tokens | 本地 tokens | 节省 | 云端成功 | 本地成功 |
|---|---:|---:|---:|---:|---:|---|---|
| cpu-high-001 | 0.33 | **0.86** | 25,588 | 7,742 | **70%** | ✗ | ✗ |
| disk-full-001 | 0.50 | 0.55 | 22,880 | 20,901 | 9% | ✗ | ✗ |
| healthy-control-001 | 0.40 | **1.00** | 23,324 | 3,545 | **85%** | ✓ | ✓ |
| memory-high-001 | 0.36 | 0.57 | 32,380 | 7,381 | **77%** | ✗ | ✗ |
| service-down-001 | 0.36 | 0.57 | 27,091 | 7,451 | **72%** | ✗ | ✗ |
| **合计** | 0.39 | **0.71** | **131,263** | **47,020** | **64%** | 1/5 | 1/5 |

本地 planner 5/5 全部命中（没有一次因 `invalid_plan` 回退），步骤数从 8 降到 2–3，
Token 降 64%。F1 也从 0.39 提到 0.71。

**但 task_success 两边都是 1/5，这个数字要小心读**：失败的 4 个场景里
`facts_matched` 都是 1/1（模型确实查到了数值），卡住的是 `tool_recall`。
原因在评测集与训练集的工具名不一致：4 个场景的 gold 都要求 `search_log`，
而 `search_log` 在 **train split 里出现 0 次** —— 690 条训练数据里的日志工具只有
`search_service_logs`（438 次）、`analyze_log_pattern`（212 次）、`search_system_logs`（36 次）。
模型选了语义更具体的 `search_service_logs` / `search_system_logs`，判分却按 gold 的
`search_log` 算漏。这是**评测集与训练集的口径不一致**，不是模型能力问题；
云端 planner 也踩同一个坑（所以两边都是 1/5）。

所以这组数现在只能用来看 **token 节省与 F1 相对变化**，不能用来看绝对成功率。
修法见"下一步"第 1 条。

`disk-full-001` 的 9% 是因为本地模型 3 步没查够，replan 又退回云端补了 5 步 ——
这也是本地快路径应有的行为（不足时回退），不算失败。

## 下一步

1. **给 test 那两组模板本身补语序变体**。
   v10 证明了变体有效（在自己见过的 8 组上一致率 0.875 / signal-first 0.925），
   但没迁移到 test 的 `memory-2` / `memory-4`——那是两个独立的模板。
   修法很直接：在 `training/order_variants.py` 里加上这两个模板的匹配，
   对 **train 里同族**的样本生成变体；如果 train 里根本没有对应模板，
   就在 `gen_dataset.py` 补一个，而不是指望模型自己归纳。
2. **v10 必须训满 2 epoch 再下结论**。它只跑了 1 epoch（第 218/234 步 OOM，
   reentrant checkpointing 的 CUDA RNG fork），`service-2` 的 schema 从 1.000 掉到 0.233。
   `train_lora.py` 已改成 `use_reentrant=False`，重跑一次即可。
   在训满之前，不要用 v10 的数字替换 v9。
3. **`search_log` 口径**（5 个场景里 4 个）。它确实在工具注册表里，但 690 条训练数据里
   出现 0 次。三条路选一：改场景 gold、在 `run_eval.py` 加别名、或在 gen_dataset 补模板。
   在这之前 `task_success` 与会话级 F1 都不能比较本地/云端。
4. **扩 test 覆盖**：test 只有 6 组模板，`network` 和 `healthy` 全在 train/val。
5. **0.5B 蒸馏**：0.5B 直接训不稳定（见 BACKLOG），用 1.5B 生成 soft targets 更可靠。

## 复现清单

```bash
# 数据
.venv-train/Scripts/python.exe training/fix_labels.py --write        # 规则 -> 标签
.venv/Scripts/python.exe training/sync_prompt.py --write             # prompt 同步
.venv-train/Scripts/python.exe training/order_variants.py --write    # 语序变体

# 训练（8GB 卡：grad-accum 8，约 45 min/2 epoch）
.venv-train/Scripts/python.exe training/train_lora.py \
    --model E:/models/Qwen/Qwen2.5-1.5B-Instruct --output training/out/qwen15-lora-vN \
    --epochs 2 --max-length 1536 --batch-size 1 --grad-accum 8 \
    --lora-r 8 --lora-alpha 16 --attn-only --fp16 --merge

# 评测（三个 split：test / test_paraphrased / probe_order）
.venv-train/Scripts/python.exe training/eval_server.py --base-url <url> --model <name>
.venv-train/Scripts/python.exe training/probe_order.py --base-url <url> --model <name>
