# 领域小模型：AIOps 分诊 + PlanStep

目标不是“把 0.5B 训练成运维专家”，而是训练一个**窄任务**模型，让它接管 Agent 里最高频、最结构化的部分：

```
告警文本 + 工具目录  ->  {severity, category, plan:[{tool,args,description}]}
```

它可以直接接进 `app/agent/aiops/planner.py` 的位置：本地模型生成 typed plan → 用 MCP registry 校验 → 不合法/低置信时升级到云端大模型。

## 为什么选这个任务

- **可验证**：tool name / args / severity / category 都能自动判分，不靠人看。
- **可接入**：输出就是项目已有的 `PlanStep`，不需要改 Agent 架构。
- **可讲取舍**：0.5B 在窄任务上可以接近大模型，但开放式根因分析不行——这正好是“端侧 + 云端”路由的故事。
- **数据便宜**：5 篇 runbook + 22 个工具 schema + 模板化告警，老师模型批量生成后做规则校验。

## 数据

```bash
# 1. 导出真实工具目录（22 个工具）
python training/dump_tools.py

# 2. 用 LLM_MODEL（默认 DeepSeek-V4.1-Flash-1017）合成；先小样本验证
python training/gen_dataset.py --per-family 1
# 3. 全量（6 类 × 4 模板 × N）
python training/gen_dataset.py --per-family 30 --concurrency 6 --names-only
```

输出：

- `training/data/train.jsonl` / `val.jsonl` / `test.jsonl`（按模板分组切分，避免泄漏）
- 每条是 `messages` 格式，可直接用于 chat SFT
- `stats.json` 记录 family / severity / category 分布

数据质量规则（`training/schema.py`）：

- 第一个工具步骤必须是 `list_all_services`
- 工具名和参数必须存在于 `tool_catalog.json`
- healthy 场景 severity=P4，其他场景不能是 P4
- 2–8 步

## 训练

建议单独环境，避免污染应用依赖：

```bash
python -m venv .venv-train
.venv-train/Scripts/python.exe -m pip install --upgrade pip
.venv-train/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu124
.venv-train/Scripts/python.exe -m pip install transformers peft accelerate

# 1.5B Instruct（质量优先）
.venv-train/Scripts/python.exe training/train_lora.py \
  --model E:/models/Qwen/Qwen2.5-1.5B-Instruct \
  --output training/out/qwen15-lora-v4 --epochs 2 --max-length 1536 --merge

# 0.5B（端侧对照）
.venv-train/Scripts/python.exe training/train_lora.py \
  --model E:/My_Agent_Projects/qwen/Qwen2___5-0___5B-Instruct \
  --output training/out/qwen05-lora --epochs 3 --merge
```

8GB 显存建议：`batch-size 1-2`、`grad-accum 8`、`max-length 1024`、开启 gradient checkpointing。OOM 就把 `max-length` 降到 768。

## 评测

```bash
.venv-train/Scripts/python.exe training/eval_lora.py \
  --model E:/models/Qwen/Qwen2.5-1.5B-Instruct \
  --adapter training/out/qwen15-lora-v4 --out training/out/eval_v4_150.json

.venv-train/Scripts/python.exe training/eval_lora.py \
  --model E:/My_Agent_Projects/qwen/Qwen2___5-0___5B-Instruct \
  --adapter training/out/qwen05-lora --out training/out/eval_qwen05.json
```

指标：JSON 合法率、schema 合法率、tool F1、arg exact、severity/category 准确率。

## 约束解码

`training/export_schema.py` 会导出两套 schema：

- `triage_schema.json`：pydantic 原始 schema（校验用）
- `triage_strict.json`：所有字段 required、无 default（llama.cpp 约束解码用）

```bash
python training/export_schema.py
llama-server -m qwen15-ops-Q4_K_M.gguf -jf training/data/triage_strict.json
python training/eval_server.py --base-url http://127.0.0.1:8080/v1   --model qwen15-ops --schema training/data/triage_strict.json
```

注意：`-jf` 作为全局默认 schema 会在 chat template 的 `<|im_start|>` 上报
`Unexpected empty grammar stack`；正确做法是通过请求里的
`response_format.json_schema` 传（`eval_server.py --schema` 已经这么做了）。

## 推理服务与基准

三条路线，产出都在 `training/RESULTS.md`：

1. **llama.cpp / Ollama（Windows，最快出结果）**
   - 合并后的 HF 模型转 GGUF Q4_K_M（`llama-quantize`，25 秒）
   - `ollama create qwen-ops -f Modelfile` 或 `llama-server -m model.gguf -t 6`
   - 只有 CPU 后端，单并发可用、8 并发 TTFT 会到 2.4 s
2. **vLLM + AWQ（WSL2，GPU）**
   - `training/make_calib.py --n 128` 导出 train split 的完整对话作校准集
   - `training/quantize_awq.py` 跑 AutoRound，产出 `quant_method: awq / version: gemm`
   - `vllm serve <dir> --quantization awq --max-model-len 2048 --enforce-eager`
3. **vLLM fp16**：作为精度上界，`--dtype float16 --enforce-eager`

`--enforce-eager` 不是性能选项，是**必需**：本机不带它会在 CUDA graph 捕获完成后
pytorch 编译 `_apply_write_kernel` 时 `ptxas` 崩（`SIGHUP` / `Unsupported .version`）。

```bash
# 量化（WSL，用 vLLM 那个 venv —— 它有 torch 2.13 + cu130）
~/vllm-venv/bin/python training/quantize_awq.py --nsamples 128 --output-dir training/out/qwen15-v8-awq
~/vllm-venv/bin/python training/quantize_awq.py --nsamples 128 --iters 200 \
    --algorithm auto_round --output-dir training/out/qwen15-v8-signround   # 推荐

# 基准
python training/serve_bench.py --base-url http://127.0.0.1:8000/v1 --model qwen-ops-signround
```

### 为什么纯 AWQ 不行、SignRound 行

`--algorithm awq` 只做 AWQ 的通道平滑，权重压缩用 RTN；`iters=0` 时没有误差补偿。
在这份 1.5B 权重上它把 `cpu-3` 一类的工具名写成了
`"tool": "query_cpu_metrics, args={service_name: 'order-service'}"`，30 条里 17–18 条 JSON 解析失败。
换成 Triton kernel（`VLLM_USE_TRITON_AWQ=1`）结果几乎相同，所以不是 kernel 的问题。

`--algorithm auto_round --iters 200`（SignRound，做逐 block 的误差补偿）在同样的
AWQ 导出格式下拿到 schema 1.000 / plan_ok 1.000 / arg exact 0.889 —— 比 fp16 的 arg exact 还高。

## 鲁棒性验证

两个脚本，解决"分数高是不是因为背模板"这个问题：

```bash
# 1. 规则改写（确定性、可进回归）
.venv-train/Scripts/python.exe training/paraphrase_rules.py
#    -> training/data/test_paraphrased.jsonl（178 条，seed 42）

# 2. 受控探针：同一句话，只换分句顺序
.venv-train/Scripts/python.exe training/probe_order.py \
    --base-url http://127.0.0.1:8000/v1 --model qwen-ops-v9-fp16
```

v9 的结果：改写后 severity 0.989 → 0.865（schema/关键工具/category 不变）；
探针里 number-first 0.966 / signal-first 0.534。结论写进 `RESULTS.md` 的
"改写鲁棒性测试"一节 —— 报 severity 之前先读那一节。

## 接入 Agent（下一步）

在 `app/agent/aiops/planner.py` 增加 `LOCAL_LLM_BASE_URL` 分支：

- 本地模型生成 plan → `normalize_plan` + registry 校验
- 校验失败 / tool F1 低于阈值 → 升级到云端 `LLM_MODEL`
- 在 trace 里记录 `planner_backend=local|cloud`、升级原因
- 用 `scripts/run_eval.py` 对比：本地处理率、P95 延迟、tokens/task、任务成功率

## 简历写法（数字全部来自 `training/RESULTS.md`，可复现）

> - 构建 719 条 AIOps 分诊/工具规划数据集：6 类故障 × 4 种表述，教师模型生成 + Pydantic/工具注册表双重校验，
>   按模板分组切分避免泄漏；用确定性规则替代教师标签消除噪声，并加单测锁住"标签与规则一致"。
> - 对 Qwen2.5-1.5B-Instruct 做 LoRA SFT：held-out 149 条上 **Schema 合法率 1.000**、
>   **关键工具召回 1.000**、**Plan OK 0.973**、Severity 0.893、Arg exact 0.831。
> - 在 RTX 4060 8GB 上做**精度/后端四向对照**（fp16 / GGUF Q4_K_M / AWQ / SignRound）：
>   证明 4-bit 的 severity 损失（−13~14 点）与运行后端无关，纯 AWQ 在 1.5B 上会崩（18/30 条 JSON 不可解析），
>   改用 SignRound（AutoRound iters=200）后 schema/plan_ok/关键工具召回全部回到 1.000。
> - 4-bit 在这台 8GB 卡上是净收益：权重 2.9 GB → 1.15 GB，8 并发吞吐 464 → 514 tok/s，TPOT 16.7 → 15.1 ms。
> - 定位并修复三个具体失效：工具名编造（prompt 显式列出合法工具名）、参数名漂移（difflib 最近邻纠错）、
>   训练时 prompt 被截断（只截 prompt 左侧、答案始终完整，单测锁住）。
> - 发现并修掉评测集污染：test 是 00:06 生成的，`labels.py` 15:54 改过，19 条 `disk-2` 带着旧规则的
>   P3 把**所有**模型的 severity 一起压低 8~12 点；补 `fix_labels.py` 与一致性单测。
> - 把 1.5B 模型接成 Agent 的本地 Planner 快路径，工具/参数校验失败自动回退云端：
>   5 个场景本地命中 5/6，健康巡检 input tokens 降低 94%，cpu-high 降低 36%。

## 诚实边界

- 1.5B 不负责开放式根因分析；它只做分诊和 typed plan。
- 训练数据是合成的，必须在 README/简历里标注；不要包装成真实工单。
- 不要用"准确率提升 30%"这种没有基线的数字；所有指标都要能指向 `eval_lora.py` / `eval_server.py` 的输出。
- **severity 不能单看总分**：`memory-2`（OOM 边界 92% vs 95%）所有后端都在 0.52–0.59，
  `cpu-3`（"低峰 + 没有用户请求超时"）train/val 里没有同类样本，规则把否定句读成了正面影响。
  这两类占 149 条里的 59 条，直接影响 severity 的可比性 —— 报数字时要拆开说。
- 纯 AWQ（`iters=0`）**不要用于生产**；它的 0.691 是"模型在某一类上直接崩掉"加权后的结果，
  不是平滑退化。
- 所有对比都是 **单次贪心解码**，没有跑多次取均值；样本量 149（每类 30），
  1 条 = 0.67 个百分点，不要解读小于 2 点的差异。
