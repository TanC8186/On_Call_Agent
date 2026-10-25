"""在 held-out test 集上评估 base / LoRA 模型的 JSON 与工具规划质量。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.schema import TriageOutput  # noqa: E402
from training.scoring import score, summarize  # noqa: E402

DATA_DIR = ROOT / "training" / "data"
CATALOG = json.loads((DATA_DIR / "tool_catalog.json").read_text(encoding="utf-8"))


def load_model(model_path: str, adapter: str | None):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return tokenizer, model


def parse_raw(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else None
    except Exception:  # noqa: BLE001
        return None


def parse_output(payload: dict | None) -> TriageOutput | None:
    if payload is None:
        return None
    try:
        return TriageOutput.model_validate(payload)
    except Exception:  # noqa: BLE001
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="E:/models/Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", default="training/out/eval.json")
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in (DATA_DIR / f"{args.split}.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    if args.limit:
        rows = rows[: args.limit]

    tokenizer, model = load_model(args.model, args.adapter)
    results = []
    for index, row in enumerate(rows, 1):
        messages = row["messages"]
        prompt = tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536)
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
            )
        text = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        raw = parse_raw(text)
        predicted = parse_output(raw)
        gold = TriageOutput.model_validate(json.loads(messages[-1]["content"]))
        result = score(
            predicted,
            gold,
            raw,
            CATALOG,
            default_service=row.get("metadata", {}).get("service", ""),
        )
        result["id"] = row.get("metadata", {}).get("template_id", str(index))
        result["predicted_plan"] = (
            [step.model_dump() for step in predicted.plan] if predicted else None
        )
        result["gold_category"] = gold.category
        results.append(result)
        print(
            f"[{index}/{len(rows)}] {result['id']} json={result['json_valid']} "
            f"schema={result['schema_valid']} f1={result['tool_f1']:.2f} "
            f"sev={result['severity_correct']} cat={result['category_correct']}"
        )

    summary = summarize(results)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.exit(main())
