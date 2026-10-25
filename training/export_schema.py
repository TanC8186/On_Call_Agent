"""导出两套 JSON Schema：

1. `triage_schema.json`   —— pydantic 原始 schema，用于校验
2. `triage_strict.json`   —— 约束解码用的严格版：所有字段 required、无 default
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.schema import TriageOutput  # noqa: E402

DATA = ROOT / "training" / "data"

base = TriageOutput.model_json_schema()
(DATA / "triage_schema.json").write_text(
    json.dumps(base, ensure_ascii=False, indent=2), encoding="utf-8"
)

strict = {
    "type": "object",
    "additionalProperties": False,
    "required": ["severity", "category", "plan"],
    "properties": {
        "severity": {"type": "string", "enum": ["P1", "P2", "P3", "P4"]},
        "category": {
            "type": "string",
            "enum": ["cpu", "memory", "disk", "network", "service", "healthy", "unknown"],
        },
        "plan": {
            "type": "array",
            "minItems": 2,
            "maxItems": 8,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "kind", "tool", "args", "description"],
                "properties": {
                    "id": {"type": "string"},
                    "kind": {"type": "string", "enum": ["tool", "reason"]},
                    "tool": {"type": "string"},
                    "args": {"type": "object"},
                    "description": {"type": "string"},
                    "success_criteria": {"type": "string"},
                },
            },
        },
    },
}
(DATA / "triage_strict.json").write_text(
    json.dumps(strict, ensure_ascii=False, indent=2), encoding="utf-8"
)
print("wrote triage_schema.json and triage_strict.json")
