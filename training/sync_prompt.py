"""把现有训练数据里的 system prompt 同步成 `app/agent/aiops/local_prompt.py` 的当前版本。

## 为什么需要它

system prompt 改了（加否定句规则、OOM 边界），但已经生成的 719 条数据还带着旧 prompt。
两条路：重新用教师跑一遍（贵，且会和旧数据不一致），或者只替换 message[0]。

后者是对的：**答案（plan）由教师生成并经过工具目录校验，与 system prompt 的具体措辞无关**；
severity/category 由 `training/labels.py` 的规则决定，也不来自教师。
system prompt 只是一段输入文本，换掉它不影响答案的正确性。

不这么做的话，训练时模型学到的是"旧 prompt -> 答案"，线上推理用新 prompt，
SFT 学到的规则说明就对不上了。

用法::

    .venv-train/Scripts/python.exe training/sync_prompt.py            # 只报告
    .venv-train/Scripts/python.exe training/sync_prompt.py --write    # 落盘
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agent.aiops.local_prompt import (  # noqa: E402
    build_system_prompt,
    compact_catalog,
)

DATA_DIR = ROOT / "training" / "data"


def current_prompt() -> str:
    catalog = json.loads((DATA_DIR / "tool_catalog.json").read_text(encoding="utf-8"))
    # 与 gen_dataset.py 保持一致：只给名字 + 一句描述，不显式重复工具名列表
    return build_system_prompt(compact_catalog(catalog, with_params=False), tool_names=None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--splits", default="train,val,test")
    args = parser.parse_args()

    prompt = current_prompt()
    for split in args.splits.split(","):
        path = DATA_DIR / f"{split}.jsonl"
        if not path.exists():
            continue
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        changed = 0
        for row in rows:
            if row["messages"][0]["role"] != "system":
                raise SystemExit(f"{split}: 第一条不是 system，数据格式变了")
            if row["messages"][0]["content"] != prompt:
                row["messages"][0]["content"] = prompt
                changed += 1
        print(f"{split}: {len(rows)} 条，需更新 {changed} 条")
        if args.write and changed:
            path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            print(f"   已写回 {path}")


if __name__ == "__main__":
    main()
