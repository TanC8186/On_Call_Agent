"""训练样本的截断逻辑（不依赖 torch / peft，便于单测）。

规则：
1. 答案（labels）必须完整保留——它是唯一的训练信号。
2. prompt 只从左侧截断，因为 system prompt 的开头是关键指令。
3. 如果答案本身超过预算，才截断答案右端。
"""

from __future__ import annotations


def build_example(
    prompt_ids: list[int], answer_ids: list[int], max_length: int
) -> tuple[list[int], list[int]]:
    """返回 (input_ids, labels)；prompt 部分 label 为 -100。"""
    if len(answer_ids) >= max_length:
        answer_ids = answer_ids[:max_length]
        prompt_ids = []
    else:
        budget = max_length - len(answer_ids)
        if len(prompt_ids) > budget:
            prompt_ids = prompt_ids[-budget:]
    input_ids = list(prompt_ids) + list(answer_ids)
    labels = [-100] * len(prompt_ids) + list(answer_ids)
    return input_ids, labels
