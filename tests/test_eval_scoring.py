"""评测判分的最小回归测试。"""

from scripts.run_eval import _fact_matches
from scripts.validate_eval import GoldFact


def test_fact_matcher_finds_metric_value():
    fact = GoldFact(metric="cpu_usage_percent", op=">", value=90)
    preview = '{"metric_name": "cpu_usage_percent", "data_points": [{"value": 95.2}]}'
    assert _fact_matches(fact, [preview]) is True


def test_fact_matcher_rejects_low_value():
    fact = GoldFact(metric="cpu_usage_percent", op=">", value=90)
    preview = '{"metric_name": "cpu_usage_percent", "data_points": [{"value": 42.0}]}'
    assert _fact_matches(fact, [preview]) is False


def test_fact_matcher_handles_missing_metric():
    fact = GoldFact(metric="disk_usage_percent", op=">", value=90)
    assert _fact_matches(fact, ['{"metric_name": "cpu_usage_percent"}']) is False
