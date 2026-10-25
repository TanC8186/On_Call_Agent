"""回归测试：模型可控参数不得逃逸到 shell 或读取白名单外文件。"""

import pytest

from servers.ssh_server import _safe_count, _safe_log_path, _safe_token


@pytest.mark.parametrize(
    "payload",
    [
        "; rm -rf /",
        "$(whoami)",
        "`id`",
        "foo' | shutdown -h now",
        "foo\nrm -rf /",
        "foo && cat /etc/shadow",
    ],
)
def test_safe_token_rejects_shell_metacharacters(payload):
    with pytest.raises(ValueError):
        _safe_token(payload, "keyword")


def test_safe_token_accepts_normal_keywords():
    assert _safe_token("ERROR timeout", "keyword") == "ERROR timeout"
    assert _safe_token("磁盘告警", "keyword") == "磁盘告警"


@pytest.mark.parametrize(
    "path",
    [
        "C:/logs/../../Windows/win.ini",
        "/var/log/../../etc/passwd",
        "C:/Users/secret.txt",
        "/etc/shadow",
        "C:/logs/a'; whoami; '",
    ],
)
def test_safe_log_path_rejects_traversal_and_injection(path):
    with pytest.raises(ValueError):
        _safe_log_path(path)


def test_safe_log_path_accepts_whitelisted_logs():
    assert _safe_log_path("/var/log/syslog") == "/var/log/syslog"
    assert _safe_log_path("C:/logs/app.log") == "C:/logs/app.log"


@pytest.mark.parametrize("value", [0, -1, 10000, "abc"])
def test_safe_count_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        _safe_count(value, "count")
