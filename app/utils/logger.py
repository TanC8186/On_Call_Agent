"""日志配置模块

使用 Loguru 配置应用日志
注意：Windows 下控制台默认 GBK 编码，emoji 等 UTF-8 字符会导致崩溃。
这里仅使用文件输出（UTF-8），避免控制台编码问题。
"""

import sys
from loguru import logger
from app.config import config


def setup_logger():
    """配置日志系统"""
    # 移除默认处理器
    logger.remove()

    # 仅在 debug 模式下添加控制台输出（简化格式，安全处理编码）
    if config.debug:
        try:
            # 尝试使用 reconfigure stdout 为 UTF-8
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass
        logger.add(
            sys.stdout,
            format="{time:HH:mm:ss} | {level: <8} | {module}:{line} | {message}",
            level="INFO",
            colorize=False,
            backtrace=False,
            diagnose=False,
        )

    # 文件输出（UTF-8 编码，主日志）
    logger.add(
        "logs/app_{time:YYYY-MM-DD}.log",
        rotation="00:00",
        retention="7 days",
        compression="zip",
        encoding="utf-8",
        enqueue=True,
        backtrace=True,
        diagnose=True,
        level="DEBUG" if config.debug else "INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {module}.{function}:{line} | {message}",
    )


setup_logger()
