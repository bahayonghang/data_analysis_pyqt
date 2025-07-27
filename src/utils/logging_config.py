"""
Logging configuration and setup for the application.
"""

import sys
from pathlib import Path
from typing import Optional

from loguru import logger


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    enable_console: bool = True,
    enable_rich: bool = True,
) -> None:
    """
    设置应用程序日志配置
    
    Args:
        log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: 日志文件路径，None表示不写入文件
        enable_console: 是否启用控制台输出
        enable_rich: 是否启用rich格式化
    """
    # 移除默认的logger
    logger.remove()
    
    # 控制台输出配置
    if enable_console:
        if enable_rich:
            logger.add(
                sys.stderr,
                level=log_level,
                format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                       "<level>{level: <8}</level> | "
                       "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                       "<level>{message}</level>",
                colorize=True,
                backtrace=True,
                diagnose=True,
            )
        else:
            logger.add(
                sys.stderr,
                level=log_level,
                format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
                colorize=False,
            )
    
    # 文件输出配置
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_path,
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
            rotation="10 MB",
            retention="30 days",
            compression="zip",
            backtrace=True,
            diagnose=True,
        )


def get_logger(name: str = None) -> "logger":
    """
    获取logger实例
    
    Args:
        name: logger名称，通常使用模块名
        
    Returns:
        logger实例
    """
    if name:
        return logger.bind(name=name)
    return logger


class LoggerMixin:
    """
    Logger混入类，为类提供日志功能
    """
    
    @property
    def logger(self) -> "logger":
        """获取当前类的logger"""
        return get_logger(self.__class__.__name__)


# 默认日志配置
DEFAULT_LOG_CONFIG = {
    "log_level": "INFO",
    "log_file": "data/logs/app.log",
    "enable_console": True,
    "enable_rich": True,
}


def init_default_logging():
    """初始化默认日志配置"""
    setup_logging(**DEFAULT_LOG_CONFIG)