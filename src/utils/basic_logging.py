"""
Basic logging configuration without external dependencies.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_basic_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    enable_console: bool = True,
) -> None:
    """
    设置基础日志配置（不依赖外部库）
    
    Args:
        log_level: 日志级别
        log_file: 日志文件路径
        enable_console: 是否启用控制台输出
    """
    # 设置日志级别
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # 创建格式器
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s'
    )
    
    # 获取根logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # 清除现有handlers
    root_logger.handlers.clear()
    
    # 控制台handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # 文件handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str = None) -> logging.Logger:
    """
    获取logger实例
    
    Args:
        name: logger名称
        
    Returns:
        logger实例
    """
    return logging.getLogger(name or __name__)


class LoggerMixin:
    """Logger混入类"""
    
    @property
    def logger(self) -> logging.Logger:
        """获取当前类的logger"""
        return get_logger(self.__class__.__name__)


def init_default_logging():
    """初始化默认日志配置"""
    setup_basic_logging(
        log_level="INFO",
        log_file="data/logs/app.log",
        enable_console=True,
    )