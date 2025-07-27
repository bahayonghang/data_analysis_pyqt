"""
Application configuration management.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from .logging_config import get_logger

logger = get_logger(__name__)


class DatabaseConfig(BaseModel):
    """数据库配置"""
    path: str = Field(default="data/analysis_history.db", description="数据库文件路径")
    backup_interval: int = Field(default=24, description="备份间隔（小时）")
    max_connections: int = Field(default=10, description="最大连接数")
    query_timeout: int = Field(default=30, description="查询超时时间（秒）")


class UIConfig(BaseModel):
    """界面配置"""
    theme: str = Field(default="auto", description="主题模式: light/dark/auto")
    theme_color: str = Field(default="#0078d4", description="主题色")
    window_width: int = Field(default=1200, description="窗口宽度")
    window_height: int = Field(default=800, description="窗口高度")
    language: str = Field(default="zh_CN", description="界面语言")
    auto_save: bool = Field(default=True, description="自动保存分析结果")


class ProcessingConfig(BaseModel):
    """数据处理配置"""
    max_memory_usage: str = Field(default="4GB", description="最大内存使用量")
    chunk_size: int = Field(default=10000, description="数据块大小")
    parallel_threads: int = Field(default=4, description="并行线程数")
    enable_streaming: bool = Field(default=True, description="启用流式处理")
    cache_size: int = Field(default=100, description="缓存大小（MB）")


class ChartConfig(BaseModel):
    """图表配置"""
    default_style: str = Field(default="nature", description="默认图表样式")
    dpi: int = Field(default=300, description="图表DPI")
    figure_width: float = Field(default=10.0, description="图表宽度（英寸）")
    figure_height: float = Field(default=6.0, description="图表高度（英寸）")
    color_palette: str = Field(default="Set2", description="颜色调色板")
    save_format: str = Field(default="png", description="默认保存格式")


class LoggingConfig(BaseModel):
    """日志配置"""
    level: str = Field(default="INFO", description="日志级别")
    file_path: str = Field(default="data/logs/app.log", description="日志文件路径")
    max_file_size: str = Field(default="10MB", description="日志文件最大大小")
    backup_count: int = Field(default=5, description="日志文件备份数量")
    enable_console: bool = Field(default=True, description="启用控制台日志")
    enable_rich: bool = Field(default=True, description="启用Rich格式化")


class AppSettings(BaseModel):
    """应用程序设置"""
    
    # 基本信息
    app_name: str = Field(default="Data Analysis PyQt", description="应用程序名称")
    version: str = Field(default="0.1.0", description="版本号")
    debug: bool = Field(default=False, description="调试模式")
    
    # 数据目录
    data_dir: str = Field(default="data", description="数据目录")
    cache_dir: str = Field(default="data/cache", description="缓存目录")
    logs_dir: str = Field(default="data/logs", description="日志目录")
    exports_dir: str = Field(default="data/exports", description="导出目录")
    
    # 配置组件
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    charts: ChartConfig = Field(default_factory=ChartConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "resources/config/app_config.json"
        self._settings: Optional[AppSettings] = None
        self._load_settings()
    
    def _load_settings(self) -> None:
        """加载设置"""
        try:
            self._settings = AppSettings()
            self._ensure_directories()
            logger.info("Configuration loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            self._settings = AppSettings()  # 使用默认配置
    
    def _ensure_directories(self) -> None:
        """确保必要的目录存在"""
        if self._settings:
            directories = [
                self._settings.data_dir,
                self._settings.cache_dir,
                self._settings.logs_dir,
                self._settings.exports_dir,
            ]
            
            for directory in directories:
                Path(directory).mkdir(parents=True, exist_ok=True)
    
    @property
    def settings(self) -> AppSettings:
        """获取当前设置"""
        if self._settings is None:
            self._load_settings()
        return self._settings
    
    def save_settings(self) -> bool:
        """保存设置到文件"""
        try:
            config_path = Path(self.config_file)
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, 'w', encoding='utf-8') as f:
                f.write(self._settings.model_dump_json(indent=2))
            
            logger.info(f"Configuration saved to {config_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False
    
    def update_setting(self, path: str, value: Any) -> bool:
        """
        更新设置值
        
        Args:
            path: 设置路径，如 "ui.theme" 或 "database.path"
            value: 新值
            
        Returns:
            bool: 是否更新成功
        """
        try:
            parts = path.split('.')
            current = self._settings
            
            # 导航到父对象
            for part in parts[:-1]:
                current = getattr(current, part)
            
            # 设置最终值
            setattr(current, parts[-1], value)
            
            logger.info(f"Setting updated: {path} = {value}")
            return True
        except Exception as e:
            logger.error(f"Failed to update setting {path}: {e}")
            return False
    
    def get_setting(self, path: str, default: Any = None) -> Any:
        """
        获取设置值
        
        Args:
            path: 设置路径
            default: 默认值
            
        Returns:
            设置值或默认值
        """
        try:
            parts = path.split('.')
            current = self._settings
            
            for part in parts:
                current = getattr(current, part)
            
            return current
        except (AttributeError, TypeError):
            logger.warning(f"Setting not found: {path}, using default: {default}")
            return default
    
    def reset_to_defaults(self) -> None:
        """重置为默认设置"""
        self._settings = AppSettings()
        logger.info("Settings reset to defaults")


# 全局配置管理器实例
config_manager = ConfigManager()

# 便捷访问函数
def get_settings() -> AppSettings:
    """获取应用设置"""
    return config_manager.settings

def get_setting(path: str, default: Any = None) -> Any:
    """获取单个设置值"""
    return config_manager.get_setting(path, default)

def update_setting(path: str, value: Any) -> bool:
    """更新单个设置值"""
    return config_manager.update_setting(path, value)