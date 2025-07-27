"""
Simple configuration management without external dependencies.
"""

import json
from pathlib import Path
from typing import Any, Dict

from .basic_logging import get_logger

logger = get_logger(__name__)


class SimpleConfig:
    """简单配置管理器"""
    
    def __init__(self):
        self.config = {
            # 基本信息
            "app_name": "Data Analysis PyQt",
            "version": "0.1.0",
            "debug": False,
            
            # 目录配置
            "data_dir": "data",
            "cache_dir": "data/cache", 
            "logs_dir": "data/logs",
            "exports_dir": "data/exports",
            
            # 日志配置
            "logging": {
                "level": "INFO",
                "file_path": "data/logs/app.log",
                "enable_console": True,
                "enable_rich": True,
            },
            
            # UI配置
            "ui": {
                "theme": "auto",
                "theme_color": "#0078d4",
                "window_width": 1200,
                "window_height": 800,
                "language": "zh_CN",
            },
            
            # 数据库配置
            "database": {
                "path": "data/analysis_history.db",
                "backup_interval": 24,
                "max_connections": 10,
            },
            
            # 处理配置
            "processing": {
                "max_memory_usage": "4GB",
                "chunk_size": 10000,
                "parallel_threads": 4,
                "enable_streaming": True,
            },
            
            # 图表配置
            "charts": {
                "default_style": "nature",
                "dpi": 300,
                "figure_width": 10.0,
                "figure_height": 6.0,
                "save_format": "png",
            }
        }
        
        self._ensure_directories()
    
    def _ensure_directories(self):
        """确保必要的目录存在"""
        directories = [
            self.config["data_dir"],
            self.config["cache_dir"],
            self.config["logs_dir"],
            self.config["exports_dir"],
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def get(self, path: str, default: Any = None) -> Any:
        """
        获取配置值，支持点号分隔的路径
        
        Args:
            path: 配置路径，如 "ui.theme"
            default: 默认值
            
        Returns:
            配置值或默认值
        """
        try:
            parts = path.split('.')
            current = self.config
            
            for part in parts:
                current = current[part]
            
            return current
        except (KeyError, TypeError):
            return default
    
    def set(self, path: str, value: Any) -> bool:
        """
        设置配置值
        
        Args:
            path: 配置路径
            value: 新值
            
        Returns:
            是否设置成功
        """
        try:
            parts = path.split('.')
            current = self.config
            
            # 导航到父对象
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            # 设置最终值
            current[parts[-1]] = value
            return True
        except Exception as e:
            logger.error(f"Failed to set config {path}: {e}")
            return False
    
    def save(self, config_file: str = "resources/config/app_config.json") -> bool:
        """保存配置到文件"""
        try:
            config_path = Path(config_file)
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Configuration saved to {config_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False
    
    def load(self, config_file: str = "resources/config/app_config.json") -> bool:
        """从文件加载配置"""
        try:
            config_path = Path(config_file)
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                    # 合并配置，保留默认值
                    self._deep_update(self.config, loaded_config)
                logger.info(f"Configuration loaded from {config_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return False
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """深度更新字典"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value


# 全局配置实例
_config = SimpleConfig()

def get_config() -> SimpleConfig:
    """获取配置实例"""
    return _config

def get_setting(path: str, default: Any = None) -> Any:
    """获取配置值"""
    return _config.get(path, default)

def set_setting(path: str, value: Any) -> bool:
    """设置配置值"""
    return _config.set(path, value)