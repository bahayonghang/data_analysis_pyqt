"""
Custom exception classes for the data analysis application.
"""

from typing import Any, Dict, Optional


class DataAnalysisError(Exception):
    """Base exception class for all application errors."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        self.original_exception = original_exception
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging or API responses."""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
            "original_exception": str(self.original_exception) if self.original_exception else None,
        }


# 数据相关异常
class DataLoadError(DataAnalysisError):
    """数据加载异常"""
    pass


class DataLoadingError(DataAnalysisError):
    """数据加载异常（别名）"""
    pass


class DataFormatError(DataAnalysisError):
    """数据格式异常"""
    pass


class DataValidationError(DataAnalysisError):
    """数据验证异常"""
    pass


class DataProcessingError(DataAnalysisError):
    """数据处理异常"""
    pass


class DataAnalysisComputationError(DataAnalysisError):
    """数据分析计算异常"""
    pass


class AnalysisError(DataAnalysisError):
    """分析引擎异常"""
    pass


class WorkflowError(DataAnalysisError):
    """工作流异常"""
    pass


# 文件相关异常
class FileNotFoundError(DataAnalysisError):
    """文件未找到异常"""
    pass


class FilePermissionError(DataAnalysisError):
    """文件权限异常"""
    pass


class FileCorruptedError(DataAnalysisError):
    """文件损坏异常"""
    pass


class FileValidationError(DataAnalysisError):
    """文件验证异常"""
    pass


class UnsupportedFileFormatError(DataAnalysisError):
    """不支持的文件格式异常"""
    pass


# 数据库相关异常
class DatabaseError(DataAnalysisError):
    """数据库异常"""
    pass


class DatabaseConnectionError(DatabaseError):
    """数据库连接异常"""
    pass


class DatabaseQueryError(DatabaseError):
    """数据库查询异常"""
    pass


class DatabaseIntegrityError(DatabaseError):
    """数据库完整性异常"""
    pass


# UI相关异常
class UIRenderError(DataAnalysisError):
    """UI渲染异常"""
    pass


class ChartGenerationError(DataAnalysisError):
    """图表生成异常"""
    pass


class ComponentInitializationError(DataAnalysisError):
    """组件初始化异常"""
    pass


# 配置相关异常
class ConfigurationError(DataAnalysisError):
    """配置异常"""
    pass


class InvalidConfigValueError(ConfigurationError):
    """无效配置值异常"""
    pass


class MissingConfigError(ConfigurationError):
    """缺失配置异常"""
    pass


# 异步和并发异常
class AsyncOperationError(DataAnalysisError):
    """异步操作异常"""
    pass


class TaskTimeoutError(AsyncOperationError):
    """任务超时异常"""
    pass


class TaskCancellationError(AsyncOperationError):
    """任务取消异常"""
    pass


# 资源相关异常
class ResourceError(DataAnalysisError):
    """资源异常"""
    pass


class InsufficientMemoryError(ResourceError):
    """内存不足异常"""
    pass


class DiskSpaceError(ResourceError):
    """磁盘空间不足异常"""
    pass


# 业务逻辑异常
class BusinessLogicError(DataAnalysisError):
    """业务逻辑异常"""
    pass


class InvalidOperationError(BusinessLogicError):
    """无效操作异常"""
    pass


class DuplicateAnalysisError(BusinessLogicError):
    """重复分析异常"""
    pass