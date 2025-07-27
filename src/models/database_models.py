"""
数据库模型定义
包含SQLite数据库的所有表结构和ORM模型
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


class AnalysisStatus(str, Enum):
    """分析状态"""
    PENDING = "pending"  # 待处理
    PROCESSING = "processing"  # 处理中
    COMPLETED = "completed"  # 已完成
    FAILED = "failed"  # 失败
    CANCELLED = "cancelled"  # 已取消


class LogLevel(str, Enum):
    """日志级别"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AnalysisHistory(BaseModel):
    """分析历史记录模型"""
    
    # 主键和唯一标识
    id: Optional[int] = Field(None, description="主键ID")
    analysis_id: str = Field(..., description="分析唯一ID")
    
    # 文件信息
    file_hash: str = Field(..., description="文件MD5哈希值")
    file_name: str = Field(..., description="文件名")
    file_path: str = Field(..., description="文件路径")
    file_size: int = Field(ge=0, description="文件大小(字节)")
    
    # 分析信息
    analysis_type: str = Field(..., description="分析类型")
    analysis_title: str = Field(default="", description="分析标题")
    analysis_description: Optional[str] = Field(None, description="分析描述")
    
    # 数据信息
    total_rows: int = Field(ge=0, description="数据总行数")
    total_columns: int = Field(ge=0, description="数据总列数")
    analyzed_columns: List[str] = Field(default_factory=list, description="分析的列名")
    time_column: Optional[str] = Field(None, description="时间列名")
    
    # 执行信息
    status: AnalysisStatus = Field(default=AnalysisStatus.PENDING, description="分析状态")
    execution_time_ms: int = Field(default=0, ge=0, description="执行时间(毫秒)")
    error_message: Optional[str] = Field(None, description="错误信息")
    
    # 结果统计
    result_summary: Dict[str, Any] = Field(default_factory=dict, description="结果摘要")
    has_charts: bool = Field(default=False, description="是否包含图表")
    chart_count: int = Field(default=0, ge=0, description="图表数量")
    
    # 时间戳
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    updated_at: datetime = Field(default_factory=datetime.now, description="更新时间")
    completed_at: Optional[datetime] = Field(None, description="完成时间")
    
    # 元数据
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    tags: List[str] = Field(default_factory=list, description="标签")
    is_favorite: bool = Field(default=False, description="是否收藏")
    
    @validator('analysis_id')
    def validate_analysis_id(cls, v):
        """验证分析ID"""
        if not v or len(v) < 8:
            raise ValueError("分析ID必须至少8个字符")
        return v
    
    @validator('file_hash')
    def validate_file_hash(cls, v):
        """验证文件哈希值"""
        if not v or len(v) != 32:
            raise ValueError("文件哈希值必须是32位MD5值")
        return v.lower()
    
    @validator('analyzed_columns', pre=True)
    def validate_analyzed_columns(cls, v):
        """验证分析列名"""
        if isinstance(v, str):
            # 如果是字符串，按逗号分割
            return [col.strip() for col in v.split(',') if col.strip()]
        return v
    
    @validator('tags', pre=True)
    def validate_tags(cls, v):
        """验证标签"""
        if isinstance(v, str):
            # 如果是字符串，按逗号分割
            return [tag.strip() for tag in v.split(',') if tag.strip()]
        return v
    
    def mark_processing(self) -> None:
        """标记为处理中"""
        self.status = AnalysisStatus.PROCESSING
        self.updated_at = datetime.now()
    
    def mark_completed(self, execution_time_ms: int, result_summary: Dict[str, Any]) -> None:
        """标记为已完成"""
        self.status = AnalysisStatus.COMPLETED
        self.execution_time_ms = execution_time_ms
        self.result_summary = result_summary
        self.completed_at = datetime.now()
        self.updated_at = datetime.now()
        self.error_message = None
    
    def mark_failed(self, error_message: str) -> None:
        """标记为失败"""
        self.status = AnalysisStatus.FAILED
        self.error_message = error_message
        self.updated_at = datetime.now()
    
    def add_tag(self, tag: str) -> None:
        """添加标签"""
        if tag and tag not in self.tags:
            self.tags.append(tag)
            self.updated_at = datetime.now()
    
    def remove_tag(self, tag: str) -> None:
        """移除标签"""
        if tag in self.tags:
            self.tags.remove(tag)
            self.updated_at = datetime.now()
    
    def toggle_favorite(self) -> None:
        """切换收藏状态"""
        self.is_favorite = not self.is_favorite
        self.updated_at = datetime.now()


class AnalysisCharts(BaseModel):
    """分析图表记录模型"""
    
    # 主键和关联
    id: Optional[int] = Field(None, description="主键ID")
    analysis_id: str = Field(..., description="关联的分析ID")
    chart_id: str = Field(..., description="图表唯一ID")
    
    # 图表信息
    chart_type: str = Field(..., description="图表类型")
    chart_title: str = Field(default="", description="图表标题")
    chart_description: Optional[str] = Field(None, description="图表描述")
    
    # 配置和数据
    chart_config: Dict[str, Any] = Field(default_factory=dict, description="图表配置")
    data_columns: List[str] = Field(default_factory=list, description="使用的数据列")
    chart_order: int = Field(default=0, ge=0, description="图表顺序")
    
    # 文件信息
    image_path: Optional[str] = Field(None, description="图表图片路径")
    image_format: str = Field(default="png", description="图片格式")
    image_size: int = Field(default=0, ge=0, description="图片大小(字节)")
    
    # 交互配置
    is_interactive: bool = Field(default=False, description="是否交互式图表")
    html_path: Optional[str] = Field(None, description="交互式图表HTML路径")
    
    # 时间戳
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    updated_at: datetime = Field(default_factory=datetime.now, description="更新时间")
    
    # 元数据
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    
    @validator('analysis_id')
    def validate_analysis_id(cls, v):
        """验证分析ID"""
        if not v or len(v) < 8:
            raise ValueError("分析ID必须至少8个字符")
        return v
    
    @validator('chart_id')
    def validate_chart_id(cls, v):
        """验证图表ID"""
        if not v or len(v) < 4:
            raise ValueError("图表ID必须至少4个字符")
        return v
    
    @validator('data_columns', pre=True)
    def validate_data_columns(cls, v):
        """验证数据列名"""
        if isinstance(v, str):
            return [col.strip() for col in v.split(',') if col.strip()]
        return v
    
    def update_image_info(self, image_path: str, image_format: str, image_size: int) -> None:
        """更新图片信息"""
        self.image_path = image_path
        self.image_format = image_format
        self.image_size = image_size
        self.updated_at = datetime.now()
    
    def update_config(self, config: Dict[str, Any]) -> None:
        """更新图表配置"""
        self.chart_config = config
        self.updated_at = datetime.now()


class UserSettings(BaseModel):
    """用户设置模型"""
    
    # 主键
    id: Optional[int] = Field(None, description="主键ID")
    setting_key: str = Field(..., description="设置键名")
    setting_value: str = Field(..., description="设置值")
    setting_type: str = Field(default="string", description="设置类型")
    
    # 分类和描述
    category: str = Field(default="general", description="设置分类")
    description: Optional[str] = Field(None, description="设置描述")
    
    # 时间戳
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    updated_at: datetime = Field(default_factory=datetime.now, description="更新时间")
    
    @validator('setting_key')
    def validate_setting_key(cls, v):
        """验证设置键名"""
        if not v or len(v) < 2:
            raise ValueError("设置键名必须至少2个字符")
        return v
    
    @validator('setting_type')
    def validate_setting_type(cls, v):
        """验证设置类型"""
        allowed_types = ['string', 'integer', 'float', 'boolean', 'json']
        if v not in allowed_types:
            raise ValueError(f"不支持的设置类型: {v}")
        return v
    
    def get_typed_value(self) -> Any:
        """获取类型化的设置值"""
        if self.setting_type == 'integer':
            return int(self.setting_value)
        elif self.setting_type == 'float':
            return float(self.setting_value)
        elif self.setting_type == 'boolean':
            return self.setting_value.lower() in ('true', '1', 'yes', 'on')
        elif self.setting_type == 'json':
            import json
            return json.loads(self.setting_value)
        else:
            return self.setting_value
    
    def set_typed_value(self, value: Any) -> None:
        """设置类型化的值"""
        if self.setting_type == 'json':
            import json
            self.setting_value = json.dumps(value)
        else:
            self.setting_value = str(value)
        self.updated_at = datetime.now()


class AppLogs(BaseModel):
    """应用日志模型"""
    
    # 主键
    id: Optional[int] = Field(None, description="主键ID")
    
    # 日志信息
    log_level: LogLevel = Field(..., description="日志级别")
    logger_name: str = Field(..., description="日志记录器名称")
    message: str = Field(..., description="日志消息")
    
    # 上下文信息
    module_name: Optional[str] = Field(None, description="模块名")
    function_name: Optional[str] = Field(None, description="函数名")
    line_number: Optional[int] = Field(None, ge=0, description="行号")
    
    # 异常信息
    exception_type: Optional[str] = Field(None, description="异常类型")
    exception_message: Optional[str] = Field(None, description="异常消息")
    traceback: Optional[str] = Field(None, description="异常堆栈")
    
    # 用户和会话信息
    user_id: Optional[str] = Field(None, description="用户ID")
    session_id: Optional[str] = Field(None, description="会话ID")
    request_id: Optional[str] = Field(None, description="请求ID")
    
    # 时间戳
    timestamp: datetime = Field(default_factory=datetime.now, description="日志时间")
    
    # 额外数据
    extra_data: Dict[str, Any] = Field(default_factory=dict, description="额外数据")
    
    @validator('logger_name')
    def validate_logger_name(cls, v):
        """验证日志记录器名称"""
        if not v:
            raise ValueError("日志记录器名称不能为空")
        return v
    
    @validator('message')
    def validate_message(cls, v):
        """验证日志消息"""
        if not v:
            raise ValueError("日志消息不能为空")
        return v
    
    @classmethod
    def create_from_record(cls, record, extra_data: Optional[Dict] = None) -> "AppLogs":
        """从日志记录创建实例"""
        return cls(
            log_level=LogLevel(record.levelname),
            logger_name=record.name,
            message=record.getMessage(),
            module_name=getattr(record, 'module', None),
            function_name=record.funcName,
            line_number=record.lineno,
            exception_type=getattr(record, 'exc_info', [None])[0].__name__ if record.exc_info and record.exc_info[0] else None,
            exception_message=str(getattr(record, 'exc_info', [None, None])[1]) if record.exc_info and record.exc_info[1] else None,
            traceback=record.exc_text if hasattr(record, 'exc_text') else None,
            timestamp=datetime.fromtimestamp(record.created),
            extra_data=extra_data or {}
        )
    
    def is_error(self) -> bool:
        """是否为错误日志"""
        return self.log_level in [LogLevel.ERROR, LogLevel.CRITICAL]
    
    def is_warning(self) -> bool:
        """是否为警告日志"""
        return self.log_level == LogLevel.WARNING
    
    def has_exception(self) -> bool:
        """是否包含异常信息"""
        return self.exception_type is not None


# 数据库表创建SQL
CREATE_TABLES_SQL = {
    "analysis_history": """
        CREATE TABLE IF NOT EXISTS analysis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_id TEXT UNIQUE NOT NULL,
            file_hash TEXT NOT NULL,
            file_name TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_size INTEGER NOT NULL,
            analysis_type TEXT NOT NULL,
            analysis_title TEXT DEFAULT '',
            analysis_description TEXT,
            total_rows INTEGER NOT NULL,
            total_columns INTEGER NOT NULL,
            analyzed_columns TEXT DEFAULT '[]',
            time_column TEXT,
            status TEXT DEFAULT 'pending',
            execution_time_ms INTEGER DEFAULT 0,
            error_message TEXT,
            result_summary TEXT DEFAULT '{}',
            has_charts BOOLEAN DEFAULT FALSE,
            chart_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP,
            metadata TEXT DEFAULT '{}',
            tags TEXT DEFAULT '[]',
            is_favorite BOOLEAN DEFAULT FALSE
        );
    """,
    
    "analysis_charts": """
        CREATE TABLE IF NOT EXISTS analysis_charts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_id TEXT NOT NULL,
            chart_id TEXT UNIQUE NOT NULL,
            chart_type TEXT NOT NULL,
            chart_title TEXT DEFAULT '',
            chart_description TEXT,
            chart_config TEXT DEFAULT '{}',
            data_columns TEXT DEFAULT '[]',
            chart_order INTEGER DEFAULT 0,
            image_path TEXT,
            image_format TEXT DEFAULT 'png',
            image_size INTEGER DEFAULT 0,
            is_interactive BOOLEAN DEFAULT FALSE,
            html_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT DEFAULT '{}',
            FOREIGN KEY (analysis_id) REFERENCES analysis_history (analysis_id) ON DELETE CASCADE
        );
    """,
    
    "user_settings": """
        CREATE TABLE IF NOT EXISTS user_settings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            setting_key TEXT UNIQUE NOT NULL,
            setting_value TEXT NOT NULL,
            setting_type TEXT DEFAULT 'string',
            category TEXT DEFAULT 'general',
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """,
    
    "app_logs": """
        CREATE TABLE IF NOT EXISTS app_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            log_level TEXT NOT NULL,
            logger_name TEXT NOT NULL,
            message TEXT NOT NULL,
            module_name TEXT,
            function_name TEXT,
            line_number INTEGER,
            exception_type TEXT,
            exception_message TEXT,
            traceback TEXT,
            user_id TEXT,
            session_id TEXT,
            request_id TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            extra_data TEXT DEFAULT '{}'
        );
    """
}

# 创建索引SQL
CREATE_INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_analysis_history_file_hash ON analysis_history(file_hash);",
    "CREATE INDEX IF NOT EXISTS idx_analysis_history_status ON analysis_history(status);",
    "CREATE INDEX IF NOT EXISTS idx_analysis_history_created_at ON analysis_history(created_at);",
    "CREATE INDEX IF NOT EXISTS idx_analysis_charts_analysis_id ON analysis_charts(analysis_id);",
    "CREATE INDEX IF NOT EXISTS idx_user_settings_category ON user_settings(category);",
    "CREATE INDEX IF NOT EXISTS idx_app_logs_level ON app_logs(log_level);",
    "CREATE INDEX IF NOT EXISTS idx_app_logs_timestamp ON app_logs(timestamp);",
]