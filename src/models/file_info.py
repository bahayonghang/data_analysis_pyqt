"""
文件信息数据模型
包含文件基本信息、数据统计和时间列检测结果
"""

import hashlib
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator, root_validator


class FileType(str, Enum):
    """支持的文件类型"""
    CSV = "csv"
    PARQUET = "parquet"
    EXCEL = "excel"
    JSON = "json"


class TimeColumnType(str, Enum):
    """时间列类型"""
    DATETIME = "datetime"
    TAG_TIME = "tagtime"
    TIMESTAMP = "timestamp"
    DATE = "date"
    UNKNOWN = "unknown"


class TimeColumnInfo(BaseModel):
    """时间列信息"""
    column_name: str = Field(..., description="列名")
    column_type: TimeColumnType = Field(..., description="时间列类型")
    sample_values: List[str] = Field(default_factory=list, description="样本值")
    format_pattern: Optional[str] = Field(None, description="时间格式模式")
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0, description="检测置信度")

    def __str__(self) -> str:
        return f"{self.column_name} ({self.column_type.value})"


class DataQualityInfo(BaseModel):
    """数据质量信息"""
    total_rows: int = Field(ge=0, description="总行数")
    total_columns: int = Field(ge=0, description="总列数")
    missing_values_count: int = Field(ge=0, description="缺失值总数")
    duplicate_rows_count: int = Field(ge=0, description="重复行数")
    memory_usage_mb: float = Field(ge=0, description="内存使用量(MB)")
    
    @property
    def missing_value_percentage(self) -> float:
        """缺失值百分比"""
        total_cells = self.total_rows * self.total_columns
        return (self.missing_values_count / total_cells * 100) if total_cells > 0 else 0.0
    
    @property
    def duplicate_percentage(self) -> float:
        """重复行百分比"""
        return (self.duplicate_rows_count / self.total_rows * 100) if self.total_rows > 0 else 0.0


class FileInfo(BaseModel):
    """文件信息模型"""
    
    # 基本文件信息
    file_path: str = Field(..., description="文件路径")
    file_name: str = Field(..., description="文件名")
    file_type: FileType = Field(..., description="文件类型")
    file_size_bytes: int = Field(ge=0, description="文件大小(字节)")
    
    # 文件哈希和时间戳
    file_hash: str = Field(..., description="文件MD5哈希值")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    modified_at: datetime = Field(..., description="文件修改时间")
    
    # 数据质量信息
    data_quality: DataQualityInfo = Field(..., description="数据质量信息")
    
    # 列信息
    column_names: List[str] = Field(default_factory=list, description="列名列表")
    column_types: Dict[str, str] = Field(default_factory=dict, description="列类型映射")
    
    # 时间列检测结果
    time_columns: List[TimeColumnInfo] = Field(default_factory=list, description="检测到的时间列")
    primary_time_column: Optional[str] = Field(None, description="主要时间列")
    
    # 处理状态
    is_loaded: bool = Field(default=False, description="是否已加载")
    load_error: Optional[str] = Field(None, description="加载错误信息")
    
    @validator('file_path')
    def validate_file_path(cls, v):
        """验证文件路径"""
        if not v:
            raise ValueError("文件路径不能为空")
        path = Path(v)
        if not path.exists():
            raise ValueError(f"文件不存在: {v}")
        return str(path.absolute())
    
    @validator('file_hash')
    def validate_file_hash(cls, v):
        """验证文件哈希值格式"""
        if not v or len(v) != 32:
            raise ValueError("文件哈希值必须是32位MD5值")
        return v.lower()
    
    @property
    def file_size_mb(self) -> float:
        """文件大小(MB)"""
        return self.file_size_bytes / (1024 * 1024)
    
    @property
    def has_time_column(self) -> bool:
        """是否包含时间列"""
        return len(self.time_columns) > 0
    
    @property
    def time_column_count(self) -> int:
        """时间列数量"""
        return len(self.time_columns)
    
    @classmethod
    def create_from_file(cls, file_path: Union[str, Path]) -> "FileInfo":
        """从文件创建FileInfo实例"""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        # 计算文件哈希
        file_hash = cls._calculate_file_hash(path)
        
        # 获取文件信息
        stat = path.stat()
        
        # 确定文件类型
        file_type = cls._detect_file_type(path)
        
        # 创建基础数据质量信息（将由数据加载器填充具体值）
        data_quality = DataQualityInfo(
            total_rows=0,
            total_columns=0,
            missing_values_count=0,
            duplicate_rows_count=0,
            memory_usage_mb=0.0
        )
        
        return cls(
            file_path=str(path),
            file_name=path.name,
            file_type=file_type,
            file_size_bytes=stat.st_size,
            file_hash=file_hash,
            modified_at=datetime.fromtimestamp(stat.st_mtime),
            data_quality=data_quality
        )
    
    @staticmethod
    def _calculate_file_hash(file_path: Path) -> str:
        """计算文件MD5哈希值"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    @staticmethod
    def _detect_file_type(file_path: Path) -> FileType:
        """检测文件类型"""
        suffix = file_path.suffix.lower()
        
        if suffix == '.csv':
            return FileType.CSV
        elif suffix == '.parquet':
            return FileType.PARQUET
        elif suffix in ['.xlsx', '.xls']:
            return FileType.EXCEL
        elif suffix == '.json':
            return FileType.JSON
        else:
            raise ValueError(f"不支持的文件类型: {suffix}")
    
    def update_data_info(
        self,
        column_names: List[str],
        column_types: Dict[str, str],
        data_quality: DataQualityInfo,
        time_columns: Optional[List[TimeColumnInfo]] = None
    ) -> None:
        """更新数据信息"""
        self.column_names = column_names
        self.column_types = column_types
        self.data_quality = data_quality
        
        if time_columns:
            self.time_columns = time_columns
            # 自动选择主要时间列（置信度最高的）
            if self.time_columns:
                primary_column = max(self.time_columns, key=lambda x: x.confidence_score)
                self.primary_time_column = primary_column.column_name
        
        self.is_loaded = True
        self.load_error = None
    
    def mark_load_error(self, error_message: str) -> None:
        """标记加载错误"""
        self.load_error = error_message
        self.is_loaded = False
    
    def get_time_column_by_name(self, column_name: str) -> Optional[TimeColumnInfo]:
        """根据列名获取时间列信息"""
        for time_col in self.time_columns:
            if time_col.column_name == column_name:
                return time_col
        return None
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return self.dict()
    
    @classmethod
    def from_dict(cls, data: Dict) -> "FileInfo":
        """从字典创建实例"""
        return cls.parse_obj(data)
    
    def __str__(self) -> str:
        return f"FileInfo({self.file_name}, {self.file_type.value}, {self.file_size_mb:.2f}MB)"
    
    def __repr__(self) -> str:
        return (f"FileInfo(file_name='{self.file_name}', file_type='{self.file_type.value}', "
                f"rows={self.data_quality.total_rows}, cols={self.data_quality.total_columns})")