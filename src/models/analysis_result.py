"""
分析结果数据模型
包含数据分析的完整结果和统计信息
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np
from pydantic import BaseModel, Field, validator, root_validator


class AnalysisType(str, Enum):
    """分析类型"""
    DESCRIPTIVE = "descriptive"  # 描述性统计
    CORRELATION = "correlation"  # 相关性分析
    DISTRIBUTION = "distribution"  # 分布分析
    OUTLIER = "outlier"  # 异常值检测
    STATIONARITY = "stationarity"  # 平稳性检验
    TREND = "trend"  # 趋势分析
    SEASONAL = "seasonal"  # 季节性分析
    CUSTOM = "custom"  # 自定义分析


class StatisticalSummary(BaseModel):
    """统计摘要信息"""
    count: int = Field(ge=0, description="数据点数量")
    mean: Optional[float] = Field(None, description="均值")
    median: Optional[float] = Field(None, description="中位数")
    std: Optional[float] = Field(None, description="标准差")
    min_value: Optional[float] = Field(None, description="最小值")
    max_value: Optional[float] = Field(None, description="最大值")
    q25: Optional[float] = Field(None, description="25%分位数")
    q75: Optional[float] = Field(None, description="75%分位数")
    skewness: Optional[float] = Field(None, description="偏度")
    kurtosis: Optional[float] = Field(None, description="峰度")
    
    @property
    def range_value(self) -> Optional[float]:
        """值域"""
        if self.min_value is not None and self.max_value is not None:
            return self.max_value - self.min_value
        return None
    
    @property
    def iqr(self) -> Optional[float]:
        """四分位距"""
        if self.q25 is not None and self.q75 is not None:
            return self.q75 - self.q25
        return None
    
    @property
    def coefficient_of_variation(self) -> Optional[float]:
        """变异系数"""
        if self.mean is not None and self.std is not None and self.mean != 0:
            return self.std / abs(self.mean)
        return None


class CorrelationResult(BaseModel):
    """相关性分析结果"""
    method: str = Field(..., description="相关性计算方法 (pearson, spearman, kendall)")
    correlation_matrix: Dict[str, Dict[str, float]] = Field(..., description="相关性矩阵")
    p_values: Optional[Dict[str, Dict[str, float]]] = Field(None, description="p值矩阵")
    significant_pairs: List[Dict[str, Any]] = Field(default_factory=list, description="显著相关对")
    
    @validator('method')
    def validate_method(cls, v):
        """验证相关性方法"""
        allowed_methods = ['pearson', 'spearman', 'kendall']
        if v not in allowed_methods:
            raise ValueError(f"不支持的相关性方法: {v}")
        return v


class OutlierResult(BaseModel):
    """异常值检测结果"""
    method: str = Field(..., description="检测方法")
    outlier_indices: List[int] = Field(default_factory=list, description="异常值索引")
    outlier_scores: Optional[List[float]] = Field(None, description="异常值分数")
    threshold: Optional[float] = Field(None, description="检测阈值")
    outlier_percentage: float = Field(ge=0, le=100, description="异常值百分比")
    
    @property
    def outlier_count(self) -> int:
        """异常值数量"""
        return len(self.outlier_indices)


class StationarityResult(BaseModel):
    """平稳性检验结果"""
    test_name: str = Field(..., description="检验方法名称")
    statistic: float = Field(..., description="检验统计量")
    p_value: float = Field(ge=0, le=1, description="p值")
    critical_values: Optional[Dict[str, float]] = Field(None, description="临界值")
    is_stationary: bool = Field(..., description="是否平稳")
    confidence_level: float = Field(default=0.05, ge=0, le=1, description="显著性水平")


class TrendResult(BaseModel):
    """趋势分析结果"""
    trend_type: str = Field(..., description="趋势类型 (increasing, decreasing, no_trend)")
    slope: Optional[float] = Field(None, description="趋势斜率")
    intercept: Optional[float] = Field(None, description="截距")
    r_squared: Optional[float] = Field(None, ge=0, le=1, description="拟合优度")
    p_value: Optional[float] = Field(None, ge=0, le=1, description="趋势显著性p值")
    seasonal_component: Optional[List[float]] = Field(None, description="季节性成分")
    residual_component: Optional[List[float]] = Field(None, description="残差成分")


class AnalysisResult(BaseModel):
    """分析结果模型"""
    
    # 基本信息
    analysis_id: str = Field(..., description="分析ID")
    analysis_type: AnalysisType = Field(..., description="分析类型")
    file_hash: str = Field(..., description="源文件哈希")
    column_names: List[str] = Field(default_factory=list, description="分析的列名")
    
    # 时间戳
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    execution_time_ms: int = Field(ge=0, description="执行时间(毫秒)")
    
    # 分析参数
    parameters: Dict[str, Any] = Field(default_factory=dict, description="分析参数")
    
    # 分析结果
    statistical_summary: Optional[Dict[str, StatisticalSummary]] = Field(None, description="统计摘要")
    correlation_result: Optional[CorrelationResult] = Field(None, description="相关性分析结果")
    outlier_result: Optional[OutlierResult] = Field(None, description="异常值检测结果")
    stationarity_result: Optional[StationarityResult] = Field(None, description="平稳性检验结果")
    trend_result: Optional[TrendResult] = Field(None, description="趋势分析结果")
    
    # 自定义结果
    custom_results: Dict[str, Any] = Field(default_factory=dict, description="自定义分析结果")
    
    # 元数据
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    success: bool = Field(default=True, description="分析是否成功")
    error_message: Optional[str] = Field(None, description="错误信息")
    
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
    
    @property
    def execution_time_seconds(self) -> float:
        """执行时间(秒)"""
        return self.execution_time_ms / 1000.0
    
    @property
    def analyzed_columns_count(self) -> int:
        """分析的列数量"""
        return len(self.column_names)
    
    @property
    def has_statistical_summary(self) -> bool:
        """是否有统计摘要"""
        return self.statistical_summary is not None and len(self.statistical_summary) > 0
    
    @property
    def has_correlation_analysis(self) -> bool:
        """是否有相关性分析"""
        return self.correlation_result is not None
    
    @property
    def has_outlier_detection(self) -> bool:
        """是否有异常值检测"""
        return self.outlier_result is not None
    
    def add_statistical_summary(self, column: str, summary: StatisticalSummary) -> None:
        """添加统计摘要"""
        if self.statistical_summary is None:
            self.statistical_summary = {}
        self.statistical_summary[column] = summary
    
    def add_correlation_result(self, result: CorrelationResult) -> None:
        """添加相关性分析结果"""
        self.correlation_result = result
    
    def add_outlier_result(self, result: OutlierResult) -> None:
        """添加异常值检测结果"""
        self.outlier_result = result
    
    def add_stationarity_result(self, result: StationarityResult) -> None:
        """添加平稳性检验结果"""
        self.stationarity_result = result
    
    def add_trend_result(self, result: TrendResult) -> None:
        """添加趋势分析结果"""
        self.trend_result = result
    
    def add_custom_result(self, key: str, value: Any) -> None:
        """添加自定义结果"""
        self.custom_results[key] = value
    
    def mark_error(self, error_message: str) -> None:
        """标记分析错误"""
        self.success = False
        self.error_message = error_message
    
    def get_summary_for_column(self, column: str) -> Optional[StatisticalSummary]:
        """获取指定列的统计摘要"""
        if self.statistical_summary:
            return self.statistical_summary.get(column)
        return None
    
    def get_result_summary(self) -> Dict[str, Any]:
        """获取结果摘要"""
        summary = {
            "analysis_type": self.analysis_type.value,
            "success": self.success,
            "execution_time_ms": self.execution_time_ms,
            "analyzed_columns": len(self.column_names),
            "has_statistical_summary": self.has_statistical_summary,
            "has_correlation_analysis": self.has_correlation_analysis,
            "has_outlier_detection": self.has_outlier_detection,
        }
        
        if self.outlier_result:
            summary["outlier_count"] = self.outlier_result.outlier_count
            summary["outlier_percentage"] = self.outlier_result.outlier_percentage
        
        if self.stationarity_result:
            summary["is_stationary"] = self.stationarity_result.is_stationary
        
        return summary
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        data = self.dict()
        
        # 处理numpy类型序列化
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            return obj
        
        return convert_numpy(data)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "AnalysisResult":
        """从字典创建实例"""
        return cls.parse_obj(data)
    
    def __str__(self) -> str:
        status = "成功" if self.success else "失败"
        return f"AnalysisResult({self.analysis_type.value}, {status}, {self.execution_time_ms}ms)"
    
    def __repr__(self) -> str:
        return (f"AnalysisResult(id='{self.analysis_id}', type='{self.analysis_type.value}', "
                f"success={self.success}, columns={len(self.column_names)})")