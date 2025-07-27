"""
扩展分析结果数据模型
为AnalysisEngine提供额外的数据结构
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from datetime import datetime


@dataclass
class DescriptiveStats:
    """描述性统计结果"""
    statistics: Dict[str, Dict[str, Any]]
    method: str
    config: Any
    computed_at: datetime = None
    
    def __post_init__(self):
        if self.computed_at is None:
            self.computed_at = datetime.now()
    
    def get_column_stats(self, column: str) -> Optional[Dict[str, Any]]:
        """获取指定列的统计信息"""
        return self.statistics.get(column)
    
    def get_summary(self) -> Dict[str, Any]:
        """获取统计摘要"""
        if not self.statistics:
            return {}
        
        total_columns = len(self.statistics)
        summary = {
            'total_columns_analyzed': total_columns,
            'method': self.method,
            'computed_at': self.computed_at
        }
        
        # 计算整体统计
        all_means = [stats.get('mean') for stats in self.statistics.values() if stats.get('mean') is not None]
        all_stds = [stats.get('std') for stats in self.statistics.values() if stats.get('std') is not None]
        
        if all_means:
            summary['overall_mean_average'] = sum(all_means) / len(all_means)
        if all_stds:
            summary['overall_std_average'] = sum(all_stds) / len(all_stds)
        
        return summary


@dataclass
class CorrelationMatrix:
    """关联矩阵结果"""
    matrix: List[List[float]]
    columns: List[str]
    method: str
    threshold: float
    computed_at: datetime = None
    
    def __post_init__(self):
        if self.computed_at is None:
            self.computed_at = datetime.now()
    
    def get_correlation(self, col1: str, col2: str) -> Optional[float]:
        """获取两列之间的相关系数"""
        try:
            idx1 = self.columns.index(col1)
            idx2 = self.columns.index(col2)
            return self.matrix[idx1][idx2]
        except (ValueError, IndexError):
            return None
    
    def get_high_correlations(self, min_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """获取高相关性对"""
        threshold = min_threshold or self.threshold
        high_corrs = []
        
        for i, col1 in enumerate(self.columns):
            for j, col2 in enumerate(self.columns):
                if i < j:  # 避免重复和自相关
                    corr_value = self.matrix[i][j]
                    if abs(corr_value) >= threshold:
                        high_corrs.append({
                            'column1': col1,
                            'column2': col2,
                            'correlation': corr_value,
                            'abs_correlation': abs(corr_value)
                        })
        
        # 按绝对相关系数排序
        high_corrs.sort(key=lambda x: x['abs_correlation'], reverse=True)
        return high_corrs
    
    def get_summary(self) -> Dict[str, Any]:
        """获取关联分析摘要"""
        import numpy as np
        
        matrix_array = np.array(self.matrix)
        
        # 获取上三角矩阵（排除对角线）
        upper_triangle = matrix_array[np.triu_indices_from(matrix_array, k=1)]
        
        summary = {
            'method': self.method,
            'matrix_size': len(self.columns),
            'total_pairs': len(upper_triangle),
            'computed_at': self.computed_at
        }
        
        if len(upper_triangle) > 0:
            summary.update({
                'max_correlation': float(np.max(np.abs(upper_triangle))),
                'min_correlation': float(np.min(np.abs(upper_triangle))),
                'mean_correlation': float(np.mean(np.abs(upper_triangle))),
                'high_correlation_count': len(self.get_high_correlations())
            })
        
        return summary


@dataclass
class AnomalyDetectionResult:
    """异常值检测结果"""
    anomalies: Dict[str, Dict[str, Any]]
    method: str
    threshold: float
    computed_at: datetime = None
    
    def __post_init__(self):
        if self.computed_at is None:
            self.computed_at = datetime.now()
    
    def get_column_anomalies(self, column: str) -> Optional[Dict[str, Any]]:
        """获取指定列的异常值信息"""
        return self.anomalies.get(column)
    
    def get_total_anomalies(self) -> int:
        """获取所有列的异常值总数"""
        total = 0
        for col_anomalies in self.anomalies.values():
            total += col_anomalies.get('outlier_count', 0)
        return total
    
    def get_anomaly_columns(self) -> List[str]:
        """获取有异常值的列名"""
        return [col for col, anomalies in self.anomalies.items() 
                if anomalies.get('outlier_count', 0) > 0]
    
    def get_summary(self) -> Dict[str, Any]:
        """获取异常值检测摘要"""
        total_anomalies = self.get_total_anomalies()
        anomaly_columns = self.get_anomaly_columns()
        
        # 计算平均异常值百分比
        percentages = [anomalies.get('outlier_percentage', 0) 
                      for anomalies in self.anomalies.values()]
        avg_percentage = sum(percentages) / len(percentages) if percentages else 0
        
        summary = {
            'method': self.method,
            'threshold': self.threshold,
            'total_columns_analyzed': len(self.anomalies),
            'columns_with_anomalies': len(anomaly_columns),
            'total_anomalies': total_anomalies,
            'average_anomaly_percentage': avg_percentage,
            'computed_at': self.computed_at
        }
        
        return summary


@dataclass
class TimeSeriesAnalysis:
    """时间序列分析结果"""
    stationarity_tests: Dict[str, Dict[str, Any]]
    trend_analysis: Dict[str, Dict[str, Any]]
    seasonality_analysis: Dict[str, Dict[str, Any]]
    time_column: str
    computed_at: datetime = None
    
    def __post_init__(self):
        if self.computed_at is None:
            self.computed_at = datetime.now()
    
    def get_column_stationarity(self, column: str) -> Optional[Dict[str, Any]]:
        """获取指定列的平稳性检验结果"""
        return self.stationarity_tests.get(column)
    
    def get_column_trend(self, column: str) -> Optional[Dict[str, Any]]:
        """获取指定列的趋势分析结果"""
        return self.trend_analysis.get(column)
    
    def get_column_seasonality(self, column: str) -> Optional[Dict[str, Any]]:
        """获取指定列的季节性分析结果"""
        return self.seasonality_analysis.get(column)
    
    def get_stationary_columns(self) -> List[str]:
        """获取平稳的列名"""
        stationary_cols = []
        for col, tests in self.stationarity_tests.items():
            # 检查ADF和KPSS检验结果
            adf_stationary = tests.get('adf', {}).get('is_stationary', False)
            kpss_stationary = tests.get('kpss', {}).get('is_stationary', False)
            
            # 如果ADF和KPSS都认为是平稳的，则认为是平稳的
            if adf_stationary and kpss_stationary:
                stationary_cols.append(col)
            # 如果只有一个检验，则以该检验为准
            elif len(tests) == 1:
                test_result = list(tests.values())[0]
                if test_result.get('is_stationary', False):
                    stationary_cols.append(col)
        
        return stationary_cols
    
    def get_trending_columns(self) -> List[str]:
        """获取有显著趋势的列名"""
        trending_cols = []
        for col, trend in self.trend_analysis.items():
            if trend.get('is_significant', False):
                trending_cols.append(col)
        return trending_cols
    
    def get_seasonal_columns(self) -> List[str]:
        """获取有季节性的列名"""
        seasonal_cols = []
        for col, seasonality in self.seasonality_analysis.items():
            for period, result in seasonality.items():
                if result.get('is_seasonal', False):
                    seasonal_cols.append(col)
                    break
        return seasonal_cols
    
    def get_summary(self) -> Dict[str, Any]:
        """获取时间序列分析摘要"""
        stationary_cols = self.get_stationary_columns()
        trending_cols = self.get_trending_columns()
        seasonal_cols = self.get_seasonal_columns()
        
        summary = {
            'time_column': self.time_column,
            'total_columns_analyzed': len(self.stationarity_tests),
            'stationary_columns_count': len(stationary_cols),
            'trending_columns_count': len(trending_cols),
            'seasonal_columns_count': len(seasonal_cols),
            'stationary_columns': stationary_cols,
            'trending_columns': trending_cols,
            'seasonal_columns': seasonal_cols,
            'computed_at': self.computed_at
        }
        
        return summary


@dataclass
class AnalysisResult:
    """完整的分析结果"""
    descriptive_stats: Optional[DescriptiveStats] = None
    correlation_matrix: Optional[CorrelationMatrix] = None
    anomaly_detection: Optional[AnomalyDetectionResult] = None
    time_series_analysis: Optional[TimeSeriesAnalysis] = None
    column_info: Optional[Dict[str, Any]] = None
    analysis_config: Optional[Any] = None
    
    # 元数据
    analysis_id: Optional[str] = None
    file_path: Optional[str] = None
    file_hash: Optional[str] = None
    created_at: datetime = None
    execution_time_ms: Optional[int] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.analysis_id is None:
            # 生成分析ID
            import uuid
            self.analysis_id = str(uuid.uuid4())[:8]
    
    def get_overall_summary(self) -> Dict[str, Any]:
        """获取整体分析摘要"""
        summary = {
            'analysis_id': self.analysis_id,
            'created_at': self.created_at,
            'execution_time_ms': self.execution_time_ms,
            'has_descriptive_stats': self.descriptive_stats is not None,
            'has_correlation_analysis': self.correlation_matrix is not None,
            'has_anomaly_detection': self.anomaly_detection is not None,
            'has_time_series_analysis': self.time_series_analysis is not None
        }
        
        # 添加各个分析模块的摘要
        if self.descriptive_stats:
            summary['descriptive_summary'] = self.descriptive_stats.get_summary()
        
        if self.correlation_matrix:
            summary['correlation_summary'] = self.correlation_matrix.get_summary()
        
        if self.anomaly_detection:
            summary['anomaly_summary'] = self.anomaly_detection.get_summary()
        
        if self.time_series_analysis:
            summary['time_series_summary'] = self.time_series_analysis.get_summary()
        
        if self.column_info:
            summary['column_info'] = self.column_info
        
        return summary
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式（用于序列化）"""
        from dataclasses import asdict
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnalysisResult":
        """从字典创建实例"""
        # 这里需要根据实际需要进行反序列化处理
        return cls(**data)