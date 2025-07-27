"""
时间列检测器
自动检测数据中的时间列（DateTime和tagTime）
"""

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl
from dataclasses import dataclass
from enum import Enum

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False
    import pandas as pd

from ..models import TimeColumnInfo, TimeColumnType
from ..models.validation import DataTypeUtils
from ..utils.basic_logging import LoggerMixin


class TimePatternType(Enum):
    """时间模式类型"""
    ISO_DATETIME = "iso_datetime"  # 2023-01-01T10:00:00
    STANDARD_DATETIME = "standard_datetime"  # 2023-01-01 10:00:00
    DATE_ONLY = "date_only"  # 2023-01-01
    TIMESTAMP_UNIX = "timestamp_unix"  # 1672574400
    TIMESTAMP_MS = "timestamp_ms"  # 1672574400000
    TAG_TIME = "tag_time"  # 特殊的tagTime格式
    CUSTOM = "custom"  # 自定义格式


@dataclass
class TimePattern:
    """时间模式定义"""
    pattern_type: TimePatternType
    regex: str
    format_string: str
    confidence_weight: float
    description: str


@dataclass 
class TimeDetectionResult:
    """时间检测结果"""
    column_name: str
    is_time_column: bool
    time_type: TimeColumnType
    confidence_score: float
    pattern_type: Optional[TimePatternType]
    format_pattern: Optional[str]
    sample_values: List[str]
    statistics: Dict[str, Any]


class TimeColumnDetector(LoggerMixin):
    """时间列检测器"""
    
    def __init__(self):
        self.time_patterns = self._initialize_patterns()
        self.time_keywords = [
            'time', 'date', 'datetime', 'timestamp', 'created', 'updated', 
            'modified', 'start', 'end', 'begin', 'finish', 'birth', 'death',
            '时间', '日期', '创建', '更新', '修改', '开始', '结束'
        ]
        self.tag_time_keywords = [
            'tagtime', 'tag_time', 'tag-time', 'tagging_time',
            'tag时间', 'tag_时间'
        ]
    
    def _initialize_patterns(self) -> List[TimePattern]:
        """初始化时间模式"""
        patterns = [
            # ISO 8601格式
            TimePattern(
                TimePatternType.ISO_DATETIME,
                r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?',
                '%Y-%m-%dT%H:%M:%S',
                0.95,
                "ISO 8601日期时间格式"
            ),
            
            # 标准日期时间格式
            TimePattern(
                TimePatternType.STANDARD_DATETIME,
                r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?',
                '%Y-%m-%d %H:%M:%S',
                0.90,
                "标准日期时间格式"
            ),
            
            # 日期格式
            TimePattern(
                TimePatternType.DATE_ONLY,
                r'\d{4}-\d{2}-\d{2}',
                '%Y-%m-%d',
                0.85,
                "日期格式"
            ),
            
            # Unix时间戳（秒）
            TimePattern(
                TimePatternType.TIMESTAMP_UNIX,
                r'^\d{10}$',
                'timestamp',
                0.80,
                "Unix时间戳（秒）"
            ),
            
            # Unix时间戳（毫秒）
            TimePattern(
                TimePatternType.TIMESTAMP_MS,
                r'^\d{13}$',
                'timestamp_ms',
                0.80,
                "Unix时间戳（毫秒）"
            ),
            
            # 其他常见格式
            TimePattern(
                TimePatternType.CUSTOM,
                r'\d{2}/\d{2}/\d{4}',
                '%m/%d/%Y',
                0.75,
                "美式日期格式"
            ),
            
            TimePattern(
                TimePatternType.CUSTOM,
                r'\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2}',
                '%m/%d/%Y %H:%M:%S',
                0.80,
                "美式日期时间格式"
            ),
            
            TimePattern(
                TimePatternType.CUSTOM,
                r'\d{4}/\d{2}/\d{2}',
                '%Y/%m/%d',
                0.75,
                "斜杠分隔日期格式"
            ),
        ]
        
        return patterns
    
    def detect_time_columns(self, df: Any, sample_size: int = 100) -> List[TimeDetectionResult]:
        """检测时间列"""
        results = []
        
        try:
            if HAS_POLARS and isinstance(df, pl.DataFrame):
                columns = df.columns
                for col in columns:
                    result = self._detect_time_column_polars(df, col, sample_size)
                    results.append(result)
            else:
                columns = df.columns
                for col in columns:
                    result = self._detect_time_column_pandas(df, col, sample_size)
                    results.append(result)
            
            # 按置信度排序
            results.sort(key=lambda x: x.confidence_score, reverse=True)
            
            return results
            
        except Exception as e:
            self.logger.error(f"时间列检测失败: {str(e)}")
            return []
    
    def _detect_time_column_polars(self, df: "pl.DataFrame", column: str, sample_size: int) -> TimeDetectionResult:
        """使用Polars检测时间列"""
        try:
            # 获取样本数据
            sample_df = df.select(column).head(sample_size)
            sample_values = [str(v) for v in sample_df[column].to_list() if v is not None]
            
            return self._analyze_column_values(column, sample_values)
            
        except Exception as e:
            self.logger.error(f"Polars时间列检测失败: {column}, 错误: {str(e)}")
            return self._create_negative_result(column, [])
    
    def _detect_time_column_pandas(self, df: Any, column: str, sample_size: int) -> TimeDetectionResult:
        """使用Pandas检测时间列"""
        try:
            # 获取样本数据
            sample_series = df[column].head(sample_size).dropna()
            sample_values = [str(v) for v in sample_series.tolist()]
            
            return self._analyze_column_values(column, sample_values)
            
        except Exception as e:
            self.logger.error(f"Pandas时间列检测失败: {column}, 错误: {str(e)}")
            return self._create_negative_result(column, [])
    
    def _analyze_column_values(self, column: str, sample_values: List[str]) -> TimeDetectionResult:
        """分析列值以检测时间特征"""
        if not sample_values:
            return self._create_negative_result(column, sample_values)
        
        # 基于列名的初步评分
        name_score = self._score_column_name(column)
        
        # 基于值的模式匹配评分
        pattern_scores = self._score_value_patterns(sample_values)
        
        # 统计信息
        statistics = self._calculate_statistics(sample_values)
        
        # 综合评分
        final_score, best_pattern, time_type = self._calculate_final_score(
            name_score, pattern_scores, statistics
        )
        
        # 确定格式模式
        format_pattern = best_pattern.format_string if best_pattern else None
        pattern_type = best_pattern.pattern_type if best_pattern else None
        
        return TimeDetectionResult(
            column_name=column,
            is_time_column=final_score >= 0.6,
            time_type=time_type,
            confidence_score=final_score,
            pattern_type=pattern_type,
            format_pattern=format_pattern,
            sample_values=sample_values[:10],  # 只保留前10个样本
            statistics=statistics
        )
    
    def _score_column_name(self, column_name: str) -> Tuple[float, TimeColumnType]:
        """基于列名评分"""
        column_lower = column_name.lower()
        
        # 检查tagTime关键词
        for keyword in self.tag_time_keywords:
            if keyword in column_lower:
                return 0.8, TimeColumnType.TAG_TIME
        
        # 检查一般时间关键词
        for keyword in self.time_keywords:
            if keyword in column_lower:
                if 'date' in column_lower and 'time' not in column_lower:
                    return 0.6, TimeColumnType.DATE
                else:
                    return 0.6, TimeColumnType.DATETIME
        
        return 0.0, TimeColumnType.UNKNOWN
    
    def _score_value_patterns(self, sample_values: List[str]) -> List[Tuple[TimePattern, float]]:
        """基于值模式评分"""
        pattern_scores = []
        
        for pattern in self.time_patterns:
            match_count = 0
            total_checked = 0
            
            for value in sample_values[:50]:  # 只检查前50个值
                if not value or value.strip() == '':
                    continue
                    
                total_checked += 1
                value_clean = value.strip()
                
                # 正则匹配
                if re.match(pattern.regex, value_clean):
                    match_count += 1
                    continue
                
                # 尝试解析时间戳
                if pattern.pattern_type in [TimePatternType.TIMESTAMP_UNIX, TimePatternType.TIMESTAMP_MS]:
                    if self._is_valid_timestamp(value_clean, pattern.pattern_type):
                        match_count += 1
            
            if total_checked > 0:
                match_ratio = match_count / total_checked
                score = match_ratio * pattern.confidence_weight
                pattern_scores.append((pattern, score))
        
        # 按得分排序
        pattern_scores.sort(key=lambda x: x[1], reverse=True)
        return pattern_scores
    
    def _is_valid_timestamp(self, value: str, pattern_type: TimePatternType) -> bool:
        """验证时间戳"""
        try:
            timestamp = int(value)
            
            if pattern_type == TimePatternType.TIMESTAMP_UNIX:
                # Unix时间戳（秒）：1970年到2100年
                return 0 <= timestamp <= 4102444800
            elif pattern_type == TimePatternType.TIMESTAMP_MS:
                # Unix时间戳（毫秒）
                return 0 <= timestamp <= 4102444800000
            
            return False
        except (ValueError, OverflowError):
            return False
    
    def _calculate_statistics(self, sample_values: List[str]) -> Dict[str, Any]:
        """计算统计信息"""
        total_values = len(sample_values)
        non_empty_values = [v for v in sample_values if v and v.strip()]
        
        statistics = {
            'total_samples': total_values,
            'non_empty_count': len(non_empty_values),
            'empty_ratio': (total_values - len(non_empty_values)) / total_values if total_values > 0 else 1.0,
            'unique_count': len(set(non_empty_values)),
            'avg_length': sum(len(v) for v in non_empty_values) / len(non_empty_values) if non_empty_values else 0,
            'min_length': min(len(v) for v in non_empty_values) if non_empty_values else 0,
            'max_length': max(len(v) for v in non_empty_values) if non_empty_values else 0,
        }
        
        # 尝试解析数值统计
        numeric_values = []
        for value in non_empty_values[:20]:
            try:
                numeric_values.append(float(value))
            except ValueError:
                pass
        
        if numeric_values:
            statistics.update({
                'numeric_ratio': len(numeric_values) / len(non_empty_values),
                'numeric_min': min(numeric_values),
                'numeric_max': max(numeric_values),
                'numeric_avg': sum(numeric_values) / len(numeric_values)
            })
        
        return statistics
    
    def _calculate_final_score(self, name_score: Tuple[float, TimeColumnType], 
                             pattern_scores: List[Tuple[TimePattern, float]], 
                             statistics: Dict[str, Any]) -> Tuple[float, Optional[TimePattern], TimeColumnType]:
        """计算最终评分"""
        base_name_score, name_time_type = name_score
        
        # 获取最佳模式评分
        best_pattern_score = 0.0
        best_pattern = None
        if pattern_scores:
            best_pattern, best_pattern_score = pattern_scores[0]
        
        # 统计信息调整
        stat_adjustment = 1.0
        
        # 空值比例过高，降低分数
        if statistics['empty_ratio'] > 0.5:
            stat_adjustment *= 0.5
        
        # 唯一值比例过低，可能不是时间列
        if statistics['non_empty_count'] > 0:
            unique_ratio = statistics['unique_count'] / statistics['non_empty_count']
            if unique_ratio < 0.1:
                stat_adjustment *= 0.3
        
        # 长度过短或过长
        avg_length = statistics['avg_length']
        if avg_length < 4 or avg_length > 30:
            stat_adjustment *= 0.7
        
        # 综合计算
        # 权重：模式匹配70%，列名30%
        final_score = (best_pattern_score * 0.7 + base_name_score * 0.3) * stat_adjustment
        
        # 确定时间类型
        time_type = TimeColumnType.UNKNOWN
        if final_score >= 0.6:
            if best_pattern:
                if best_pattern.pattern_type == TimePatternType.DATE_ONLY:
                    time_type = TimeColumnType.DATE
                elif best_pattern.pattern_type in [TimePatternType.TIMESTAMP_UNIX, TimePatternType.TIMESTAMP_MS]:
                    time_type = TimeColumnType.TIMESTAMP
                else:
                    time_type = TimeColumnType.DATETIME
            
            # tagTime列名优先级高
            if name_time_type == TimeColumnType.TAG_TIME:
                time_type = TimeColumnType.TAG_TIME
        
        return final_score, best_pattern, time_type
    
    def _create_negative_result(self, column: str, sample_values: List[str]) -> TimeDetectionResult:
        """创建负面检测结果"""
        return TimeDetectionResult(
            column_name=column,
            is_time_column=False,
            time_type=TimeColumnType.UNKNOWN,
            confidence_score=0.0,
            pattern_type=None,
            format_pattern=None,
            sample_values=sample_values[:10],
            statistics={}
        )
    
    def convert_to_time_column_info(self, result: TimeDetectionResult) -> TimeColumnInfo:
        """转换为TimeColumnInfo模型"""
        return TimeColumnInfo(
            column_name=result.column_name,
            column_type=result.time_type,
            sample_values=result.sample_values,
            format_pattern=result.format_pattern,
            confidence_score=result.confidence_score
        )
    
    def validate_time_parsing(self, sample_values: List[str], format_pattern: str) -> Tuple[bool, int]:
        """验证时间解析"""
        if not sample_values or not format_pattern:
            return False, 0
        
        success_count = 0
        total_count = 0
        
        for value in sample_values[:20]:
            if not value or value.strip() == '':
                continue
                
            total_count += 1
            
            try:
                if format_pattern == 'timestamp':
                    timestamp = int(value)
                    datetime.fromtimestamp(timestamp)
                    success_count += 1
                elif format_pattern == 'timestamp_ms':
                    timestamp = int(value) / 1000
                    datetime.fromtimestamp(timestamp)
                    success_count += 1
                else:
                    datetime.strptime(value.strip(), format_pattern)
                    success_count += 1
            except (ValueError, OSError, OverflowError):
                pass
        
        if total_count == 0:
            return False, 0
        
        success_rate = success_count / total_count
        return success_rate >= 0.8, success_count
    
    def suggest_format_improvements(self, result: TimeDetectionResult) -> List[str]:
        """建议格式改进"""
        suggestions = []
        
        if not result.is_time_column:
            return ["该列不被识别为时间列"]
        
        stats = result.statistics
        
        # 检查空值比例
        if stats.get('empty_ratio', 0) > 0.3:
            suggestions.append(f"空值比例较高 ({stats['empty_ratio']:.1%})，建议清理数据")
        
        # 检查格式一致性
        if stats.get('unique_count', 0) / max(stats.get('non_empty_count', 1), 1) > 0.8:
            suggestions.append("时间格式较为分散，建议标准化格式")
        
        # 检查长度一致性
        min_len = stats.get('min_length', 0)
        max_len = stats.get('max_length', 0)
        if max_len - min_len > 5:
            suggestions.append("时间字符串长度不一致，建议统一格式")
        
        if not suggestions:
            suggestions.append("时间列格式良好，无需特殊处理")
        
        return suggestions