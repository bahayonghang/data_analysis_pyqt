"""
数据验证器
提供数据质量检查和验证功能
"""

from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl
from dataclasses import dataclass
from enum import Enum
import re

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False
    import pandas as pd
    import numpy as np

from ..utils.basic_logging import LoggerMixin


class ValidationLevel(Enum):
    """验证级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationRule(Enum):
    """验证规则"""
    NOT_EMPTY = "not_empty"
    NO_NULL_VALUES = "no_null_values"
    UNIQUE_VALUES = "unique_values"
    NUMERIC_RANGE = "numeric_range"
    STRING_LENGTH = "string_length"
    REGEX_PATTERN = "regex_pattern"
    DATE_RANGE = "date_range"
    DATA_TYPE = "data_type"
    OUTLIER_CHECK = "outlier_check"
    DUPLICATE_CHECK = "duplicate_check"


@dataclass
class ValidationIssue:
    """验证问题"""
    rule: ValidationRule
    level: ValidationLevel
    column: Optional[str]
    message: str
    count: int
    percentage: float
    details: Dict[str, Any]


@dataclass
class ValidationResult:
    """验证结果"""
    is_valid: bool
    issues: List[ValidationIssue]
    statistics: Dict[str, Any]
    score: float  # 数据质量分数 (0-100)
    
    def get_issues_by_level(self, level: ValidationLevel) -> List[ValidationIssue]:
        """按级别获取问题"""
        return [issue for issue in self.issues if issue.level == level]
    
    def get_issues_by_column(self, column: str) -> List[ValidationIssue]:
        """按列获取问题"""
        return [issue for issue in self.issues if issue.column == column]
    
    def has_critical_issues(self) -> bool:
        """是否有严重问题"""
        return any(issue.level == ValidationLevel.CRITICAL for issue in self.issues)
    
    def get_summary(self) -> Dict[str, int]:
        """获取问题摘要"""
        summary = {level.value: 0 for level in ValidationLevel}
        for issue in self.issues:
            summary[issue.level.value] += 1
        return summary


class DataValidator(LoggerMixin):
    """数据验证器"""
    
    def __init__(self):
        self.use_polars = HAS_POLARS
    
    def validate(self, df: Any, rules: Optional[List[ValidationRule]] = None) -> ValidationResult:
        """执行数据验证"""
        if rules is None:
            rules = [
                ValidationRule.NOT_EMPTY,
                ValidationRule.DATA_TYPE,
                ValidationRule.DUPLICATE_CHECK,
                ValidationRule.OUTLIER_CHECK
            ]
        
        issues = []
        statistics = {}
        
        try:
            self.logger.info("开始数据验证")
            
            # 基础统计信息
            statistics.update(self._get_basic_statistics(df))
            
            # 执行各项验证规则
            for rule in rules:
                rule_issues = self._apply_rule(df, rule)
                issues.extend(rule_issues)
            
            # 计算数据质量分数
            score = self._calculate_quality_score(df, issues)
            
            # 判断是否有效
            is_valid = not any(issue.level == ValidationLevel.CRITICAL for issue in issues)
            
            result = ValidationResult(
                is_valid=is_valid,
                issues=issues,
                statistics=statistics,
                score=score
            )
            
            self.logger.info(f"数据验证完成，质量分数: {score:.1f}, 问题数: {len(issues)}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"数据验证失败: {str(e)}")
            return ValidationResult(
                is_valid=False,
                issues=[ValidationIssue(
                    rule=ValidationRule.NOT_EMPTY,
                    level=ValidationLevel.CRITICAL,
                    column=None,
                    message=f"验证过程失败: {str(e)}",
                    count=1,
                    percentage=100.0,
                    details={}
                )],
                statistics={},
                score=0.0
            )
    
    def _get_basic_statistics(self, df: Any) -> Dict[str, Any]:
        """获取基础统计信息"""
        try:
            if self.use_polars and isinstance(df, pl.DataFrame):
                return self._get_statistics_polars(df)
            else:
                return self._get_statistics_pandas(df)
        except Exception as e:
            self.logger.warning(f"获取统计信息失败: {str(e)}")
            return {}
    
    def _get_statistics_polars(self, df: "pl.DataFrame") -> Dict[str, Any]:
        """Polars统计信息"""
        stats = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'memory_usage_mb': df.estimated_size() / (1024 * 1024),
            'column_types': {col: str(df[col].dtype) for col in df.columns},
            'null_counts': {col: df[col].null_count() for col in df.columns},
        }
        
        # 数值列统计
        numeric_columns = [col for col in df.columns if df[col].dtype.is_numeric()]
        if numeric_columns:
            stats['numeric_columns'] = numeric_columns
            stats['numeric_statistics'] = {}
            
            for col in numeric_columns:
                col_stats = {
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'null_count': df[col].null_count()
                }
                stats['numeric_statistics'][col] = col_stats
        
        return stats
    
    def _get_statistics_pandas(self, df: Any) -> Dict[str, Any]:
        """Pandas统计信息"""
        stats = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
            'column_types': {col: str(df[col].dtype) for col in df.columns},
            'null_counts': df.isnull().sum().to_dict(),
        }
        
        # 数值列统计
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_columns:
            stats['numeric_columns'] = numeric_columns
            stats['numeric_statistics'] = df[numeric_columns].describe().to_dict()
        
        return stats
    
    def _apply_rule(self, df: Any, rule: ValidationRule) -> List[ValidationIssue]:
        """应用验证规则"""
        try:
            if rule == ValidationRule.NOT_EMPTY:
                return self._check_not_empty(df)
            elif rule == ValidationRule.NO_NULL_VALUES:
                return self._check_null_values(df)
            elif rule == ValidationRule.DATA_TYPE:
                return self._check_data_types(df)
            elif rule == ValidationRule.DUPLICATE_CHECK:
                return self._check_duplicates(df)
            elif rule == ValidationRule.OUTLIER_CHECK:
                return self._check_outliers(df)
            elif rule == ValidationRule.UNIQUE_VALUES:
                return self._check_unique_values(df)
            else:
                return []
        except Exception as e:
            self.logger.warning(f"验证规则 {rule.value} 执行失败: {str(e)}")
            return []
    
    def _check_not_empty(self, df: Any) -> List[ValidationIssue]:
        """检查数据是否为空"""
        issues = []
        
        total_rows = len(df) if df is not None else 0
        total_columns = len(df.columns) if df is not None and hasattr(df, 'columns') else 0
        
        if total_rows == 0:
            issues.append(ValidationIssue(
                rule=ValidationRule.NOT_EMPTY,
                level=ValidationLevel.CRITICAL,
                column=None,
                message="数据集为空，没有任何行",
                count=0,
                percentage=100.0,
                details={}
            ))
        
        if total_columns == 0:
            issues.append(ValidationIssue(
                rule=ValidationRule.NOT_EMPTY,
                level=ValidationLevel.CRITICAL,
                column=None,
                message="数据集没有列",
                count=0,
                percentage=100.0,
                details={}
            ))
        
        return issues
    
    def _check_null_values(self, df: Any) -> List[ValidationIssue]:
        """检查空值"""
        issues = []
        
        try:
            if self.use_polars and isinstance(df, pl.DataFrame):
                total_rows = len(df)
                for col in df.columns:
                    null_count = df[col].null_count()
                    if null_count > 0:
                        percentage = (null_count / total_rows) * 100
                        level = self._get_null_value_level(percentage)
                        
                        issues.append(ValidationIssue(
                            rule=ValidationRule.NO_NULL_VALUES,
                            level=level,
                            column=col,
                            message=f"列 '{col}' 包含 {null_count} 个空值",
                            count=null_count,
                            percentage=percentage,
                            details={'null_count': null_count, 'total_rows': total_rows}
                        ))
            else:
                total_rows = len(df)
                null_counts = df.isnull().sum()
                
                for col, null_count in null_counts.items():
                    if null_count > 0:
                        percentage = (null_count / total_rows) * 100
                        level = self._get_null_value_level(percentage)
                        
                        issues.append(ValidationIssue(
                            rule=ValidationRule.NO_NULL_VALUES,
                            level=level,
                            column=col,
                            message=f"列 '{col}' 包含 {null_count} 个空值",
                            count=int(null_count),
                            percentage=float(percentage),
                            details={'null_count': int(null_count), 'total_rows': total_rows}
                        ))
        
        except Exception as e:
            self.logger.warning(f"空值检查失败: {str(e)}")
        
        return issues
    
    def _get_null_value_level(self, percentage: float) -> ValidationLevel:
        """根据空值比例确定问题级别"""
        if percentage >= 50:
            return ValidationLevel.CRITICAL
        elif percentage >= 20:
            return ValidationLevel.ERROR
        elif percentage >= 5:
            return ValidationLevel.WARNING
        else:
            return ValidationLevel.INFO
    
    def _check_data_types(self, df: Any) -> List[ValidationIssue]:
        """检查数据类型"""
        issues = []
        
        try:
            if self.use_polars and isinstance(df, pl.DataFrame):
                for col in df.columns:
                    dtype = str(df[col].dtype)
                    # 检查是否有混合类型的问题
                    if dtype == 'Object':
                        issues.append(ValidationIssue(
                            rule=ValidationRule.DATA_TYPE,
                            level=ValidationLevel.WARNING,
                            column=col,
                            message=f"列 '{col}' 数据类型为 Object，可能包含混合类型",
                            count=1,
                            percentage=0.0,
                            details={'dtype': dtype}
                        ))
            else:
                for col in df.columns:
                    dtype = str(df[col].dtype)
                    # 检查 object 类型列
                    if dtype == 'object':
                        # 尝试判断是否应该是数值类型
                        sample = df[col].dropna().head(100)
                        if len(sample) > 0:
                            try:
                                pd.to_numeric(sample, errors='coerce')
                                numeric_ratio = pd.to_numeric(sample, errors='coerce').notna().sum() / len(sample)
                                if numeric_ratio > 0.8:
                                    issues.append(ValidationIssue(
                                        rule=ValidationRule.DATA_TYPE,
                                        level=ValidationLevel.WARNING,
                                        column=col,
                                        message=f"列 '{col}' 可能应该是数值类型",
                                        count=1,
                                        percentage=numeric_ratio * 100,
                                        details={'current_dtype': dtype, 'suggested_dtype': 'numeric'}
                                    ))
                            except:
                                pass
        
        except Exception as e:
            self.logger.warning(f"数据类型检查失败: {str(e)}")
        
        return issues
    
    def _check_duplicates(self, df: Any) -> List[ValidationIssue]:
        """检查重复值"""
        issues = []
        
        try:
            if self.use_polars and isinstance(df, pl.DataFrame):
                total_rows = len(df)
                unique_rows = df.unique().height
                duplicate_count = total_rows - unique_rows
            else:
                total_rows = len(df)
                duplicate_count = df.duplicated().sum()
            
            if duplicate_count > 0:
                percentage = (duplicate_count / total_rows) * 100
                level = ValidationLevel.WARNING if percentage < 10 else ValidationLevel.ERROR
                
                issues.append(ValidationIssue(
                    rule=ValidationRule.DUPLICATE_CHECK,
                    level=level,
                    column=None,
                    message=f"发现 {duplicate_count} 行重复数据",
                    count=duplicate_count,
                    percentage=percentage,
                    details={'duplicate_count': duplicate_count, 'total_rows': total_rows}
                ))
        
        except Exception as e:
            self.logger.warning(f"重复值检查失败: {str(e)}")
        
        return issues
    
    def _check_outliers(self, df: Any) -> List[ValidationIssue]:
        """检查异常值"""
        issues = []
        
        try:
            if self.use_polars and isinstance(df, pl.DataFrame):
                numeric_columns = [col for col in df.columns if df[col].dtype.is_numeric()]
            else:
                numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
            for col in numeric_columns:
                outlier_count = self._count_outliers(df, col)
                if outlier_count > 0:
                    total_rows = len(df)
                    percentage = (outlier_count / total_rows) * 100
                    level = ValidationLevel.INFO if percentage < 5 else ValidationLevel.WARNING
                    
                    issues.append(ValidationIssue(
                        rule=ValidationRule.OUTLIER_CHECK,
                        level=level,
                        column=col,
                        message=f"列 '{col}' 发现 {outlier_count} 个异常值",
                        count=outlier_count,
                        percentage=percentage,
                        details={'outlier_count': outlier_count}
                    ))
        
        except Exception as e:
            self.logger.warning(f"异常值检查失败: {str(e)}")
        
        return issues
    
    def _count_outliers(self, df: Any, column: str) -> int:
        """使用IQR方法计算异常值数量"""
        try:
            if self.use_polars and isinstance(df, pl.DataFrame):
                q1 = df[column].quantile(0.25)
                q3 = df[column].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outliers = df.filter(
                    (pl.col(column) < lower_bound) | (pl.col(column) > upper_bound)
                )
                return len(outliers)
            else:
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
                return outliers.sum()
        
        except Exception:
            return 0
    
    def _check_unique_values(self, df: Any) -> List[ValidationIssue]:
        """检查唯一值"""
        issues = []
        
        try:
            for col in df.columns:
                if self.use_polars and isinstance(df, pl.DataFrame):
                    total_count = len(df)
                    unique_count = df[col].n_unique()
                else:
                    total_count = len(df)
                    unique_count = df[col].nunique()
                
                unique_ratio = unique_count / total_count if total_count > 0 else 0
                
                # 如果唯一值比例太低，可能有问题
                if unique_ratio < 0.01 and total_count > 100:
                    issues.append(ValidationIssue(
                        rule=ValidationRule.UNIQUE_VALUES,
                        level=ValidationLevel.WARNING,
                        column=col,
                        message=f"列 '{col}' 唯一值比例过低 ({unique_ratio:.2%})",
                        count=unique_count,
                        percentage=unique_ratio * 100,
                        details={'unique_count': unique_count, 'total_count': total_count}
                    ))
        
        except Exception as e:
            self.logger.warning(f"唯一值检查失败: {str(e)}")
        
        return issues
    
    def _calculate_quality_score(self, df: Any, issues: List[ValidationIssue]) -> float:
        """计算数据质量分数"""
        base_score = 100.0
        
        # 根据问题级别扣分
        for issue in issues:
            if issue.level == ValidationLevel.CRITICAL:
                base_score -= 30
            elif issue.level == ValidationLevel.ERROR:
                base_score -= 15
            elif issue.level == ValidationLevel.WARNING:
                base_score -= 5
            elif issue.level == ValidationLevel.INFO:
                base_score -= 1
        
        # 确保分数不小于0
        return max(0.0, base_score)
    
    def validate_column_rules(self, df: Any, column_rules: Dict[str, List[Dict]]) -> ValidationResult:
        """根据列规则验证"""
        issues = []
        
        for column, rules in column_rules.items():
            if column not in df.columns:
                issues.append(ValidationIssue(
                    rule=ValidationRule.DATA_TYPE,
                    level=ValidationLevel.ERROR,
                    column=column,
                    message=f"列 '{column}' 不存在",
                    count=1,
                    percentage=0.0,
                    details={}
                ))
                continue
            
            for rule_config in rules:
                rule_type = rule_config.get('type')
                
                if rule_type == 'numeric_range':
                    min_val = rule_config.get('min')
                    max_val = rule_config.get('max')
                    issues.extend(self._validate_numeric_range(df, column, min_val, max_val))
                
                elif rule_type == 'string_length':
                    min_len = rule_config.get('min_length', 0)
                    max_len = rule_config.get('max_length', float('inf'))
                    issues.extend(self._validate_string_length(df, column, min_len, max_len))
                
                elif rule_type == 'regex_pattern':
                    pattern = rule_config.get('pattern')
                    if pattern:
                        issues.extend(self._validate_regex_pattern(df, column, pattern))
        
        score = self._calculate_quality_score(df, issues)
        is_valid = not any(issue.level == ValidationLevel.CRITICAL for issue in issues)
        
        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            statistics={},
            score=score
        )
    
    def _validate_numeric_range(self, df: Any, column: str, min_val: Optional[float], max_val: Optional[float]) -> List[ValidationIssue]:
        """验证数值范围"""
        issues = []
        
        try:
            if self.use_polars and isinstance(df, pl.DataFrame):
                if min_val is not None:
                    below_min = df.filter(pl.col(column) < min_val).height
                    if below_min > 0:
                        issues.append(ValidationIssue(
                            rule=ValidationRule.NUMERIC_RANGE,
                            level=ValidationLevel.WARNING,
                            column=column,
                            message=f"列 '{column}' 有 {below_min} 个值小于最小值 {min_val}",
                            count=below_min,
                            percentage=(below_min / len(df)) * 100,
                            details={'min_val': min_val, 'below_min_count': below_min}
                        ))
                
                if max_val is not None:
                    above_max = df.filter(pl.col(column) > max_val).height
                    if above_max > 0:
                        issues.append(ValidationIssue(
                            rule=ValidationRule.NUMERIC_RANGE,
                            level=ValidationLevel.WARNING,
                            column=column,
                            message=f"列 '{column}' 有 {above_max} 个值大于最大值 {max_val}",
                            count=above_max,
                            percentage=(above_max / len(df)) * 100,
                            details={'max_val': max_val, 'above_max_count': above_max}
                        ))
            else:
                if min_val is not None:
                    below_min = (df[column] < min_val).sum()
                    if below_min > 0:
                        issues.append(ValidationIssue(
                            rule=ValidationRule.NUMERIC_RANGE,
                            level=ValidationLevel.WARNING,
                            column=column,
                            message=f"列 '{column}' 有 {below_min} 个值小于最小值 {min_val}",
                            count=int(below_min),
                            percentage=float((below_min / len(df)) * 100),
                            details={'min_val': min_val, 'below_min_count': int(below_min)}
                        ))
                
                if max_val is not None:
                    above_max = (df[column] > max_val).sum()
                    if above_max > 0:
                        issues.append(ValidationIssue(
                            rule=ValidationRule.NUMERIC_RANGE,
                            level=ValidationLevel.WARNING,
                            column=column,
                            message=f"列 '{column}' 有 {above_max} 个值大于最大值 {max_val}",
                            count=int(above_max),
                            percentage=float((above_max / len(df)) * 100),
                            details={'max_val': max_val, 'above_max_count': int(above_max)}
                        ))
        
        except Exception as e:
            self.logger.warning(f"数值范围验证失败: {str(e)}")
        
        return issues
    
    def _validate_string_length(self, df: Any, column: str, min_length: int, max_length: int) -> List[ValidationIssue]:
        """验证字符串长度"""
        issues = []
        
        try:
            if self.use_polars and isinstance(df, pl.DataFrame):
                # Polars字符串长度验证
                too_short = df.filter(pl.col(column).str.lengths() < min_length).height
                too_long = df.filter(pl.col(column).str.lengths() > max_length).height
            else:
                # Pandas字符串长度验证
                string_lengths = df[column].astype(str).str.len()
                too_short = (string_lengths < min_length).sum()
                too_long = (string_lengths > max_length).sum()
            
            if too_short > 0:
                issues.append(ValidationIssue(
                    rule=ValidationRule.STRING_LENGTH,
                    level=ValidationLevel.WARNING,
                    column=column,
                    message=f"列 '{column}' 有 {too_short} 个值长度小于 {min_length}",
                    count=too_short,
                    percentage=(too_short / len(df)) * 100,
                    details={'min_length': min_length, 'too_short_count': too_short}
                ))
            
            if too_long > 0:
                issues.append(ValidationIssue(
                    rule=ValidationRule.STRING_LENGTH,
                    level=ValidationLevel.WARNING,
                    column=column,
                    message=f"列 '{column}' 有 {too_long} 个值长度大于 {max_length}",
                    count=too_long,
                    percentage=(too_long / len(df)) * 100,
                    details={'max_length': max_length, 'too_long_count': too_long}
                ))
        
        except Exception as e:
            self.logger.warning(f"字符串长度验证失败: {str(e)}")
        
        return issues
    
    def _validate_regex_pattern(self, df: Any, column: str, pattern: str) -> List[ValidationIssue]:
        """验证正则表达式模式"""
        issues = []
        
        try:
            compiled_pattern = re.compile(pattern)
            
            if self.use_polars and isinstance(df, pl.DataFrame):
                # Polars正则验证
                not_matching = df.filter(~pl.col(column).str.contains(pattern)).height
            else:
                # Pandas正则验证
                matches = df[column].astype(str).str.match(compiled_pattern)
                not_matching = (~matches).sum()
            
            if not_matching > 0:
                issues.append(ValidationIssue(
                    rule=ValidationRule.REGEX_PATTERN,
                    level=ValidationLevel.WARNING,
                    column=column,
                    message=f"列 '{column}' 有 {not_matching} 个值不匹配模式 '{pattern}'",
                    count=not_matching,
                    percentage=(not_matching / len(df)) * 100,
                    details={'pattern': pattern, 'not_matching_count': not_matching}
                ))
        
        except Exception as e:
            self.logger.warning(f"正则模式验证失败: {str(e)}")
        
        return issues