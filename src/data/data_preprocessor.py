"""
数据预处理器
提供数据清洗、转换和预处理功能
"""

from typing import Any, Dict, List, Optional, Union, Tuple, TYPE_CHECKING

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
from ..utils.exceptions import DataProcessingError


class CleaningMethod(Enum):
    """数据清洗方法"""
    DROP_MISSING = "drop_missing"  # 删除缺失值
    FILL_MEAN = "fill_mean"  # 均值填充
    FILL_MEDIAN = "fill_median"  # 中位数填充
    FILL_MODE = "fill_mode"  # 众数填充
    FILL_FORWARD = "fill_forward"  # 前向填充
    FILL_BACKWARD = "fill_backward"  # 后向填充
    FILL_INTERPOLATE = "fill_interpolate"  # 插值填充
    FILL_CONSTANT = "fill_constant"  # 常数填充


class OutlierMethod(Enum):
    """异常值处理方法"""
    IQR = "iqr"  # 四分位距方法
    Z_SCORE = "z_score"  # Z分数方法
    ISOLATION_FOREST = "isolation_forest"  # 孤立森林
    PERCENTILE = "percentile"  # 百分位数方法


@dataclass
class PreprocessingConfig:
    """预处理配置"""
    # 缺失值处理
    handle_missing: bool = True
    missing_method: CleaningMethod = CleaningMethod.DROP_MISSING
    missing_threshold: float = 0.5  # 缺失值比例阈值
    fill_value: Optional[Any] = None  # 常数填充值
    
    # 重复值处理
    handle_duplicates: bool = True
    keep_first: bool = True  # 保留第一个重复值
    
    # 异常值处理
    handle_outliers: bool = False
    outlier_method: OutlierMethod = OutlierMethod.IQR
    outlier_threshold: float = 1.5  # IQR倍数或Z分数阈值
    outlier_percentile: Tuple[float, float] = (0.01, 0.99)  # 百分位数范围
    
    # 数据类型转换
    auto_convert_types: bool = True
    convert_strings_to_numeric: bool = True
    convert_to_datetime: bool = True
    
    # 文本清洗
    clean_text: bool = True
    trim_whitespace: bool = True
    remove_special_chars: bool = False
    lowercase: bool = False
    
    # 数值处理
    normalize_numeric: bool = False
    standardize_numeric: bool = False
    
    # 内存优化
    optimize_memory: bool = True
    downcast_integers: bool = True
    downcast_floats: bool = True


class DataPreprocessor(LoggerMixin):
    """数据预处理器"""
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        self.use_polars = HAS_POLARS
    
    def preprocess(self, df: Any, column_info: Optional[Dict[str, str]] = None) -> Tuple[Any, Dict[str, Any]]:
        """执行完整的数据预处理"""
        preprocessing_log = {
            'steps_applied': [],
            'original_shape': self._get_shape(df),
            'statistics': {}
        }
        
        try:
            self.logger.info("开始数据预处理")
            
            # 1. 基础信息记录
            preprocessing_log['original_shape'] = self._get_shape(df)
            preprocessing_log['original_memory'] = self._get_memory_usage(df)
            
            # 2. 数据类型转换
            if self.config.auto_convert_types:
                df = self._convert_data_types(df, column_info)
                preprocessing_log['steps_applied'].append('数据类型转换')
            
            # 3. 文本清洗
            if self.config.clean_text:
                df = self._clean_text_columns(df)
                preprocessing_log['steps_applied'].append('文本清洗')
            
            # 4. 缺失值处理
            if self.config.handle_missing:
                df, missing_stats = self._handle_missing_values(df)
                preprocessing_log['steps_applied'].append('缺失值处理')
                preprocessing_log['statistics']['missing_values'] = missing_stats
            
            # 5. 重复值处理
            if self.config.handle_duplicates:
                df, duplicate_stats = self._handle_duplicates(df)
                preprocessing_log['steps_applied'].append('重复值处理')
                preprocessing_log['statistics']['duplicates'] = duplicate_stats
            
            # 6. 异常值处理
            if self.config.handle_outliers:
                df, outlier_stats = self._handle_outliers(df)
                preprocessing_log['steps_applied'].append('异常值处理')
                preprocessing_log['statistics']['outliers'] = outlier_stats
            
            # 7. 数值标准化/归一化
            if self.config.normalize_numeric or self.config.standardize_numeric:
                df = self._normalize_numeric_columns(df)
                preprocessing_log['steps_applied'].append('数值标准化')
            
            # 8. 内存优化
            if self.config.optimize_memory:
                df = self._optimize_memory_usage(df)
                preprocessing_log['steps_applied'].append('内存优化')
            
            # 9. 最终统计
            preprocessing_log['final_shape'] = self._get_shape(df)
            preprocessing_log['final_memory'] = self._get_memory_usage(df)
            preprocessing_log['memory_reduction'] = self._calculate_memory_reduction(
                preprocessing_log['original_memory'], 
                preprocessing_log['final_memory']
            )
            
            self.logger.info(f"数据预处理完成，应用了 {len(preprocessing_log['steps_applied'])} 个步骤")
            
            return df, preprocessing_log
            
        except Exception as e:
            self.logger.error(f"数据预处理失败: {str(e)}")
            raise DataProcessingError(f"预处理失败: {str(e)}") from e
    
    def _convert_data_types(self, df: Any, column_info: Optional[Dict[str, str]] = None) -> Any:
        """数据类型转换"""
        try:
            if self.use_polars and isinstance(df, pl.DataFrame):
                return self._convert_types_polars(df, column_info)
            else:
                return self._convert_types_pandas(df, column_info)
        except Exception as e:
            self.logger.warning(f"数据类型转换失败: {str(e)}")
            return df
    
    def _convert_types_polars(self, df: "pl.DataFrame", column_info: Optional[Dict[str, str]] = None) -> "pl.DataFrame":
        """Polars数据类型转换"""
        conversions = []
        
        for col in df.columns:
            current_type = str(df[col].dtype)
            
            # 跳过已经是数值类型的列
            if current_type in ['Int64', 'Float64', 'Int32', 'Float32']:
                continue
            
            # 尝试转换为数值类型
            if self.config.convert_strings_to_numeric:
                try:
                    # 检查是否可以转换为数值
                    sample = df[col].head(100).drop_nulls()
                    if len(sample) > 0:
                        # 尝试转换为数值
                        numeric_sample = sample.cast(pl.Float64, strict=False)
                        if numeric_sample.null_count() < len(sample) * 0.8:
                            conversions.append(pl.col(col).cast(pl.Float64, strict=False))
                            continue
                except:
                    pass
            
            # 尝试转换为日期时间
            if self.config.convert_to_datetime:
                try:
                    datetime_sample = df[col].head(100).str.strptime(pl.Datetime, strict=False)
                    if datetime_sample.null_count() < len(datetime_sample) * 0.8:
                        conversions.append(pl.col(col).str.strptime(pl.Datetime, strict=False))
                        continue
                except:
                    pass
        
        # 应用转换
        if conversions:
            df = df.with_columns(conversions)
        
        return df
    
    def _convert_types_pandas(self, df: Any, column_info: Optional[Dict[str, str]] = None) -> Any:
        """Pandas数据类型转换"""
        for col in df.columns:
            current_type = str(df[col].dtype)
            
            # 跳过已经是数值类型的列
            if pd.api.types.is_numeric_dtype(df[col]):
                continue
            
            # 尝试转换为数值类型
            if self.config.convert_strings_to_numeric:
                try:
                    numeric_series = pd.to_numeric(df[col], errors='coerce')
                    # 如果转换成功率高于80%，则应用转换
                    if numeric_series.notna().sum() / len(numeric_series) > 0.8:
                        df[col] = numeric_series
                        continue
                except:
                    pass
            
            # 尝试转换为日期时间
            if self.config.convert_to_datetime:
                try:
                    datetime_series = pd.to_datetime(df[col], errors='coerce')
                    # 如果转换成功率高于80%，则应用转换
                    if datetime_series.notna().sum() / len(datetime_series) > 0.8:
                        df[col] = datetime_series
                        continue
                except:
                    pass
        
        return df
    
    def _clean_text_columns(self, df: Any) -> Any:
        """清洗文本列"""
        try:
            if self.use_polars and isinstance(df, pl.DataFrame):
                return self._clean_text_polars(df)
            else:
                return self._clean_text_pandas(df)
        except Exception as e:
            self.logger.warning(f"文本清洗失败: {str(e)}")
            return df
    
    def _clean_text_polars(self, df: "pl.DataFrame") -> "pl.DataFrame":
        """Polars文本清洗"""
        text_columns = [col for col in df.columns if df[col].dtype == pl.Utf8]
        
        if not text_columns:
            return df
        
        expressions = []
        for col in text_columns:
            expr = pl.col(col)
            
            if self.config.trim_whitespace:
                expr = expr.str.strip()
            
            if self.config.remove_special_chars:
                expr = expr.str.replace_all(r'[^\w\s]', '')
            
            if self.config.lowercase:
                expr = expr.str.to_lowercase()
            
            expressions.append(expr.alias(col))
        
        # 保持其他列不变
        other_columns = [col for col in df.columns if col not in text_columns]
        expressions.extend([pl.col(col) for col in other_columns])
        
        return df.select(expressions)
    
    def _clean_text_pandas(self, df: Any) -> Any:
        """Pandas文本清洗"""
        for col in df.columns:
            if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == 'object':
                if self.config.trim_whitespace:
                    df[col] = df[col].astype(str).str.strip()
                
                if self.config.remove_special_chars:
                    df[col] = df[col].astype(str).str.replace(r'[^\w\s]', '', regex=True)
                
                if self.config.lowercase:
                    df[col] = df[col].astype(str).str.lower()
        
        return df
    
    def _handle_missing_values(self, df: Any) -> Tuple[Any, Dict[str, Any]]:
        """处理缺失值"""
        try:
            if self.use_polars and isinstance(df, pl.DataFrame):
                return self._handle_missing_polars(df)
            else:
                return self._handle_missing_pandas(df)
        except Exception as e:
            self.logger.warning(f"缺失值处理失败: {str(e)}")
            return df, {}
    
    def _handle_missing_polars(self, df: "pl.DataFrame") -> Tuple["pl.DataFrame", Dict[str, Any]]:
        """Polars缺失值处理"""
        original_rows = len(df)
        missing_stats = {}
        
        # 计算每列的缺失值比例
        for col in df.columns:
            null_count = df[col].null_count()
            missing_ratio = null_count / original_rows if original_rows > 0 else 0
            missing_stats[col] = {
                'null_count': null_count,
                'missing_ratio': missing_ratio
            }
        
        # 根据配置处理缺失值
        if self.config.missing_method == CleaningMethod.DROP_MISSING:
            # 删除缺失值比例过高的列
            cols_to_keep = [
                col for col in df.columns 
                if missing_stats[col]['missing_ratio'] <= self.config.missing_threshold
            ]
            if cols_to_keep:
                df = df.select(cols_to_keep)
            
            # 删除包含缺失值的行
            df = df.drop_nulls()
        
        elif self.config.missing_method == CleaningMethod.FILL_CONSTANT:
            fill_value = self.config.fill_value or 0
            df = df.fill_null(fill_value)
        
        elif self.config.missing_method == CleaningMethod.FILL_FORWARD:
            df = df.fill_null(strategy="forward")
        
        elif self.config.missing_method == CleaningMethod.FILL_BACKWARD:
            df = df.fill_null(strategy="backward")
        
        # 其他填充方法需要按列类型处理
        else:
            expressions = []
            for col in df.columns:
                expr = pl.col(col)
                
                if df[col].dtype.is_numeric():
                    if self.config.missing_method == CleaningMethod.FILL_MEAN:
                        expr = expr.fill_null(expr.mean())
                    elif self.config.missing_method == CleaningMethod.FILL_MEDIAN:
                        expr = expr.fill_null(expr.median())
                else:
                    if self.config.missing_method == CleaningMethod.FILL_MODE:
                        expr = expr.fill_null(expr.mode().first())
                
                expressions.append(expr)
            
            df = df.with_columns(expressions)
        
        missing_stats['rows_before'] = original_rows
        missing_stats['rows_after'] = len(df)
        missing_stats['rows_removed'] = original_rows - len(df)
        
        return df, missing_stats
    
    def _handle_missing_pandas(self, df: Any) -> Tuple[Any, Dict[str, Any]]:
        """Pandas缺失值处理"""
        original_rows = len(df)
        missing_stats = {}
        
        # 计算缺失值统计
        for col in df.columns:
            null_count = df[col].isnull().sum()
            missing_ratio = null_count / original_rows if original_rows > 0 else 0
            missing_stats[col] = {
                'null_count': int(null_count),
                'missing_ratio': float(missing_ratio)
            }
        
        # 根据配置处理缺失值
        if self.config.missing_method == CleaningMethod.DROP_MISSING:
            # 删除缺失值比例过高的列
            cols_to_drop = [
                col for col in df.columns 
                if missing_stats[col]['missing_ratio'] > self.config.missing_threshold
            ]
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)
            
            # 删除包含缺失值的行
            df = df.dropna()
        
        elif self.config.missing_method == CleaningMethod.FILL_CONSTANT:
            fill_value = self.config.fill_value or 0
            df = df.fillna(fill_value)
        
        elif self.config.missing_method == CleaningMethod.FILL_FORWARD:
            df = df.fillna(method='ffill')
        
        elif self.config.missing_method == CleaningMethod.FILL_BACKWARD:
            df = df.fillna(method='bfill')
        
        else:
            # 按列类型填充
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    if self.config.missing_method == CleaningMethod.FILL_MEAN:
                        df[col] = df[col].fillna(df[col].mean())
                    elif self.config.missing_method == CleaningMethod.FILL_MEDIAN:
                        df[col] = df[col].fillna(df[col].median())
                else:
                    if self.config.missing_method == CleaningMethod.FILL_MODE:
                        mode_value = df[col].mode()
                        if len(mode_value) > 0:
                            df[col] = df[col].fillna(mode_value[0])
        
        missing_stats['rows_before'] = original_rows
        missing_stats['rows_after'] = len(df)
        missing_stats['rows_removed'] = original_rows - len(df)
        
        return df, missing_stats
    
    def _handle_duplicates(self, df: Any) -> Tuple[Any, Dict[str, Any]]:
        """处理重复值"""
        try:
            if self.use_polars and isinstance(df, pl.DataFrame):
                return self._handle_duplicates_polars(df)
            else:
                return self._handle_duplicates_pandas(df)
        except Exception as e:
            self.logger.warning(f"重复值处理失败: {str(e)}")
            return df, {}
    
    def _handle_duplicates_polars(self, df: "pl.DataFrame") -> Tuple["pl.DataFrame", Dict[str, Any]]:
        """Polars重复值处理"""
        original_rows = len(df)
        
        # 计算重复值
        duplicate_count = original_rows - df.unique().height
        
        # 删除重复值
        if self.config.keep_first:
            df_cleaned = df.unique(maintain_order=True)
        else:
            df_cleaned = df.unique()
        
        stats = {
            'duplicate_count': duplicate_count,
            'rows_before': original_rows,
            'rows_after': len(df_cleaned),
            'duplicate_ratio': duplicate_count / original_rows if original_rows > 0 else 0
        }
        
        return df_cleaned, stats
    
    def _handle_duplicates_pandas(self, df: Any) -> Tuple[Any, Dict[str, Any]]:
        """Pandas重复值处理"""
        original_rows = len(df)
        
        # 计算重复值
        duplicate_count = df.duplicated().sum()
        
        # 删除重复值
        keep_option = 'first' if self.config.keep_first else 'last'
        df_cleaned = df.drop_duplicates(keep=keep_option)
        
        stats = {
            'duplicate_count': int(duplicate_count),
            'rows_before': original_rows,
            'rows_after': len(df_cleaned),
            'duplicate_ratio': float(duplicate_count / original_rows) if original_rows > 0 else 0.0
        }
        
        return df_cleaned, stats
    
    def _handle_outliers(self, df: Any) -> Tuple[Any, Dict[str, Any]]:
        """处理异常值"""
        try:
            if self.use_polars and isinstance(df, pl.DataFrame):
                return self._handle_outliers_polars(df)
            else:
                return self._handle_outliers_pandas(df)
        except Exception as e:
            self.logger.warning(f"异常值处理失败: {str(e)}")
            return df, {}
    
    def _handle_outliers_polars(self, df: "pl.DataFrame") -> Tuple["pl.DataFrame", Dict[str, Any]]:
        """Polars异常值处理"""
        # 简化实现，主要处理数值列
        numeric_columns = [col for col in df.columns if df[col].dtype.is_numeric()]
        outlier_stats = {}
        
        if self.config.outlier_method == OutlierMethod.IQR:
            conditions = []
            for col in numeric_columns:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - self.config.outlier_threshold * iqr
                upper_bound = q3 + self.config.outlier_threshold * iqr
                
                condition = (pl.col(col) >= lower_bound) & (pl.col(col) <= upper_bound)
                conditions.append(condition)
                
                # 统计异常值
                outlier_count = df.filter(~condition).height
                outlier_stats[col] = {
                    'outlier_count': outlier_count,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }
            
            if conditions:
                final_condition = conditions[0]
                for condition in conditions[1:]:
                    final_condition = final_condition & condition
                df = df.filter(final_condition)
        
        return df, outlier_stats
    
    def _handle_outliers_pandas(self, df: Any) -> Tuple[Any, Dict[str, Any]]:
        """Pandas异常值处理"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        outlier_stats = {}
        original_rows = len(df)
        
        if self.config.outlier_method == OutlierMethod.IQR:
            for col in numeric_columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - self.config.outlier_threshold * IQR
                upper_bound = Q3 + self.config.outlier_threshold * IQR
                
                # 标记异常值
                outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
                outlier_count = outliers.sum()
                
                outlier_stats[col] = {
                    'outlier_count': int(outlier_count),
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound)
                }
                
                # 移除异常值
                df = df[~outliers]
        
        elif self.config.outlier_method == OutlierMethod.Z_SCORE:
            for col in numeric_columns:
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers = z_scores > self.config.outlier_threshold
                outlier_count = outliers.sum()
                
                outlier_stats[col] = {
                    'outlier_count': int(outlier_count),
                    'threshold': self.config.outlier_threshold
                }
                
                df = df[~outliers]
        
        outlier_stats['rows_before'] = original_rows
        outlier_stats['rows_after'] = len(df)
        outlier_stats['total_outliers_removed'] = original_rows - len(df)
        
        return df, outlier_stats
    
    def _normalize_numeric_columns(self, df: Any) -> Any:
        """标准化数值列"""
        try:
            if self.use_polars and isinstance(df, pl.DataFrame):
                return self._normalize_polars(df)
            else:
                return self._normalize_pandas(df)
        except Exception as e:
            self.logger.warning(f"数值标准化失败: {str(e)}")
            return df
    
    def _normalize_polars(self, df: "pl.DataFrame") -> "pl.DataFrame":
        """Polars数值标准化"""
        numeric_columns = [col for col in df.columns if df[col].dtype.is_numeric()]
        
        expressions = []
        for col in df.columns:
            if col in numeric_columns:
                if self.config.normalize_numeric:
                    # Min-Max归一化
                    min_val = df[col].min()
                    max_val = df[col].max()
                    expr = (pl.col(col) - min_val) / (max_val - min_val)
                elif self.config.standardize_numeric:
                    # Z-score标准化
                    mean_val = df[col].mean()
                    std_val = df[col].std()
                    expr = (pl.col(col) - mean_val) / std_val
                else:
                    expr = pl.col(col)
                
                expressions.append(expr.alias(col))
            else:
                expressions.append(pl.col(col))
        
        return df.select(expressions)
    
    def _normalize_pandas(self, df: Any) -> Any:
        """Pandas数值标准化"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if self.config.normalize_numeric:
                # Min-Max归一化
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val != min_val:
                    df[col] = (df[col] - min_val) / (max_val - min_val)
            elif self.config.standardize_numeric:
                # Z-score标准化
                mean_val = df[col].mean()
                std_val = df[col].std()
                if std_val != 0:
                    df[col] = (df[col] - mean_val) / std_val
        
        return df
    
    def _optimize_memory_usage(self, df: Any) -> Any:
        """优化内存使用"""
        try:
            if not self.use_polars and hasattr(df, 'select_dtypes'):
                return self._optimize_memory_pandas(df)
            return df
        except Exception as e:
            self.logger.warning(f"内存优化失败: {str(e)}")
            return df
    
    def _optimize_memory_pandas(self, df: Any) -> Any:
        """Pandas内存优化"""
        if self.config.downcast_integers:
            int_cols = df.select_dtypes(include=['int']).columns
            for col in int_cols:
                df[col] = pd.to_numeric(df[col], downcast='integer')
        
        if self.config.downcast_floats:
            float_cols = df.select_dtypes(include=['float']).columns
            for col in float_cols:
                df[col] = pd.to_numeric(df[col], downcast='float')
        
        return df
    
    def _get_shape(self, df: Any) -> Tuple[int, int]:
        """获取数据形状"""
        try:
            if hasattr(df, 'shape'):
                return df.shape
            elif hasattr(df, 'height') and hasattr(df, 'width'):
                return (df.height, df.width)
            else:
                return (len(df), len(df.columns))
        except:
            return (0, 0)
    
    def _get_memory_usage(self, df: Any) -> float:
        """获取内存使用量(MB)"""
        try:
            if self.use_polars and isinstance(df, pl.DataFrame):
                return df.estimated_size() / (1024 * 1024)
            elif hasattr(df, 'memory_usage'):
                return df.memory_usage(deep=True).sum() / (1024 * 1024)
            else:
                return 0.0
        except:
            return 0.0
    
    def _calculate_memory_reduction(self, before: float, after: float) -> float:
        """计算内存减少百分比"""
        if before == 0:
            return 0.0
        return ((before - after) / before) * 100