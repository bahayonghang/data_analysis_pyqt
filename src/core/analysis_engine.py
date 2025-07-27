"""
数据分析引擎
实现描述性统计、关联分析、异常值检测和时间序列分析功能
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING
import warnings
import numpy as np

if TYPE_CHECKING:
    import polars as pl
    import pandas as pd

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    from scipy import stats
    from scipy.stats import jarque_bera, shapiro, anderson
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from statsmodels.tsa.stattools import adfuller, kpss
    from statsmodels.stats.diagnostic import acorr_ljungbox
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

from ..models.extended_analysis_result import (
    DescriptiveStats, CorrelationMatrix, AnomalyDetectionResult,
    TimeSeriesAnalysis, AnalysisResult
)
from ..utils.exceptions import AnalysisError
from ..utils.basic_logging import LoggerMixin


@dataclass
class AnalysisConfig:
    """分析配置"""
    # 描述性统计配置
    include_percentiles: List[float] = None
    confidence_interval: float = 0.95
    
    # 异常值检测配置
    outlier_method: str = "iqr"  # iqr, zscore, isolation_forest
    outlier_threshold: float = 3.0
    iqr_multiplier: float = 1.5
    
    # 关联分析配置
    correlation_method: str = "pearson"  # pearson, spearman, kendall
    min_correlation_threshold: float = 0.1
    
    # 时间序列分析配置
    stationarity_tests: List[str] = None
    max_lags: int = 12
    alpha: float = 0.05
    
    # 性能配置
    n_threads: int = 4
    max_memory_mb: int = 1000
    enable_parallel: bool = True
    
    def __post_init__(self):
        if self.include_percentiles is None:
            self.include_percentiles = [0.25, 0.5, 0.75, 0.95, 0.99]
        if self.stationarity_tests is None:
            self.stationarity_tests = ["adf", "kpss"]


class AnalysisEngine(LoggerMixin):
    """数据分析引擎"""
    
    def __init__(self, config: Optional[AnalysisConfig] = None):
        self.config = config or AnalysisConfig()
        self.executor = ThreadPoolExecutor(max_workers=self.config.n_threads)
        
        # 检查可用的库
        self.use_polars = HAS_POLARS
        self.has_scipy = HAS_SCIPY
        self.has_statsmodels = HAS_STATSMODELS
        
        if not self.has_scipy:
            self.logger.warning("SciPy not available, some statistical functions will be limited")
        if not self.has_statsmodels:
            self.logger.warning("Statsmodels not available, time series analysis will be limited")
    
    async def analyze_dataset_async(
        self, 
        df: Any, 
        time_column: Optional[str] = None,
        exclude_columns: Optional[List[str]] = None
    ) -> AnalysisResult:
        """异步分析数据集"""
        loop = asyncio.get_event_loop()
        
        return await loop.run_in_executor(
            self.executor,
            self.analyze_dataset,
            df, time_column, exclude_columns
        )
    
    def analyze_dataset(
        self, 
        df: Any, 
        time_column: Optional[str] = None,
        exclude_columns: Optional[List[str]] = None
    ) -> AnalysisResult:
        """分析数据集"""
        try:
            self.logger.info("开始数据分析")
            
            # 数据预处理
            numeric_df, column_info = self._prepare_numeric_data(df, time_column, exclude_columns)
            
            # 并行执行各种分析
            if self.config.enable_parallel:
                results = self._parallel_analysis(numeric_df, df, time_column)
            else:
                results = self._sequential_analysis(numeric_df, df, time_column)
            
            # 创建分析结果
            analysis_result = AnalysisResult(
                descriptive_stats=results['descriptive'],
                correlation_matrix=results['correlation'],
                anomaly_detection=results['anomaly'],
                time_series_analysis=results['time_series'],
                column_info=column_info,
                analysis_config=self.config
            )
            
            self.logger.info("数据分析完成")
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"数据分析失败: {str(e)}")
            raise AnalysisError(f"数据分析失败: {str(e)}") from e
    
    def _prepare_numeric_data(
        self, 
        df: Any, 
        time_column: Optional[str] = None,
        exclude_columns: Optional[List[str]] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """准备数值数据"""
        try:
            exclude_cols = set(exclude_columns or [])
            if time_column:
                exclude_cols.add(time_column)
            
            if self.use_polars and isinstance(df, pl.DataFrame):
                return self._prepare_numeric_data_polars(df, exclude_cols)
            else:
                return self._prepare_numeric_data_pandas(df, exclude_cols)
                
        except Exception as e:
            raise AnalysisError(f"数据预处理失败: {str(e)}") from e
    
    def _prepare_numeric_data_polars(self, df: "pl.DataFrame", exclude_cols: set) -> Tuple["pl.DataFrame", Dict[str, Any]]:
        """Polars数据预处理"""
        # 获取数值列
        numeric_columns = []
        column_info = {
            'total_columns': len(df.columns),
            'numeric_columns': [],
            'excluded_columns': list(exclude_cols),
            'data_types': {}
        }
        
        for col in df.columns:
            if col not in exclude_cols:
                dtype = df[col].dtype
                column_info['data_types'][col] = str(dtype)
                
                if dtype in [pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64]:
                    numeric_columns.append(col)
                    column_info['numeric_columns'].append(col)
        
        if not numeric_columns:
            raise AnalysisError("没有找到可分析的数值列")
        
        numeric_df = df.select(numeric_columns)
        return numeric_df, column_info
    
    def _prepare_numeric_data_pandas(self, df: Any, exclude_cols: set) -> Tuple[Any, Dict[str, Any]]:
        """Pandas数据预处理"""
        # 获取数值列
        numeric_columns = []
        column_info = {
            'total_columns': len(df.columns),
            'numeric_columns': [],
            'excluded_columns': list(exclude_cols),
            'data_types': {}
        }
        
        for col in df.columns:
            if col not in exclude_cols:
                dtype = df[col].dtype
                column_info['data_types'][col] = str(dtype)
                
                if np.issubdtype(dtype, np.number):
                    numeric_columns.append(col)
                    column_info['numeric_columns'].append(col)
        
        if not numeric_columns:
            raise AnalysisError("没有找到可分析的数值列")
        
        numeric_df = df[numeric_columns]
        return numeric_df, column_info
    
    def _parallel_analysis(self, numeric_df: Any, original_df: Any, time_column: Optional[str]) -> Dict[str, Any]:
        """并行执行分析"""
        import concurrent.futures
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.n_threads) as executor:
            # 提交任务
            futures = {
                'descriptive': executor.submit(self.compute_descriptive_stats, numeric_df),
                'correlation': executor.submit(self.compute_correlation_matrix, numeric_df),
                'anomaly': executor.submit(self.detect_anomalies, numeric_df)
            }
            
            # 如果有时间列，添加时间序列分析
            if time_column and time_column in (original_df.columns if hasattr(original_df, 'columns') else []):
                futures['time_series'] = executor.submit(
                    self.analyze_time_series, original_df, time_column
                )
            
            # 收集结果
            results = {}
            for key, future in futures.items():
                try:
                    results[key] = future.result(timeout=30)
                except Exception as e:
                    self.logger.error(f"{key}分析失败: {str(e)}")
                    results[key] = None
            
            # 如果没有时间序列分析，设置为None
            if 'time_series' not in results:
                results['time_series'] = None
                
            return results
    
    def _sequential_analysis(self, numeric_df: Any, original_df: Any, time_column: Optional[str]) -> Dict[str, Any]:
        """顺序执行分析"""
        results = {}
        
        try:
            results['descriptive'] = self.compute_descriptive_stats(numeric_df)
        except Exception as e:
            self.logger.error(f"描述性统计分析失败: {str(e)}")
            results['descriptive'] = None
        
        try:
            results['correlation'] = self.compute_correlation_matrix(numeric_df)
        except Exception as e:
            self.logger.error(f"关联分析失败: {str(e)}")
            results['correlation'] = None
        
        try:
            results['anomaly'] = self.detect_anomalies(numeric_df)
        except Exception as e:
            self.logger.error(f"异常值检测失败: {str(e)}")
            results['anomaly'] = None
        
        # 时间序列分析
        if time_column and time_column in (original_df.columns if hasattr(original_df, 'columns') else []):
            try:
                results['time_series'] = self.analyze_time_series(original_df, time_column)
            except Exception as e:
                self.logger.error(f"时间序列分析失败: {str(e)}")
                results['time_series'] = None
        else:
            results['time_series'] = None
        
        return results
    
    def compute_descriptive_stats(self, df: Any) -> DescriptiveStats:
        """计算描述性统计"""
        try:
            if self.use_polars and isinstance(df, pl.DataFrame):
                return self._compute_descriptive_stats_polars(df)
            else:
                return self._compute_descriptive_stats_pandas(df)
                
        except Exception as e:
            raise AnalysisError(f"描述性统计计算失败: {str(e)}") from e
    
    def _compute_descriptive_stats_polars(self, df: "pl.DataFrame") -> DescriptiveStats:
        """Polars描述性统计"""
        stats_dict = {}
        
        for col in df.columns:
            col_data = df[col].drop_nulls()
            
            if len(col_data) == 0:
                continue
            
            # 基础统计
            stats_info = {
                'count': len(col_data),
                'mean': float(col_data.mean()),
                'std': float(col_data.std()),
                'min': float(col_data.min()),
                'max': float(col_data.max()),
                'median': float(col_data.median()),
                'variance': float(col_data.var()),
                'skewness': None,
                'kurtosis': None
            }
            
            # 分位数
            for p in self.config.include_percentiles:
                stats_info[f'percentile_{int(p*100)}'] = float(col_data.quantile(p))
            
            # 如果有scipy，计算偏度和峰度
            if self.has_scipy:
                values = col_data.to_numpy()
                stats_info['skewness'] = float(stats.skew(values, nan_policy='omit'))
                stats_info['kurtosis'] = float(stats.kurtosis(values, nan_policy='omit'))
            
            stats_dict[col] = stats_info
        
        return DescriptiveStats(
            statistics=stats_dict,
            method="polars",
            config=self.config
        )
    
    def _compute_descriptive_stats_pandas(self, df: Any) -> DescriptiveStats:
        """Pandas描述性统计"""
        stats_dict = {}
        
        for col in df.columns:
            col_data = df[col].dropna()
            
            if len(col_data) == 0:
                continue
            
            # 基础统计
            stats_info = {
                'count': len(col_data),
                'mean': float(col_data.mean()),
                'std': float(col_data.std()),
                'min': float(col_data.min()),
                'max': float(col_data.max()),
                'median': float(col_data.median()),
                'variance': float(col_data.var()),
                'skewness': None,
                'kurtosis': None
            }
            
            # 分位数
            for p in self.config.include_percentiles:
                stats_info[f'percentile_{int(p*100)}'] = float(col_data.quantile(p))
            
            # 偏度和峰度
            stats_info['skewness'] = float(col_data.skew())
            stats_info['kurtosis'] = float(col_data.kurtosis())
            
            stats_dict[col] = stats_info
        
        return DescriptiveStats(
            statistics=stats_dict,
            method="pandas",
            config=self.config
        )
    
    def compute_correlation_matrix(self, df: Any) -> CorrelationMatrix:
        """计算关联矩阵"""
        try:
            if self.use_polars and isinstance(df, pl.DataFrame):
                return self._compute_correlation_polars(df)
            else:
                return self._compute_correlation_pandas(df)
                
        except Exception as e:
            raise AnalysisError(f"关联矩阵计算失败: {str(e)}") from e
    
    def _compute_correlation_polars(self, df: "pl.DataFrame") -> CorrelationMatrix:
        """Polars关联分析"""
        # 转换为numpy进行相关性计算
        numeric_data = df.to_numpy()
        column_names = df.columns
        
        if self.config.correlation_method == "pearson":
            corr_matrix = np.corrcoef(numeric_data.T)
        elif self.config.correlation_method == "spearman" and self.has_scipy:
            corr_matrix, _ = stats.spearmanr(numeric_data, axis=0, nan_policy='omit')
        elif self.config.correlation_method == "kendall" and self.has_scipy:
            n_cols = numeric_data.shape[1]
            corr_matrix = np.ones((n_cols, n_cols))
            for i in range(n_cols):
                for j in range(i+1, n_cols):
                    corr, _ = stats.kendalltau(numeric_data[:, i], numeric_data[:, j], nan_policy='omit')
                    corr_matrix[i, j] = corr_matrix[j, i] = corr
        else:
            # 默认使用pearson
            corr_matrix = np.corrcoef(numeric_data.T)
        
        return CorrelationMatrix(
            matrix=corr_matrix.tolist(),
            columns=column_names,
            method=self.config.correlation_method,
            threshold=self.config.min_correlation_threshold
        )
    
    def _compute_correlation_pandas(self, df: Any) -> CorrelationMatrix:
        """Pandas关联分析"""
        if self.config.correlation_method == "pearson":
            corr_matrix = df.corr(method='pearson')
        elif self.config.correlation_method == "spearman":
            corr_matrix = df.corr(method='spearman')
        elif self.config.correlation_method == "kendall":
            corr_matrix = df.corr(method='kendall')
        else:
            corr_matrix = df.corr(method='pearson')
        
        return CorrelationMatrix(
            matrix=corr_matrix.values.tolist(),
            columns=list(corr_matrix.columns),
            method=self.config.correlation_method,
            threshold=self.config.min_correlation_threshold
        )
    
    def detect_anomalies(self, df: Any) -> AnomalyDetectionResult:
        """异常值检测"""
        try:
            if self.use_polars and isinstance(df, pl.DataFrame):
                return self._detect_anomalies_polars(df)
            else:
                return self._detect_anomalies_pandas(df)
                
        except Exception as e:
            raise AnalysisError(f"异常值检测失败: {str(e)}") from e
    
    def _detect_anomalies_polars(self, df: "pl.DataFrame") -> AnomalyDetectionResult:
        """Polars异常值检测"""
        anomalies = {}
        
        for col in df.columns:
            col_data = df[col].drop_nulls()
            
            if len(col_data) == 0:
                continue
            
            values = col_data.to_numpy()
            outliers = self._detect_outliers(values, col)
            anomalies[col] = outliers
        
        return AnomalyDetectionResult(
            anomalies=anomalies,
            method=self.config.outlier_method,
            threshold=self.config.outlier_threshold
        )
    
    def _detect_anomalies_pandas(self, df: Any) -> AnomalyDetectionResult:
        """Pandas异常值检测"""
        anomalies = {}
        
        for col in df.columns:
            col_data = df[col].dropna()
            
            if len(col_data) == 0:
                continue
            
            values = col_data.values
            outliers = self._detect_outliers(values, col)
            anomalies[col] = outliers
        
        return AnomalyDetectionResult(
            anomalies=anomalies,
            method=self.config.outlier_method,
            threshold=self.config.outlier_threshold
        )
    
    def _detect_outliers(self, values: np.ndarray, column_name: str) -> Dict[str, Any]:
        """检测异常值"""
        outliers_info = {
            'outlier_indices': [],
            'outlier_values': [],
            'outlier_count': 0,
            'outlier_percentage': 0.0,
            'method_details': {}
        }
        
        if self.config.outlier_method == "iqr":
            outliers_info.update(self._detect_outliers_iqr(values))
        elif self.config.outlier_method == "zscore":
            outliers_info.update(self._detect_outliers_zscore(values))
        elif self.config.outlier_method == "isolation_forest" and self.has_scipy:
            outliers_info.update(self._detect_outliers_isolation_forest(values))
        else:
            # 默认使用IQR方法
            outliers_info.update(self._detect_outliers_iqr(values))
        
        return outliers_info
    
    def _detect_outliers_iqr(self, values: np.ndarray) -> Dict[str, Any]:
        """IQR方法检测异常值"""
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - self.config.iqr_multiplier * iqr
        upper_bound = q3 + self.config.iqr_multiplier * iqr
        
        outlier_mask = (values < lower_bound) | (values > upper_bound)
        outlier_indices = np.where(outlier_mask)[0].tolist()
        outlier_values = values[outlier_mask].tolist()
        
        return {
            'outlier_indices': outlier_indices,
            'outlier_values': outlier_values,
            'outlier_count': len(outlier_indices),
            'outlier_percentage': len(outlier_indices) / len(values) * 100,
            'method_details': {
                'q1': q1,
                'q3': q3,
                'iqr': iqr,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'multiplier': self.config.iqr_multiplier
            }
        }
    
    def _detect_outliers_zscore(self, values: np.ndarray) -> Dict[str, Any]:
        """Z-score方法检测异常值"""
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        if std_val == 0:
            return {
                'outlier_indices': [],
                'outlier_values': [],
                'outlier_count': 0,
                'outlier_percentage': 0.0,
                'method_details': {'mean': mean_val, 'std': std_val, 'threshold': self.config.outlier_threshold}
            }
        
        z_scores = np.abs((values - mean_val) / std_val)
        outlier_mask = z_scores > self.config.outlier_threshold
        outlier_indices = np.where(outlier_mask)[0].tolist()
        outlier_values = values[outlier_mask].tolist()
        
        return {
            'outlier_indices': outlier_indices,
            'outlier_values': outlier_values,
            'outlier_count': len(outlier_indices),
            'outlier_percentage': len(outlier_indices) / len(values) * 100,
            'method_details': {
                'mean': mean_val,
                'std': std_val,
                'threshold': self.config.outlier_threshold,
                'z_scores_max': float(np.max(z_scores))
            }
        }
    
    def _detect_outliers_isolation_forest(self, values: np.ndarray) -> Dict[str, Any]:
        """孤立森林方法检测异常值"""
        try:
            from sklearn.ensemble import IsolationForest
            
            # 重塑数据
            X = values.reshape(-1, 1)
            
            # 创建孤立森林模型
            iso_forest = IsolationForest(contamination='auto', random_state=42)
            outlier_labels = iso_forest.fit_predict(X)
            
            # -1表示异常值，1表示正常值
            outlier_mask = outlier_labels == -1
            outlier_indices = np.where(outlier_mask)[0].tolist()
            outlier_values = values[outlier_mask].tolist()
            
            return {
                'outlier_indices': outlier_indices,
                'outlier_values': outlier_values,
                'outlier_count': len(outlier_indices),
                'outlier_percentage': len(outlier_indices) / len(values) * 100,
                'method_details': {
                    'contamination': 'auto',
                    'decision_scores': iso_forest.decision_function(X).tolist()
                }
            }
            
        except ImportError:
            self.logger.warning("Scikit-learn not available, falling back to IQR method")
            return self._detect_outliers_iqr(values)
    
    def analyze_time_series(self, df: Any, time_column: str) -> Optional[TimeSeriesAnalysis]:
        """时间序列分析"""
        if not self.has_statsmodels:
            self.logger.warning("Statsmodels not available, skipping time series analysis")
            return None
        
        try:
            if self.use_polars and isinstance(df, pl.DataFrame):
                return self._analyze_time_series_polars(df, time_column)
            else:
                return self._analyze_time_series_pandas(df, time_column)
                
        except Exception as e:
            self.logger.error(f"时间序列分析失败: {str(e)}")
            return None
    
    def _analyze_time_series_polars(self, df: "pl.DataFrame", time_column: str) -> TimeSeriesAnalysis:
        """Polars时间序列分析"""
        # 转换为pandas进行时间序列分析
        pandas_df = df.to_pandas()
        return self._analyze_time_series_pandas(pandas_df, time_column)
    
    def _analyze_time_series_pandas(self, df: Any, time_column: str) -> TimeSeriesAnalysis:
        """Pandas时间序列分析"""
        if not HAS_PANDAS:
            raise AnalysisError("Pandas not available for time series analysis")
            
        # 确保时间列是datetime类型
        if not pd.api.types.is_datetime64_any_dtype(df[time_column]):
            df[time_column] = pd.to_datetime(df[time_column])
        
        # 设置时间索引
        ts_df = df.set_index(time_column).select_dtypes(include=[np.number])
        
        if ts_df.empty:
            return TimeSeriesAnalysis(
                stationarity_tests={},
                trend_analysis={},
                seasonality_analysis={},
                time_column=time_column
            )
        
        stationarity_results = {}
        trend_results = {}
        seasonality_results = {}
        
        for col in ts_df.columns:
            series = ts_df[col].dropna()
            
            if len(series) < 10:  # 需要足够的数据点
                continue
            
            # 平稳性检验
            stationarity_results[col] = self._test_stationarity(series)
            
            # 趋势分析
            trend_results[col] = self._analyze_trend(series)
            
            # 季节性分析
            seasonality_results[col] = self._analyze_seasonality(series)
        
        return TimeSeriesAnalysis(
            stationarity_tests=stationarity_results,
            trend_analysis=trend_results,
            seasonality_analysis=seasonality_results,
            time_column=time_column
        )
    
    def _test_stationarity(self, series: Any) -> Dict[str, Any]:
        """平稳性检验"""
        results = {}
        
        # ADF检验
        if "adf" in self.config.stationarity_tests:
            try:
                adf_result = adfuller(series, maxlag=self.config.max_lags)
                results['adf'] = {
                    'statistic': adf_result[0],
                    'p_value': adf_result[1],
                    'critical_values': adf_result[4],
                    'is_stationary': adf_result[1] < self.config.alpha
                }
            except Exception as e:
                self.logger.warning(f"ADF检验失败: {str(e)}")
        
        # KPSS检验
        if "kpss" in self.config.stationarity_tests:
            try:
                kpss_result = kpss(series, regression='c', nlags=self.config.max_lags)
                results['kpss'] = {
                    'statistic': kpss_result[0],
                    'p_value': kpss_result[1],
                    'critical_values': kpss_result[3],
                    'is_stationary': kpss_result[1] > self.config.alpha
                }
            except Exception as e:
                self.logger.warning(f"KPSS检验失败: {str(e)}")
        
        return results
    
    def _analyze_trend(self, series: Any) -> Dict[str, Any]:
        """趋势分析"""
        try:
            from scipy.stats import linregress
            
            x = np.arange(len(series))
            slope, intercept, r_value, p_value, std_err = linregress(x, series)
            
            return {
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_value ** 2,
                'p_value': p_value,
                'std_error': std_err,
                'trend_direction': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable',
                'is_significant': p_value < self.config.alpha
            }
            
        except Exception as e:
            self.logger.warning(f"趋势分析失败: {str(e)}")
            return {}
    
    def _analyze_seasonality(self, series: Any) -> Dict[str, Any]:
        """季节性分析"""
        try:
            # 简单的季节性检测
            if len(series) < 24:  # 需要足够的数据点
                return {}
            
            # 计算不同周期的自相关
            periods_to_test = [7, 30, 365] if len(series) >= 365 else [7, 30] if len(series) >= 30 else [7]
            seasonality_results = {}
            
            for period in periods_to_test:
                if len(series) >= 2 * period:
                    # 计算指定周期的自相关
                    autocorr = series.autocorr(lag=period)
                    seasonality_results[f'period_{period}'] = {
                        'autocorrelation': autocorr,
                        'is_seasonal': abs(autocorr) > 0.3 if not np.isnan(autocorr) else False
                    }
            
            return seasonality_results
            
        except Exception as e:
            self.logger.warning(f"季节性分析失败: {str(e)}")
            return {}
    
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)