"""
高级时间序列分析模块
基于statsmodels实现趋势分析、季节性检测、预测模型
"""

from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import warnings

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.api import SARIMAX
    from statsmodels.tsa.statespace import SARIMAX as StateSpaceSARIMAX
    from statsmodels.tsa.seasonal import STL
    from statsmodels.tsa.forecasting.stl import STLForecast
    from statsmodels.tsa.statespace.exponential_smoothing import ExponentialSmoothing
    from statsmodels.tsa.stattools import adfuller, kpss
    from statsmodels.tsa.x13 import x13_arima_analysis
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.holtwinters import ExponentialSmoothing as HoltWinters
    import scipy.stats as stats
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

from ..utils.basic_logging import LoggerMixin
from ..utils.exceptions import DataProcessingError


class TSAnalysisType(str, Enum):
    """时间序列分析类型"""
    TREND_ANALYSIS = "trend_analysis"
    SEASONALITY_DETECTION = "seasonality_detection"
    FORECASTING = "forecasting"
    STATIONARITY_TEST = "stationarity_test"
    DECOMPOSITION = "decomposition"
    ANOMALY_DETECTION = "anomaly_detection"


class TSModelType(str, Enum):
    """时间序列模型类型"""
    # ARIMA族模型
    ARIMA = "arima"
    SARIMA = "sarima"
    AUTO_ARIMA = "auto_arima"
    
    # 指数平滑模型
    SIMPLE_EXP_SMOOTHING = "simple_exponential_smoothing"
    DOUBLE_EXP_SMOOTHING = "double_exponential_smoothing"
    TRIPLE_EXP_SMOOTHING = "triple_exponential_smoothing"
    HOLT_WINTERS = "holt_winters"
    
    # 分解模型
    STL_FORECAST = "stl_forecast"
    SEASONAL_DECOMPOSE = "seasonal_decompose"
    
    # 状态空间模型
    UNOBSERVED_COMPONENTS = "unobserved_components"
    STATE_SPACE_MODEL = "state_space_model"


class SeasonalityType(str, Enum):
    """季节性类型"""
    ADDITIVE = "additive"
    MULTIPLICATIVE = "multiplicative"
    NONE = "none"


@dataclass
class TSConfig:
    """时间序列配置"""
    # 基础设置
    freq: Optional[str] = None
    seasonal_periods: Optional[int] = None
    
    # 分析设置
    confidence_level: float = 0.95
    significance_level: float = 0.05
    
    # 预测设置
    forecast_steps: int = 12
    enable_confidence_intervals: bool = True
    
    # ARIMA设置
    arima_order: Optional[Tuple[int, int, int]] = None
    seasonal_order: Optional[Tuple[int, int, int, int]] = None
    
    # STL分解设置
    stl_seasonal: Optional[int] = None
    stl_trend: Optional[int] = None
    stl_robust: bool = True
    
    # 指数平滑设置
    trend_type: Optional[str] = None  # 'add', 'mul', None
    seasonal_type: Optional[str] = None  # 'add', 'mul', None
    damped_trend: bool = False
    
    # 模型参数
    model_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TSResult:
    """时间序列分析结果"""
    analysis_type: TSAnalysisType
    model_type: Optional[TSModelType] = None
    model: Any = None
    
    # 原始数据
    original_data: Optional[pd.Series] = None
    transformed_data: Optional[pd.Series] = None
    
    # 趋势分析
    trend_component: Optional[pd.Series] = None
    trend_slope: Optional[float] = None
    trend_direction: Optional[str] = None
    
    # 季节性分析
    seasonal_component: Optional[pd.Series] = None
    seasonal_strength: Optional[float] = None
    seasonal_period: Optional[int] = None
    seasonality_type: Optional[SeasonalityType] = None
    
    # 分解结果
    decomposition: Optional[Any] = None
    residuals: Optional[pd.Series] = None
    
    # 预测结果
    forecast: Optional[pd.Series] = None
    forecast_conf_int: Optional[pd.DataFrame] = None
    fitted_values: Optional[pd.Series] = None
    
    # 统计检验
    stationarity_tests: Dict[str, Dict[str, float]] = field(default_factory=dict)
    normality_tests: Dict[str, Dict[str, float]] = field(default_factory=dict)
    residual_tests: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # 模型性能
    metrics: Dict[str, float] = field(default_factory=dict)
    information_criteria: Dict[str, float] = field(default_factory=dict)
    
    # 异常检测
    anomalies: Optional[pd.Series] = None
    anomaly_scores: Optional[pd.Series] = None
    
    # 执行信息
    execution_time: float = 0.0
    error_message: Optional[str] = None


class TimeSeriesEngine(LoggerMixin):
    """时间序列分析引擎"""
    
    def __init__(self, config: Optional[TSConfig] = None):
        self.config = config or TSConfig()
        
        if not HAS_STATSMODELS:
            raise ImportError("statsmodels未安装，无法使用时间序列功能")
        if not HAS_PANDAS:
            raise ImportError("pandas未安装，无法处理时间序列数据")
    
    def analyze_trend(
        self,
        data: pd.Series,
        method: str = "linear"
    ) -> TSResult:
        """趋势分析"""
        try:
            import time
            start_time = time.time()
            
            # 数据准备
            data = self._prepare_time_series(data)
            
            # 趋势检测
            if method == "linear":
                trend_component, trend_slope = self._linear_trend_analysis(data)
            elif method == "stl":
                trend_component, trend_slope = self._stl_trend_analysis(data)
            else:
                raise ValueError(f"不支持的趋势分析方法: {method}")
            
            # 趋势方向判断
            trend_direction = self._determine_trend_direction(trend_slope)
            
            # 统计检验
            stationarity_tests = self._test_stationarity(data)
            
            execution_time = time.time() - start_time
            
            result = TSResult(
                analysis_type=TSAnalysisType.TREND_ANALYSIS,
                original_data=data,
                trend_component=trend_component,
                trend_slope=trend_slope,
                trend_direction=trend_direction,
                stationarity_tests=stationarity_tests,
                execution_time=execution_time
            )
            
            self.logger.info(f"趋势分析完成: 方向={trend_direction}, 斜率={trend_slope:.4f}")
            return result
            
        except Exception as e:
            self.logger.error(f"趋势分析失败: {str(e)}")
            return TSResult(
                analysis_type=TSAnalysisType.TREND_ANALYSIS,
                error_message=str(e)
            )
    
    def detect_seasonality(
        self,
        data: pd.Series,
        max_period: Optional[int] = None
    ) -> TSResult:
        """季节性检测"""
        try:
            import time
            start_time = time.time()
            
            # 数据准备
            data = self._prepare_time_series(data)
            
            # 自动检测季节性周期
            if max_period is None:
                max_period = min(len(data) // 2, 365)
            
            seasonal_period = self._detect_seasonal_period(data, max_period)
            
            if seasonal_period:
                # STL分解检测季节性
                decomposition = self._stl_decomposition(data, seasonal_period)
                seasonal_component = decomposition.seasonal
                seasonal_strength = self._calculate_seasonal_strength(decomposition)
                seasonality_type = self._determine_seasonality_type(data, seasonal_component)
            else:
                seasonal_component = None
                seasonal_strength = 0.0
                seasonality_type = SeasonalityType.NONE
                decomposition = None
            
            execution_time = time.time() - start_time
            
            result = TSResult(
                analysis_type=TSAnalysisType.SEASONALITY_DETECTION,
                original_data=data,
                seasonal_component=seasonal_component,
                seasonal_strength=seasonal_strength,
                seasonal_period=seasonal_period,
                seasonality_type=seasonality_type,
                decomposition=decomposition,
                execution_time=execution_time
            )
            
            self.logger.info(f"季节性检测完成: 周期={seasonal_period}, 强度={seasonal_strength:.3f}")
            return result
            
        except Exception as e:
            self.logger.error(f"季节性检测失败: {str(e)}")
            return TSResult(
                analysis_type=TSAnalysisType.SEASONALITY_DETECTION,
                error_message=str(e)
            )
    
    def forecast(
        self,
        data: pd.Series,
        model_type: TSModelType = TSModelType.ARIMA,
        steps: Optional[int] = None
    ) -> TSResult:
        """时间序列预测"""
        try:
            import time
            start_time = time.time()
            
            # 数据准备
            data = self._prepare_time_series(data)
            
            # 预测步数
            steps = steps or self.config.forecast_steps
            
            # 模型训练和预测
            if model_type == TSModelType.ARIMA:
                result = self._arima_forecast(data, steps)
            elif model_type == TSModelType.SARIMA:
                result = self._sarima_forecast(data, steps)
            elif model_type == TSModelType.STL_FORECAST:
                result = self._stl_forecast(data, steps)
            elif model_type == TSModelType.HOLT_WINTERS:
                result = self._holt_winters_forecast(data, steps)
            else:
                raise ValueError(f"不支持的预测模型: {model_type}")
            
            # 添加通用信息
            result.analysis_type = TSAnalysisType.FORECASTING
            result.model_type = model_type
            result.original_data = data
            result.execution_time = time.time() - start_time
            
            self.logger.info(f"预测完成: 模型={model_type.value}, 步数={steps}")
            return result
            
        except Exception as e:
            self.logger.error(f"时间序列预测失败: {str(e)}")
            return TSResult(
                analysis_type=TSAnalysisType.FORECASTING,
                model_type=model_type,
                error_message=str(e)
            )
    
    def decompose(
        self,
        data: pd.Series,
        model: str = "stl",
        seasonal_period: Optional[int] = None
    ) -> TSResult:
        """时间序列分解"""
        try:
            import time
            start_time = time.time()
            
            # 数据准备
            data = self._prepare_time_series(data)
            
            # 自动检测季节性周期
            if seasonal_period is None:
                seasonal_period = self._detect_seasonal_period(data)
            
            # 执行分解
            if model == "stl":
                decomposition = self._stl_decomposition(data, seasonal_period)
            elif model == "seasonal":
                decomposition = self._seasonal_decomposition(data, seasonal_period)
            else:
                raise ValueError(f"不支持的分解模型: {model}")
            
            # 提取组件
            trend_component = decomposition.trend
            seasonal_component = decomposition.seasonal
            residuals = decomposition.resid
            
            # 计算季节性强度
            seasonal_strength = self._calculate_seasonal_strength(decomposition)
            
            execution_time = time.time() - start_time
            
            result = TSResult(
                analysis_type=TSAnalysisType.DECOMPOSITION,
                original_data=data,
                decomposition=decomposition,
                trend_component=trend_component,
                seasonal_component=seasonal_component,
                seasonal_strength=seasonal_strength,
                seasonal_period=seasonal_period,
                residuals=residuals,
                execution_time=execution_time
            )
            
            self.logger.info(f"时间序列分解完成: 模型={model}, 季节性周期={seasonal_period}")
            return result
            
        except Exception as e:
            self.logger.error(f"时间序列分解失败: {str(e)}")
            return TSResult(
                analysis_type=TSAnalysisType.DECOMPOSITION,
                error_message=str(e)
            )
    
    def test_stationarity(self, data: pd.Series) -> TSResult:
        """平稳性检验"""
        try:
            import time
            start_time = time.time()
            
            # 数据准备
            data = self._prepare_time_series(data)
            
            # 平稳性检验
            stationarity_tests = self._test_stationarity(data)
            
            execution_time = time.time() - start_time
            
            result = TSResult(
                analysis_type=TSAnalysisType.STATIONARITY_TEST,
                original_data=data,
                stationarity_tests=stationarity_tests,
                execution_time=execution_time
            )
            
            self.logger.info("平稳性检验完成")
            return result
            
        except Exception as e:
            self.logger.error(f"平稳性检验失败: {str(e)}")
            return TSResult(
                analysis_type=TSAnalysisType.STATIONARITY_TEST,
                error_message=str(e)
            )
    
    def _prepare_time_series(self, data: pd.Series) -> pd.Series:
        """准备时间序列数据"""
        if not isinstance(data, pd.Series):
            raise ValueError("数据必须是pandas Series")
        
        # 移除缺失值
        data = data.dropna()
        
        if len(data) < 10:
            raise ValueError("数据点数量不足（最少需要10个点）")
        
        # 设置频率（如果未设置）
        if data.index.freq is None and hasattr(data.index, 'inferred_freq'):
            data.index.freq = data.index.inferred_freq
        
        return data
    
    def _linear_trend_analysis(self, data: pd.Series) -> Tuple[pd.Series, float]:
        """线性趋势分析"""
        from scipy import stats
        
        x = np.arange(len(data))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, data.values)
        
        trend_component = pd.Series(
            intercept + slope * x,
            index=data.index,
            name='trend'
        )
        
        return trend_component, slope
    
    def _stl_trend_analysis(self, data: pd.Series) -> Tuple[pd.Series, float]:
        """STL趋势分析"""
        # 自动检测季节性周期
        seasonal_period = self._detect_seasonal_period(data)
        
        if seasonal_period:
            stl = STL(data, seasonal=seasonal_period, robust=self.config.stl_robust)
            decomposition = stl.fit()
            trend_component = decomposition.trend
        else:
            # 如果没有季节性，使用移动平均
            window = min(len(data) // 4, 12)
            trend_component = data.rolling(window=window, center=True).mean()
        
        # 计算趋势斜率
        x = np.arange(len(trend_component))
        valid_trend = trend_component.dropna()
        valid_x = x[trend_component.notna()]
        
        if len(valid_trend) > 2:
            slope, _, _, _, _ = stats.linregress(valid_x, valid_trend.values)
        else:
            slope = 0.0
        
        return trend_component, slope
    
    def _determine_trend_direction(self, slope: float) -> str:
        """判断趋势方向"""
        if abs(slope) < 1e-6:
            return "无趋势"
        elif slope > 0:
            return "上升趋势"
        else:
            return "下降趋势"
    
    def _detect_seasonal_period(self, data: pd.Series, max_period: Optional[int] = None) -> Optional[int]:
        """检测季节性周期"""
        if max_period is None:
            max_period = min(len(data) // 2, 365)
        
        if len(data) < 20:
            return None
        
        # 使用自相关函数检测周期性
        autocorr_values = []
        periods = range(2, min(max_period + 1, len(data) // 2))
        
        for period in periods:
            if len(data) > period:
                autocorr = data.autocorr(lag=period)
                if not np.isnan(autocorr):
                    autocorr_values.append((period, abs(autocorr)))
        
        if autocorr_values:
            # 选择自相关最大的周期
            autocorr_values.sort(key=lambda x: x[1], reverse=True)
            best_period = autocorr_values[0][0]
            
            # 检查是否显著
            if autocorr_values[0][1] > 0.3:  # 阈值可调
                return best_period
        
        return None
    
    def _stl_decomposition(self, data: pd.Series, seasonal_period: Optional[int]) -> Any:
        """STL分解"""
        if seasonal_period is None or seasonal_period <= 1:
            raise ValueError("需要有效的季节性周期进行STL分解")
        
        stl = STL(
            data,
            seasonal=seasonal_period,
            trend=self.config.stl_trend,
            robust=self.config.stl_robust
        )
        return stl.fit()
    
    def _seasonal_decomposition(self, data: pd.Series, seasonal_period: Optional[int]) -> Any:
        """传统季节性分解"""
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        if seasonal_period is None or seasonal_period <= 1:
            raise ValueError("需要有效的季节性周期进行季节性分解")
        
        return seasonal_decompose(
            data,
            model='additive',
            period=seasonal_period,
            extrapolate_trend='freq'
        )
    
    def _calculate_seasonal_strength(self, decomposition: Any) -> float:
        """计算季节性强度"""
        try:
            seasonal_var = np.var(decomposition.seasonal.dropna())
            residual_var = np.var(decomposition.resid.dropna())
            
            if residual_var > 0:
                strength = seasonal_var / (seasonal_var + residual_var)
                return min(strength, 1.0)
            else:
                return 1.0
        except:
            return 0.0
    
    def _determine_seasonality_type(self, data: pd.Series, seasonal: pd.Series) -> SeasonalityType:
        """判断季节性类型"""
        try:
            # 简单判断：如果季节性变化与数据水平相关，则为乘性
            correlation = np.corrcoef(data.dropna(), np.abs(seasonal.dropna()))[0, 1]
            
            if np.isnan(correlation) or abs(correlation) < 0.3:
                return SeasonalityType.ADDITIVE
            elif correlation > 0.3:
                return SeasonalityType.MULTIPLICATIVE
            else:
                return SeasonalityType.ADDITIVE
        except:
            return SeasonalityType.ADDITIVE
    
    def _test_stationarity(self, data: pd.Series) -> Dict[str, Dict[str, float]]:
        """平稳性检验"""
        tests = {}
        
        try:
            # ADF检验
            adf_result = adfuller(data.dropna())
            tests['adf'] = {
                'statistic': adf_result[0],
                'pvalue': adf_result[1],
                'critical_values_1%': adf_result[4]['1%'],
                'critical_values_5%': adf_result[4]['5%'],
                'critical_values_10%': adf_result[4]['10%']
            }
        except Exception as e:
            self.logger.warning(f"ADF检验失败: {e}")
        
        try:
            # KPSS检验
            kpss_result = kpss(data.dropna())
            tests['kpss'] = {
                'statistic': kpss_result[0],
                'pvalue': kpss_result[1],
                'critical_values_1%': kpss_result[3]['1%'],
                'critical_values_5%': kpss_result[3]['5%'],
                'critical_values_10%': kpss_result[3]['10%']
            }
        except Exception as e:
            self.logger.warning(f"KPSS检验失败: {e}")
        
        return tests
    
    def _arima_forecast(self, data: pd.Series, steps: int) -> TSResult:
        """ARIMA预测"""
        # 自动选择ARIMA参数（简化版本）
        order = self.config.arima_order or (1, 1, 1)
        
        model = ARIMA(data, order=order)
        fitted_model = model.fit()
        
        # 预测
        forecast_result = fitted_model.get_forecast(steps=steps)
        forecast = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int()
        
        # 拟合值
        fitted_values = fitted_model.fittedvalues
        
        # 残差
        residuals = fitted_model.resid
        
        # 模型性能
        metrics = {
            'aic': fitted_model.aic,
            'bic': fitted_model.bic,
            'mse': np.mean(residuals**2),
            'mae': np.mean(np.abs(residuals))
        }
        
        return TSResult(
            model=fitted_model,
            forecast=forecast,
            forecast_conf_int=conf_int,
            fitted_values=fitted_values,
            residuals=residuals,
            metrics=metrics,
            information_criteria={'aic': fitted_model.aic, 'bic': fitted_model.bic}
        )
    
    def _sarima_forecast(self, data: pd.Series, steps: int) -> TSResult:
        """SARIMA预测"""
        # 自动检测季节性
        seasonal_period = self._detect_seasonal_period(data)
        
        if seasonal_period:
            order = self.config.arima_order or (1, 1, 1)
            seasonal_order = self.config.seasonal_order or (1, 1, 1, seasonal_period)
        else:
            # 如果没有季节性，回退到ARIMA
            return self._arima_forecast(data, steps)
        
        model = SARIMAX(data, order=order, seasonal_order=seasonal_order)
        fitted_model = model.fit()
        
        # 预测
        forecast_result = fitted_model.get_forecast(steps=steps)
        forecast = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int()
        
        # 拟合值
        fitted_values = fitted_model.fittedvalues
        
        # 残差
        residuals = fitted_model.resid
        
        # 模型性能
        metrics = {
            'aic': fitted_model.aic,
            'bic': fitted_model.bic,
            'mse': np.mean(residuals**2),
            'mae': np.mean(np.abs(residuals))
        }
        
        return TSResult(
            model=fitted_model,
            forecast=forecast,
            forecast_conf_int=conf_int,
            fitted_values=fitted_values,
            residuals=residuals,
            metrics=metrics,
            information_criteria={'aic': fitted_model.aic, 'bic': fitted_model.bic}
        )
    
    def _stl_forecast(self, data: pd.Series, steps: int) -> TSResult:
        """STL预测"""
        # 检测季节性周期
        seasonal_period = self._detect_seasonal_period(data)
        
        if not seasonal_period:
            raise ValueError("STL预测需要明显的季节性模式")
        
        # 使用ARIMA作为STL的预测模型
        arima_order = self.config.arima_order or (1, 1, 0)
        
        stlf = STLForecast(
            data,
            ARIMA,
            model_kwargs=dict(order=arima_order, trend="c"),
            period=seasonal_period
        )
        stlf_result = stlf.fit()
        
        # 预测
        forecast = stlf_result.forecast(steps)
        
        # 获取STL分解结果
        stl_decomposition = stlf_result.stl_res
        
        # 简化的拟合值和残差
        fitted_values = stlf_result.model.fittedvalues
        residuals = stlf_result.model.resid
        
        return TSResult(
            model=stlf_result,
            forecast=forecast,
            fitted_values=fitted_values,
            residuals=residuals,
            decomposition=stl_decomposition,
            seasonal_period=seasonal_period
        )
    
    def _holt_winters_forecast(self, data: pd.Series, steps: int) -> TSResult:
        """Holt-Winters预测"""
        # 检测季节性
        seasonal_period = self._detect_seasonal_period(data)
        
        if seasonal_period and seasonal_period > 1:
            # 有季节性的Holt-Winters
            model = HoltWinters(
                data,
                trend=self.config.trend_type or 'add',
                seasonal=self.config.seasonal_type or 'add',
                seasonal_periods=seasonal_period,
                damped_trend=self.config.damped_trend
            )
        else:
            # 无季节性的指数平滑
            model = HoltWinters(
                data,
                trend=self.config.trend_type,
                seasonal=None,
                damped_trend=self.config.damped_trend
            )
        
        fitted_model = model.fit()
        
        # 预测
        forecast = fitted_model.forecast(steps)
        
        # 拟合值和残差
        fitted_values = fitted_model.fittedvalues
        residuals = fitted_model.resid
        
        return TSResult(
            model=fitted_model,
            forecast=forecast,
            fitted_values=fitted_values,
            residuals=residuals,
            seasonal_period=seasonal_period
        )


def create_ts_engine(config: Optional[TSConfig] = None) -> TimeSeriesEngine:
    """创建时间序列分析引擎的工厂函数"""
    try:
        return TimeSeriesEngine(config)
    except Exception as e:
        raise DataProcessingError(f"创建时间序列分析引擎失败: {str(e)}") from e