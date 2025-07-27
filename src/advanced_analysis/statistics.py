"""
高级统计分析模块
基于scipy实现多元统计、假设检验、回归分析
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
    import scipy.stats as stats
    from scipy.stats import (
        # 假设检验
        ttest_1samp, ttest_ind, ttest_rel,
        mannwhitneyu, wilcoxon, kruskal,
        chi2_contingency, fisher_exact,
        shapiro, normaltest, jarque_bera,
        levene, bartlett, fligner,
        pearsonr, spearmanr, kendalltau,
        # 多元统计
        multivariate_normal,
        # 回归分析
        linregress, theilslopes
    )
    from scipy import linalg
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    # 尝试导入sklearn用于更高级的回归分析
    from sklearn.linear_model import (
        LinearRegression, Ridge, Lasso, ElasticNet,
        LogisticRegression, PoissonRegressor
    )
    from sklearn.metrics import r2_score, mean_squared_error
    from sklearn.preprocessing import PolynomialFeatures
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from ..utils.basic_logging import LoggerMixin
from ..utils.exceptions import DataProcessingError


class StatTestType(str, Enum):
    """统计检验类型"""
    # 参数检验
    T_TEST_ONE_SAMPLE = "t_test_one_sample"
    T_TEST_INDEPENDENT = "t_test_independent"
    T_TEST_PAIRED = "t_test_paired"
    
    # 非参数检验
    MANN_WHITNEY_U = "mann_whitney_u"
    WILCOXON_SIGNED_RANK = "wilcoxon_signed_rank"
    KRUSKAL_WALLIS = "kruskal_wallis"
    
    # 正态性检验
    SHAPIRO_WILK = "shapiro_wilk"
    JARQUE_BERA = "jarque_bera"
    NORMALTEST = "normaltest"
    
    # 方差齐性检验
    LEVENE = "levene"
    BARTLETT = "bartlett"
    FLIGNER = "fligner"
    
    # 关联性检验
    CHI_SQUARE = "chi_square"
    FISHER_EXACT = "fisher_exact"
    
    # 相关性检验
    PEARSON_CORRELATION = "pearson_correlation"
    SPEARMAN_CORRELATION = "spearman_correlation"
    KENDALL_TAU = "kendall_tau"


class RegressionType(str, Enum):
    """回归分析类型"""
    LINEAR = "linear"
    MULTIPLE_LINEAR = "multiple_linear"
    POLYNOMIAL = "polynomial"
    RIDGE = "ridge"
    LASSO = "lasso"
    ELASTIC_NET = "elastic_net"
    LOGISTIC = "logistic"
    POISSON = "poisson"
    THEIL_SEN = "theil_sen"


class MultivariateTestType(str, Enum):
    """多元统计检验类型"""
    MULTIVARIATE_NORMALITY = "multivariate_normality"
    HOTELLING_T2 = "hotelling_t2"
    MANOVA = "manova"
    PRINCIPAL_COMPONENT = "principal_component"


@dataclass
class StatTestConfig:
    """统计检验配置"""
    # 基础设置
    alpha: float = 0.05
    alternative: str = "two-sided"  # 'two-sided', 'less', 'greater'
    
    # 样本设置
    equal_var: bool = True
    paired: bool = False
    
    # 多重比较设置
    correction_method: Optional[str] = None  # 'bonferroni', 'holm', 'fdr_bh'
    
    # 效应量设置
    effect_size: bool = True
    confidence_interval: bool = True
    
    # 引导法设置
    bootstrap_samples: int = 1000
    random_state: Optional[int] = None


@dataclass
class RegressionConfig:
    """回归分析配置"""
    # 基础设置
    fit_intercept: bool = True
    normalize: bool = False
    
    # 正则化设置
    alpha: float = 1.0
    l1_ratio: float = 0.5  # ElasticNet参数
    
    # 多项式设置
    degree: int = 2
    include_bias: bool = True
    
    # 验证设置
    cross_validation: bool = True
    cv_folds: int = 5
    
    # 诊断设置
    residual_analysis: bool = True
    influence_analysis: bool = True


@dataclass
class StatResult:
    """统计分析结果"""
    test_type: Union[StatTestType, RegressionType, MultivariateTestType]
    
    # 检验统计量
    statistic: Optional[float] = None
    pvalue: Optional[float] = None
    degrees_of_freedom: Optional[Union[int, Tuple[int, ...]]] = None
    
    # 置信区间
    confidence_interval: Optional[Tuple[float, float]] = None
    confidence_level: float = 0.95
    
    # 效应量
    effect_size: Optional[float] = None
    effect_size_type: Optional[str] = None
    
    # 回归结果
    coefficients: Optional[Union[float, np.ndarray]] = None
    intercept: Optional[float] = None
    r_squared: Optional[float] = None
    adjusted_r_squared: Optional[float] = None
    
    # 回归诊断
    residuals: Optional[np.ndarray] = None
    fitted_values: Optional[np.ndarray] = None
    standardized_residuals: Optional[np.ndarray] = None
    leverage: Optional[np.ndarray] = None
    cooks_distance: Optional[np.ndarray] = None
    
    # 模型评估
    aic: Optional[float] = None
    bic: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    
    # 假设检验信息
    null_hypothesis: Optional[str] = None
    alternative_hypothesis: Optional[str] = None
    test_assumptions: Optional[List[str]] = None
    
    # 原始数据
    sample_data: Optional[Dict[str, np.ndarray]] = None
    sample_statistics: Optional[Dict[str, float]] = None
    
    # 多元分析
    multivariate_stats: Optional[Dict[str, Any]] = None
    
    # 执行信息
    execution_time: float = 0.0
    error_message: Optional[str] = None


class AdvancedStatisticsEngine(LoggerMixin):
    """高级统计分析引擎"""
    
    def __init__(self, 
                 test_config: Optional[StatTestConfig] = None,
                 regression_config: Optional[RegressionConfig] = None):
        self.test_config = test_config or StatTestConfig()
        self.regression_config = regression_config or RegressionConfig()
        
        if not HAS_SCIPY:
            raise ImportError("scipy未安装，无法使用统计分析功能")
        if not HAS_PANDAS:
            raise ImportError("pandas未安装，无法处理数据")
    
    def hypothesis_test(
        self,
        test_type: StatTestType,
        *args,
        **kwargs
    ) -> StatResult:
        """假设检验"""
        try:
            import time
            start_time = time.time()
            
            # 根据检验类型调用相应方法
            if test_type == StatTestType.T_TEST_ONE_SAMPLE:
                result = self._t_test_one_sample(*args, **kwargs)
            elif test_type == StatTestType.T_TEST_INDEPENDENT:
                result = self._t_test_independent(*args, **kwargs)
            elif test_type == StatTestType.T_TEST_PAIRED:
                result = self._t_test_paired(*args, **kwargs)
            elif test_type == StatTestType.MANN_WHITNEY_U:
                result = self._mann_whitney_u(*args, **kwargs)
            elif test_type == StatTestType.WILCOXON_SIGNED_RANK:
                result = self._wilcoxon_signed_rank(*args, **kwargs)
            elif test_type == StatTestType.KRUSKAL_WALLIS:
                result = self._kruskal_wallis(*args, **kwargs)
            elif test_type == StatTestType.SHAPIRO_WILK:
                result = self._shapiro_wilk(*args, **kwargs)
            elif test_type == StatTestType.JARQUE_BERA:
                result = self._jarque_bera(*args, **kwargs)
            elif test_type == StatTestType.NORMALTEST:
                result = self._normaltest(*args, **kwargs)
            elif test_type == StatTestType.LEVENE:
                result = self._levene(*args, **kwargs)
            elif test_type == StatTestType.BARTLETT:
                result = self._bartlett(*args, **kwargs)
            elif test_type == StatTestType.FLIGNER:
                result = self._fligner(*args, **kwargs)
            elif test_type == StatTestType.CHI_SQUARE:
                result = self._chi_square(*args, **kwargs)
            elif test_type == StatTestType.FISHER_EXACT:
                result = self._fisher_exact(*args, **kwargs)
            elif test_type == StatTestType.PEARSON_CORRELATION:
                result = self._pearson_correlation(*args, **kwargs)
            elif test_type == StatTestType.SPEARMAN_CORRELATION:
                result = self._spearman_correlation(*args, **kwargs)
            elif test_type == StatTestType.KENDALL_TAU:
                result = self._kendall_tau(*args, **kwargs)
            else:
                raise ValueError(f"不支持的检验类型: {test_type}")
            
            result.test_type = test_type
            result.execution_time = time.time() - start_time
            
            self.logger.info(f"假设检验完成: {test_type.value}, p值={result.pvalue:.4f}")
            return result
            
        except Exception as e:
            self.logger.error(f"假设检验失败: {str(e)}")
            return StatResult(
                test_type=test_type,
                error_message=str(e)
            )
    
    def regression_analysis(
        self,
        regression_type: RegressionType,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        **kwargs
    ) -> StatResult:
        """回归分析"""
        try:
            import time
            start_time = time.time()
            
            # 数据预处理
            X, y = self._prepare_regression_data(X, y)
            
            # 根据回归类型调用相应方法
            if regression_type == RegressionType.LINEAR:
                result = self._simple_linear_regression(X, y, **kwargs)
            elif regression_type == RegressionType.MULTIPLE_LINEAR:
                result = self._multiple_linear_regression(X, y, **kwargs)
            elif regression_type == RegressionType.POLYNOMIAL:
                result = self._polynomial_regression(X, y, **kwargs)
            elif regression_type == RegressionType.RIDGE:
                result = self._ridge_regression(X, y, **kwargs)
            elif regression_type == RegressionType.LASSO:
                result = self._lasso_regression(X, y, **kwargs)
            elif regression_type == RegressionType.ELASTIC_NET:
                result = self._elastic_net_regression(X, y, **kwargs)
            elif regression_type == RegressionType.LOGISTIC:
                result = self._logistic_regression(X, y, **kwargs)
            elif regression_type == RegressionType.POISSON:
                result = self._poisson_regression(X, y, **kwargs)
            elif regression_type == RegressionType.THEIL_SEN:
                result = self._theil_sen_regression(X, y, **kwargs)
            else:
                raise ValueError(f"不支持的回归类型: {regression_type}")
            
            result.test_type = regression_type
            result.execution_time = time.time() - start_time
            
            self.logger.info(f"回归分析完成: {regression_type.value}, R²={result.r_squared:.4f}")
            return result
            
        except Exception as e:
            self.logger.error(f"回归分析失败: {str(e)}")
            return StatResult(
                test_type=regression_type,
                error_message=str(e)
            )
    
    def multivariate_analysis(
        self,
        test_type: MultivariateTestType,
        data: Union[np.ndarray, pd.DataFrame],
        **kwargs
    ) -> StatResult:
        """多元统计分析"""
        try:
            import time
            start_time = time.time()
            
            # 数据预处理
            if isinstance(data, pd.DataFrame):
                data = data.values
            
            # 根据分析类型调用相应方法
            if test_type == MultivariateTestType.MULTIVARIATE_NORMALITY:
                result = self._multivariate_normality_test(data, **kwargs)
            elif test_type == MultivariateTestType.HOTELLING_T2:
                result = self._hotelling_t2_test(data, **kwargs)
            elif test_type == MultivariateTestType.MANOVA:
                result = self._manova_test(data, **kwargs)
            elif test_type == MultivariateTestType.PRINCIPAL_COMPONENT:
                result = self._principal_component_analysis(data, **kwargs)
            else:
                raise ValueError(f"不支持的多元分析类型: {test_type}")
            
            result.test_type = test_type
            result.execution_time = time.time() - start_time
            
            self.logger.info(f"多元分析完成: {test_type.value}")
            return result
            
        except Exception as e:
            self.logger.error(f"多元分析失败: {str(e)}")
            return StatResult(
                test_type=test_type,
                error_message=str(e)
            )
    
    # =========================
    # 假设检验实现
    # =========================
    
    def _t_test_one_sample(self, data: np.ndarray, popmean: float) -> StatResult:
        """单样本t检验"""
        stat, pval = ttest_1samp(data, popmean, alternative=self.test_config.alternative)
        
        # 计算效应量 (Cohen's d)
        effect_size = (np.mean(data) - popmean) / np.std(data, ddof=1)
        
        # 置信区间
        n = len(data)
        se = stats.sem(data)
        ci = stats.t.interval(1 - self.test_config.alpha, n - 1, loc=np.mean(data), scale=se)
        
        return StatResult(
            statistic=stat,
            pvalue=pval,
            degrees_of_freedom=n - 1,
            effect_size=effect_size,
            effect_size_type="Cohen's d",
            confidence_interval=ci,
            null_hypothesis=f"μ = {popmean}",
            alternative_hypothesis=f"μ {self.test_config.alternative} {popmean}",
            sample_data={"sample": data},
            sample_statistics={
                "mean": np.mean(data),
                "std": np.std(data, ddof=1),
                "n": n
            }
        )
    
    def _t_test_independent(self, group1: np.ndarray, group2: np.ndarray) -> StatResult:
        """独立样本t检验"""
        stat, pval = ttest_ind(
            group1, group2,
            equal_var=self.test_config.equal_var,
            alternative=self.test_config.alternative
        )
        
        # 计算效应量 (Cohen's d)
        pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1, ddof=1) + 
                             (len(group2) - 1) * np.var(group2, ddof=1)) / 
                            (len(group1) + len(group2) - 2))
        effect_size = (np.mean(group1) - np.mean(group2)) / pooled_std
        
        # 自由度
        if self.test_config.equal_var:
            df = len(group1) + len(group2) - 2
        else:
            # Welch's t-test
            s1, s2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
            n1, n2 = len(group1), len(group2)
            df = (s1/n1 + s2/n2)**2 / ((s1/n1)**2/(n1-1) + (s2/n2)**2/(n2-1))
        
        return StatResult(
            statistic=stat,
            pvalue=pval,
            degrees_of_freedom=df,
            effect_size=effect_size,
            effect_size_type="Cohen's d",
            null_hypothesis="μ₁ = μ₂",
            alternative_hypothesis=f"μ₁ {self.test_config.alternative} μ₂",
            sample_data={"group1": group1, "group2": group2},
            sample_statistics={
                "mean1": np.mean(group1),
                "mean2": np.mean(group2),
                "std1": np.std(group1, ddof=1),
                "std2": np.std(group2, ddof=1),
                "n1": len(group1),
                "n2": len(group2)
            }
        )
    
    def _t_test_paired(self, before: np.ndarray, after: np.ndarray) -> StatResult:
        """配对样本t检验"""
        stat, pval = ttest_rel(before, after, alternative=self.test_config.alternative)
        
        # 计算差值
        diff = before - after
        
        # 计算效应量 (Cohen's d for paired samples)
        effect_size = np.mean(diff) / np.std(diff, ddof=1)
        
        return StatResult(
            statistic=stat,
            pvalue=pval,
            degrees_of_freedom=len(diff) - 1,
            effect_size=effect_size,
            effect_size_type="Cohen's d (paired)",
            null_hypothesis="μ_diff = 0",
            alternative_hypothesis=f"μ_diff {self.test_config.alternative} 0",
            sample_data={"before": before, "after": after, "difference": diff},
            sample_statistics={
                "mean_before": np.mean(before),
                "mean_after": np.mean(after),
                "mean_diff": np.mean(diff),
                "std_diff": np.std(diff, ddof=1),
                "n": len(diff)
            }
        )
    
    def _mann_whitney_u(self, group1: np.ndarray, group2: np.ndarray) -> StatResult:
        """Mann-Whitney U检验"""
        stat, pval = mannwhitneyu(
            group1, group2,
            alternative=self.test_config.alternative
        )
        
        # 计算效应量 (rank-biserial correlation)
        n1, n2 = len(group1), len(group2)
        effect_size = 1 - (2 * stat) / (n1 * n2)
        
        return StatResult(
            statistic=stat,
            pvalue=pval,
            effect_size=effect_size,
            effect_size_type="rank-biserial correlation",
            null_hypothesis="两组分布相同",
            alternative_hypothesis="两组分布不同",
            sample_data={"group1": group1, "group2": group2},
            sample_statistics={
                "median1": np.median(group1),
                "median2": np.median(group2),
                "n1": n1,
                "n2": n2
            }
        )
    
    def _wilcoxon_signed_rank(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> StatResult:
        """Wilcoxon符号秩检验"""
        stat, pval = wilcoxon(x, y, alternative=self.test_config.alternative)
        
        # 计算效应量
        n = len(x)
        z_score = (stat - n*(n+1)/4) / np.sqrt(n*(n+1)*(2*n+1)/24)
        effect_size = z_score / np.sqrt(n)
        
        return StatResult(
            statistic=stat,
            pvalue=pval,
            effect_size=effect_size,
            effect_size_type="r (effect size)",
            null_hypothesis="中位数差为0",
            sample_data={"x": x, "y": y},
            sample_statistics={
                "median_x": np.median(x),
                "median_y": np.median(y) if y is not None else None,
                "n": n
            }
        )
    
    def _kruskal_wallis(self, *groups) -> StatResult:
        """Kruskal-Wallis检验"""
        stat, pval = kruskal(*groups)
        
        # 计算效应量 (eta squared)
        N = sum(len(group) for group in groups)
        k = len(groups)
        effect_size = (stat - k + 1) / (N - k)
        
        return StatResult(
            statistic=stat,
            pvalue=pval,
            degrees_of_freedom=k - 1,
            effect_size=effect_size,
            effect_size_type="eta squared",
            null_hypothesis="所有组的分布相同",
            sample_data={f"group_{i+1}": group for i, group in enumerate(groups)},
            sample_statistics={
                f"median_group_{i+1}": np.median(group) 
                for i, group in enumerate(groups)
            }
        )
    
    def _shapiro_wilk(self, data: np.ndarray) -> StatResult:
        """Shapiro-Wilk正态性检验"""
        stat, pval = shapiro(data)
        
        return StatResult(
            statistic=stat,
            pvalue=pval,
            null_hypothesis="数据服从正态分布",
            test_assumptions=["样本大小应在3-5000之间"],
            sample_data={"data": data},
            sample_statistics={
                "mean": np.mean(data),
                "std": np.std(data, ddof=1),
                "skewness": stats.skew(data),
                "kurtosis": stats.kurtosis(data),
                "n": len(data)
            }
        )
    
    def _jarque_bera(self, data: np.ndarray) -> StatResult:
        """Jarque-Bera正态性检验"""
        stat, pval = jarque_bera(data)
        
        return StatResult(
            statistic=stat,
            pvalue=pval,
            degrees_of_freedom=2,
            null_hypothesis="数据服从正态分布",
            test_assumptions=["大样本检验", "基于偏度和峰度"],
            sample_data={"data": data},
            sample_statistics={
                "mean": np.mean(data),
                "std": np.std(data, ddof=1),
                "skewness": stats.skew(data),
                "kurtosis": stats.kurtosis(data),
                "n": len(data)
            }
        )
    
    def _normaltest(self, data: np.ndarray) -> StatResult:
        """D'Agostino正态性检验"""
        stat, pval = normaltest(data)
        
        return StatResult(
            statistic=stat,
            pvalue=pval,
            degrees_of_freedom=2,
            null_hypothesis="数据服从正态分布",
            test_assumptions=["样本大小≥8"],
            sample_data={"data": data},
            sample_statistics={
                "mean": np.mean(data),
                "std": np.std(data, ddof=1),
                "skewness": stats.skew(data),
                "kurtosis": stats.kurtosis(data),
                "n": len(data)
            }
        )
    
    def _levene(self, *groups) -> StatResult:
        """Levene方差齐性检验"""
        stat, pval = levene(*groups)
        
        k = len(groups)
        N = sum(len(group) for group in groups)
        
        return StatResult(
            statistic=stat,
            pvalue=pval,
            degrees_of_freedom=(k - 1, N - k),
            null_hypothesis="各组方差相等",
            test_assumptions=["对非正态分布较稳健"],
            sample_data={f"group_{i+1}": group for i, group in enumerate(groups)},
            sample_statistics={
                f"var_group_{i+1}": np.var(group, ddof=1) 
                for i, group in enumerate(groups)
            }
        )
    
    def _bartlett(self, *groups) -> StatResult:
        """Bartlett方差齐性检验"""
        stat, pval = bartlett(*groups)
        
        k = len(groups)
        
        return StatResult(
            statistic=stat,
            pvalue=pval,
            degrees_of_freedom=k - 1,
            null_hypothesis="各组方差相等",
            test_assumptions=["假设各组数据服从正态分布"],
            sample_data={f"group_{i+1}": group for i, group in enumerate(groups)},
            sample_statistics={
                f"var_group_{i+1}": np.var(group, ddof=1) 
                for i, group in enumerate(groups)
            }
        )
    
    def _fligner(self, *groups) -> StatResult:
        """Fligner-Killeen方差齐性检验"""
        stat, pval = fligner(*groups)
        
        k = len(groups)
        N = sum(len(group) for group in groups)
        
        return StatResult(
            statistic=stat,
            pvalue=pval,
            degrees_of_freedom=k - 1,
            null_hypothesis="各组方差相等",
            test_assumptions=["对非正态分布稳健"],
            sample_data={f"group_{i+1}": group for i, group in enumerate(groups)},
            sample_statistics={
                f"var_group_{i+1}": np.var(group, ddof=1) 
                for i, group in enumerate(groups)
            }
        )
    
    def _chi_square(self, observed: np.ndarray, expected: Optional[np.ndarray] = None) -> StatResult:
        """卡方检验"""
        if expected is None:
            # 拟合优度检验
            expected = np.full_like(observed, np.mean(observed))
        
        if observed.ndim == 1:
            # 一维卡方检验
            stat, pval = stats.chisquare(observed, expected)
            df = len(observed) - 1
        else:
            # 独立性检验
            stat, pval, df, expected_freq = chi2_contingency(observed)
        
        return StatResult(
            statistic=stat,
            pvalue=pval,
            degrees_of_freedom=df,
            null_hypothesis="观测频数与期望频数无显著差异" if observed.ndim == 1 else "变量独立",
            test_assumptions=["期望频数≥5"],
            sample_data={"observed": observed, "expected": expected}
        )
    
    def _fisher_exact(self, table: np.ndarray) -> StatResult:
        """Fisher精确检验"""
        stat, pval = fisher_exact(table, alternative=self.test_config.alternative)
        
        return StatResult(
            statistic=stat,
            pvalue=pval,
            null_hypothesis="两变量独立",
            test_assumptions=["适用于小样本", "2x2列联表"],
            sample_data={"contingency_table": table}
        )
    
    def _pearson_correlation(self, x: np.ndarray, y: np.ndarray) -> StatResult:
        """Pearson相关性检验"""
        stat, pval = pearsonr(x, y, alternative=self.test_config.alternative)
        
        n = len(x)
        # Fisher变换置信区间
        z = np.arctanh(stat)
        se = 1 / np.sqrt(n - 3)
        z_ci = stats.norm.interval(1 - self.test_config.alpha, loc=z, scale=se)
        ci = tuple(np.tanh(z_ci))
        
        return StatResult(
            statistic=stat,
            pvalue=pval,
            degrees_of_freedom=n - 2,
            confidence_interval=ci,
            null_hypothesis="ρ = 0",
            test_assumptions=["线性关系", "双变量正态分布"],
            sample_data={"x": x, "y": y},
            sample_statistics={
                "mean_x": np.mean(x),
                "mean_y": np.mean(y),
                "std_x": np.std(x, ddof=1),
                "std_y": np.std(y, ddof=1),
                "n": n
            }
        )
    
    def _spearman_correlation(self, x: np.ndarray, y: np.ndarray) -> StatResult:
        """Spearman秩相关检验"""
        stat, pval = spearmanr(x, y, alternative=self.test_config.alternative)
        
        return StatResult(
            statistic=stat,
            pvalue=pval,
            null_hypothesis="ρₛ = 0",
            test_assumptions=["单调关系", "对异常值稳健"],
            sample_data={"x": x, "y": y},
            sample_statistics={
                "median_x": np.median(x),
                "median_y": np.median(y),
                "n": len(x)
            }
        )
    
    def _kendall_tau(self, x: np.ndarray, y: np.ndarray) -> StatResult:
        """Kendall's tau相关检验"""
        stat, pval = kendalltau(x, y, alternative=self.test_config.alternative)
        
        return StatResult(
            statistic=stat,
            pvalue=pval,
            null_hypothesis="τ = 0",
            test_assumptions=["单调关系", "对异常值稳健"],
            sample_data={"x": x, "y": y},
            sample_statistics={
                "median_x": np.median(x),
                "median_y": np.median(y),
                "n": len(x)
            }
        )
    
    # =========================
    # 回归分析实现
    # =========================
    
    def _prepare_regression_data(self, X, y):
        """准备回归数据"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        return X, y
    
    def _simple_linear_regression(self, X: np.ndarray, y: np.ndarray) -> StatResult:
        """简单线性回归"""
        if X.shape[1] != 1:
            raise ValueError("简单线性回归只适用于单变量")
        
        x = X.flatten()
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        
        # 计算拟合值和残差
        fitted_values = slope * x + intercept
        residuals = y - fitted_values
        
        # 模型评估
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        n = len(y)
        adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - 2)
        
        rmse = np.sqrt(np.mean(residuals ** 2))
        mae = np.mean(np.abs(residuals))
        
        return StatResult(
            statistic=slope / std_err,  # t统计量
            pvalue=p_value,
            coefficients=np.array([slope]),
            intercept=intercept,
            r_squared=r_squared,
            adjusted_r_squared=adjusted_r_squared,
            residuals=residuals,
            fitted_values=fitted_values,
            rmse=rmse,
            mae=mae,
            degrees_of_freedom=n - 2
        )
    
    def _multiple_linear_regression(self, X: np.ndarray, y: np.ndarray) -> StatResult:
        """多元线性回归"""
        # 添加截距项
        if self.regression_config.fit_intercept:
            X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        else:
            X_with_intercept = X
        
        # 最小二乘估计
        try:
            coeffs = linalg.lstsq(X_with_intercept, y)[0]
        except linalg.LinAlgError:
            raise ValueError("设计矩阵奇异，无法进行回归")
        
        # 分离截距和系数
        if self.regression_config.fit_intercept:
            intercept = coeffs[0]
            coefficients = coeffs[1:]
        else:
            intercept = 0
            coefficients = coeffs
        
        # 预测和残差
        fitted_values = X_with_intercept @ coeffs
        residuals = y - fitted_values
        
        # 模型评估
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        n, p = X.shape[0], X.shape[1]
        if self.regression_config.fit_intercept:
            p += 1
        
        adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p)
        
        rmse = np.sqrt(np.mean(residuals ** 2))
        mae = np.mean(np.abs(residuals))
        
        # 计算AIC和BIC
        log_likelihood = -0.5 * n * np.log(2 * np.pi * ss_res / n) - 0.5 * ss_res / (ss_res / n)
        aic = 2 * p - 2 * log_likelihood
        bic = p * np.log(n) - 2 * log_likelihood
        
        return StatResult(
            coefficients=coefficients,
            intercept=intercept,
            r_squared=r_squared,
            adjusted_r_squared=adjusted_r_squared,
            residuals=residuals,
            fitted_values=fitted_values,
            rmse=rmse,
            mae=mae,
            aic=aic,
            bic=bic,
            degrees_of_freedom=n - p
        )
    
    def _polynomial_regression(self, X: np.ndarray, y: np.ndarray) -> StatResult:
        """多项式回归"""
        if X.shape[1] != 1:
            raise ValueError("多项式回归只适用于单变量")
        
        # 生成多项式特征
        poly_features = PolynomialFeatures(
            degree=self.regression_config.degree,
            include_bias=self.regression_config.include_bias
        )
        X_poly = poly_features.fit_transform(X)
        
        # 使用多元线性回归
        original_fit_intercept = self.regression_config.fit_intercept
        self.regression_config.fit_intercept = not self.regression_config.include_bias
        
        result = self._multiple_linear_regression(X_poly, y)
        
        # 恢复原设置
        self.regression_config.fit_intercept = original_fit_intercept
        
        return result
    
    def _ridge_regression(self, X: np.ndarray, y: np.ndarray) -> StatResult:
        """岭回归"""
        if not HAS_SKLEARN:
            raise ImportError("sklearn未安装，无法使用岭回归")
        
        model = Ridge(
            alpha=self.regression_config.alpha,
            fit_intercept=self.regression_config.fit_intercept
        )
        model.fit(X, y)
        
        # 预测和残差
        fitted_values = model.predict(X)
        residuals = y - fitted_values
        
        # 模型评估
        r_squared = model.score(X, y)
        
        n, p = X.shape
        if self.regression_config.fit_intercept:
            p += 1
        
        adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p)
        
        rmse = np.sqrt(mean_squared_error(y, fitted_values))
        mae = np.mean(np.abs(residuals))
        
        return StatResult(
            coefficients=model.coef_,
            intercept=model.intercept_ if self.regression_config.fit_intercept else 0,
            r_squared=r_squared,
            adjusted_r_squared=adjusted_r_squared,
            residuals=residuals,
            fitted_values=fitted_values,
            rmse=rmse,
            mae=mae
        )
    
    def _lasso_regression(self, X: np.ndarray, y: np.ndarray) -> StatResult:
        """Lasso回归"""
        if not HAS_SKLEARN:
            raise ImportError("sklearn未安装，无法使用Lasso回归")
        
        model = Lasso(
            alpha=self.regression_config.alpha,
            fit_intercept=self.regression_config.fit_intercept
        )
        model.fit(X, y)
        
        # 预测和残差
        fitted_values = model.predict(X)
        residuals = y - fitted_values
        
        # 模型评估
        r_squared = model.score(X, y)
        
        n, p = X.shape
        if self.regression_config.fit_intercept:
            p += 1
        
        adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p)
        
        rmse = np.sqrt(mean_squared_error(y, fitted_values))
        mae = np.mean(np.abs(residuals))
        
        return StatResult(
            coefficients=model.coef_,
            intercept=model.intercept_ if self.regression_config.fit_intercept else 0,
            r_squared=r_squared,
            adjusted_r_squared=adjusted_r_squared,
            residuals=residuals,
            fitted_values=fitted_values,
            rmse=rmse,
            mae=mae
        )
    
    def _elastic_net_regression(self, X: np.ndarray, y: np.ndarray) -> StatResult:
        """弹性网回归"""
        if not HAS_SKLEARN:
            raise ImportError("sklearn未安装，无法使用弹性网回归")
        
        model = ElasticNet(
            alpha=self.regression_config.alpha,
            l1_ratio=self.regression_config.l1_ratio,
            fit_intercept=self.regression_config.fit_intercept
        )
        model.fit(X, y)
        
        # 预测和残差
        fitted_values = model.predict(X)
        residuals = y - fitted_values
        
        # 模型评估
        r_squared = model.score(X, y)
        
        n, p = X.shape
        if self.regression_config.fit_intercept:
            p += 1
        
        adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p)
        
        rmse = np.sqrt(mean_squared_error(y, fitted_values))
        mae = np.mean(np.abs(residuals))
        
        return StatResult(
            coefficients=model.coef_,
            intercept=model.intercept_ if self.regression_config.fit_intercept else 0,
            r_squared=r_squared,
            adjusted_r_squared=adjusted_r_squared,
            residuals=residuals,
            fitted_values=fitted_values,
            rmse=rmse,
            mae=mae
        )
    
    def _logistic_regression(self, X: np.ndarray, y: np.ndarray) -> StatResult:
        """逻辑回归"""
        if not HAS_SKLEARN:
            raise ImportError("sklearn未安装，无法使用逻辑回归")
        
        model = LogisticRegression(
            fit_intercept=self.regression_config.fit_intercept
        )
        model.fit(X, y)
        
        # 预测概率和分类
        fitted_probs = model.predict_proba(X)[:, 1]
        fitted_classes = model.predict(X)
        
        # 伪R²
        from sklearn.metrics import log_loss
        null_deviance = -2 * log_loss(y, [np.mean(y)] * len(y))
        model_deviance = -2 * log_loss(y, fitted_probs)
        pseudo_r_squared = 1 - (model_deviance / null_deviance)
        
        return StatResult(
            coefficients=model.coef_[0],
            intercept=model.intercept_[0] if self.regression_config.fit_intercept else 0,
            r_squared=pseudo_r_squared,
            fitted_values=fitted_probs,
            sample_data={"predicted_classes": fitted_classes}
        )
    
    def _poisson_regression(self, X: np.ndarray, y: np.ndarray) -> StatResult:
        """泊松回归"""
        if not HAS_SKLEARN:
            raise ImportError("sklearn未安装，无法使用泊松回归")
        
        model = PoissonRegressor(
            fit_intercept=self.regression_config.fit_intercept
        )
        model.fit(X, y)
        
        # 预测
        fitted_values = model.predict(X)
        
        # 偏差
        deviance = 2 * np.sum(y * np.log(y / fitted_values + 1e-10) - (y - fitted_values))
        null_deviance = 2 * np.sum(y * np.log(y / np.mean(y) + 1e-10) - (y - np.mean(y)))
        pseudo_r_squared = 1 - (deviance / null_deviance)
        
        return StatResult(
            coefficients=model.coef_,
            intercept=model.intercept_ if self.regression_config.fit_intercept else 0,
            r_squared=pseudo_r_squared,
            fitted_values=fitted_values
        )
    
    def _theil_sen_regression(self, X: np.ndarray, y: np.ndarray) -> StatResult:
        """Theil-Sen回归"""
        if X.shape[1] != 1:
            raise ValueError("Theil-Sen回归只适用于单变量")
        
        x = X.flatten()
        slope, intercept, low_slope, high_slope = theilslopes(y, x, alpha=self.test_config.alpha)
        
        # 预测和残差
        fitted_values = slope * x + intercept
        residuals = y - fitted_values
        
        # R²
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        return StatResult(
            coefficients=np.array([slope]),
            intercept=intercept,
            r_squared=r_squared,
            residuals=residuals,
            fitted_values=fitted_values,
            confidence_interval=(low_slope, high_slope)
        )
    
    # =========================
    # 多元统计分析实现
    # =========================
    
    def _multivariate_normality_test(self, data: np.ndarray) -> StatResult:
        """多元正态性检验"""
        n, p = data.shape
        
        # 计算马氏距离
        mean = np.mean(data, axis=0)
        cov = np.cov(data.T)
        
        try:
            inv_cov = linalg.inv(cov)
        except linalg.LinAlgError:
            raise ValueError("协方差矩阵奇异")
        
        mahal_dist = []
        for i in range(n):
            diff = data[i] - mean
            dist = diff.T @ inv_cov @ diff
            mahal_dist.append(dist)
        
        mahal_dist = np.array(mahal_dist)
        
        # Mardia检验
        # 偏度检验
        skewness_stat = n * np.sum((mahal_dist - p) ** 3) / (6 * p * (p + 1) * (p + 2))
        skewness_pval = 1 - stats.chi2.cdf(skewness_stat, df=p*(p+1)*(p+2)/6)
        
        # 峰度检验
        kurtosis_stat = (np.mean(mahal_dist) - p * (p + 2)) / np.sqrt(8 * p * (p + 2) / n)
        kurtosis_pval = 2 * (1 - stats.norm.cdf(abs(kurtosis_stat)))
        
        return StatResult(
            multivariate_stats={
                "mardia_skewness": {"statistic": skewness_stat, "pvalue": skewness_pval},
                "mardia_kurtosis": {"statistic": kurtosis_stat, "pvalue": kurtosis_pval},
                "mahalanobis_distances": mahal_dist
            },
            null_hypothesis="数据服从多元正态分布"
        )
    
    def _hotelling_t2_test(self, data: np.ndarray, mu0: Optional[np.ndarray] = None) -> StatResult:
        """Hotelling T²检验"""
        n, p = data.shape
        
        if mu0 is None:
            mu0 = np.zeros(p)
        
        # 计算样本统计量
        sample_mean = np.mean(data, axis=0)
        sample_cov = np.cov(data.T)
        
        # T²统计量
        diff = sample_mean - mu0
        try:
            inv_cov = linalg.inv(sample_cov)
        except linalg.LinAlgError:
            raise ValueError("样本协方差矩阵奇异")
        
        t2_stat = n * diff.T @ inv_cov @ diff
        
        # F统计量
        f_stat = (n - p) * t2_stat / (p * (n - 1))
        pval = 1 - stats.f.cdf(f_stat, p, n - p)
        
        return StatResult(
            statistic=t2_stat,
            pvalue=pval,
            degrees_of_freedom=(p, n - p),
            null_hypothesis=f"μ = {mu0}",
            multivariate_stats={
                "f_statistic": f_stat,
                "sample_mean": sample_mean,
                "sample_covariance": sample_cov
            }
        )
    
    def _manova_test(self, data: np.ndarray, groups: np.ndarray) -> StatResult:
        """多元方差分析(MANOVA)"""
        # 简化的MANOVA实现
        unique_groups = np.unique(groups)
        k = len(unique_groups)
        n, p = data.shape
        
        # 计算组内和组间平方和矩阵
        grand_mean = np.mean(data, axis=0)
        
        # 组间平方和矩阵(H)
        H = np.zeros((p, p))
        for group in unique_groups:
            group_data = data[groups == group]
            group_mean = np.mean(group_data, axis=0)
            n_group = len(group_data)
            diff = group_mean - grand_mean
            H += n_group * np.outer(diff, diff)
        
        # 组内平方和矩阵(E)
        E = np.zeros((p, p))
        for group in unique_groups:
            group_data = data[groups == group]
            group_mean = np.mean(group_data, axis=0)
            for i in range(len(group_data)):
                diff = group_data[i] - group_mean
                E += np.outer(diff, diff)
        
        # Wilks' Lambda
        try:
            wilks_lambda = linalg.det(E) / linalg.det(E + H)
        except (linalg.LinAlgError, ZeroDivisionError):
            raise ValueError("矩阵计算错误")
        
        # 近似F统计量
        df1 = p * (k - 1)
        df2 = n - k - p + 1
        f_stat = ((1 - wilks_lambda) / wilks_lambda) * (df2 / df1)
        pval = 1 - stats.f.cdf(f_stat, df1, df2)
        
        return StatResult(
            statistic=wilks_lambda,
            pvalue=pval,
            degrees_of_freedom=(df1, df2),
            null_hypothesis="各组均值向量相等",
            multivariate_stats={
                "wilks_lambda": wilks_lambda,
                "f_statistic": f_stat,
                "between_group_matrix": H,
                "within_group_matrix": E
            }
        )
    
    def _principal_component_analysis(self, data: np.ndarray) -> StatResult:
        """主成分分析"""
        n, p = data.shape
        
        # 标准化数据
        data_centered = data - np.mean(data, axis=0)
        
        # 计算协方差矩阵
        cov_matrix = np.cov(data_centered.T)
        
        # 特征值分解
        eigenvalues, eigenvectors = linalg.eigh(cov_matrix)
        
        # 按特征值降序排列
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # 方差解释比例
        explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
        
        # 主成分得分
        pc_scores = data_centered @ eigenvectors
        
        return StatResult(
            multivariate_stats={
                "eigenvalues": eigenvalues,
                "eigenvectors": eigenvectors,
                "explained_variance_ratio": explained_variance_ratio,
                "cumulative_variance_ratio": cumulative_variance_ratio,
                "principal_component_scores": pc_scores,
                "covariance_matrix": cov_matrix
            }
        )


def create_advanced_stats_engine(
    test_config: Optional[StatTestConfig] = None,
    regression_config: Optional[RegressionConfig] = None
) -> AdvancedStatisticsEngine:
    """创建高级统计分析引擎的工厂函数"""
    try:
        return AdvancedStatisticsEngine(test_config, regression_config)
    except Exception as e:
        raise DataProcessingError(f"创建高级统计分析引擎失败: {str(e)}") from e