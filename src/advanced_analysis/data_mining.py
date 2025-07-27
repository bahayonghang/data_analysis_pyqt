"""
数据挖掘工具模块
实现特征选择、降维、异常检测等数据挖掘算法
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
    from sklearn.feature_selection import (
        SelectKBest, SelectPercentile, RFE, RFECV,
        f_classif, f_regression, chi2, mutual_info_classif, mutual_info_regression,
        VarianceThreshold, SelectFromModel
    )
    from sklearn.decomposition import (
        PCA, FastICA, TruncatedSVD, FactorAnalysis,
        LatentDirichletAllocation, NMF
    )
    from sklearn.manifold import (
        TSNE, MDS, Isomap, LocallyLinearEmbedding
    )
    from sklearn.cluster import DBSCAN, IsolationForest
    from sklearn.ensemble import IsolationForest as IsolationForestAnomaly
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.svm import OneClassSVM
    from sklearn.covariance import EllipticEnvelope
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, LinearRegression
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import scipy.stats as stats
    from scipy.spatial.distance import pdist, squareform
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from ..utils.basic_logging import LoggerMixin
from ..utils.exceptions import DataProcessingError


class FeatureSelectionMethod(str, Enum):
    """特征选择方法"""
    # 统计方法
    VARIANCE_THRESHOLD = "variance_threshold"
    UNIVARIATE_SELECTION = "univariate_selection"
    MUTUAL_INFORMATION = "mutual_information"
    CHI_SQUARE = "chi_square"
    
    # 模型选择
    RFE = "rfe"  # 递归特征消除
    RFECV = "rfecv"  # 交叉验证递归特征消除
    SELECT_FROM_MODEL = "select_from_model"
    
    # 相关性分析
    CORRELATION_FILTER = "correlation_filter"
    MULTICOLLINEARITY = "multicollinearity"


class DimensionReductionMethod(str, Enum):
    """降维方法"""
    # 线性方法
    PCA = "pca"
    FACTOR_ANALYSIS = "factor_analysis"
    TRUNCATED_SVD = "truncated_svd"
    ICA = "ica"
    
    # 非线性方法
    TSNE = "tsne"
    MDS = "mds"
    ISOMAP = "isomap"
    LLE = "lle"  # 局部线性嵌入
    
    # 矩阵分解
    NMF = "nmf"  # 非负矩阵分解
    LDA_TOPIC = "lda_topic"  # 主题模型


class AnomalyDetectionMethod(str, Enum):
    """异常检测方法"""
    # 统计方法
    Z_SCORE = "z_score"
    IQR = "iqr"
    MODIFIED_Z_SCORE = "modified_z_score"
    
    # 机器学习方法
    ISOLATION_FOREST = "isolation_forest"
    ONE_CLASS_SVM = "one_class_svm"
    LOCAL_OUTLIER_FACTOR = "local_outlier_factor"
    ELLIPTIC_ENVELOPE = "elliptic_envelope"
    
    # 聚类方法
    DBSCAN_OUTLIER = "dbscan_outlier"


@dataclass
class DataMiningConfig:
    """数据挖掘配置"""
    # 特征选择设置
    k_features: int = 10
    percentile: float = 50.0
    variance_threshold: float = 0.0
    correlation_threshold: float = 0.95
    vif_threshold: float = 5.0  # 方差膨胀因子阈值
    
    # 降维设置
    n_components: int = 2
    random_state: Optional[int] = None
    
    # PCA设置
    whiten: bool = False
    
    # t-SNE设置
    perplexity: float = 30.0
    learning_rate: float = 200.0
    n_iter: int = 1000
    
    # 异常检测设置
    contamination: float = 0.1
    n_neighbors: int = 20
    novelty: bool = False
    
    # 通用设置
    n_jobs: int = -1
    verbose: bool = False


@dataclass
class DataMiningResult:
    """数据挖掘结果"""
    method: Union[FeatureSelectionMethod, DimensionReductionMethod, AnomalyDetectionMethod]
    
    # 特征选择结果
    selected_features: Optional[List[str]] = None
    feature_scores: Optional[Dict[str, float]] = None
    feature_ranking: Optional[Dict[str, int]] = None
    
    # 降维结果
    transformed_data: Optional[np.ndarray] = None
    explained_variance_ratio: Optional[np.ndarray] = None
    components: Optional[np.ndarray] = None
    loadings: Optional[np.ndarray] = None
    
    # 异常检测结果
    anomaly_labels: Optional[np.ndarray] = None  # -1为异常，1为正常
    anomaly_scores: Optional[np.ndarray] = None
    anomaly_indices: Optional[List[int]] = None
    
    # 模型信息
    model: Optional[Any] = None
    feature_names: Optional[List[str]] = None
    
    # 评估指标
    metrics: Dict[str, float] = field(default_factory=dict)
    
    # 可视化数据
    visualization_data: Optional[Dict[str, Any]] = None
    
    # 执行信息
    execution_time: float = 0.0
    error_message: Optional[str] = None


class DataMiningEngine(LoggerMixin):
    """数据挖掘引擎"""
    
    def __init__(self, config: Optional[DataMiningConfig] = None):
        self.config = config or DataMiningConfig()
        
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn未安装，无法使用数据挖掘功能")
        if not HAS_PANDAS:
            raise ImportError("pandas未安装，无法处理数据")
        if not HAS_SCIPY:
            raise ImportError("scipy未安装，无法使用统计功能")
    
    def feature_selection(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        target: Optional[Union[pd.Series, np.ndarray]] = None,
        method: FeatureSelectionMethod = FeatureSelectionMethod.UNIVARIATE_SELECTION,
        **kwargs
    ) -> DataMiningResult:
        """特征选择"""
        try:
            import time
            start_time = time.time()
            
            # 数据预处理
            X, feature_names = self._prepare_feature_data(data)
            
            # 根据方法选择特征
            if method == FeatureSelectionMethod.VARIANCE_THRESHOLD:
                result = self._variance_threshold_selection(X, feature_names, **kwargs)
            elif method == FeatureSelectionMethod.UNIVARIATE_SELECTION:
                if target is None:
                    raise ValueError("单变量选择需要目标变量")
                result = self._univariate_selection(X, target, feature_names, **kwargs)
            elif method == FeatureSelectionMethod.MUTUAL_INFORMATION:
                if target is None:
                    raise ValueError("互信息选择需要目标变量")
                result = self._mutual_information_selection(X, target, feature_names, **kwargs)
            elif method == FeatureSelectionMethod.CHI_SQUARE:
                if target is None:
                    raise ValueError("卡方检验需要目标变量")
                result = self._chi_square_selection(X, target, feature_names, **kwargs)
            elif method == FeatureSelectionMethod.RFE:
                if target is None:
                    raise ValueError("RFE需要目标变量")
                result = self._rfe_selection(X, target, feature_names, **kwargs)
            elif method == FeatureSelectionMethod.RFECV:
                if target is None:
                    raise ValueError("RFECV需要目标变量")
                result = self._rfecv_selection(X, target, feature_names, **kwargs)
            elif method == FeatureSelectionMethod.SELECT_FROM_MODEL:
                if target is None:
                    raise ValueError("基于模型的选择需要目标变量")
                result = self._model_based_selection(X, target, feature_names, **kwargs)
            elif method == FeatureSelectionMethod.CORRELATION_FILTER:
                result = self._correlation_filter(X, feature_names, **kwargs)
            elif method == FeatureSelectionMethod.MULTICOLLINEARITY:
                result = self._multicollinearity_filter(X, feature_names, **kwargs)
            else:
                raise ValueError(f"不支持的特征选择方法: {method}")
            
            result.method = method
            result.feature_names = feature_names
            result.execution_time = time.time() - start_time
            
            self.logger.info(f"特征选择完成: {method.value}, 选择了{len(result.selected_features)}个特征")
            return result
            
        except Exception as e:
            self.logger.error(f"特征选择失败: {str(e)}")
            return DataMiningResult(
                method=method,
                error_message=str(e)
            )
    
    def dimension_reduction(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        method: DimensionReductionMethod = DimensionReductionMethod.PCA,
        **kwargs
    ) -> DataMiningResult:
        """降维"""
        try:
            import time
            start_time = time.time()
            
            # 数据预处理
            X, feature_names = self._prepare_feature_data(data)
            
            # 标准化数据（某些方法需要）
            if method in [DimensionReductionMethod.PCA, DimensionReductionMethod.FACTOR_ANALYSIS, 
                         DimensionReductionMethod.ICA]:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
            else:
                X_scaled = X
            
            # 根据方法进行降维
            if method == DimensionReductionMethod.PCA:
                result = self._pca_reduction(X_scaled, feature_names, **kwargs)
            elif method == DimensionReductionMethod.FACTOR_ANALYSIS:
                result = self._factor_analysis_reduction(X_scaled, feature_names, **kwargs)
            elif method == DimensionReductionMethod.TRUNCATED_SVD:
                result = self._svd_reduction(X, feature_names, **kwargs)
            elif method == DimensionReductionMethod.ICA:
                result = self._ica_reduction(X_scaled, feature_names, **kwargs)
            elif method == DimensionReductionMethod.TSNE:
                result = self._tsne_reduction(X_scaled, feature_names, **kwargs)
            elif method == DimensionReductionMethod.MDS:
                result = self._mds_reduction(X_scaled, feature_names, **kwargs)
            elif method == DimensionReductionMethod.ISOMAP:
                result = self._isomap_reduction(X_scaled, feature_names, **kwargs)
            elif method == DimensionReductionMethod.LLE:
                result = self._lle_reduction(X_scaled, feature_names, **kwargs)
            elif method == DimensionReductionMethod.NMF:
                # NMF需要非负数据
                X_nonneg = X - X.min() + 1e-6
                result = self._nmf_reduction(X_nonneg, feature_names, **kwargs)
            elif method == DimensionReductionMethod.LDA_TOPIC:
                # 主题模型需要整数数据
                X_int = np.round(np.abs(X)).astype(int)
                result = self._lda_reduction(X_int, feature_names, **kwargs)
            else:
                raise ValueError(f"不支持的降维方法: {method}")
            
            result.method = method
            result.feature_names = feature_names
            result.execution_time = time.time() - start_time
            
            self.logger.info(f"降维完成: {method.value}, 降至{result.transformed_data.shape[1]}维")
            return result
            
        except Exception as e:
            self.logger.error(f"降维失败: {str(e)}")
            return DataMiningResult(
                method=method,
                error_message=str(e)
            )
    
    def anomaly_detection(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        method: AnomalyDetectionMethod = AnomalyDetectionMethod.ISOLATION_FOREST,
        **kwargs
    ) -> DataMiningResult:
        """异常检测"""
        try:
            import time
            start_time = time.time()
            
            # 数据预处理
            X, feature_names = self._prepare_feature_data(data)
            
            # 根据方法进行异常检测
            if method == AnomalyDetectionMethod.Z_SCORE:
                result = self._z_score_detection(X, feature_names, **kwargs)
            elif method == AnomalyDetectionMethod.IQR:
                result = self._iqr_detection(X, feature_names, **kwargs)
            elif method == AnomalyDetectionMethod.MODIFIED_Z_SCORE:
                result = self._modified_z_score_detection(X, feature_names, **kwargs)
            elif method == AnomalyDetectionMethod.ISOLATION_FOREST:
                result = self._isolation_forest_detection(X, feature_names, **kwargs)
            elif method == AnomalyDetectionMethod.ONE_CLASS_SVM:
                result = self._one_class_svm_detection(X, feature_names, **kwargs)
            elif method == AnomalyDetectionMethod.LOCAL_OUTLIER_FACTOR:
                result = self._lof_detection(X, feature_names, **kwargs)
            elif method == AnomalyDetectionMethod.ELLIPTIC_ENVELOPE:
                result = self._elliptic_envelope_detection(X, feature_names, **kwargs)
            elif method == AnomalyDetectionMethod.DBSCAN_OUTLIER:
                result = self._dbscan_outlier_detection(X, feature_names, **kwargs)
            else:
                raise ValueError(f"不支持的异常检测方法: {method}")
            
            result.method = method
            result.feature_names = feature_names
            result.execution_time = time.time() - start_time
            
            anomaly_count = np.sum(result.anomaly_labels == -1)
            self.logger.info(f"异常检测完成: {method.value}, 检测到{anomaly_count}个异常点")
            return result
            
        except Exception as e:
            self.logger.error(f"异常检测失败: {str(e)}")
            return DataMiningResult(
                method=method,
                error_message=str(e)
            )
    
    # =========================
    # 数据预处理
    # =========================
    
    def _prepare_feature_data(self, data):
        """准备特征数据"""
        if isinstance(data, pd.DataFrame):
            feature_names = data.columns.tolist()
            X = data.values
        else:
            X = np.array(data)
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        # 检查数据
        if X.ndim != 2:
            raise ValueError("数据必须是二维的")
        
        if np.any(np.isnan(X)):
            self.logger.warning("数据包含缺失值，将进行填充")
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='mean')
            X = imputer.fit_transform(X)
        
        return X, feature_names
    
    # =========================
    # 特征选择实现
    # =========================
    
    def _variance_threshold_selection(self, X, feature_names, **kwargs):
        """方差阈值特征选择"""
        threshold = kwargs.get('threshold', self.config.variance_threshold)
        
        selector = VarianceThreshold(threshold=threshold)
        X_selected = selector.fit_transform(X)
        
        selected_mask = selector.get_support()
        selected_features = [feature_names[i] for i in range(len(feature_names)) if selected_mask[i]]
        
        # 计算特征方差
        feature_scores = {}
        for i, feature in enumerate(feature_names):
            feature_scores[feature] = np.var(X[:, i])
        
        return DataMiningResult(
            selected_features=selected_features,
            feature_scores=feature_scores,
            model=selector
        )
    
    def _univariate_selection(self, X, y, feature_names, **kwargs):
        """单变量特征选择"""
        k = kwargs.get('k', self.config.k_features)
        task_type = kwargs.get('task_type', 'auto')
        
        # 自动判断任务类型
        if task_type == 'auto':
            if len(np.unique(y)) <= 10:
                score_func = f_classif
                task_type = 'classification'
            else:
                score_func = f_regression
                task_type = 'regression'
        elif task_type == 'classification':
            score_func = f_classif
        else:
            score_func = f_regression
        
        selector = SelectKBest(score_func=score_func, k=k)
        X_selected = selector.fit_transform(X, y)
        
        selected_mask = selector.get_support()
        selected_features = [feature_names[i] for i in range(len(feature_names)) if selected_mask[i]]
        
        # 获取特征得分
        scores = selector.scores_
        feature_scores = dict(zip(feature_names, scores))
        
        return DataMiningResult(
            selected_features=selected_features,
            feature_scores=feature_scores,
            model=selector
        )
    
    def _mutual_information_selection(self, X, y, feature_names, **kwargs):
        """互信息特征选择"""
        k = kwargs.get('k', self.config.k_features)
        task_type = kwargs.get('task_type', 'auto')
        
        # 自动判断任务类型
        if task_type == 'auto':
            if len(np.unique(y)) <= 10:
                score_func = mutual_info_classif
            else:
                score_func = mutual_info_regression
        elif task_type == 'classification':
            score_func = mutual_info_classif
        else:
            score_func = mutual_info_regression
        
        selector = SelectKBest(score_func=score_func, k=k)
        X_selected = selector.fit_transform(X, y)
        
        selected_mask = selector.get_support()
        selected_features = [feature_names[i] for i in range(len(feature_names)) if selected_mask[i]]
        
        scores = selector.scores_
        feature_scores = dict(zip(feature_names, scores))
        
        return DataMiningResult(
            selected_features=selected_features,
            feature_scores=feature_scores,
            model=selector
        )
    
    def _chi_square_selection(self, X, y, feature_names, **kwargs):
        """卡方检验特征选择"""
        k = kwargs.get('k', self.config.k_features)
        
        # 确保数据为非负
        X_nonneg = X - X.min(axis=0) + 1e-6
        
        selector = SelectKBest(score_func=chi2, k=k)
        X_selected = selector.fit_transform(X_nonneg, y)
        
        selected_mask = selector.get_support()
        selected_features = [feature_names[i] for i in range(len(feature_names)) if selected_mask[i]]
        
        scores = selector.scores_
        feature_scores = dict(zip(feature_names, scores))
        
        return DataMiningResult(
            selected_features=selected_features,
            feature_scores=feature_scores,
            model=selector
        )
    
    def _rfe_selection(self, X, y, feature_names, **kwargs):
        """递归特征消除"""
        n_features = kwargs.get('n_features_to_select', self.config.k_features)
        task_type = kwargs.get('task_type', 'auto')
        
        # 选择基础估计器
        if task_type == 'auto':
            if len(np.unique(y)) <= 10:
                estimator = RandomForestClassifier(n_estimators=100, random_state=self.config.random_state)
            else:
                estimator = RandomForestRegressor(n_estimators=100, random_state=self.config.random_state)
        elif task_type == 'classification':
            estimator = RandomForestClassifier(n_estimators=100, random_state=self.config.random_state)
        else:
            estimator = RandomForestRegressor(n_estimators=100, random_state=self.config.random_state)
        
        selector = RFE(estimator=estimator, n_features_to_select=n_features)
        X_selected = selector.fit_transform(X, y)
        
        selected_mask = selector.get_support()
        selected_features = [feature_names[i] for i in range(len(feature_names)) if selected_mask[i]]
        
        ranking = dict(zip(feature_names, selector.ranking_))
        
        return DataMiningResult(
            selected_features=selected_features,
            feature_ranking=ranking,
            model=selector
        )
    
    def _rfecv_selection(self, X, y, feature_names, **kwargs):
        """交叉验证递归特征消除"""
        cv = kwargs.get('cv', 5)
        task_type = kwargs.get('task_type', 'auto')
        
        # 选择基础估计器
        if task_type == 'auto':
            if len(np.unique(y)) <= 10:
                estimator = RandomForestClassifier(n_estimators=100, random_state=self.config.random_state)
            else:
                estimator = RandomForestRegressor(n_estimators=100, random_state=self.config.random_state)
        elif task_type == 'classification':
            estimator = RandomForestClassifier(n_estimators=100, random_state=self.config.random_state)
        else:
            estimator = RandomForestRegressor(n_estimators=100, random_state=self.config.random_state)
        
        selector = RFECV(estimator=estimator, cv=cv)
        X_selected = selector.fit_transform(X, y)
        
        selected_mask = selector.get_support()
        selected_features = [feature_names[i] for i in range(len(feature_names)) if selected_mask[i]]
        
        ranking = dict(zip(feature_names, selector.ranking_))
        
        return DataMiningResult(
            selected_features=selected_features,
            feature_ranking=ranking,
            model=selector,
            metrics={'optimal_features': selector.n_features_}
        )
    
    def _model_based_selection(self, X, y, feature_names, **kwargs):
        """基于模型的特征选择"""
        task_type = kwargs.get('task_type', 'auto')
        threshold = kwargs.get('threshold', 'mean')
        
        # 选择基础模型
        if task_type == 'auto':
            if len(np.unique(y)) <= 10:
                estimator = RandomForestClassifier(n_estimators=100, random_state=self.config.random_state)
            else:
                estimator = RandomForestRegressor(n_estimators=100, random_state=self.config.random_state)
        elif task_type == 'classification':
            estimator = RandomForestClassifier(n_estimators=100, random_state=self.config.random_state)
        else:
            estimator = RandomForestRegressor(n_estimators=100, random_state=self.config.random_state)
        
        selector = SelectFromModel(estimator=estimator, threshold=threshold)
        X_selected = selector.fit_transform(X, y)
        
        selected_mask = selector.get_support()
        selected_features = [feature_names[i] for i in range(len(feature_names)) if selected_mask[i]]
        
        # 获取特征重要性
        estimator.fit(X, y)
        if hasattr(estimator, 'feature_importances_'):
            feature_scores = dict(zip(feature_names, estimator.feature_importances_))
        else:
            feature_scores = dict(zip(feature_names, np.abs(estimator.coef_)))
        
        return DataMiningResult(
            selected_features=selected_features,
            feature_scores=feature_scores,
            model=selector
        )
    
    def _correlation_filter(self, X, feature_names, **kwargs):
        """相关性过滤"""
        threshold = kwargs.get('threshold', self.config.correlation_threshold)
        
        # 计算相关系数矩阵
        corr_matrix = np.corrcoef(X.T)
        
        # 找到高相关性的特征对
        high_corr_pairs = []
        for i in range(len(feature_names)):
            for j in range(i + 1, len(feature_names)):
                if abs(corr_matrix[i, j]) > threshold:
                    high_corr_pairs.append((feature_names[i], feature_names[j], corr_matrix[i, j]))
        
        # 移除高相关性特征（保留方差较大的）
        features_to_remove = set()
        for feat1, feat2, corr in high_corr_pairs:
            idx1 = feature_names.index(feat1)
            idx2 = feature_names.index(feat2)
            
            # 保留方差较大的特征
            if np.var(X[:, idx1]) >= np.var(X[:, idx2]):
                features_to_remove.add(feat2)
            else:
                features_to_remove.add(feat1)
        
        selected_features = [f for f in feature_names if f not in features_to_remove]
        
        # 相关性得分
        feature_scores = {}
        for i, feature in enumerate(feature_names):
            # 计算该特征与其他特征的平均相关性
            other_corrs = [abs(corr_matrix[i, j]) for j in range(len(feature_names)) if i != j]
            feature_scores[feature] = np.mean(other_corrs)
        
        return DataMiningResult(
            selected_features=selected_features,
            feature_scores=feature_scores,
            metrics={'removed_pairs': len(high_corr_pairs)}
        )
    
    def _multicollinearity_filter(self, X, feature_names, **kwargs):
        """多重共线性过滤（VIF）"""
        threshold = kwargs.get('threshold', self.config.vif_threshold)
        
        selected_features = feature_names.copy()
        vif_scores = {}
        
        while True:
            # 计算当前特征的VIF
            current_vifs = {}
            for i, feature in enumerate(selected_features):
                feat_idx = feature_names.index(feature)
                X_others = np.delete(X, feat_idx, axis=1)
                X_target = X[:, feat_idx]
                
                # 简单线性回归计算R²
                try:
                    from sklearn.linear_model import LinearRegression
                    reg = LinearRegression()
                    reg.fit(X_others, X_target)
                    r_squared = reg.score(X_others, X_target)
                    
                    if r_squared >= 0.999:
                        vif = float('inf')
                    else:
                        vif = 1 / (1 - r_squared)
                    
                    current_vifs[feature] = vif
                except:
                    current_vifs[feature] = 1.0
            
            # 找到VIF最大的特征
            max_vif_feature = max(current_vifs, key=current_vifs.get)
            max_vif = current_vifs[max_vif_feature]
            
            if max_vif > threshold:
                selected_features.remove(max_vif_feature)
                vif_scores[max_vif_feature] = max_vif
                
                # 更新X矩阵
                remove_idx = feature_names.index(max_vif_feature)
                X = np.delete(X, remove_idx, axis=1)
                feature_names = [f for f in feature_names if f != max_vif_feature]
            else:
                # 所有特征的VIF都在阈值以下
                for feature, vif in current_vifs.items():
                    vif_scores[feature] = vif
                break
        
        return DataMiningResult(
            selected_features=selected_features,
            feature_scores=vif_scores,
            metrics={'vif_threshold': threshold}
        )
    
    # =========================
    # 降维实现
    # =========================
    
    def _pca_reduction(self, X, feature_names, **kwargs):
        """主成分分析降维"""
        n_components = kwargs.get('n_components', self.config.n_components)
        
        pca = PCA(
            n_components=n_components,
            whiten=self.config.whiten,
            random_state=self.config.random_state
        )
        
        X_transformed = pca.fit_transform(X)
        
        return DataMiningResult(
            transformed_data=X_transformed,
            explained_variance_ratio=pca.explained_variance_ratio_,
            components=pca.components_,
            model=pca,
            metrics={
                'total_explained_variance': np.sum(pca.explained_variance_ratio_),
                'n_components': n_components
            }
        )
    
    def _factor_analysis_reduction(self, X, feature_names, **kwargs):
        """因子分析降维"""
        n_components = kwargs.get('n_components', self.config.n_components)
        
        fa = FactorAnalysis(
            n_components=n_components,
            random_state=self.config.random_state
        )
        
        X_transformed = fa.fit_transform(X)
        
        return DataMiningResult(
            transformed_data=X_transformed,
            components=fa.components_,
            loadings=fa.components_.T,
            model=fa,
            metrics={
                'log_likelihood': fa.loglike_[-1],
                'n_components': n_components
            }
        )
    
    def _svd_reduction(self, X, feature_names, **kwargs):
        """截断SVD降维"""
        n_components = kwargs.get('n_components', self.config.n_components)
        
        svd = TruncatedSVD(
            n_components=n_components,
            random_state=self.config.random_state
        )
        
        X_transformed = svd.fit_transform(X)
        
        return DataMiningResult(
            transformed_data=X_transformed,
            explained_variance_ratio=svd.explained_variance_ratio_,
            components=svd.components_,
            model=svd,
            metrics={
                'total_explained_variance': np.sum(svd.explained_variance_ratio_),
                'n_components': n_components
            }
        )
    
    def _ica_reduction(self, X, feature_names, **kwargs):
        """独立成分分析降维"""
        n_components = kwargs.get('n_components', self.config.n_components)
        
        ica = FastICA(
            n_components=n_components,
            random_state=self.config.random_state,
            whiten='unit-variance'
        )
        
        X_transformed = ica.fit_transform(X)
        
        return DataMiningResult(
            transformed_data=X_transformed,
            components=ica.components_,
            model=ica,
            metrics={'n_components': n_components}
        )
    
    def _tsne_reduction(self, X, feature_names, **kwargs):
        """t-SNE降维"""
        n_components = kwargs.get('n_components', self.config.n_components)
        perplexity = kwargs.get('perplexity', self.config.perplexity)
        learning_rate = kwargs.get('learning_rate', self.config.learning_rate)
        n_iter = kwargs.get('n_iter', self.config.n_iter)
        
        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            learning_rate=learning_rate,
            n_iter=n_iter,
            random_state=self.config.random_state
        )
        
        X_transformed = tsne.fit_transform(X)
        
        return DataMiningResult(
            transformed_data=X_transformed,
            model=tsne,
            metrics={
                'kl_divergence': tsne.kl_divergence_,
                'n_iter': tsne.n_iter_,
                'perplexity': perplexity
            }
        )
    
    def _mds_reduction(self, X, feature_names, **kwargs):
        """多维标度降维"""
        n_components = kwargs.get('n_components', self.config.n_components)
        
        mds = MDS(
            n_components=n_components,
            random_state=self.config.random_state
        )
        
        X_transformed = mds.fit_transform(X)
        
        return DataMiningResult(
            transformed_data=X_transformed,
            model=mds,
            metrics={
                'stress': mds.stress_,
                'n_iter': mds.n_iter_
            }
        )
    
    def _isomap_reduction(self, X, feature_names, **kwargs):
        """Isomap降维"""
        n_components = kwargs.get('n_components', self.config.n_components)
        n_neighbors = kwargs.get('n_neighbors', self.config.n_neighbors)
        
        isomap = Isomap(
            n_components=n_components,
            n_neighbors=n_neighbors
        )
        
        X_transformed = isomap.fit_transform(X)
        
        return DataMiningResult(
            transformed_data=X_transformed,
            model=isomap,
            metrics={'reconstruction_error': isomap.reconstruction_error()}
        )
    
    def _lle_reduction(self, X, feature_names, **kwargs):
        """局部线性嵌入降维"""
        n_components = kwargs.get('n_components', self.config.n_components)
        n_neighbors = kwargs.get('n_neighbors', self.config.n_neighbors)
        
        lle = LocallyLinearEmbedding(
            n_components=n_components,
            n_neighbors=n_neighbors,
            random_state=self.config.random_state
        )
        
        X_transformed = lle.fit_transform(X)
        
        return DataMiningResult(
            transformed_data=X_transformed,
            model=lle,
            metrics={'reconstruction_error': lle.reconstruction_error_}
        )
    
    def _nmf_reduction(self, X, feature_names, **kwargs):
        """非负矩阵分解降维"""
        n_components = kwargs.get('n_components', self.config.n_components)
        
        nmf = NMF(
            n_components=n_components,
            random_state=self.config.random_state
        )
        
        X_transformed = nmf.fit_transform(X)
        
        return DataMiningResult(
            transformed_data=X_transformed,
            components=nmf.components_,
            model=nmf,
            metrics={'reconstruction_error': nmf.reconstruction_err_}
        )
    
    def _lda_reduction(self, X, feature_names, **kwargs):
        """LDA主题模型降维"""
        n_components = kwargs.get('n_components', self.config.n_components)
        
        lda = LatentDirichletAllocation(
            n_components=n_components,
            random_state=self.config.random_state
        )
        
        X_transformed = lda.fit_transform(X)
        
        return DataMiningResult(
            transformed_data=X_transformed,
            components=lda.components_,
            model=lda,
            metrics={
                'perplexity': lda.perplexity(X),
                'log_likelihood': lda.score(X)
            }
        )
    
    # =========================
    # 异常检测实现
    # =========================
    
    def _z_score_detection(self, X, feature_names, **kwargs):
        """Z分数异常检测"""
        threshold = kwargs.get('threshold', 3.0)
        
        z_scores = np.abs(stats.zscore(X, axis=0))
        anomaly_mask = np.any(z_scores > threshold, axis=1)
        
        anomaly_labels = np.where(anomaly_mask, -1, 1)
        anomaly_scores = np.max(z_scores, axis=1)
        anomaly_indices = np.where(anomaly_mask)[0].tolist()
        
        return DataMiningResult(
            anomaly_labels=anomaly_labels,
            anomaly_scores=anomaly_scores,
            anomaly_indices=anomaly_indices,
            metrics={
                'threshold': threshold,
                'anomaly_count': np.sum(anomaly_mask)
            }
        )
    
    def _iqr_detection(self, X, feature_names, **kwargs):
        """IQR异常检测"""
        multiplier = kwargs.get('multiplier', 1.5)
        
        Q1 = np.percentile(X, 25, axis=0)
        Q3 = np.percentile(X, 75, axis=0)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        anomaly_mask = np.any((X < lower_bound) | (X > upper_bound), axis=1)
        
        anomaly_labels = np.where(anomaly_mask, -1, 1)
        
        # 计算异常分数（距离边界的最大距离）
        anomaly_scores = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            distances = []
            for j in range(X.shape[1]):
                if X[i, j] < lower_bound[j]:
                    distances.append(lower_bound[j] - X[i, j])
                elif X[i, j] > upper_bound[j]:
                    distances.append(X[i, j] - upper_bound[j])
                else:
                    distances.append(0)
            anomaly_scores[i] = max(distances)
        
        anomaly_indices = np.where(anomaly_mask)[0].tolist()
        
        return DataMiningResult(
            anomaly_labels=anomaly_labels,
            anomaly_scores=anomaly_scores,
            anomaly_indices=anomaly_indices,
            metrics={
                'multiplier': multiplier,
                'anomaly_count': np.sum(anomaly_mask)
            }
        )
    
    def _modified_z_score_detection(self, X, feature_names, **kwargs):
        """修正Z分数异常检测"""
        threshold = kwargs.get('threshold', 3.5)
        
        median = np.median(X, axis=0)
        mad = np.median(np.abs(X - median), axis=0)
        
        # 避免除零
        mad[mad == 0] = np.finfo(float).eps
        
        modified_z_scores = 0.6745 * (X - median) / mad
        anomaly_mask = np.any(np.abs(modified_z_scores) > threshold, axis=1)
        
        anomaly_labels = np.where(anomaly_mask, -1, 1)
        anomaly_scores = np.max(np.abs(modified_z_scores), axis=1)
        anomaly_indices = np.where(anomaly_mask)[0].tolist()
        
        return DataMiningResult(
            anomaly_labels=anomaly_labels,
            anomaly_scores=anomaly_scores,
            anomaly_indices=anomaly_indices,
            metrics={
                'threshold': threshold,
                'anomaly_count': np.sum(anomaly_mask)
            }
        )
    
    def _isolation_forest_detection(self, X, feature_names, **kwargs):
        """孤立森林异常检测"""
        contamination = kwargs.get('contamination', self.config.contamination)
        
        iso_forest = IsolationForestAnomaly(
            contamination=contamination,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs
        )
        
        anomaly_labels = iso_forest.fit_predict(X)
        anomaly_scores = iso_forest.decision_function(X)
        anomaly_indices = np.where(anomaly_labels == -1)[0].tolist()
        
        return DataMiningResult(
            anomaly_labels=anomaly_labels,
            anomaly_scores=-anomaly_scores,  # 转换为正值，值越大越异常
            anomaly_indices=anomaly_indices,
            model=iso_forest,
            metrics={
                'contamination': contamination,
                'anomaly_count': np.sum(anomaly_labels == -1)
            }
        )
    
    def _one_class_svm_detection(self, X, feature_names, **kwargs):
        """单类SVM异常检测"""
        nu = kwargs.get('nu', self.config.contamination)
        kernel = kwargs.get('kernel', 'rbf')
        
        svm = OneClassSVM(
            nu=nu,
            kernel=kernel,
            gamma='scale'
        )
        
        anomaly_labels = svm.fit_predict(X)
        anomaly_scores = svm.decision_function(X)
        anomaly_indices = np.where(anomaly_labels == -1)[0].tolist()
        
        return DataMiningResult(
            anomaly_labels=anomaly_labels,
            anomaly_scores=-anomaly_scores,
            anomaly_indices=anomaly_indices,
            model=svm,
            metrics={
                'nu': nu,
                'kernel': kernel,
                'anomaly_count': np.sum(anomaly_labels == -1)
            }
        )
    
    def _lof_detection(self, X, feature_names, **kwargs):
        """局部异常因子检测"""
        n_neighbors = kwargs.get('n_neighbors', self.config.n_neighbors)
        contamination = kwargs.get('contamination', self.config.contamination)
        
        lof = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            novelty=self.config.novelty,
            n_jobs=self.config.n_jobs
        )
        
        anomaly_labels = lof.fit_predict(X)
        anomaly_scores = -lof.negative_outlier_factor_
        anomaly_indices = np.where(anomaly_labels == -1)[0].tolist()
        
        return DataMiningResult(
            anomaly_labels=anomaly_labels,
            anomaly_scores=anomaly_scores,
            anomaly_indices=anomaly_indices,
            model=lof,
            metrics={
                'n_neighbors': n_neighbors,
                'contamination': contamination,
                'anomaly_count': np.sum(anomaly_labels == -1)
            }
        )
    
    def _elliptic_envelope_detection(self, X, feature_names, **kwargs):
        """椭圆包络异常检测"""
        contamination = kwargs.get('contamination', self.config.contamination)
        
        envelope = EllipticEnvelope(
            contamination=contamination,
            random_state=self.config.random_state
        )
        
        anomaly_labels = envelope.fit_predict(X)
        anomaly_scores = envelope.decision_function(X)
        anomaly_indices = np.where(anomaly_labels == -1)[0].tolist()
        
        return DataMiningResult(
            anomaly_labels=anomaly_labels,
            anomaly_scores=-anomaly_scores,
            anomaly_indices=anomaly_indices,
            model=envelope,
            metrics={
                'contamination': contamination,
                'anomaly_count': np.sum(anomaly_labels == -1)
            }
        )
    
    def _dbscan_outlier_detection(self, X, feature_names, **kwargs):
        """DBSCAN异常检测"""
        eps = kwargs.get('eps', 0.5)
        min_samples = kwargs.get('min_samples', 5)
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=self.config.n_jobs)
        cluster_labels = dbscan.fit_predict(X)
        
        # -1标签表示噪声点（异常）
        anomaly_labels = np.where(cluster_labels == -1, -1, 1)
        anomaly_indices = np.where(cluster_labels == -1)[0].tolist()
        
        # 计算异常分数（到最近核心点的距离）
        anomaly_scores = np.zeros(X.shape[0])
        core_samples = dbscan.core_sample_indices_
        
        if len(core_samples) > 0:
            core_points = X[core_samples]
            for i in range(X.shape[0]):
                if cluster_labels[i] == -1:
                    distances = np.sqrt(np.sum((X[i] - core_points) ** 2, axis=1))
                    anomaly_scores[i] = np.min(distances)
        
        return DataMiningResult(
            anomaly_labels=anomaly_labels,
            anomaly_scores=anomaly_scores,
            anomaly_indices=anomaly_indices,
            model=dbscan,
            metrics={
                'eps': eps,
                'min_samples': min_samples,
                'n_clusters': len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0),
                'anomaly_count': np.sum(cluster_labels == -1)
            }
        )


def create_data_mining_engine(config: Optional[DataMiningConfig] = None) -> DataMiningEngine:
    """创建数据挖掘引擎的工厂函数"""
    try:
        return DataMiningEngine(config)
    except Exception as e:
        raise DataProcessingError(f"创建数据挖掘引擎失败: {str(e)}") from e