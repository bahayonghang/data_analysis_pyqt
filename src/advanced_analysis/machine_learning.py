"""
高级机器学习分析模块
集成scikit-learn，提供分类、回归、聚类算法
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
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
    from sklearn.svm import SVC, SVR
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.naive_bayes import GaussianNB, MultinomialNB
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
    from sklearn.feature_selection import SelectKBest, f_classif, f_regression, RFE
    from sklearn.decomposition import PCA, FastICA
    from sklearn.metrics import (
        classification_report, confusion_matrix, accuracy_score,
        mean_squared_error, r2_score, silhouette_score
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


from ..utils.basic_logging import LoggerMixin
from ..utils.exceptions import DataProcessingError


class MLAlgorithmType(str, Enum):
    """机器学习算法类型"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"  
    CLUSTERING = "clustering"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"
    FEATURE_SELECTION = "feature_selection"


class MLModelType(str, Enum):
    """具体模型类型"""
    # 分类算法
    RANDOM_FOREST_CLASSIFIER = "random_forest_classifier"
    LOGISTIC_REGRESSION = "logistic_regression"
    SVM_CLASSIFIER = "svm_classifier"
    NAIVE_BAYES = "naive_bayes"
    DECISION_TREE_CLASSIFIER = "decision_tree_classifier"
    KNN_CLASSIFIER = "knn_classifier"
    GRADIENT_BOOSTING = "gradient_boosting"
    
    # 回归算法
    RANDOM_FOREST_REGRESSOR = "random_forest_regressor"
    LINEAR_REGRESSION = "linear_regression"
    RIDGE_REGRESSION = "ridge_regression"
    LASSO_REGRESSION = "lasso_regression"
    SVM_REGRESSOR = "svm_regressor"
    DECISION_TREE_REGRESSOR = "decision_tree_regressor"
    KNN_REGRESSOR = "knn_regressor"
    
    # 聚类算法
    KMEANS = "kmeans"
    DBSCAN = "dbscan"
    HIERARCHICAL = "hierarchical"
    
    # 降维算法
    PCA = "pca"
    ICA = "ica"


@dataclass
class MLConfig:
    """机器学习配置"""
    test_size: float = 0.2
    random_state: int = 42
    cross_validation_folds: int = 5
    enable_feature_scaling: bool = True
    enable_grid_search: bool = False
    grid_search_cv: int = 3
    n_jobs: int = -1
    
    # 特征选择
    enable_feature_selection: bool = False
    feature_selection_k: int = 10
    
    # 模型参数
    model_params: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class MLResult:
    """机器学习结果"""
    algorithm_type: MLAlgorithmType
    model_type: MLModelType
    model: Any = None
    
    # 训练数据
    X_train: Optional[Any] = None
    X_test: Optional[Any] = None  
    y_train: Optional[Any] = None
    y_test: Optional[Any] = None
    
    # 预测结果
    y_pred: Optional[Any] = None
    y_pred_proba: Optional[Any] = None
    
    # 评估指标
    metrics: Dict[str, float] = field(default_factory=dict)
    cross_val_scores: Optional[List[float]] = None
    
    # 特征信息
    feature_names: Optional[List[str]] = None
    feature_importance: Optional[Dict[str, float]] = None
    selected_features: Optional[List[str]] = None
    
    # 聚类专用
    cluster_labels: Optional[Any] = None
    cluster_centers: Optional[Any] = None
    
    # 降维专用
    transformed_data: Optional[Any] = None
    explained_variance_ratio: Optional[List[float]] = None
    
    # 模型参数
    best_params: Optional[Dict[str, Any]] = None
    
    # 执行信息
    training_time: float = 0.0
    prediction_time: float = 0.0
    error_message: Optional[str] = None


class MachineLearningEngine(LoggerMixin):
    """机器学习分析引擎"""
    
    def __init__(self, config: Optional[MLConfig] = None):
        self.config = config or MLConfig()
        self.scalers = {}
        self.label_encoders = {}
        
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn未安装，无法使用机器学习功能")
        if not HAS_PANDAS:
            raise ImportError("pandas未安装，无法处理数据")
    
    def run_classification(
        self, 
        data: pd.DataFrame, 
        target_column: str,
        features: Optional[List[str]] = None,
        model_type: MLModelType = MLModelType.RANDOM_FOREST_CLASSIFIER
    ) -> MLResult:
        """运行分类分析"""
        try:
            import time
            start_time = time.time()
            
            # 准备数据
            X, y, feature_names = self._prepare_classification_data(data, target_column, features)
            
            # 数据分割
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.config.test_size, 
                random_state=self.config.random_state, stratify=y
            )
            
            # 特征缩放
            if self.config.enable_feature_scaling:
                X_train, X_test = self._scale_features(X_train, X_test, "classification")
            
            # 特征选择
            if self.config.enable_feature_selection:
                X_train, X_test, selected_features = self._select_features(
                    X_train, X_test, y_train, feature_names, "classification"
                )
            else:
                selected_features = feature_names
            
            # 创建模型
            model = self._create_classification_model(model_type)
            
            # 网格搜索优化
            if self.config.enable_grid_search:
                model = self._grid_search_optimize(model, X_train, y_train, model_type)
            
            # 训练模型
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # 预测
            pred_start = time.time()
            y_pred = model.predict(X_test)
            y_pred_proba = None
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)
            prediction_time = time.time() - pred_start
            
            # 交叉验证
            cv_scores = cross_val_score(
                model, X_train, y_train, 
                cv=self.config.cross_validation_folds,
                n_jobs=self.config.n_jobs
            )
            
            # 计算指标
            metrics = self._calculate_classification_metrics(y_test, y_pred, y_pred_proba)
            
            # 特征重要性
            feature_importance = self._get_feature_importance(model, selected_features)
            
            result = MLResult(
                algorithm_type=MLAlgorithmType.CLASSIFICATION,
                model_type=model_type,
                model=model,
                X_train=X_train, X_test=X_test,
                y_train=y_train, y_test=y_test,
                y_pred=y_pred, y_pred_proba=y_pred_proba,
                metrics=metrics,
                cross_val_scores=cv_scores.tolist(),
                feature_names=feature_names,
                feature_importance=feature_importance,
                selected_features=selected_features,
                best_params=getattr(model, 'best_params_', None),
                training_time=training_time,
                prediction_time=prediction_time
            )
            
            self.logger.info(f"分类分析完成: {model_type.value}, 准确率: {metrics.get('accuracy', 0):.3f}")
            return result
            
        except Exception as e:
            self.logger.error(f"分类分析失败: {str(e)}")
            return MLResult(
                algorithm_type=MLAlgorithmType.CLASSIFICATION,
                model_type=model_type,
                error_message=str(e)
            )
    
    def run_regression(
        self,
        data: pd.DataFrame,
        target_column: str,
        features: Optional[List[str]] = None,
        model_type: MLModelType = MLModelType.RANDOM_FOREST_REGRESSOR
    ) -> MLResult:
        """运行回归分析"""
        try:
            import time
            start_time = time.time()
            
            # 准备数据
            X, y, feature_names = self._prepare_regression_data(data, target_column, features)
            
            # 数据分割
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.config.test_size,
                random_state=self.config.random_state
            )
            
            # 特征缩放
            if self.config.enable_feature_scaling:
                X_train, X_test = self._scale_features(X_train, X_test, "regression")
            
            # 特征选择
            if self.config.enable_feature_selection:
                X_train, X_test, selected_features = self._select_features(
                    X_train, X_test, y_train, feature_names, "regression"
                )
            else:
                selected_features = feature_names
            
            # 创建模型
            model = self._create_regression_model(model_type)
            
            # 网格搜索优化
            if self.config.enable_grid_search:
                model = self._grid_search_optimize(model, X_train, y_train, model_type)
            
            # 训练模型
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # 预测
            pred_start = time.time()
            y_pred = model.predict(X_test)
            prediction_time = time.time() - pred_start
            
            # 交叉验证
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=self.config.cross_validation_folds,
                scoring='r2',
                n_jobs=self.config.n_jobs
            )
            
            # 计算指标
            metrics = self._calculate_regression_metrics(y_test, y_pred)
            
            # 特征重要性
            feature_importance = self._get_feature_importance(model, selected_features)
            
            result = MLResult(
                algorithm_type=MLAlgorithmType.REGRESSION,
                model_type=model_type,
                model=model,
                X_train=X_train, X_test=X_test,
                y_train=y_train, y_test=y_test,
                y_pred=y_pred,
                metrics=metrics,
                cross_val_scores=cv_scores.tolist(),
                feature_names=feature_names,
                feature_importance=feature_importance,
                selected_features=selected_features,
                best_params=getattr(model, 'best_params_', None),
                training_time=training_time,
                prediction_time=prediction_time
            )
            
            self.logger.info(f"回归分析完成: {model_type.value}, R²: {metrics.get('r2_score', 0):.3f}")
            return result
            
        except Exception as e:
            self.logger.error(f"回归分析失败: {str(e)}")
            return MLResult(
                algorithm_type=MLAlgorithmType.REGRESSION,
                model_type=model_type,
                error_message=str(e)
            )
    
    def run_clustering(
        self,
        data: pd.DataFrame,
        features: Optional[List[str]] = None,
        model_type: MLModelType = MLModelType.KMEANS,
        n_clusters: int = 3
    ) -> MLResult:
        """运行聚类分析"""
        try:
            import time
            start_time = time.time()
            
            # 准备数据
            X, feature_names = self._prepare_clustering_data(data, features)
            
            # 特征缩放
            if self.config.enable_feature_scaling:
                X_scaled = self._scale_features_for_clustering(X)
            else:
                X_scaled = X
            
            # 创建模型
            model = self._create_clustering_model(model_type, n_clusters)
            
            # 训练模型
            if model_type == MLModelType.KMEANS:
                cluster_labels = model.fit_predict(X_scaled)
                cluster_centers = model.cluster_centers_
            else:
                cluster_labels = model.fit_predict(X_scaled)
                cluster_centers = None
            
            training_time = time.time() - start_time
            
            # 计算指标
            metrics = {}
            if len(set(cluster_labels)) > 1:  # 需要至少2个聚类
                metrics['silhouette_score'] = silhouette_score(X_scaled, cluster_labels)
            
            if hasattr(model, 'inertia_'):
                metrics['inertia'] = model.inertia_
            
            result = MLResult(
                algorithm_type=MLAlgorithmType.CLUSTERING,
                model_type=model_type,
                model=model,
                X_train=X_scaled,
                metrics=metrics,
                feature_names=feature_names,
                cluster_labels=cluster_labels,
                cluster_centers=cluster_centers,
                training_time=training_time
            )
            
            self.logger.info(f"聚类分析完成: {model_type.value}, 聚类数: {len(set(cluster_labels))}")
            return result
            
        except Exception as e:
            self.logger.error(f"聚类分析失败: {str(e)}")
            return MLResult(
                algorithm_type=MLAlgorithmType.CLUSTERING,
                model_type=model_type,
                error_message=str(e)
            )
    
    def run_dimensionality_reduction(
        self,
        data: pd.DataFrame,
        features: Optional[List[str]] = None,
        model_type: MLModelType = MLModelType.PCA,
        n_components: int = 2
    ) -> MLResult:
        """运行降维分析"""
        try:
            import time
            start_time = time.time()
            
            # 准备数据
            X, feature_names = self._prepare_clustering_data(data, features)
            
            # 特征缩放
            if self.config.enable_feature_scaling:
                X_scaled = self._scale_features_for_clustering(X)
            else:
                X_scaled = X
            
            # 创建模型
            if model_type == MLModelType.PCA:
                model = PCA(n_components=n_components, random_state=self.config.random_state)
            elif model_type == MLModelType.ICA:
                model = FastICA(n_components=n_components, random_state=self.config.random_state)
            else:
                raise ValueError(f"不支持的降维算法: {model_type}")
            
            # 训练和转换
            transformed_data = model.fit_transform(X_scaled)
            training_time = time.time() - start_time
            
            # 获取解释方差比例
            explained_variance_ratio = None
            if hasattr(model, 'explained_variance_ratio_'):
                explained_variance_ratio = model.explained_variance_ratio_.tolist()
            
            result = MLResult(
                algorithm_type=MLAlgorithmType.DIMENSIONALITY_REDUCTION,
                model_type=model_type,
                model=model,
                X_train=X_scaled,
                transformed_data=transformed_data,
                feature_names=feature_names,
                explained_variance_ratio=explained_variance_ratio,
                training_time=training_time
            )
            
            self.logger.info(f"降维分析完成: {model_type.value}, 降至 {n_components} 维")
            return result
            
        except Exception as e:
            self.logger.error(f"降维分析失败: {str(e)}")
            return MLResult(
                algorithm_type=MLAlgorithmType.DIMENSIONALITY_REDUCTION,
                model_type=model_type,
                error_message=str(e)
            )
    
    def _prepare_classification_data(
        self, data: pd.DataFrame, target_column: str, features: Optional[List[str]]
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """准备分类数据"""
        if target_column not in data.columns:
            raise ValueError(f"目标列 '{target_column}' 不存在")
        
        if features is None:
            features = [col for col in data.columns if col != target_column and data[col].dtype in ['int64', 'float64']]
        
        if not features:
            raise ValueError("没有可用的特征列")
        
        # 处理缺失值
        data_clean = data[features + [target_column]].dropna()
        
        X = data_clean[features].values
        y = data_clean[target_column].values
        
        # 编码目标变量
        if data_clean[target_column].dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
            self.label_encoders[target_column] = le
        
        return X, y, features
    
    def _prepare_regression_data(
        self, data: pd.DataFrame, target_column: str, features: Optional[List[str]]
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """准备回归数据"""
        if target_column not in data.columns:
            raise ValueError(f"目标列 '{target_column}' 不存在")
        
        if data[target_column].dtype not in ['int64', 'float64']:
            raise ValueError(f"目标列 '{target_column}' 必须是数值类型")
        
        if features is None:
            features = [col for col in data.columns if col != target_column and data[col].dtype in ['int64', 'float64']]
        
        if not features:
            raise ValueError("没有可用的特征列")
        
        # 处理缺失值
        data_clean = data[features + [target_column]].dropna()
        
        X = data_clean[features].values
        y = data_clean[target_column].values
        
        return X, y, features
    
    def _prepare_clustering_data(
        self, data: pd.DataFrame, features: Optional[List[str]]
    ) -> Tuple[np.ndarray, List[str]]:
        """准备聚类数据"""
        if features is None:
            features = [col for col in data.columns if data[col].dtype in ['int64', 'float64']]
        
        if not features:
            raise ValueError("没有可用的特征列")
        
        # 处理缺失值
        data_clean = data[features].dropna()
        X = data_clean.values
        
        return X, features
    
    def _scale_features(self, X_train: np.ndarray, X_test: np.ndarray, task_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """特征缩放"""
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers[task_type] = scaler
        return X_train_scaled, X_test_scaled
    
    def _scale_features_for_clustering(self, X: np.ndarray) -> np.ndarray:
        """聚类特征缩放"""
        scaler = StandardScaler()
        return scaler.fit_transform(X)
    
    def _select_features(
        self, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray,
        feature_names: List[str], task_type: str
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """特征选择"""
        if task_type == "classification":
            selector = SelectKBest(score_func=f_classif, k=min(self.config.feature_selection_k, X_train.shape[1]))
        else:
            selector = SelectKBest(score_func=f_regression, k=min(self.config.feature_selection_k, X_train.shape[1]))
        
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
        
        selected_indices = selector.get_support(indices=True)
        selected_features = [feature_names[i] for i in selected_indices]
        
        return X_train_selected, X_test_selected, selected_features
    
    def _create_classification_model(self, model_type: MLModelType):
        """创建分类模型"""
        params = self.config.model_params.get(model_type.value, {})
        
        if model_type == MLModelType.RANDOM_FOREST_CLASSIFIER:
            return RandomForestClassifier(random_state=self.config.random_state, n_jobs=self.config.n_jobs, **params)
        elif model_type == MLModelType.LOGISTIC_REGRESSION:
            return LogisticRegression(random_state=self.config.random_state, max_iter=1000, **params)
        elif model_type == MLModelType.SVM_CLASSIFIER:
            return SVC(random_state=self.config.random_state, probability=True, **params)
        elif model_type == MLModelType.NAIVE_BAYES:
            return GaussianNB(**params)
        elif model_type == MLModelType.DECISION_TREE_CLASSIFIER:
            return DecisionTreeClassifier(random_state=self.config.random_state, **params)
        elif model_type == MLModelType.KNN_CLASSIFIER:
            return KNeighborsClassifier(n_jobs=self.config.n_jobs, **params)
        elif model_type == MLModelType.GRADIENT_BOOSTING:
            return GradientBoostingClassifier(random_state=self.config.random_state, **params)
        else:
            raise ValueError(f"不支持的分类模型: {model_type}")
    
    def _create_regression_model(self, model_type: MLModelType):
        """创建回归模型"""
        params = self.config.model_params.get(model_type.value, {})
        
        if model_type == MLModelType.RANDOM_FOREST_REGRESSOR:
            return RandomForestRegressor(random_state=self.config.random_state, n_jobs=self.config.n_jobs, **params)
        elif model_type == MLModelType.LINEAR_REGRESSION:
            return LinearRegression(n_jobs=self.config.n_jobs, **params)
        elif model_type == MLModelType.RIDGE_REGRESSION:
            return Ridge(random_state=self.config.random_state, **params)
        elif model_type == MLModelType.LASSO_REGRESSION:
            return Lasso(random_state=self.config.random_state, **params)
        elif model_type == MLModelType.SVM_REGRESSOR:
            return SVR(**params)
        elif model_type == MLModelType.DECISION_TREE_REGRESSOR:
            return DecisionTreeRegressor(random_state=self.config.random_state, **params)
        elif model_type == MLModelType.KNN_REGRESSOR:
            return KNeighborsRegressor(n_jobs=self.config.n_jobs, **params)
        else:
            raise ValueError(f"不支持的回归模型: {model_type}")
    
    def _create_clustering_model(self, model_type: MLModelType, n_clusters: int):
        """创建聚类模型"""
        params = self.config.model_params.get(model_type.value, {})
        
        if model_type == MLModelType.KMEANS:
            return KMeans(n_clusters=n_clusters, random_state=self.config.random_state, n_init=10, **params)
        elif model_type == MLModelType.DBSCAN:
            return DBSCAN(**params)
        elif model_type == MLModelType.HIERARCHICAL:
            return AgglomerativeClustering(n_clusters=n_clusters, **params)
        else:
            raise ValueError(f"不支持的聚类模型: {model_type}")
    
    def _grid_search_optimize(self, model, X_train: np.ndarray, y_train: np.ndarray, model_type: MLModelType):
        """网格搜索优化"""
        param_grids = {
            MLModelType.RANDOM_FOREST_CLASSIFIER: {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            },
            MLModelType.SVM_CLASSIFIER: {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            },
            MLModelType.LOGISTIC_REGRESSION: {
                'C': [0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        }
        
        if model_type in param_grids:
            grid_search = GridSearchCV(
                model, param_grids[model_type],
                cv=self.config.grid_search_cv,
                n_jobs=self.config.n_jobs,
                scoring='accuracy' if 'classifier' in model_type.value else 'r2'
            )
            grid_search.fit(X_train, y_train)
            return grid_search
        
        return model
    
    def _calculate_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: Optional[np.ndarray]) -> Dict[str, float]:
        """计算分类指标"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred)
            }
            
            # 计算详细报告
            try:
                report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
                metrics.update({
                    'precision_macro': report['macro avg']['precision'],
                    'recall_macro': report['macro avg']['recall'],
                    'f1_macro': report['macro avg']['f1-score']
                })
            except:
                pass
            
            return metrics
    
    def _calculate_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """计算回归指标"""
        return {
            'r2_score': r2_score(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': np.mean(np.abs(y_true - y_pred))
        }
    
    def _get_feature_importance(self, model, feature_names: List[str]) -> Optional[Dict[str, float]]:
        """获取特征重要性"""
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_).flatten()
            else:
                return None
            
            # 处理网格搜索模型
            if hasattr(model, 'best_estimator_'):
                if hasattr(model.best_estimator_, 'feature_importances_'):
                    importances = model.best_estimator_.feature_importances_
                elif hasattr(model.best_estimator_, 'coef_'):
                    importances = np.abs(model.best_estimator_.coef_).flatten()
            
            return dict(zip(feature_names, importances))
        except:
            return None


def create_ml_engine(config: Optional[MLConfig] = None) -> MachineLearningEngine:
    """创建机器学习引擎的工厂函数"""
    try:
        return MachineLearningEngine(config)
    except Exception as e:
        raise DataProcessingError(f"创建机器学习引擎失败: {str(e)}") from e