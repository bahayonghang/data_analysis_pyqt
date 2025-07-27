#!/usr/bin/env python3
"""
任务13.1: 数据处理算法准确性测试

测试所有数据处理和分析算法的准确性：
1. 描述性统计算法测试
2. 相关性分析算法测试
3. 异常值检测算法测试
4. 时间序列分析算法测试
5. 数据预处理算法测试
"""

import sys
import unittest
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 导入测试配置
from .test_config import TestConfig, TestDataGenerator, TestAssertions

# 添加src目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


class TestDescriptiveStatistics(unittest.TestCase):
    """测试描述性统计算法"""
    
    def setUp(self):
        """测试设置"""
        TestConfig.setup()
    
    def tearDown(self):
        """测试清理"""
        TestConfig.teardown()
    
    def test_descriptive_stats_accuracy(self):
        """测试描述性统计的准确性"""
        from src.core.analysis_engine import AnalysisEngine, AnalysisConfig
        
        # 创建已知统计特性的数据
        np.random.seed(42)
        data = np.random.normal(100, 15, 1000)  # 均值100，标准差15
        df = pd.DataFrame({'value': data})
        
        # 执行分析
        config = AnalysisConfig(include_descriptive=True)
        engine = AnalysisEngine(config)
        result = engine.analyze(df)
        
        # 验证统计结果的准确性
        stats_dict = result.descriptive_stats.get('value', {})
        
        # 检查均值（允许5%误差）
        calculated_mean = stats_dict.get('mean', 0)
        expected_mean = 100
        mean_error = abs(calculated_mean - expected_mean) / expected_mean
        self.assertLess(mean_error, 0.05, f"均值误差过大: {mean_error:.3f}")
        
        # 检查标准差（允许10%误差）
        calculated_std = stats_dict.get('std', 0)
        expected_std = 15
        std_error = abs(calculated_std - expected_std) / expected_std
        self.assertLess(std_error, 0.10, f"标准差误差过大: {std_error:.3f}")
        
        print(f"✅ 描述性统计准确性测试通过")
        print(f"   计算均值: {calculated_mean:.2f} (期望: {expected_mean})")
        print(f"   计算标准差: {calculated_std:.2f} (期望: {expected_std})")
    
    def test_quartiles_calculation(self):
        """测试四分位数计算"""
        from src.core.analysis_engine import AnalysisEngine, AnalysisConfig
        
        # 创建已知分位数的数据
        data = list(range(1, 101))  # 1到100
        df = pd.DataFrame({'value': data})
        
        config = AnalysisConfig(include_descriptive=True)
        engine = AnalysisEngine(config)
        result = engine.analyze(df)
        
        stats_dict = result.descriptive_stats.get('value', {})
        
        # 验证分位数
        q25 = stats_dict.get('25%', 0)
        q50 = stats_dict.get('50%', 0)  # 中位数
        q75 = stats_dict.get('75%', 0)
        
        # 对于1-100的数据，期望的分位数
        expected_q25 = 25.75  # pandas默认插值方法
        expected_q50 = 50.5
        expected_q75 = 75.25
        
        self.assertAlmostEqual(q25, expected_q25, places=1, msg="25%分位数不准确")
        self.assertAlmostEqual(q50, expected_q50, places=1, msg="中位数不准确")
        self.assertAlmostEqual(q75, expected_q75, places=1, msg="75%分位数不准确")
        
        print("✅ 四分位数计算准确性测试通过")


class TestCorrelationAnalysis(unittest.TestCase):
    """测试相关性分析算法"""
    
    def setUp(self):
        """测试设置"""
        TestConfig.setup()
    
    def tearDown(self):
        """测试清理"""
        TestConfig.teardown()
    
    def test_perfect_correlation(self):
        """测试完全相关的情况"""
        from src.core.analysis_engine import AnalysisEngine, AnalysisConfig
        
        # 创建完全正相关的数据
        x = np.array([1, 2, 3, 4, 5])
        y = 2 * x + 1  # 完全线性相关
        df = pd.DataFrame({'x': x, 'y': y})
        
        config = AnalysisConfig(
            include_correlation=True,
            correlation_method='pearson'
        )
        engine = AnalysisEngine(config)
        result = engine.analyze(df)
        
        # 检查相关系数
        corr_matrix = result.correlation_matrix
        if isinstance(corr_matrix, dict):
            # 提取x和y之间的相关系数
            xy_correlation = None
            if 'x' in corr_matrix and 'y' in corr_matrix['x']:
                xy_correlation = corr_matrix['x']['y']
            elif 'y' in corr_matrix and 'x' in corr_matrix['y']:
                xy_correlation = corr_matrix['y']['x']
            
            if xy_correlation is not None:
                self.assertAlmostEqual(xy_correlation, 1.0, places=3, 
                                     msg=f"完全正相关检测失败: {xy_correlation}")
                print(f"✅ 完全相关检测通过: r = {xy_correlation:.6f}")
    
    def test_no_correlation(self):
        """测试无相关的情况"""
        from src.core.analysis_engine import AnalysisEngine, AnalysisConfig
        
        # 创建无相关的数据
        np.random.seed(42)
        x = np.random.normal(0, 1, 1000)
        y = np.random.normal(0, 1, 1000)  # 独立随机数
        df = pd.DataFrame({'x': x, 'y': y})
        
        config = AnalysisConfig(
            include_correlation=True,
            correlation_method='pearson'
        )
        engine = AnalysisEngine(config)
        result = engine.analyze(df)
        
        # 检查相关系数应该接近0
        corr_matrix = result.correlation_matrix
        if isinstance(corr_matrix, dict):
            xy_correlation = None
            if 'x' in corr_matrix and 'y' in corr_matrix['x']:
                xy_correlation = corr_matrix['x']['y']
            elif 'y' in corr_matrix and 'x' in corr_matrix['y']:
                xy_correlation = corr_matrix['y']['x']
            
            if xy_correlation is not None:
                self.assertLess(abs(xy_correlation), 0.1, 
                               f"无相关检测失败: |r| = {abs(xy_correlation)}")
                print(f"✅ 无相关检测通过: r = {xy_correlation:.6f}")
    
    def test_correlation_methods(self):
        """测试不同相关性计算方法"""
        from src.core.analysis_engine import AnalysisEngine, AnalysisConfig
        
        # 创建单调但非线性的关系
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        y = x ** 2  # 非线性但单调关系
        df = pd.DataFrame({'x': x, 'y': y})
        
        # 测试Pearson相关（对非线性不敏感）
        config_pearson = AnalysisConfig(
            include_correlation=True,
            correlation_method='pearson'
        )
        engine_pearson = AnalysisEngine(config_pearson)
        result_pearson = engine_pearson.analyze(df)
        
        # 测试Spearman相关（对单调关系敏感）
        config_spearman = AnalysisConfig(
            include_correlation=True,
            correlation_method='spearman'
        )
        engine_spearman = AnalysisEngine(config_spearman)
        result_spearman = engine_spearman.analyze(df)
        
        print("✅ 不同相关性方法测试完成")


class TestOutlierDetection(unittest.TestCase):
    """测试异常值检测算法"""
    
    def setUp(self):
        """测试设置"""
        TestConfig.setup()
    
    def tearDown(self):
        """测试清理"""
        TestConfig.teardown()
    
    def test_zscore_outlier_detection(self):
        """测试Z-score异常值检测"""
        from src.core.analysis_engine import AnalysisEngine, AnalysisConfig
        
        # 创建包含明显异常值的数据
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, 100)
        outliers = [10, -10, 15]  # 明显的异常值
        data = np.concatenate([normal_data, outliers])
        df = pd.DataFrame({'value': data})
        
        config = AnalysisConfig(
            include_outlier_detection=True,
            outlier_method='zscore',
            outlier_threshold=3.0
        )
        engine = AnalysisEngine(config)
        result = engine.analyze(df)
        
        # 验证异常值检测结果
        outliers_result = result.outliers
        if isinstance(outliers_result, dict):
            outlier_count = outliers_result.get('count', 0)
            self.assertGreater(outlier_count, 0, "应该检测到异常值")
            
            # 异常值数量应该合理（不超过总数的10%）
            self.assertLess(outlier_count, len(data) * 0.1, "异常值检测过于敏感")
            
            print(f"✅ Z-score异常值检测通过: 检测到 {outlier_count} 个异常值")
    
    def test_iqr_outlier_detection(self):
        """测试IQR异常值检测"""
        from src.core.analysis_engine import AnalysisEngine, AnalysisConfig
        
        # 创建包含异常值的数据
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100]  # 100是明显异常值
        df = pd.DataFrame({'value': data})
        
        config = AnalysisConfig(
            include_outlier_detection=True,
            outlier_method='iqr',
            outlier_threshold=1.5
        )
        engine = AnalysisEngine(config)
        result = engine.analyze(df)
        
        # 验证异常值检测
        outliers_result = result.outliers
        if isinstance(outliers_result, dict):
            outlier_count = outliers_result.get('count', 0)
            self.assertGreater(outlier_count, 0, "IQR方法应该检测到异常值")
            print(f"✅ IQR异常值检测通过: 检测到 {outlier_count} 个异常值")
    
    def test_outlier_detection_sensitivity(self):
        """测试异常值检测的敏感性"""
        from src.core.analysis_engine import AnalysisEngine, AnalysisConfig
        
        # 创建正态分布数据
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)
        df = pd.DataFrame({'value': data})
        
        # 测试不同阈值的敏感性
        thresholds = [2.0, 3.0, 4.0]
        outlier_counts = []
        
        for threshold in thresholds:
            config = AnalysisConfig(
                include_outlier_detection=True,
                outlier_method='zscore',
                outlier_threshold=threshold
            )
            engine = AnalysisEngine(config)
            result = engine.analyze(df)
            
            outliers_result = result.outliers
            if isinstance(outliers_result, dict):
                count = outliers_result.get('count', 0)
                outlier_counts.append(count)
        
        # 验证阈值越高，检测到的异常值越少
        for i in range(len(outlier_counts) - 1):
            self.assertGreaterEqual(outlier_counts[i], outlier_counts[i + 1],
                                   "高阈值应该检测到更少的异常值")
        
        print(f"✅ 异常值检测敏感性测试通过: 阈值 {thresholds} -> 异常值 {outlier_counts}")


class TestTimeSeriesAnalysis(unittest.TestCase):
    """测试时间序列分析算法"""
    
    def setUp(self):
        """测试设置"""
        TestConfig.setup()
    
    def tearDown(self):
        """测试清理"""
        TestConfig.teardown()
    
    def test_stationarity_test_stationary(self):
        """测试平稳性检验 - 平稳序列"""
        from src.core.analysis_engine import AnalysisEngine, AnalysisConfig
        
        # 创建平稳时间序列（白噪声）
        np.random.seed(42)
        stationary_data = np.random.normal(0, 1, 500)
        
        # 添加时间索引
        dates = pd.date_range('2023-01-01', periods=500, freq='D')
        df = pd.DataFrame({
            'datetime': dates,
            'value': stationary_data
        })
        
        config = AnalysisConfig(
            include_stationarity=True,
            time_column='datetime'
        )
        engine = AnalysisEngine(config)
        result = engine.analyze(df)
        
        # 验证平稳性检验结果
        stationarity_result = result.stationarity_test
        if isinstance(stationarity_result, dict):
            is_stationary = stationarity_result.get('is_stationary', False)
            p_value = stationarity_result.get('p_value', 1.0)
            
            self.assertTrue(is_stationary, "平稳序列应该被检测为平稳")
            self.assertLess(p_value, 0.05, f"平稳序列的p值应该小于0.05: {p_value}")
            print(f"✅ 平稳序列检测通过: p-value = {p_value:.6f}")
    
    def test_stationarity_test_nonstationary(self):
        """测试平稳性检验 - 非平稳序列"""
        from src.core.analysis_engine import AnalysisEngine, AnalysisConfig
        
        # 创建非平稳时间序列（随机游走）
        np.random.seed(42)
        random_walk = np.cumsum(np.random.normal(0, 1, 500))
        
        dates = pd.date_range('2023-01-01', periods=500, freq='D')
        df = pd.DataFrame({
            'datetime': dates,
            'value': random_walk
        })
        
        config = AnalysisConfig(
            include_stationarity=True,
            time_column='datetime'
        )
        engine = AnalysisEngine(config)
        result = engine.analyze(df)
        
        # 验证非平稳性检验结果
        stationarity_result = result.stationarity_test
        if isinstance(stationarity_result, dict):
            is_stationary = stationarity_result.get('is_stationary', True)
            p_value = stationarity_result.get('p_value', 0.0)
            
            self.assertFalse(is_stationary, "随机游走应该被检测为非平稳")
            self.assertGreater(p_value, 0.05, f"非平稳序列的p值应该大于0.05: {p_value}")
            print(f"✅ 非平稳序列检测通过: p-value = {p_value:.6f}")


class TestDataPreprocessing(unittest.TestCase):
    """测试数据预处理算法"""
    
    def setUp(self):
        """测试设置"""
        TestConfig.setup()
    
    def tearDown(self):
        """测试清理"""
        TestConfig.teardown()
    
    def test_data_cleaning(self):
        """测试数据清洗算法"""
        from src.data.data_preprocessor import DataPreprocessor
        
        # 创建包含缺失值和重复值的数据
        data = pd.DataFrame({
            'A': [1, 2, None, 4, 5, 5],  # 包含缺失值和重复
            'B': [1, 2, 3, 4, 5, 5],
            'C': ['a', 'b', 'c', 'd', 'e', 'e']
        })
        
        preprocessor = DataPreprocessor()
        
        # 测试缺失值处理
        cleaned_data = preprocessor.handle_missing_values(data, method='drop')
        self.assertFalse(cleaned_data.isnull().any().any(), "清洗后不应有缺失值")
        
        # 测试重复值处理
        deduped_data = preprocessor.remove_duplicates(data)
        self.assertEqual(len(deduped_data), 5, "应该移除重复行")
        
        print("✅ 数据清洗算法测试通过")
    
    def test_data_transformation(self):
        """测试数据变换算法"""
        from src.data.data_preprocessor import DataPreprocessor
        
        # 创建需要变换的数据
        np.random.seed(42)
        data = pd.DataFrame({
            'normal': np.random.normal(100, 15, 1000),
            'skewed': np.random.exponential(2, 1000)
        })
        
        preprocessor = DataPreprocessor()
        
        # 测试标准化
        normalized_data = preprocessor.normalize_data(data)
        
        # 验证标准化结果
        for col in normalized_data.select_dtypes(include=[np.number]).columns:
            mean = normalized_data[col].mean()
            std = normalized_data[col].std()
            self.assertAlmostEqual(mean, 0, places=1, msg=f"{col}列均值应该接近0")
            self.assertAlmostEqual(std, 1, places=1, msg=f"{col}列标准差应该接近1")
        
        print("✅ 数据变换算法测试通过")


class TestNumericalAccuracy(unittest.TestCase):
    """测试数值计算精度"""
    
    def test_floating_point_precision(self):
        """测试浮点数精度"""
        from src.core.analysis_engine import AnalysisEngine, AnalysisConfig
        
        # 创建高精度数值数据
        data = pd.DataFrame({
            'precise': [1.123456789, 2.987654321, 3.141592653]
        })
        
        config = AnalysisConfig(include_descriptive=True)
        engine = AnalysisEngine(config)
        result = engine.analyze(data)
        
        # 验证精度保持
        stats_dict = result.descriptive_stats.get('precise', {})
        calculated_mean = stats_dict.get('mean', 0)
        expected_mean = data['precise'].mean()
        
        self.assertAlmostEqual(calculated_mean, expected_mean, places=6,
                              msg="高精度数值计算误差过大")
        print("✅ 浮点数精度测试通过")
    
    def test_large_numbers(self):
        """测试大数值处理"""
        from src.core.analysis_engine import AnalysisEngine, AnalysisConfig
        
        # 创建大数值数据
        large_numbers = [1e9, 1e10, 1e11]
        data = pd.DataFrame({'large': large_numbers})
        
        config = AnalysisConfig(include_descriptive=True)
        engine = AnalysisEngine(config)
        result = engine.analyze(data)
        
        # 验证大数值计算正确
        stats_dict = result.descriptive_stats.get('large', {})
        calculated_mean = stats_dict.get('mean', 0)
        expected_mean = np.mean(large_numbers)
        
        relative_error = abs(calculated_mean - expected_mean) / expected_mean
        self.assertLess(relative_error, 1e-10, "大数值计算精度不足")
        print("✅ 大数值处理测试通过")


def run_algorithm_accuracy_tests():
    """运行算法准确性测试"""
    print("\n🧮 开始数据处理算法准确性测试...")
    
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试类
    test_classes = [
        TestDescriptiveStatistics,
        TestCorrelationAnalysis,
        TestOutlierDetection,
        TestTimeSeriesAnalysis,
        TestDataPreprocessing,
        TestNumericalAccuracy
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 统计结果
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    passed = total_tests - failures - errors
    
    print(f"\n{'='*60}")
    print(f"📊 算法准确性测试结果汇总:")
    print(f"总测试数: {total_tests}")
    print(f"通过测试: {passed}")
    print(f"失败测试: {failures}")
    print(f"错误测试: {errors}")
    print(f"通过率: {passed/total_tests*100:.1f}%")
    
    if failures > 0:
        print(f"\n❌ 失败的测试:")
        for test, error in result.failures:
            print(f"  - {test}: {error.split('AssertionError:')[-1].strip()}")
    
    if errors > 0:
        print(f"\n💥 错误的测试:")
        for test, error in result.errors:
            print(f"  - {test}: {error.split('Exception:')[-1].strip()}")
    
    if passed == total_tests:
        print("🎉 所有算法准确性测试通过！")
        return True
    else:
        print("⚠️  部分测试失败，需要检查算法实现")
        return False


if __name__ == "__main__":
    success = run_algorithm_accuracy_tests()
    sys.exit(0 if success else 1)