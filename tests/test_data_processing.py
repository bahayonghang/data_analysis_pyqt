"""
数据处理模块测试
测试数据加载、时间检测、预处理和验证功能
"""

import tempfile
import unittest
from pathlib import Path
from datetime import datetime
import io

import pytest

# 尝试导入pandas作为测试数据源
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from src.data import (
    DataLoader, LoaderConfig,
    TimeColumnDetector, TimeDetectionResult,
    DataPreprocessor, PreprocessingConfig, CleaningMethod,
    DataValidator, ValidationLevel, ValidationRule
)
from src.models import FileInfo, FileType


class TestDataLoader(unittest.TestCase):
    """数据加载器测试"""
    
    def setUp(self):
        """设置测试数据"""
        # 创建测试CSV文件
        self.csv_content = """name,age,salary,join_date,tagtime
John,25,50000,2023-01-01,2023-01-01 09:00:00
Jane,30,60000,2023-01-02,2023-01-02 10:00:00
Bob,35,70000,2023-01-03,2023-01-03 11:00:00
Alice,,80000,2023-01-04,2023-01-04 12:00:00
Charlie,40,90000,,2023-01-05 13:00:00"""
        
        self.temp_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.temp_csv.write(self.csv_content)
        self.temp_csv.close()
        
        self.csv_file_info = FileInfo.create_from_file(self.temp_csv.name)
        
        # 创建加载器配置
        self.config = LoaderConfig(
            chunk_size=1000,
            max_memory_mb=100,
            use_polars=False  # 使用pandas进行测试
        )
        
        self.loader = DataLoader(self.config)
    
    def tearDown(self):
        """清理测试数据"""
        Path(self.temp_csv.name).unlink(missing_ok=True)
    
    def test_load_csv_file(self):
        """测试CSV文件加载"""
        df, quality_info = self.loader.load_file(self.csv_file_info)
        
        self.assertIsNotNone(df)
        self.assertEqual(quality_info.total_rows, 5)
        self.assertEqual(quality_info.total_columns, 5)
        self.assertTrue(quality_info.missing_values_count > 0)  # 有缺失值
    
    def test_get_column_info(self):
        """测试获取列信息"""
        df, _ = self.loader.load_file(self.csv_file_info)
        column_names, column_types = self.loader.get_column_info(df)
        
        expected_columns = ['name', 'age', 'salary', 'join_date', 'tagtime']
        self.assertEqual(column_names, expected_columns)
        self.assertEqual(len(column_types), 5)
    
    def test_get_sample_data(self):
        """测试获取样本数据"""
        df, _ = self.loader.load_file(self.csv_file_info)
        sample_data = self.loader.get_sample_data(df, n_rows=3)
        
        self.assertIsInstance(sample_data, list)
        self.assertEqual(len(sample_data), 3)
        self.assertEqual(len(sample_data[0]), 5)  # 5列
    
    @unittest.skipIf(not HAS_PANDAS, "Pandas not available")
    def test_load_with_pandas_fallback(self):
        """测试Pandas后备加载"""
        # 强制使用pandas
        config = LoaderConfig(use_polars=False)
        loader = DataLoader(config)
        
        df, quality_info = loader.load_file(self.csv_file_info)
        
        self.assertIsNotNone(df)
        self.assertEqual(quality_info.total_rows, 5)


class TestTimeColumnDetector(unittest.TestCase):
    """时间列检测器测试"""
    
    def setUp(self):
        """设置测试数据"""
        if HAS_PANDAS:
            self.test_df = pd.DataFrame({
                'id': [1, 2, 3, 4, 5],
                'name': ['John', 'Jane', 'Bob', 'Alice', 'Charlie'],
                'created_at': ['2023-01-01 10:00:00', '2023-01-02 11:00:00', 
                              '2023-01-03 12:00:00', '2023-01-04 13:00:00', 
                              '2023-01-05 14:00:00'],
                'birth_date': ['1990-01-01', '1985-05-15', '1980-12-30', 
                              '1995-07-20', '1988-03-10'],
                'tagtime': ['2023-01-01T10:00:00Z', '2023-01-02T11:00:00Z',
                           '2023-01-03T12:00:00Z', '2023-01-04T13:00:00Z',
                           '2023-01-05T14:00:00Z'],
                'timestamp': [1672574400, 1672660800, 1672747200, 1672833600, 1672920000],
                'random_text': ['abc', 'def', 'ghi', 'jkl', 'mno']
            })
        
        self.detector = TimeColumnDetector()
    
    @unittest.skipIf(not HAS_PANDAS, "Pandas not available")
    def test_detect_time_columns(self):
        """测试时间列检测"""
        results = self.detector.detect_time_columns(self.test_df)
        
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 7)  # 7列
        
        # 检查已知的时间列
        time_columns = [r for r in results if r.is_time_column]
        self.assertTrue(len(time_columns) >= 3)  # 至少检测到3个时间列
        
        # 检查tagtime列
        tagtime_result = next((r for r in results if r.column_name == 'tagtime'), None)
        self.assertIsNotNone(tagtime_result)
        self.assertTrue(tagtime_result.is_time_column)
        self.assertEqual(tagtime_result.time_type.value, 'tagtime')
    
    @unittest.skipIf(not HAS_PANDAS, "Pandas not available")
    def test_convert_to_time_column_info(self):
        """测试转换为TimeColumnInfo"""
        results = self.detector.detect_time_columns(self.test_df)
        
        for result in results:
            if result.is_time_column:
                time_col_info = self.detector.convert_to_time_column_info(result)
                
                self.assertEqual(time_col_info.column_name, result.column_name)
                self.assertEqual(time_col_info.column_type, result.time_type)
                self.assertEqual(time_col_info.confidence_score, result.confidence_score)
    
    @unittest.skipIf(not HAS_PANDAS, "Pandas not available")
    def test_validate_time_parsing(self):
        """测试时间解析验证"""
        sample_values = ['2023-01-01 10:00:00', '2023-01-02 11:00:00', '2023-01-03 12:00:00']
        format_pattern = '%Y-%m-%d %H:%M:%S'
        
        is_valid, success_count = self.detector.validate_time_parsing(sample_values, format_pattern)
        
        self.assertTrue(is_valid)
        self.assertEqual(success_count, 3)
    
    @unittest.skipIf(not HAS_PANDAS, "Pandas not available")
    def test_suggest_format_improvements(self):
        """测试格式改进建议"""
        results = self.detector.detect_time_columns(self.test_df)
        
        for result in results:
            suggestions = self.detector.suggest_format_improvements(result)
            self.assertIsInstance(suggestions, list)
            self.assertTrue(len(suggestions) > 0)


class TestDataPreprocessor(unittest.TestCase):
    """数据预处理器测试"""
    
    def setUp(self):
        """设置测试数据"""
        if HAS_PANDAS:
            self.test_df = pd.DataFrame({
                'id': [1, 2, 3, 4, 5, 5],  # 包含重复值
                'name': ['John', 'Jane', None, 'Alice', 'Charlie', 'Charlie'],  # 包含空值
                'age': [25, 30, 35, None, 40, 40],  # 包含空值
                'salary': [50000, 60000, 70000, 80000, 1000000, 1000000],  # 包含异常值
                'score': ['85.5', '90.0', '78.5', '95.0', '88.0', '88.0']  # 字符串数值
            })
        
        self.config = PreprocessingConfig(
            handle_missing=True,
            missing_method=CleaningMethod.DROP_MISSING,
            handle_duplicates=True,
            handle_outliers=True,
            convert_strings_to_numeric=True
        )
        
        self.preprocessor = DataPreprocessor(self.config)
    
    @unittest.skipIf(not HAS_PANDAS, "Pandas not available")
    def test_preprocess_data(self):
        """测试完整预处理流程"""
        df_processed, log = self.preprocessor.preprocess(self.test_df)
        
        self.assertIsNotNone(df_processed)
        self.assertIsInstance(log, dict)
        self.assertIn('steps_applied', log)
        self.assertIn('original_shape', log)
        self.assertIn('final_shape', log)
        
        # 检查数据清洗效果
        original_rows, _ = log['original_shape']
        final_rows, _ = log['final_shape']
        self.assertLessEqual(final_rows, original_rows)  # 清洗后行数应该减少或相等
    
    @unittest.skipIf(not HAS_PANDAS, "Pandas not available")
    def test_handle_missing_values(self):
        """测试缺失值处理"""
        df_processed, log = self.preprocessor.preprocess(self.test_df)
        
        # 检查是否有缺失值统计
        self.assertIn('statistics', log)
        if 'missing_values' in log['statistics']:
            missing_stats = log['statistics']['missing_values']
            self.assertIn('rows_before', missing_stats)
            self.assertIn('rows_after', missing_stats)
    
    @unittest.skipIf(not HAS_PANDAS, "Pandas not available")
    def test_different_missing_methods(self):
        """测试不同的缺失值处理方法"""
        methods = [CleaningMethod.FILL_MEAN, CleaningMethod.FILL_MEDIAN, CleaningMethod.FILL_CONSTANT]
        
        for method in methods:
            config = PreprocessingConfig(
                handle_missing=True,
                missing_method=method,
                fill_value=999 if method == CleaningMethod.FILL_CONSTANT else None
            )
            preprocessor = DataPreprocessor(config)
            
            try:
                df_processed, log = preprocessor.preprocess(self.test_df.copy())
                self.assertIsNotNone(df_processed)
            except Exception as e:
                self.fail(f"预处理失败，方法: {method.value}, 错误: {str(e)}")


class TestDataValidator(unittest.TestCase):
    """数据验证器测试"""
    
    def setUp(self):
        """设置测试数据"""
        if HAS_PANDAS:
            # 创建有问题的测试数据
            self.problematic_df = pd.DataFrame({
                'id': [1, 2, 3, 3, 5],  # 重复值
                'name': ['John', 'Jane', None, 'Alice', ''],  # 空值
                'age': [25, 30, 35, -5, 200],  # 异常值
                'email': ['john@test.com', 'invalid-email', 'jane@test.com', '', None],  # 格式问题
                'score': [85, 90, 78, 95, 88]
            })
            
            # 创建良好的测试数据
            self.good_df = pd.DataFrame({
                'id': [1, 2, 3, 4, 5],
                'name': ['John', 'Jane', 'Bob', 'Alice', 'Charlie'],
                'age': [25, 30, 35, 28, 40],
                'score': [85, 90, 78, 95, 88]
            })
        
        self.validator = DataValidator()
    
    @unittest.skipIf(not HAS_PANDAS, "Pandas not available")
    def test_validate_problematic_data(self):
        """测试验证有问题的数据"""
        result = self.validator.validate(self.problematic_df)
        
        self.assertIsInstance(result, type(result))
        self.assertFalse(result.is_valid or len(result.issues) > 0)
        self.assertTrue(result.score < 100)
        
        # 检查是否检测到重复值
        duplicate_issues = [issue for issue in result.issues 
                          if issue.rule == ValidationRule.DUPLICATE_CHECK]
        self.assertTrue(len(duplicate_issues) > 0)
        
        # 检查是否检测到空值
        null_issues = [issue for issue in result.issues 
                      if issue.rule == ValidationRule.NO_NULL_VALUES]
        self.assertTrue(len(null_issues) > 0)
    
    @unittest.skipIf(not HAS_PANDAS, "Pandas not available")
    def test_validate_good_data(self):
        """测试验证良好的数据"""
        result = self.validator.validate(self.good_df)
        
        self.assertIsInstance(result, type(result))
        self.assertTrue(result.score > 80)  # 应该有较高的质量分数
    
    @unittest.skipIf(not HAS_PANDAS, "Pandas not available")
    def test_validation_result_methods(self):
        """测试验证结果的方法"""
        result = self.validator.validate(self.problematic_df)
        
        # 测试按级别获取问题
        warning_issues = result.get_issues_by_level(ValidationLevel.WARNING)
        self.assertIsInstance(warning_issues, list)
        
        # 测试获取问题摘要
        summary = result.get_summary()
        self.assertIsInstance(summary, dict)
        self.assertIn('warning', summary)
        self.assertIn('error', summary)
    
    @unittest.skipIf(not HAS_PANDAS, "Pandas not available")
    def test_column_rules_validation(self):
        """测试列规则验证"""
        column_rules = {
            'age': [
                {'type': 'numeric_range', 'min': 0, 'max': 120}
            ],
            'name': [
                {'type': 'string_length', 'min_length': 1, 'max_length': 50}
            ]
        }
        
        result = self.validator.validate_column_rules(self.problematic_df, column_rules)
        
        self.assertIsInstance(result, type(result))
        # 应该检测到age列的范围问题
        age_issues = result.get_issues_by_column('age')
        self.assertTrue(len(age_issues) > 0)
    
    def test_empty_dataframe_validation(self):
        """测试空数据框验证"""
        if HAS_PANDAS:
            empty_df = pd.DataFrame()
            result = self.validator.validate(empty_df)
            
            self.assertFalse(result.is_valid)
            self.assertTrue(result.has_critical_issues())
            self.assertEqual(result.score, 0.0)


class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    def setUp(self):
        """设置集成测试"""
        # 创建完整的测试CSV文件
        self.csv_content = """id,name,age,salary,join_date,tagtime,email
1,John,25,50000,2023-01-01,2023-01-01T09:00:00Z,john@test.com
2,Jane,30,60000,2023-01-02,2023-01-02T10:00:00Z,jane@test.com
3,Bob,35,70000,2023-01-03,2023-01-03T11:00:00Z,bob@test.com
4,Alice,,80000,2023-01-04,2023-01-04T12:00:00Z,alice@test.com
5,Charlie,40,90000,,2023-01-05T13:00:00Z,charlie@test.com
6,David,45,95000,2023-01-06,2023-01-06T14:00:00Z,invalid-email"""
        
        self.temp_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.temp_csv.write(self.csv_content)
        self.temp_csv.close()
    
    def tearDown(self):
        """清理资源"""
        Path(self.temp_csv.name).unlink(missing_ok=True)
    
    def test_complete_data_processing_pipeline(self):
        """测试完整的数据处理管道"""
        # 1. 创建文件信息
        file_info = FileInfo.create_from_file(self.temp_csv.name)
        self.assertEqual(file_info.file_type, FileType.CSV)
        
        # 2. 加载数据
        loader = DataLoader(LoaderConfig(use_polars=False))
        df, quality_info = loader.load_file(file_info)
        
        self.assertIsNotNone(df)
        self.assertEqual(quality_info.total_rows, 6)
        self.assertEqual(quality_info.total_columns, 7)
        
        # 3. 检测时间列
        time_detector = TimeColumnDetector()
        time_results = time_detector.detect_time_columns(df)
        
        time_columns = [r for r in time_results if r.is_time_column]
        self.assertTrue(len(time_columns) >= 2)  # 应该检测到至少2个时间列
        
        # 检查tagtime列
        tagtime_result = next((r for r in time_results if r.column_name == 'tagtime'), None)
        self.assertIsNotNone(tagtime_result)
        self.assertTrue(tagtime_result.is_time_column)
        
        # 4. 数据预处理
        preprocessor = DataPreprocessor(PreprocessingConfig(
            handle_missing=True,
            missing_method=CleaningMethod.DROP_MISSING,
            handle_duplicates=True,
            convert_strings_to_numeric=True
        ))
        
        df_processed, preprocess_log = preprocessor.preprocess(df)
        self.assertIsNotNone(df_processed)
        self.assertIn('steps_applied', preprocess_log)
        
        # 5. 数据验证
        validator = DataValidator()
        validation_result = validator.validate(df_processed)
        
        self.assertIsNotNone(validation_result)
        self.assertTrue(validation_result.score > 0)
        
        # 6. 更新文件信息
        column_names, column_types = loader.get_column_info(df_processed)
        time_column_infos = [time_detector.convert_to_time_column_info(r) 
                           for r in time_results if r.is_time_column]
        
        # 重新计算质量信息（基于处理后的数据）
        if hasattr(df_processed, 'shape'):
            rows, cols = df_processed.shape
        else:
            rows, cols = len(df_processed), len(df_processed.columns)
        
        from src.models import DataQualityInfo
        final_quality = DataQualityInfo(
            total_rows=rows,
            total_columns=cols,
            missing_values_count=0,  # 预处理后应该没有缺失值
            duplicate_rows_count=0,  # 预处理后应该没有重复值
            memory_usage_mb=preprocess_log.get('final_memory', 0)
        )
        
        file_info.update_data_info(
            column_names=column_names,
            column_types=column_types,
            data_quality=final_quality,
            time_columns=time_column_infos
        )
        
        # 验证更新后的文件信息
        self.assertTrue(file_info.is_loaded)
        self.assertTrue(file_info.has_time_column)
        self.assertIsNotNone(file_info.primary_time_column)
        
        print(f"✅ 完整数据处理管道测试成功")
        print(f"   原始数据: {quality_info.total_rows}行 x {quality_info.total_columns}列")
        print(f"   处理后数据: {final_quality.total_rows}行 x {final_quality.total_columns}列")
        print(f"   检测到时间列: {len(time_column_infos)}个")
        print(f"   主要时间列: {file_info.primary_time_column}")
        print(f"   数据质量分数: {validation_result.score:.1f}")


if __name__ == "__main__":
    # 运行所有测试
    unittest.main(verbosity=2)