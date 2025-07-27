#!/usr/bin/env python3
"""
任务13.1: 核心组件单元测试

全面测试所有核心组件的功能：
1. 数据模型测试
2. 数据处理引擎测试
3. 分析引擎测试
4. 历史管理器测试
5. 导出系统测试
6. 工作流系统测试
"""

import sys
import unittest
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

# 导入测试配置
from .test_config import TestConfig, TestDataGenerator, TestAssertions, with_test_data

# 添加src目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


class TestDataModels(unittest.TestCase):
    """测试数据模型"""
    
    def setUp(self):
        """测试设置"""
        TestConfig.setup()
    
    def tearDown(self):
        """测试清理"""
        TestConfig.teardown()
    
    def test_file_info_creation(self):
        """测试FileInfo创建"""
        from src.models.file_info import FileInfo
        
        # 测试从文件创建
        if TestConfig.CSV_FILE and TestConfig.CSV_FILE.exists():
            file_info = FileInfo.create_from_file(str(TestConfig.CSV_FILE))
            
            self.assertIsNotNone(file_info)
            self.assertEqual(file_info.file_type, "csv")
            self.assertTrue(file_info.file_size > 0)
            self.assertTrue(len(file_info.columns) > 0)
            self.assertTrue(file_info.row_count > 0)
            print(f"✅ FileInfo创建测试通过: {file_info.file_name}")
    
    def test_analysis_result_creation(self):
        """测试AnalysisResult创建"""
        from src.models.extended_analysis_result import AnalysisResult
        
        # 创建模拟数据
        df = TestDataGenerator.create_simple_dataframe(100)
        
        # 创建分析结果
        result = AnalysisResult()
        result.descriptive_stats = df.describe().to_dict()
        result.correlation_matrix = df.corr().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 1 else {}
        result.outliers = {'count': 5, 'indices': [1, 2, 3, 4, 5]}
        result.stationarity_test = {'is_stationary': True, 'p_value': 0.01}
        
        TestAssertions.assert_analysis_result_valid(result)
        print("✅ AnalysisResult创建测试通过")
    
    def test_analysis_history_record(self):
        """测试AnalysisHistoryRecord"""
        from src.models.analysis_history import AnalysisHistoryRecord, AnalysisStatus
        from src.models.file_info import FileInfo
        from src.models.extended_analysis_result import AnalysisResult
        
        # 创建模拟组件
        file_info = FileInfo(
            file_path="/tmp/test.csv",
            file_name="test.csv",
            file_size=1024,
            file_type="csv"
        )
        
        analysis_result = AnalysisResult()
        analysis_result.descriptive_stats = {'mean': 50}
        
        # 创建历史记录
        record = AnalysisHistoryRecord(
            file_info=file_info,
            analysis_result=analysis_result,
            analysis_config={'test': True},
            status=AnalysisStatus.COMPLETED
        )
        
        self.assertIsNotNone(record.analysis_id)
        self.assertEqual(record.status, AnalysisStatus.COMPLETED)
        self.assertIsNotNone(record.created_at)
        print("✅ AnalysisHistoryRecord测试通过")


class TestDataProcessing(unittest.TestCase):
    """测试数据处理组件"""
    
    def setUp(self):
        """测试设置"""
        TestConfig.setup()
    
    def tearDown(self):
        """测试清理"""
        TestConfig.teardown()
    
    def test_data_loader(self):
        """测试数据加载器"""
        from src.data.data_loader import DataLoader
        
        loader = DataLoader()
        
        # 测试CSV加载
        if TestConfig.CSV_FILE and TestConfig.CSV_FILE.exists():
            df = loader.load_file(str(TestConfig.CSV_FILE))
            TestAssertions.assert_dataframe_valid(df, min_rows=10)
            print(f"✅ CSV加载测试通过: {len(df)} 行")
        
        # 测试Excel加载
        if TestConfig.EXCEL_FILE and TestConfig.EXCEL_FILE.exists():
            try:
                df = loader.load_file(str(TestConfig.EXCEL_FILE))
                TestAssertions.assert_dataframe_valid(df, min_rows=10)
                print(f"✅ Excel加载测试通过: {len(df)} 行")
            except ImportError:
                print("⚠️ Excel加载跳过: openpyxl未安装")
    
    def test_time_detector(self):
        """测试时间列检测"""
        from src.data.time_detector import TimeDetector
        
        # 创建包含时间列的数据
        df = TestDataGenerator.create_time_series_dataframe(100)
        detector = TimeDetector()
        
        time_info = detector.detect_time_columns(df)
        
        self.assertIsNotNone(time_info)
        self.assertTrue(len(time_info) > 0)
        print(f"✅ 时间检测测试通过: 发现 {len(time_info)} 个时间列")
    
    def test_data_validator(self):
        """测试数据验证器"""
        from src.data.data_validator import DataValidator
        
        # 创建测试数据
        df = TestDataGenerator.create_simple_dataframe(100)
        validator = DataValidator()
        
        # 验证数据
        validation_result = validator.validate_dataframe(df)
        
        self.assertIsNotNone(validation_result)
        self.assertIn('is_valid', validation_result)
        self.assertIn('issues', validation_result)
        print("✅ 数据验证测试通过")


class TestAnalysisEngine(unittest.TestCase):
    """测试分析引擎"""
    
    def setUp(self):
        """测试设置"""
        TestConfig.setup()
    
    def tearDown(self):
        """测试清理"""
        TestConfig.teardown()
    
    def test_analysis_engine_basic(self):
        """测试分析引擎基本功能"""
        from src.core.analysis_engine import AnalysisEngine, AnalysisConfig
        
        # 创建测试数据
        df = TestDataGenerator.create_correlation_dataframe(200)
        
        # 创建分析配置
        config = AnalysisConfig(
            include_descriptive=True,
            include_correlation=True,
            include_outlier_detection=True
        )
        
        # 执行分析
        engine = AnalysisEngine(config)
        result = engine.analyze(df)
        
        # 验证结果
        TestAssertions.assert_analysis_result_valid(result)
        self.assertIsNotNone(result.descriptive_stats)
        self.assertIsNotNone(result.correlation_matrix)
        print("✅ 分析引擎基本功能测试通过")
    
    def test_correlation_analysis(self):
        """测试相关性分析"""
        from src.core.analysis_engine import AnalysisEngine, AnalysisConfig
        
        # 创建有明确相关性的数据
        df = TestDataGenerator.create_correlation_dataframe(300)
        
        config = AnalysisConfig(include_correlation=True)
        engine = AnalysisEngine(config)
        result = engine.analyze(df)
        
        # 验证相关性矩阵存在且合理
        self.assertIsNotNone(result.correlation_matrix)
        self.assertTrue(len(result.correlation_matrix) > 0)
        print("✅ 相关性分析测试通过")
    
    def test_outlier_detection(self):
        """测试异常值检测"""
        from src.core.analysis_engine import AnalysisEngine, AnalysisConfig
        
        # 创建包含异常值的数据
        df = TestDataGenerator.create_simple_dataframe(100)
        # 添加明显的异常值
        df.loc[0, 'value'] = 1000  # 极端值
        
        config = AnalysisConfig(
            include_outlier_detection=True,
            outlier_method='zscore',
            outlier_threshold=2.0
        )
        
        engine = AnalysisEngine(config)
        result = engine.analyze(df)
        
        # 验证异常值检测结果
        self.assertIsNotNone(result.outliers)
        if isinstance(result.outliers, dict):
            self.assertIn('count', result.outliers)
        print("✅ 异常值检测测试通过")


class TestHistoryManager(unittest.TestCase):
    """测试历史管理器"""
    
    def setUp(self):
        """测试设置"""
        TestConfig.setup()
    
    def tearDown(self):
        """测试清理"""
        TestConfig.teardown()
    
    def test_history_manager_crud(self):
        """测试历史管理器CRUD操作"""
        from src.core.history_manager import get_history_manager
        from src.models.analysis_history import AnalysisHistoryRecord, AnalysisStatus
        from src.models.file_info import FileInfo
        from src.models.extended_analysis_result import AnalysisResult
        
        # 获取历史管理器
        manager = get_history_manager(db_path=str(TestConfig.TEST_DB_PATH))
        
        # 创建测试记录
        file_info = FileInfo(
            file_path="/tmp/test.csv",
            file_name="test.csv",
            file_size=1024,
            file_type="csv"
        )
        
        analysis_result = AnalysisResult()
        analysis_result.descriptive_stats = {'mean': 50}
        
        record = AnalysisHistoryRecord(
            file_info=file_info,
            analysis_result=analysis_result,
            analysis_config={'test': True},
            status=AnalysisStatus.COMPLETED
        )
        
        # 测试保存
        saved_record = manager.save_record(record)
        self.assertIsNotNone(saved_record.analysis_id)
        print(f"✅ 历史记录保存测试通过: {saved_record.analysis_id}")
        
        # 测试查询
        retrieved_record = manager.get_record(saved_record.analysis_id)
        self.assertIsNotNone(retrieved_record)
        self.assertEqual(retrieved_record.analysis_id, saved_record.analysis_id)
        print("✅ 历史记录查询测试通过")
        
        # 测试列表
        records = manager.list_records(limit=10)
        self.assertTrue(len(records) >= 1)
        print(f"✅ 历史记录列表测试通过: {len(records)} 条记录")


class TestExportSystem(unittest.TestCase):
    """测试导出系统"""
    
    def setUp(self):
        """测试设置"""
        TestConfig.setup()
    
    def tearDown(self):
        """测试清理"""
        TestConfig.teardown()
    
    def test_data_exporter(self):
        """测试数据导出器"""
        from src.export.data_exporter import DataExporter
        
        # 创建测试数据
        df = TestDataGenerator.create_simple_dataframe(50)
        exporter = DataExporter()
        
        # 测试CSV导出
        csv_file = TestConfig.TEMP_DATA_DIR / "test_export.csv"
        success = exporter.export_to_csv(df, str(csv_file))
        self.assertTrue(success)
        TestAssertions.assert_file_exists(csv_file)
        print("✅ CSV导出测试通过")
        
        # 测试JSON导出
        json_file = TestConfig.TEMP_DATA_DIR / "test_export.json"
        success = exporter.export_to_json(df, str(json_file))
        self.assertTrue(success)
        TestAssertions.assert_file_exists(json_file)
        print("✅ JSON导出测试通过")
    
    def test_chart_exporter(self):
        """测试图表导出器"""
        from src.export.chart_exporter import ChartExporter
        
        exporter = ChartExporter()
        
        # 创建简单的matplotlib图表
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3, 4], [1, 4, 2, 3])
            
            # 测试PNG导出
            png_file = TestConfig.TEMP_DATA_DIR / "test_chart.png"
            success = exporter.export_matplotlib_chart(fig, str(png_file), format='png')
            self.assertTrue(success)
            TestAssertions.assert_file_exists(png_file)
            print("✅ 图表PNG导出测试通过")
            
            plt.close(fig)
            
        except ImportError:
            print("⚠️ 图表导出跳过: matplotlib未安装")
    
    def test_export_manager(self):
        """测试导出管理器"""
        from src.export.export_manager import ExportManager
        from src.models.file_info import FileInfo
        from src.models.extended_analysis_result import AnalysisResult
        
        manager = ExportManager()
        
        # 创建模拟数据
        file_info = FileInfo(
            file_path="/tmp/test.csv",
            file_name="test.csv",
            file_size=1024,
            file_type="csv"
        )
        
        analysis_result = AnalysisResult()
        analysis_result.descriptive_stats = {'mean': 50, 'std': 10}
        
        data = TestDataGenerator.create_simple_dataframe(50)
        
        # 测试数据导出
        output_file = TestConfig.TEMP_DATA_DIR / "export_test.csv"
        success = manager.export_data_only(data, str(output_file), "csv")
        self.assertTrue(success)
        print("✅ 导出管理器测试通过")


class TestWorkflowSystem(unittest.TestCase):
    """测试工作流系统"""
    
    def setUp(self):
        """测试设置"""
        TestConfig.setup()
    
    def tearDown(self):
        """测试清理"""
        TestConfig.teardown()
    
    def test_performance_optimizer(self):
        """测试性能优化器"""
        from src.workflow.performance_optimizer import PerformanceOptimizer, OptimizationStrategy
        
        optimizer = PerformanceOptimizer()
        
        # 测试策略设置
        for strategy in OptimizationStrategy:
            optimizer.set_optimization_strategy(strategy)
            self.assertEqual(optimizer.current_strategy, strategy)
        
        # 测试性能摘要
        summary = optimizer.get_performance_summary()
        self.assertIsInstance(summary, dict)
        
        # 测试UI优化建议
        recommendations = optimizer.optimize_ui_rendering(100)
        self.assertIsInstance(recommendations, dict)
        self.assertIn('batch_updates', recommendations)
        
        optimizer.cleanup()
        print("✅ 性能优化器测试通过")
    
    def test_workflow_integrator(self):
        """测试工作流集成器"""
        from src.workflow.workflow_integrator import WorkflowIntegrator, WorkflowState
        
        integrator = WorkflowIntegrator()
        
        # 测试工作流状态管理
        active_workflows = integrator.get_active_workflows()
        self.assertIsInstance(active_workflows, list)
        
        # 测试性能摘要集成
        summary = integrator.get_performance_summary()
        self.assertIsInstance(summary, dict)
        
        integrator.cleanup()
        print("✅ 工作流集成器测试通过")


def run_core_component_tests():
    """运行核心组件测试"""
    print("\n🧪 开始核心组件单元测试...")
    
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试类
    test_classes = [
        TestDataModels,
        TestDataProcessing,
        TestAnalysisEngine,
        TestHistoryManager,
        TestExportSystem,
        TestWorkflowSystem
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
    print(f"📊 核心组件测试结果汇总:")
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
        print("🎉 所有核心组件测试通过！")
        return True
    else:
        print("⚠️  部分测试失败，需要检查实现")
        return False


if __name__ == "__main__":
    success = run_core_component_tests()
    sys.exit(0 if success else 1)