#!/usr/bin/env python3
"""
任务13.2: 完整工作流集成测试

测试端到端的数据分析工作流：
1. 文件格式兼容性测试
2. 完整分析流程测试
3. 导出功能集成测试
4. 工作流系统集成测试
5. 性能基准测试
"""

import sys
import unittest
import tempfile
import shutil
import time
import asyncio
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import json

# 导入测试配置
sys.path.insert(0, str(Path(__file__).parent))
from test_config import TestConfig, TestDataGenerator, TestAssertions

# 添加src目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


class TestFileFormatCompatibility(unittest.TestCase):
    """测试文件格式兼容性"""
    
    def setUp(self):
        """测试设置"""
        TestConfig.setup()
    
    def tearDown(self):
        """测试清理"""
        TestConfig.teardown()
    
    def test_csv_file_processing(self):
        """测试CSV文件处理"""
        if not TestConfig.CSV_FILE or not TestConfig.CSV_FILE.exists():
            self.skipTest("CSV测试文件不存在")
        
        try:
            # 测试文件读取
            df = pd.read_csv(TestConfig.CSV_FILE)
            TestAssertions.assert_dataframe_valid(df, min_rows=10)
            
            # 验证数据内容
            self.assertIn('datetime', df.columns, "应包含datetime列")
            self.assertIn('temperature', df.columns, "应包含temperature列")
            self.assertTrue(len(df) > 100, f"数据行数应大于100，实际: {len(df)}")
            
            print(f"✅ CSV文件处理测试通过: {len(df)} 行, {len(df.columns)} 列")
            
        except Exception as e:
            self.fail(f"CSV文件处理失败: {e}")
    
    def test_excel_file_processing(self):
        """测试Excel文件处理"""
        if not TestConfig.EXCEL_FILE or not TestConfig.EXCEL_FILE.exists():
            self.skipTest("Excel测试文件不存在")
        
        try:
            # 测试Excel读取
            df = pd.read_excel(TestConfig.EXCEL_FILE)
            TestAssertions.assert_dataframe_valid(df, min_rows=10)
            
            # 验证数据类型一致性
            self.assertTrue(df['temperature'].dtype in [np.float64, np.int64], 
                           f"temperature列类型错误: {df['temperature'].dtype}")
            
            print(f"✅ Excel文件处理测试通过: {len(df)} 行, {len(df.columns)} 列")
            
        except ImportError:
            self.skipTest("openpyxl未安装，跳过Excel测试")
        except Exception as e:
            self.fail(f"Excel文件处理失败: {e}")
    
    def test_parquet_file_processing(self):
        """测试Parquet文件处理"""
        if not TestConfig.PARQUET_FILE or not TestConfig.PARQUET_FILE.exists():
            self.skipTest("Parquet测试文件不存在")
        
        try:
            # 测试Parquet读取
            df = pd.read_parquet(TestConfig.PARQUET_FILE)
            TestAssertions.assert_dataframe_valid(df, min_rows=10)
            
            # 验证数据完整性
            self.assertFalse(df.empty, "Parquet文件不应为空")
            
            print(f"✅ Parquet文件处理测试通过: {len(df)} 行, {len(df.columns)} 列")
            
        except ImportError:
            self.skipTest("pyarrow未安装，跳过Parquet测试")
        except Exception as e:
            self.fail(f"Parquet文件处理失败: {e}")
    
    def test_invalid_file_handling(self):
        """测试无效文件处理"""
        if not TestConfig.INVALID_FILE or not TestConfig.INVALID_FILE.exists():
            self.skipTest("无效测试文件不存在")
        
        # 测试无效文件应该正确处理错误
        try:
            df = pd.read_csv(TestConfig.INVALID_FILE)
            # 如果能读取，检查是否为空或无效
            if not df.empty:
                self.fail("无效文件不应该被成功解析")
        except Exception:
            # 期望出现异常
            print("✅ 无效文件处理测试通过: 正确识别并处理了无效文件")


class TestCompleteAnalysisWorkflow(unittest.TestCase):
    """测试完整分析工作流"""
    
    def setUp(self):
        """测试设置"""
        TestConfig.setup()
    
    def tearDown(self):
        """测试清理"""
        TestConfig.teardown()
    
    def test_end_to_end_analysis(self):
        """测试端到端分析流程"""
        if not TestConfig.CSV_FILE or not TestConfig.CSV_FILE.exists():
            self.skipTest("测试数据文件不存在")
        
        try:
            # 步骤1: 数据加载
            df = pd.read_csv(TestConfig.CSV_FILE)
            self.assertIsNotNone(df)
            self.assertTrue(len(df) > 0)
            print(f"✅ 步骤1: 数据加载成功 - {len(df)} 行")
            
            # 步骤2: 基本统计分析
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                stats = df[numeric_columns].describe()
                self.assertIsNotNone(stats)
                print(f"✅ 步骤2: 描述性统计完成 - {len(numeric_columns)} 个数值列")
            
            # 步骤3: 相关性分析
            if len(numeric_columns) > 1:
                corr_matrix = df[numeric_columns].corr()
                self.assertEqual(corr_matrix.shape[0], corr_matrix.shape[1])
                print(f"✅ 步骤3: 相关性分析完成 - {corr_matrix.shape[0]}x{corr_matrix.shape[1]} 矩阵")
            
            # 步骤4: 异常值检测
            for col in numeric_columns:
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers = df[z_scores > 3]
                outlier_count = len(outliers)
                print(f"✅ 步骤4: {col} 异常值检测完成 - 发现 {outlier_count} 个异常值")
            
            # 步骤5: 数据导出测试
            output_file = TestConfig.TEMP_DATA_DIR / "analysis_result.csv"
            df.to_csv(output_file, index=False)
            TestAssertions.assert_file_exists(output_file)
            print("✅ 步骤5: 数据导出成功")
            
            print("🎉 端到端分析流程完整测试通过！")
            
        except Exception as e:
            self.fail(f"端到端分析流程失败: {e}")
    
    def test_time_series_analysis_workflow(self):
        """测试时间序列分析工作流"""
        try:
            # 创建时间序列数据
            ts_df = TestDataGenerator.create_time_series_dataframe(500)
            
            # 时间列检测
            time_columns = []
            for col in ts_df.columns:
                if ts_df[col].dtype.name.startswith('datetime'):
                    time_columns.append(col)
            
            self.assertTrue(len(time_columns) > 0, "应该检测到时间列")
            print(f"✅ 时间列检测: 发现 {len(time_columns)} 个时间列")
            
            # 基础时间序列统计
            time_col = time_columns[0]
            value_col = 'value'
            
            if value_col in ts_df.columns:
                # 计算基本统计
                mean_value = ts_df[value_col].mean()
                std_value = ts_df[value_col].std()
                
                self.assertIsNotNone(mean_value)
                self.assertIsNotNone(std_value)
                print(f"✅ 时间序列统计: 均值={mean_value:.2f}, 标准差={std_value:.2f}")
            
            print("✅ 时间序列分析工作流测试通过")
            
        except Exception as e:
            self.fail(f"时间序列分析工作流失败: {e}")


class TestExportIntegration(unittest.TestCase):
    """测试导出功能集成"""
    
    def setUp(self):
        """测试设置"""
        TestConfig.setup()
    
    def tearDown(self):
        """测试清理"""
        TestConfig.teardown()
    
    def test_multi_format_export(self):
        """测试多格式导出"""
        try:
            # 创建测试数据
            df = TestDataGenerator.create_simple_dataframe(100)
            
            # 测试CSV导出
            csv_file = TestConfig.TEMP_DATA_DIR / "export_test.csv"
            df.to_csv(csv_file, index=False)
            TestAssertions.assert_file_exists(csv_file)
            
            # 验证导出文件内容
            exported_df = pd.read_csv(csv_file)
            self.assertEqual(len(exported_df), len(df))
            print("✅ CSV导出测试通过")
            
            # 测试JSON导出
            json_file = TestConfig.TEMP_DATA_DIR / "export_test.json"
            df.to_json(json_file, orient='records')
            TestAssertions.assert_file_exists(json_file)
            
            # 验证JSON内容
            with open(json_file, 'r') as f:
                json_data = json.load(f)
            self.assertEqual(len(json_data), len(df))
            print("✅ JSON导出测试通过")
            
            # 测试Excel导出（如果可用）
            try:
                excel_file = TestConfig.TEMP_DATA_DIR / "export_test.xlsx"
                df.to_excel(excel_file, index=False)
                TestAssertions.assert_file_exists(excel_file)
                print("✅ Excel导出测试通过")
            except ImportError:
                print("⚠️ Excel导出跳过: openpyxl未安装")
            
            print("✅ 多格式导出集成测试通过")
            
        except Exception as e:
            self.fail(f"多格式导出测试失败: {e}")
    
    def test_chart_export_integration(self):
        """测试图表导出集成"""
        try:
            import matplotlib.pyplot as plt
            
            # 创建测试图表
            fig, ax = plt.subplots(figsize=(8, 6))
            x = np.linspace(0, 10, 100)
            y = np.sin(x)
            ax.plot(x, y, label='sin(x)')
            ax.set_title('测试图表')
            ax.legend()
            
            # 测试PNG导出
            png_file = TestConfig.TEMP_DATA_DIR / "test_chart.png"
            fig.savefig(png_file, dpi=150, bbox_inches='tight')
            TestAssertions.assert_file_exists(png_file)
            print("✅ PNG图表导出测试通过")
            
            # 测试SVG导出
            svg_file = TestConfig.TEMP_DATA_DIR / "test_chart.svg"
            fig.savefig(svg_file, format='svg', bbox_inches='tight')
            TestAssertions.assert_file_exists(svg_file)
            print("✅ SVG图表导出测试通过")
            
            plt.close(fig)
            print("✅ 图表导出集成测试通过")
            
        except ImportError:
            self.skipTest("matplotlib未安装，跳过图表导出测试")
        except Exception as e:
            self.fail(f"图表导出测试失败: {e}")


class TestWorkflowSystemIntegration(unittest.TestCase):
    """测试工作流系统集成"""
    
    def setUp(self):
        """测试设置"""
        TestConfig.setup()
    
    def tearDown(self):
        """测试清理"""
        TestConfig.teardown()
    
    def test_workflow_state_management(self):
        """测试工作流状态管理"""
        try:
            from src.workflow import WorkflowIntegrator, WorkflowState, WorkflowContext
            
            # 创建工作流集成器
            integrator = WorkflowIntegrator()
            
            # 创建工作流上下文
            context = WorkflowContext()
            self.assertEqual(context.current_state, WorkflowState.IDLE)
            
            # 测试状态变更
            context.current_state = WorkflowState.ANALYZING
            self.assertEqual(context.current_state, WorkflowState.ANALYZING)
            self.assertTrue(context.is_active)
            
            # 测试完成状态
            context.current_state = WorkflowState.COMPLETED
            context.completed_at = datetime.now()
            self.assertFalse(context.is_active)
            self.assertIsNotNone(context.duration)
            
            integrator.cleanup()
            print("✅ 工作流状态管理测试通过")
            
        except Exception as e:
            self.fail(f"工作流状态管理测试失败: {e}")
    
    def test_performance_optimization_integration(self):
        """测试性能优化集成"""
        try:
            from src.workflow import PerformanceOptimizer, OptimizationStrategy
            
            # 创建性能优化器
            optimizer = PerformanceOptimizer()
            
            # 测试优化策略设置
            strategies = [
                OptimizationStrategy.BALANCED,
                OptimizationStrategy.MEMORY_AGGRESSIVE,
                OptimizationStrategy.RESPONSIVE
            ]
            
            for strategy in strategies:
                optimizer.set_optimization_strategy(strategy)
                self.assertEqual(optimizer.current_strategy, strategy)
            
            # 测试UI优化建议
            recommendations = optimizer.optimize_ui_rendering(50)
            self.assertIsInstance(recommendations, dict)
            self.assertIn('batch_updates', recommendations)
            
            # 测试性能摘要
            summary = optimizer.get_performance_summary()
            self.assertIsInstance(summary, dict)
            
            optimizer.cleanup()
            print("✅ 性能优化集成测试通过")
            
        except Exception as e:
            self.fail(f"性能优化集成测试失败: {e}")


class TestPerformanceBenchmarks(unittest.TestCase):
    """测试性能基准"""
    
    def setUp(self):
        """测试设置"""
        TestConfig.setup()
    
    def tearDown(self):
        """测试清理"""
        TestConfig.teardown()
    
    def test_data_loading_performance(self):
        """测试数据加载性能"""
        if not TestConfig.CSV_FILE or not TestConfig.CSV_FILE.exists():
            self.skipTest("测试数据文件不存在")
        
        try:
            # 基准测试：数据加载时间
            start_time = time.time()
            df = pd.read_csv(TestConfig.CSV_FILE)
            load_time = time.time() - start_time
            
            # 性能断言（加载时间应该合理）
            self.assertLess(load_time, 5.0, f"数据加载时间过长: {load_time:.2f}秒")
            
            # 内存使用检查
            memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
            self.assertLess(memory_usage, 100, f"内存使用过多: {memory_usage:.2f}MB")
            
            print(f"✅ 数据加载性能测试通过: {load_time:.3f}秒, {memory_usage:.2f}MB")
            
        except Exception as e:
            self.fail(f"数据加载性能测试失败: {e}")
    
    def test_analysis_performance(self):
        """测试分析性能"""
        try:
            # 创建大数据集进行性能测试
            large_df = TestDataGenerator.create_correlation_dataframe(10000)
            
            # 基准测试：描述性统计
            start_time = time.time()
            stats = large_df.describe()
            stats_time = time.time() - start_time
            
            self.assertLess(stats_time, 2.0, f"描述性统计时间过长: {stats_time:.2f}秒")
            
            # 基准测试：相关性分析
            start_time = time.time()
            corr_matrix = large_df.select_dtypes(include=[np.number]).corr()
            corr_time = time.time() - start_time
            
            self.assertLess(corr_time, 3.0, f"相关性分析时间过长: {corr_time:.2f}秒")
            
            print(f"✅ 分析性能测试通过: 统计={stats_time:.3f}s, 相关性={corr_time:.3f}s")
            
        except Exception as e:
            self.fail(f"分析性能测试失败: {e}")
    
    def test_export_performance(self):
        """测试导出性能"""
        try:
            # 创建测试数据
            df = TestDataGenerator.create_simple_dataframe(5000)
            
            # 基准测试：CSV导出
            csv_file = TestConfig.TEMP_DATA_DIR / "perf_test.csv"
            start_time = time.time()
            df.to_csv(csv_file, index=False)
            csv_time = time.time() - start_time
            
            self.assertLess(csv_time, 2.0, f"CSV导出时间过长: {csv_time:.2f}秒")
            TestAssertions.assert_file_exists(csv_file)
            
            # 基准测试：JSON导出
            json_file = TestConfig.TEMP_DATA_DIR / "perf_test.json"
            start_time = time.time()
            df.to_json(json_file, orient='records')
            json_time = time.time() - start_time
            
            self.assertLess(json_time, 3.0, f"JSON导出时间过长: {json_time:.2f}秒")
            TestAssertions.assert_file_exists(json_file)
            
            print(f"✅ 导出性能测试通过: CSV={csv_time:.3f}s, JSON={json_time:.3f}s")
            
        except Exception as e:
            self.fail(f"导出性能测试失败: {e}")


class TestRegressionSuite(unittest.TestCase):
    """回归测试套件"""
    
    def test_data_consistency(self):
        """测试数据一致性"""
        try:
            # 创建相同种子的数据，应该产生一致结果
            np.random.seed(12345)
            df1 = TestDataGenerator.create_simple_dataframe(100)
            
            np.random.seed(12345)
            df2 = TestDataGenerator.create_simple_dataframe(100)
            
            # 验证数据一致性
            pd.testing.assert_frame_equal(df1, df2)
            print("✅ 数据一致性回归测试通过")
            
        except Exception as e:
            self.fail(f"数据一致性测试失败: {e}")
    
    def test_statistical_consistency(self):
        """测试统计计算一致性"""
        try:
            # 创建已知统计特性的数据
            np.random.seed(42)
            data = np.random.normal(100, 15, 1000)
            df = pd.DataFrame({'value': data})
            
            # 多次计算应该产生相同结果
            stats1 = df.describe()
            stats2 = df.describe()
            
            pd.testing.assert_frame_equal(stats1, stats2)
            print("✅ 统计计算一致性回归测试通过")
            
        except Exception as e:
            self.fail(f"统计计算一致性测试失败: {e}")


def run_integration_tests():
    """运行集成测试"""
    print("\n🔗 开始完整工作流集成测试...")
    
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试类
    test_classes = [
        TestFileFormatCompatibility,
        TestCompleteAnalysisWorkflow,
        TestExportIntegration,
        TestWorkflowSystemIntegration,
        TestPerformanceBenchmarks,
        TestRegressionSuite
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
    skipped = len(result.skipped) if hasattr(result, 'skipped') else 0
    passed = total_tests - failures - errors - skipped
    
    print(f"\n{'='*60}")
    print(f"📊 集成测试结果汇总:")
    print(f"总测试数: {total_tests}")
    print(f"通过测试: {passed}")
    print(f"失败测试: {failures}")
    print(f"错误测试: {errors}")
    print(f"跳过测试: {skipped}")
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
        print("🎉 所有集成测试通过！")
        return True
    else:
        print("⚠️  部分测试失败，需要检查实现")
        return False


if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)