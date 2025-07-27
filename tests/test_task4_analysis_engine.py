"""
任务4测试：数据处理引擎
测试AnalysisEngine分析引擎和AsyncWorker异步工作流管理
"""

import asyncio
import tempfile
import unittest
from pathlib import Path
import time

import pytest

# 尝试导入pandas作为测试数据源
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from src.core.analysis_engine import AnalysisEngine, AnalysisConfig
from src.core.async_worker import AsyncWorker, AsyncTask, TaskStatus, TaskPriority, WorkflowConfig
from src.data import DataLoader, LoaderConfig
from src.models import FileInfo, FileType


class TestAnalysisEngine(unittest.TestCase):
    """分析引擎测试"""
    
    def setUp(self):
        """设置测试数据"""
        if HAS_PANDAS and HAS_NUMPY:
            # 创建测试数据
            np.random.seed(42)  # 固定随机种子
            self.test_df = pd.DataFrame({
                'value1': np.random.normal(100, 15, 100),  # 正态分布
                'value2': np.random.exponential(2, 100),   # 指数分布
                'value3': np.random.uniform(0, 100, 100),  # 均匀分布
                'category': np.random.choice(['A', 'B', 'C'], 100),  # 分类变量
                'timestamp': pd.date_range('2023-01-01', periods=100, freq='D')
            })
            
            # 添加一些异常值
            self.test_df.loc[95:99, 'value1'] = 1000  # 异常值
            
            # 添加缺失值
            self.test_df.loc[90:92, 'value2'] = np.nan
        
        self.config = AnalysisConfig(
            outlier_method="iqr",
            correlation_method="pearson",
            stationarity_tests=["adf", "kpss"],
            n_threads=2
        )
        
        self.engine = AnalysisEngine(self.config)
    
    @unittest.skipIf(not (HAS_PANDAS and HAS_NUMPY), "Pandas or NumPy not available")
    def test_descriptive_stats(self):
        """测试描述性统计"""
        descriptive_stats = self.engine.compute_descriptive_stats(
            self.test_df[['value1', 'value2', 'value3']]
        )
        
        self.assertIsNotNone(descriptive_stats)
        self.assertEqual(len(descriptive_stats.statistics), 3)
        
        # 检查value1的统计信息
        value1_stats = descriptive_stats.get_column_stats('value1')
        self.assertIsNotNone(value1_stats)
        self.assertIn('mean', value1_stats)
        self.assertIn('std', value1_stats)
        self.assertIn('min', value1_stats)
        self.assertIn('max', value1_stats)
        
        print(f"✅ 描述性统计测试成功")
        print(f"   分析列数: {len(descriptive_stats.statistics)}")
        print(f"   value1均值: {value1_stats['mean']:.2f}")
    
    @unittest.skipIf(not (HAS_PANDAS and HAS_NUMPY), "Pandas or NumPy not available")
    def test_correlation_matrix(self):
        """测试关联矩阵计算"""
        correlation_matrix = self.engine.compute_correlation_matrix(
            self.test_df[['value1', 'value2', 'value3']]
        )
        
        self.assertIsNotNone(correlation_matrix)
        self.assertEqual(len(correlation_matrix.matrix), 3)
        self.assertEqual(len(correlation_matrix.columns), 3)
        
        # 检查相关系数
        corr_12 = correlation_matrix.get_correlation('value1', 'value2')
        self.assertIsNotNone(corr_12)
        self.assertTrue(-1 <= corr_12 <= 1)
        
        # 检查高相关性对
        high_corrs = correlation_matrix.get_high_correlations(0.5)
        self.assertIsInstance(high_corrs, list)
        
        print(f"✅ 关联矩阵测试成功")
        print(f"   矩阵维度: {len(correlation_matrix.matrix)}x{len(correlation_matrix.columns)}")
        print(f"   value1-value2相关系数: {corr_12:.3f}")
    
    @unittest.skipIf(not (HAS_PANDAS and HAS_NUMPY), "Pandas or NumPy not available")
    def test_anomaly_detection(self):
        """测试异常值检测"""
        anomaly_result = self.engine.detect_anomalies(
            self.test_df[['value1', 'value2', 'value3']]
        )
        
        self.assertIsNotNone(anomaly_result)
        self.assertEqual(len(anomaly_result.anomalies), 3)
        
        # 检查value1的异常值（应该检测到我们添加的异常值）
        value1_anomalies = anomaly_result.get_column_anomalies('value1')
        self.assertIsNotNone(value1_anomalies)
        self.assertGreater(value1_anomalies['outlier_count'], 0)  # 应该检测到异常值
        
        total_anomalies = anomaly_result.get_total_anomalies()
        anomaly_columns = anomaly_result.get_anomaly_columns()
        
        print(f"✅ 异常值检测测试成功")
        print(f"   总异常值数量: {total_anomalies}")
        print(f"   有异常值的列: {anomaly_columns}")
        print(f"   value1异常值: {value1_anomalies['outlier_count']}个 ({value1_anomalies['outlier_percentage']:.1f}%)")
    
    @unittest.skipIf(not (HAS_PANDAS and HAS_NUMPY), "Pandas or NumPy not available")
    def test_time_series_analysis(self):
        """测试时间序列分析"""
        # 创建时间序列数据
        ts_df = self.test_df.copy()
        
        time_series_result = self.engine.analyze_time_series(ts_df, 'timestamp')
        
        if time_series_result is not None:  # 如果有statsmodels库
            self.assertIsNotNone(time_series_result.stationarity_tests)
            self.assertEqual(time_series_result.time_column, 'timestamp')
            
            # 检查平稳性检验结果
            for col in ['value1', 'value2', 'value3']:
                if col in time_series_result.stationarity_tests:
                    stationarity = time_series_result.get_column_stationarity(col)
                    self.assertIsNotNone(stationarity)
            
            stationary_cols = time_series_result.get_stationary_columns()
            trending_cols = time_series_result.get_trending_columns()
            
            print(f"✅ 时间序列分析测试成功")
            print(f"   分析列数: {len(time_series_result.stationarity_tests)}")
            print(f"   平稳列数: {len(stationary_cols)}")
            print(f"   趋势列数: {len(trending_cols)}")
        else:
            print("⚠️  时间序列分析跳过（缺少statsmodels库）")
    
    @unittest.skipIf(not (HAS_PANDAS and HAS_NUMPY), "Pandas or NumPy not available")
    def test_full_analysis(self):
        """测试完整分析流程"""
        analysis_result = self.engine.analyze_dataset(
            self.test_df,
            time_column='timestamp',
            exclude_columns=['category']
        )
        
        self.assertIsNotNone(analysis_result)
        self.assertIsNotNone(analysis_result.descriptive_stats)
        self.assertIsNotNone(analysis_result.correlation_matrix)
        self.assertIsNotNone(analysis_result.anomaly_detection)
        
        # 检查整体摘要
        summary = analysis_result.get_overall_summary()
        self.assertIn('analysis_id', summary)
        self.assertIn('has_descriptive_stats', summary)
        self.assertIn('has_correlation_analysis', summary)
        self.assertIn('has_anomaly_detection', summary)
        
        print(f"✅ 完整分析测试成功")
        print(f"   分析ID: {analysis_result.analysis_id}")
        print(f"   执行时间: {analysis_result.execution_time_ms}ms")
        print(f"   包含描述性统计: {summary['has_descriptive_stats']}")
        print(f"   包含关联分析: {summary['has_correlation_analysis']}")
        print(f"   包含异常值检测: {summary['has_anomaly_detection']}")
        print(f"   包含时间序列分析: {summary['has_time_series_analysis']}")


class TestAsyncWorker(unittest.TestCase):
    """异步工作流管理器测试"""
    
    def setUp(self):
        """设置测试"""
        self.config = WorkflowConfig(
            max_concurrent_tasks=2,
            max_queue_size=10,
            default_task_timeout=10.0
        )
        self.worker = AsyncWorker(self.config)
    
    def tearDown(self):
        """清理资源"""
        if hasattr(self, 'worker'):
            asyncio.run(self.worker.stop())
    
    def test_basic_task_execution(self):
        """测试基本任务执行"""
        async def run_test():
            await self.worker.start()
            
            # 简单的同步任务
            def simple_task(x, y):
                time.sleep(0.1)  # 模拟耗时操作
                return x + y
            
            task_id = await self.worker.submit_task(
                name="simple_addition",
                func=simple_task,
                args=(5, 3),
                priority=TaskPriority.NORMAL
            )
            
            # 等待任务完成
            await asyncio.sleep(0.5)
            
            task = self.worker.get_task_status(task_id)
            self.assertIsNotNone(task)
            self.assertEqual(task.status, TaskStatus.COMPLETED)
            self.assertEqual(task.result, 8)
            
            print(f"✅ 基本任务执行测试成功")
            print(f"   任务ID: {task_id}")
            print(f"   执行时间: {task.execution_time:.3f}s")
            print(f"   结果: {task.result}")
        
        asyncio.run(run_test())
    
    def test_async_task_execution(self):
        """测试异步任务执行"""
        async def run_test():
            await self.worker.start()
            
            # 异步任务
            async def async_task(message):
                await asyncio.sleep(0.1)
                return f"Processed: {message}"
            
            task_id = await self.worker.submit_task(
                name="async_processing",
                func=async_task,
                args=("Hello World",),
                priority=TaskPriority.HIGH
            )
            
            # 等待任务完成
            await asyncio.sleep(0.5)
            
            task = self.worker.get_task_status(task_id)
            self.assertIsNotNone(task)
            self.assertEqual(task.status, TaskStatus.COMPLETED)
            self.assertEqual(task.result, "Processed: Hello World")
            
            print(f"✅ 异步任务执行测试成功")
            print(f"   任务ID: {task_id}")
            print(f"   结果: {task.result}")
        
        asyncio.run(run_test())
    
    def test_progress_callback(self):
        """测试进度回调"""
        async def run_test():
            await self.worker.start()
            
            progress_updates = []
            
            def progress_callback(progress):
                progress_updates.append(progress.percentage)
            
            def task_with_progress():
                task = self.worker.get_running_tasks()[0] if self.worker.get_running_tasks() else None
                if task:
                    for i in range(5):
                        task.update_progress(i + 1, 5, f"Step {i + 1}")
                        time.sleep(0.05)
                return "Completed with progress"
            
            task_id = await self.worker.submit_task(
                name="progress_task",
                func=task_with_progress,
                progress_callback=progress_callback
            )
            
            # 等待任务完成
            await asyncio.sleep(1.0)
            
            task = self.worker.get_task_status(task_id)
            self.assertEqual(task.status, TaskStatus.COMPLETED)
            self.assertGreater(len(progress_updates), 0)
            
            print(f"✅ 进度回调测试成功")
            print(f"   进度更新次数: {len(progress_updates)}")
            print(f"   最终进度: {progress_updates[-1] if progress_updates else 0}%")
        
        asyncio.run(run_test())
    
    def test_task_cancellation(self):
        """测试任务取消"""
        async def run_test():
            await self.worker.start()
            
            def long_running_task():
                for i in range(100):
                    time.sleep(0.01)
                    # 检查取消状态（在实际实现中需要任务检查取消状态）
                return "Should not complete"
            
            task_id = await self.worker.submit_task(
                name="long_task",
                func=long_running_task
            )
            
            # 等待任务开始
            await asyncio.sleep(0.1)
            
            # 取消任务
            cancelled = await self.worker.cancel_task(task_id)
            self.assertTrue(cancelled)
            
            # 等待取消生效
            await asyncio.sleep(0.2)
            
            task = self.worker.get_task_status(task_id)
            self.assertEqual(task.status, TaskStatus.CANCELLED)
            
            print(f"✅ 任务取消测试成功")
            print(f"   任务状态: {task.status.value}")
        
        asyncio.run(run_test())
    
    def test_multiple_tasks_execution(self):
        """测试多任务并发执行"""
        async def run_test():
            await self.worker.start()
            
            def numbered_task(number):
                time.sleep(0.1)
                return f"Task {number} completed"
            
            # 提交多个任务
            task_ids = []
            for i in range(5):
                task_id = await self.worker.submit_task(
                    name=f"task_{i}",
                    func=numbered_task,
                    args=(i,)
                )
                task_ids.append(task_id)
            
            # 等待所有任务完成
            completed = await self.worker.wait_for_completion(timeout=5.0)
            self.assertTrue(completed)
            
            # 检查所有任务都完成了
            completed_count = 0
            for task_id in task_ids:
                task = self.worker.get_task_status(task_id)
                if task.status == TaskStatus.COMPLETED:
                    completed_count += 1
            
            self.assertEqual(completed_count, 5)
            
            stats = self.worker.get_stats()
            print(f"✅ 多任务执行测试成功")
            print(f"   完成任务数: {stats['completed_tasks']}")
            print(f"   平均执行时间: {stats['average_execution_time']:.3f}s")
        
        asyncio.run(run_test())
    
    def test_error_handling_and_retry(self):
        """测试错误处理和重试"""
        async def run_test():
            await self.worker.start()
            
            def failing_task(attempt_count=[0]):
                attempt_count[0] += 1
                if attempt_count[0] < 3:  # 前两次失败
                    raise ValueError(f"Attempt {attempt_count[0]} failed")
                return f"Success on attempt {attempt_count[0]}"
            
            task_id = await self.worker.submit_task(
                name="failing_task",
                func=failing_task,
                max_retries=3
            )
            
            # 等待任务完成（包括重试）
            await asyncio.sleep(2.0)
            
            task = self.worker.get_task_status(task_id)
            # 任务应该在第3次尝试时成功
            self.assertEqual(task.status, TaskStatus.COMPLETED)
            self.assertEqual(task.result, "Success on attempt 3")
            self.assertEqual(task.retry_count, 2)  # 重试了2次
            
            print(f"✅ 错误处理和重试测试成功")
            print(f"   重试次数: {task.retry_count}")
            print(f"   最终结果: {task.result}")
        
        asyncio.run(run_test())


class TestIntegrationTask4(unittest.TestCase):
    """任务4集成测试"""
    
    def setUp(self):
        """设置集成测试"""
        if HAS_PANDAS and HAS_NUMPY:
            # 创建测试CSV文件
            np.random.seed(42)
            test_data = pd.DataFrame({
                'timestamp': pd.date_range('2023-01-01', periods=50, freq='D'),
                'sales': np.random.normal(1000, 200, 50) + np.sin(np.arange(50) * 0.1) * 100,
                'temperature': np.random.normal(20, 5, 50),
                'humidity': np.random.uniform(30, 90, 50),
                'category': np.random.choice(['A', 'B', 'C'], 50)
            })
            
            # 添加异常值
            test_data.loc[45:47, 'sales'] = 5000
            
            self.csv_content = test_data.to_csv(index=False)
            
            self.temp_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
            self.temp_csv.write(self.csv_content)
            self.temp_csv.close()
    
    def tearDown(self):
        """清理资源"""
        if hasattr(self, 'temp_csv'):
            Path(self.temp_csv.name).unlink(missing_ok=True)
    
    @unittest.skipIf(not (HAS_PANDAS and HAS_NUMPY), "Pandas or NumPy not available")
    def test_complete_analysis_workflow(self):
        """测试完整的分析工作流"""
        async def run_analysis():
            # 1. 创建异步工作器
            worker = AsyncWorker(WorkflowConfig(max_concurrent_tasks=2))
            await worker.start()
            
            try:
                # 2. 数据加载任务
                def load_data_task():
                    file_info = FileInfo.create_from_file(self.temp_csv.name)
                    loader = DataLoader(LoaderConfig(use_polars=False))
                    df, quality_info = loader.load_file(file_info)
                    return df, quality_info, file_info
                
                load_task_id = await worker.submit_task(
                    name="load_data",
                    func=load_data_task,
                    priority=TaskPriority.HIGH
                )
                
                # 等待数据加载完成
                await asyncio.sleep(1.0)
                load_task = worker.get_task_status(load_task_id)
                self.assertEqual(load_task.status, TaskStatus.COMPLETED)
                
                df, quality_info, file_info = load_task.result
                
                # 3. 分析任务
                def analysis_task():
                    engine = AnalysisEngine(AnalysisConfig(
                        outlier_method="iqr",
                        correlation_method="pearson",
                        n_threads=1
                    ))
                    
                    return engine.analyze_dataset(
                        df,
                        time_column='timestamp',
                        exclude_columns=['category']
                    )
                
                analysis_task_id = await worker.submit_task(
                    name="data_analysis",
                    func=analysis_task,
                    priority=TaskPriority.NORMAL
                )
                
                # 等待分析完成
                await asyncio.sleep(3.0)
                analysis_task = worker.get_task_status(analysis_task_id)
                self.assertEqual(analysis_task.status, TaskStatus.COMPLETED)
                
                analysis_result = analysis_task.result
                
                # 4. 验证分析结果
                self.assertIsNotNone(analysis_result)
                self.assertIsNotNone(analysis_result.descriptive_stats)
                self.assertIsNotNone(analysis_result.correlation_matrix)
                self.assertIsNotNone(analysis_result.anomaly_detection)
                
                # 检查描述性统计
                desc_summary = analysis_result.descriptive_stats.get_summary()
                self.assertGreater(desc_summary['total_columns_analyzed'], 0)
                
                # 检查异常值检测
                anomaly_summary = analysis_result.anomaly_detection.get_summary()
                self.assertGreaterEqual(anomaly_summary['total_anomalies'], 1)  # 应该检测到sales的异常值
                
                # 检查关联分析
                corr_summary = analysis_result.correlation_matrix.get_summary()
                self.assertEqual(corr_summary['matrix_size'], 3)  # sales, temperature, humidity
                
                # 获取统计信息
                stats = worker.get_stats()
                
                print(f"✅ 完整分析工作流测试成功")
                print(f"   数据行数: {quality_info.total_rows}")
                print(f"   分析列数: {desc_summary['total_columns_analyzed']}")
                print(f"   检测到异常值: {anomaly_summary['total_anomalies']}个")
                print(f"   关联矩阵维度: {corr_summary['matrix_size']}x{corr_summary['matrix_size']}")
                print(f"   工作器统计:")
                print(f"     总任务数: {stats['total_tasks']}")
                print(f"     完成任务数: {stats['completed_tasks']}")
                print(f"     平均执行时间: {stats['average_execution_time']:.3f}s")
                
            finally:
                await worker.stop()
        
        asyncio.run(run_analysis())


if __name__ == "__main__":
    # 运行所有测试
    unittest.main(verbosity=2)