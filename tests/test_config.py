"""
测试配置和工具

为所有测试提供统一的配置、工具函数和测试数据
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加src目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

class TestConfig:
    """测试配置类"""
    
    # 测试数据目录
    TEST_DATA_DIR = project_root / "tests" / "test_data"
    TEMP_DATA_DIR = None
    
    # 测试文件路径
    CSV_FILE = None
    EXCEL_FILE = None
    PARQUET_FILE = None
    INVALID_FILE = None
    
    # 数据库测试配置
    TEST_DB_PATH = None
    
    @classmethod
    def setup(cls):
        """设置测试环境"""
        # 创建临时目录
        cls.TEMP_DATA_DIR = Path(tempfile.mkdtemp(prefix="data_analysis_test_"))
        
        # 创建测试数据目录
        cls.TEST_DATA_DIR.mkdir(exist_ok=True)
        
        # 生成测试文件
        cls._create_test_files()
        
        # 设置测试数据库路径
        cls.TEST_DB_PATH = cls.TEMP_DATA_DIR / "test_database.db"
        
        print(f"测试环境已设置，临时目录: {cls.TEMP_DATA_DIR}")
    
    @classmethod
    def teardown(cls):
        """清理测试环境"""
        if cls.TEMP_DATA_DIR and cls.TEMP_DATA_DIR.exists():
            shutil.rmtree(cls.TEMP_DATA_DIR)
            print(f"测试环境已清理")
    
    @classmethod
    def _create_test_files(cls):
        """创建测试数据文件"""
        try:
            # 生成测试数据
            dates = pd.date_range('2023-01-01', periods=1000, freq='H')
            np.random.seed(42)
            
            test_data = pd.DataFrame({
                'datetime': dates,
                'tagTime': dates.astype(str),
                'temperature': np.random.normal(25, 5, 1000),
                'humidity': np.random.normal(60, 10, 1000),
                'pressure': np.random.normal(1013, 20, 1000),
                'wind_speed': np.random.exponential(10, 1000),
                'category': np.random.choice(['A', 'B', 'C'], 1000),
                'value': np.random.randint(1, 100, 1000)
            })
            
            # 添加一些异常值
            test_data.loc[50:55, 'temperature'] = 100  # 异常高温
            test_data.loc[100:105, 'humidity'] = -10   # 异常湿度
            
            # 保存为不同格式
            cls.CSV_FILE = cls.TEMP_DATA_DIR / "test_data.csv"
            test_data.to_csv(cls.CSV_FILE, index=False)
            
            cls.EXCEL_FILE = cls.TEMP_DATA_DIR / "test_data.xlsx"
            test_data.to_excel(cls.EXCEL_FILE, index=False)
            
            # 如果有pyarrow，创建Parquet文件
            try:
                cls.PARQUET_FILE = cls.TEMP_DATA_DIR / "test_data.parquet"
                test_data.to_parquet(cls.PARQUET_FILE, index=False)
            except ImportError:
                print("警告: pyarrow未安装，跳过Parquet文件创建")
            
            # 创建无效文件
            cls.INVALID_FILE = cls.TEMP_DATA_DIR / "invalid.txt"
            with open(cls.INVALID_FILE, 'w') as f:
                f.write("这不是一个有效的数据文件")
            
            print(f"测试数据文件已创建")
            
        except Exception as e:
            print(f"创建测试文件失败: {e}")
    
    @classmethod
    def get_sample_data(cls) -> pd.DataFrame:
        """获取样本数据DataFrame"""
        if cls.CSV_FILE and cls.CSV_FILE.exists():
            return pd.read_csv(cls.CSV_FILE)
        else:
            # 如果文件不存在，生成简单的样本数据
            return pd.DataFrame({
                'datetime': pd.date_range('2023-01-01', periods=100, freq='H'),
                'value': np.random.normal(50, 10, 100),
                'category': np.random.choice(['A', 'B'], 100)
            })


class TestDataGenerator:
    """测试数据生成器"""
    
    @staticmethod
    def create_simple_dataframe(rows: int = 100) -> pd.DataFrame:
        """创建简单的测试DataFrame"""
        return pd.DataFrame({
            'id': range(rows),
            'value': np.random.normal(0, 1, rows),
            'category': np.random.choice(['A', 'B', 'C'], rows)
        })
    
    @staticmethod
    def create_time_series_dataframe(rows: int = 1000) -> pd.DataFrame:
        """创建时间序列测试DataFrame"""
        dates = pd.date_range('2023-01-01', periods=rows, freq='H')
        return pd.DataFrame({
            'datetime': dates,
            'value': np.random.normal(100, 15, rows) + np.sin(np.arange(rows) * 0.1) * 10,
            'secondary': np.random.exponential(5, rows)
        })
    
    @staticmethod
    def create_correlation_dataframe(rows: int = 500) -> pd.DataFrame:
        """创建有相关性的测试DataFrame"""
        x = np.random.normal(0, 1, rows)
        y = 2 * x + np.random.normal(0, 0.5, rows)  # 强相关
        z = np.random.normal(0, 1, rows)  # 无相关
        
        return pd.DataFrame({
            'x': x,
            'y': y,
            'z': z,
            'category': np.random.choice(['Type1', 'Type2'], rows)
        })


class TestAssertions:
    """测试断言工具"""
    
    @staticmethod
    def assert_dataframe_valid(df: pd.DataFrame, min_rows: int = 1):
        """断言DataFrame有效"""
        assert df is not None, "DataFrame不能为None"
        assert isinstance(df, pd.DataFrame), f"期望DataFrame，得到{type(df)}"
        assert len(df) >= min_rows, f"DataFrame行数{len(df)}少于最小要求{min_rows}"
        assert len(df.columns) > 0, "DataFrame必须有列"
    
    @staticmethod
    def assert_analysis_result_valid(result: Any):
        """断言分析结果有效"""
        assert result is not None, "分析结果不能为None"
        assert hasattr(result, 'descriptive_stats'), "分析结果必须包含描述性统计"
        assert hasattr(result, 'correlation_matrix'), "分析结果必须包含相关矩阵"
    
    @staticmethod
    def assert_file_exists(file_path: Path):
        """断言文件存在"""
        assert file_path.exists(), f"文件不存在: {file_path}"
        assert file_path.is_file(), f"路径不是文件: {file_path}"
        assert file_path.stat().st_size > 0, f"文件为空: {file_path}"
    
    @staticmethod
    def assert_performance_metrics(metrics: Dict[str, Any]):
        """断言性能指标有效"""
        required_keys = ['memory_usage_mb', 'processing_time', 'cpu_percent']
        for key in required_keys:
            assert key in metrics, f"性能指标缺少{key}"
            assert isinstance(metrics[key], (int, float)), f"{key}必须是数值"


class MockData:
    """模拟数据类"""
    
    @staticmethod
    def create_file_info():
        """创建模拟文件信息"""
        from src.models.file_info import FileInfo
        return FileInfo(
            file_path=str(TestConfig.CSV_FILE or "/tmp/test.csv"),
            file_name="test.csv",
            file_size=1024,
            file_type="csv",
            columns=['datetime', 'value', 'category'],
            row_count=100,
            has_time_column=True,
            time_column='datetime'
        )
    
    @staticmethod
    def create_analysis_config():
        """创建模拟分析配置"""
        return {
            'include_descriptive': True,
            'include_correlation': True,
            'include_outlier_detection': True,
            'include_stationarity': True,
            'outlier_method': 'zscore',
            'outlier_threshold': 3.0,
            'correlation_method': 'pearson'
        }


# 测试装饰器
def with_test_data(func):
    """测试数据装饰器"""
    def wrapper(*args, **kwargs):
        TestConfig.setup()
        try:
            return func(*args, **kwargs)
        finally:
            TestConfig.teardown()
    return wrapper


def skip_if_no_dependency(dependency_name: str):
    """如果依赖不存在则跳过测试的装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                __import__(dependency_name)
                return func(*args, **kwargs)
            except ImportError:
                print(f"跳过测试 {func.__name__}: {dependency_name} 未安装")
                return True  # 跳过测试视为通过
        return wrapper
    return decorator


# 全局测试设置
def setup_test_environment():
    """设置全局测试环境"""
    TestConfig.setup()
    
    # 设置日志级别
    import logging
    logging.basicConfig(level=logging.WARNING)  # 减少测试时的日志输出
    
    return TestConfig


def teardown_test_environment():
    """清理全局测试环境"""
    TestConfig.teardown()


if __name__ == "__main__":
    # 测试配置
    print("测试配置验证...")
    setup_test_environment()
    
    # 验证测试数据
    print(f"CSV文件: {TestConfig.CSV_FILE}")
    print(f"Excel文件: {TestConfig.EXCEL_FILE}")
    print(f"Parquet文件: {TestConfig.PARQUET_FILE}")
    
    # 验证数据生成器
    df = TestDataGenerator.create_simple_dataframe(50)
    TestAssertions.assert_dataframe_valid(df, min_rows=50)
    print("✅ 测试数据生成器正常")
    
    # 验证时间序列数据
    ts_df = TestDataGenerator.create_time_series_dataframe(100)
    TestAssertions.assert_dataframe_valid(ts_df, min_rows=100)
    print("✅ 时间序列数据生成正常")
    
    teardown_test_environment()
    print("✅ 测试配置验证完成")