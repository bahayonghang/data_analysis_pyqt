"""
测试历史管理功能
"""

import os
import tempfile
import pytest
from pathlib import Path
from datetime import datetime

from src.models.analysis_history import AnalysisHistoryDB, AnalysisHistoryRecord, AnalysisStatus
from src.core.history_manager import HistoryManager
from src.core.analysis_engine import AnalysisConfig


class MockResult:
    """模拟分析结果"""
    def get_overall_summary(self):
        return {
            'total_records': 3,
            'numeric_columns': 2,
            'analysis_completed': True
        }


class TestHistoryManagement:
    """历史管理测试类"""
    
    def setup_method(self):
        """测试前设置"""
        # 创建临时目录和文件
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_history.db")
        self.results_dir = os.path.join(self.temp_dir, "results")
        
        # 初始化历史管理器
        self.history_manager = HistoryManager(
            db_path=self.db_path,
            results_dir=self.results_dir
        )
        
        # 创建测试数据
        self.test_file_path = os.path.join(self.temp_dir, "test_data.csv")
        with open(self.test_file_path, 'w') as f:
            f.write("name,age,score\nAlice,25,85\nBob,30,92\nCharlie,35,78\n")
        
        self.test_config = AnalysisConfig(
            correlation_method="pearson",
            outlier_method="iqr",
            outlier_threshold=3.0
        )
    
    def teardown_method(self):
        """测试后清理"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_database_initialization(self):
        """测试数据库初始化"""
        assert os.path.exists(self.db_path)
        assert self.history_manager.db is not None
        
        # 检查表是否创建
        import sqlite3
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='analysis_history'")
            table_exists = cursor.fetchone() is not None
            assert table_exists
    
    def test_create_record_from_data(self):
        """测试从数据创建记录"""
        record = self.history_manager.create_record_from_data(
            self.test_file_path,
            self.test_config,
            analysis_type="comprehensive",
            time_column=None
        )
        
        assert record is not None
        assert record.file_name == "test_data.csv"
        assert record.file_path == str(Path(self.test_file_path).absolute())
        assert record.analysis_type == "comprehensive"
        assert record.status == AnalysisStatus.PENDING
        assert record.analysis_id != ""
        assert record.file_hash != ""
    
    def test_save_and_retrieve_record(self):
        """测试保存和检索记录"""
        # 创建记录
        record = self.history_manager.create_record_from_data(
            self.test_file_path,
            self.test_config
        )
        
        # 保存记录
        saved_record = self.history_manager.save_record(record)
        assert saved_record.id is not None
        
        # 通过ID检索
        retrieved_record = self.history_manager.db.get_record_by_id(saved_record.id)
        assert retrieved_record is not None
        assert retrieved_record.analysis_id == record.analysis_id
        assert retrieved_record.file_name == record.file_name
        
        # 通过分析ID检索
        retrieved_by_analysis_id = self.history_manager.db.get_record_by_analysis_id(record.analysis_id)
        assert retrieved_by_analysis_id is not None
        assert retrieved_by_analysis_id.id == saved_record.id
    
    def test_analysis_lifecycle(self):
        """测试分析生命周期"""
        # 创建记录（使用不同的分析类型以确保唯一性）
        record = self.history_manager.create_record_from_data(
            self.test_file_path,
            self.test_config,
            analysis_type="lifecycle_test"
        )
        
        # 保存记录
        record = self.history_manager.save_record(record)
        assert record.status == AnalysisStatus.PENDING
        
        # 开始分析
        record = self.history_manager.start_analysis(record)
        assert record.status == AnalysisStatus.RUNNING
        assert record.started_at is not None
        
        # 模拟分析完成
        from src.models.extended_analysis_result import AnalysisResult
        mock_result = self._create_mock_result()
        
        record = self.history_manager.complete_analysis(
            record,
            mock_result,
            execution_time_ms=1500
        )
        
        assert record.status == AnalysisStatus.COMPLETED
        assert record.completed_at is not None
        assert record.execution_time_ms == 1500
        assert record.result_file_path is not None
        assert os.path.exists(record.result_file_path)
    
    def test_analysis_failure(self):
        """测试分析失败"""
        # 创建记录
        record = self.history_manager.create_record_from_data(
            self.test_file_path,
            self.test_config,
            analysis_type="failure_test"
        )
        
        # 保存并开始分析
        record = self.history_manager.save_record(record)
        record = self.history_manager.start_analysis(record)
        
        # 标记失败
        error_message = "模拟分析失败"
        record = self.history_manager.fail_analysis(
            record,
            error_message,
            execution_time_ms=500
        )
        
        assert record.status == AnalysisStatus.FAILED
        assert record.error_message == error_message
        assert record.execution_time_ms == 500
        assert record.completed_at is not None
    
    def test_analysis_cancellation(self):
        """测试分析取消"""
        # 创建记录
        record = self.history_manager.create_record_from_data(
            self.test_file_path,
            self.test_config,
            analysis_type="cancellation_test"
        )
        
        # 保存并开始分析
        record = self.history_manager.save_record(record)
        record = self.history_manager.start_analysis(record)
        
        # 取消分析
        record = self.history_manager.cancel_analysis(record)
        
        assert record.status == AnalysisStatus.CANCELLED
        assert record.completed_at is not None
    
    def test_find_existing_analysis(self):
        """测试查找已存在的分析"""
        # 创建特定配置用于测试查找
        test_config = AnalysisConfig(
            correlation_method="spearman",  # 使用不同的配置
            outlier_method="zscore",
            outlier_threshold=2.5
        )
        
        # 创建并完成分析
        record = self.history_manager.create_record_from_data(
            self.test_file_path,
            test_config,
            analysis_type="existing_test"
        )
        
        record = self.history_manager.save_record(record)
        record = self.history_manager.start_analysis(record)
        
        mock_result = self._create_mock_result()
        record = self.history_manager.complete_analysis(
            record,
            mock_result,
            execution_time_ms=1000
        )
        
        # 查找相同配置的分析
        existing_record = self.history_manager.find_existing_analysis(
            self.test_file_path,
            test_config,  # 使用相同的配置
            "existing_test"  # 添加分析类型
        )
        
        assert existing_record is not None
        assert existing_record.analysis_id == record.analysis_id
        assert existing_record.status == AnalysisStatus.COMPLETED
    
    def test_load_analysis_result(self):
        """测试加载分析结果"""
        # 创建并完成分析
        record = self.history_manager.create_record_from_data(
            self.test_file_path,
            self.test_config,
            analysis_type="load_test"
        )
        
        record = self.history_manager.save_record(record)
        record = self.history_manager.start_analysis(record)
        
        mock_result = self._create_mock_result()
        record = self.history_manager.complete_analysis(
            record,
            mock_result,
            execution_time_ms=1000
        )
        
        # 加载结果
        loaded_result = self.history_manager.load_analysis_result(record)
        assert loaded_result is not None
        assert loaded_result.get_overall_summary() == mock_result.get_overall_summary()
    
    def test_get_recent_records(self):
        """测试获取最近记录"""
        # 创建多个记录
        records = []
        for i in range(5):
            record = self.history_manager.create_record_from_data(
                self.test_file_path,
                self.test_config,
                analysis_type=f"test_{i}"
            )
            record = self.history_manager.save_record(record)
            records.append(record)
        
        # 获取最近记录
        recent_records = self.history_manager.get_recent_records(limit=3)
        assert len(recent_records) == 3
        
        # 检查顺序（最新的在前）
        for i in range(len(recent_records) - 1):
            assert recent_records[i].created_at >= recent_records[i + 1].created_at
    
    def test_delete_record(self):
        """测试删除记录"""
        # 创建并完成分析
        record = self.history_manager.create_record_from_data(
            self.test_file_path,
            self.test_config
        )
        
        record = self.history_manager.save_record(record)
        record = self.history_manager.start_analysis(record)
        
        mock_result = self._create_mock_result()
        record = self.history_manager.complete_analysis(
            record,
            mock_result,
            execution_time_ms=1000
        )
        
        # 删除记录
        success = self.history_manager.delete_record(record.id)
        assert success
        
        # 检查记录已删除
        retrieved_record = self.history_manager.db.get_record_by_id(record.id)
        assert retrieved_record is None
        
        # 检查结果文件已删除
        if record.result_file_path:
            assert not os.path.exists(record.result_file_path)
    
    def test_get_statistics(self):
        """测试获取统计信息"""
        # 创建多个不同状态的记录
        records = []
        
        # 完成的记录
        for i in range(3):
            record = self.history_manager.create_record_from_data(
                self.test_file_path,
                self.test_config,
                analysis_type=f"completed_{i}"
            )
            record = self.history_manager.save_record(record)
            record = self.history_manager.start_analysis(record)
            
            mock_result = self._create_mock_result()
            record = self.history_manager.complete_analysis(
                record,
                mock_result,
                execution_time_ms=1000 + i * 500
            )
            records.append(record)
        
        # 失败的记录
        record = self.history_manager.create_record_from_data(
            self.test_file_path,
            self.test_config,
            analysis_type="failed"
        )
        record = self.history_manager.save_record(record)
        record = self.history_manager.start_analysis(record)
        record = self.history_manager.fail_analysis(record, "test error")
        records.append(record)
        
        # 获取统计信息
        stats = self.history_manager.get_statistics()
        
        assert stats['total_count'] == 4
        assert stats['result_files_count'] == 3  # 只有完成的有结果文件
        assert 'total_result_size_mb' in stats
    
    def _create_mock_result(self):
        """创建模拟分析结果"""
        return MockResult()


if __name__ == "__main__":
    # 运行单个测试
    test = TestHistoryManagement()
    test.setup_method()
    
    try:
        print("测试数据库初始化...")
        test.test_database_initialization()
        print("✓ 数据库初始化测试通过")
        
        print("测试创建记录...")
        test.test_create_record_from_data()
        print("✓ 创建记录测试通过")
        
        print("测试保存和检索记录...")
        test.test_save_and_retrieve_record()
        print("✓ 保存和检索记录测试通过")
        
        print("测试分析生命周期...")
        test.test_analysis_lifecycle()
        print("✓ 分析生命周期测试通过")
        
        print("测试分析失败...")
        test.test_analysis_failure()
        print("✓ 分析失败测试通过")
        
        print("测试查找已存在分析...")
        test.test_find_existing_analysis()
        print("✓ 查找已存在分析测试通过")
        
        print("\n✅ 所有历史管理功能测试通过！")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        test.teardown_method()