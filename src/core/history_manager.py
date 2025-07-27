"""
历史记录管理器
负责分析结果的自动保存、加载和管理
"""

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from ..models.analysis_history import AnalysisHistoryDB, AnalysisHistoryRecord, AnalysisStatus
from ..models.extended_analysis_result import AnalysisResult
from ..core.analysis_engine import AnalysisConfig
from ..utils.basic_logging import LoggerMixin
from ..utils.exceptions import AnalysisError


class HistoryManager(LoggerMixin):
    """历史记录管理器"""
    
    def __init__(self, db_path: str = "data/analysis_history.db", results_dir: str = "data/results"):
        self.db_path = Path(db_path)
        self.results_dir = Path(results_dir)
        
        # 确保目录存在
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化数据库
        self.db = AnalysisHistoryDB(self.db_path)
        
        self.logger.info(f"HistoryManager初始化完成: db={self.db_path}, results={self.results_dir}")
    
    def create_record_from_data(
        self,
        file_path: str,
        analysis_config: AnalysisConfig,
        analysis_type: str = "comprehensive",
        time_column: Optional[str] = None
    ) -> AnalysisHistoryRecord:
        """从数据文件创建历史记录"""
        try:
            file_path_obj = Path(file_path)
            
            # 计算文件哈希
            file_hash = self._calculate_file_hash(file_path_obj)
            
            # 生成分析ID
            analysis_id = self._generate_analysis_id(file_hash, analysis_config, analysis_type, time_column)
            
            # 创建记录
            record = AnalysisHistoryRecord(
                analysis_id=analysis_id,
                file_name=file_path_obj.name,
                file_path=str(file_path_obj.absolute()),
                file_size=file_path_obj.stat().st_size if file_path_obj.exists() else 0,
                file_hash=file_hash,
                analysis_config=self._config_to_dict(analysis_config),
                analysis_type=analysis_type,
                time_column=time_column,
                status=AnalysisStatus.PENDING,
                created_at=datetime.now()
            )
            
            self.logger.info(f"创建历史记录: {analysis_id}")
            return record
            
        except Exception as e:
            self.logger.error(f"创建历史记录失败: {str(e)}")
            raise AnalysisError(f"创建历史记录失败: {str(e)}")
    
    def save_record(self, record: AnalysisHistoryRecord) -> AnalysisHistoryRecord:
        """保存历史记录"""
        return self.db.save_record(record)
    
    def start_analysis(self, record: AnalysisHistoryRecord) -> AnalysisHistoryRecord:
        """标记分析开始"""
        record.status = AnalysisStatus.RUNNING
        record.started_at = datetime.now()
        return self.save_record(record)
    
    def complete_analysis(
        self,
        record: AnalysisHistoryRecord,
        result: AnalysisResult,
        execution_time_ms: int
    ) -> AnalysisHistoryRecord:
        """标记分析完成并保存结果"""
        try:
            # 保存分析结果到文件
            result_file_path = self._save_analysis_result(record.analysis_id, result)
            
            # 更新记录
            record.status = AnalysisStatus.COMPLETED
            record.completed_at = datetime.now()
            record.execution_time_ms = execution_time_ms
            record.result_file_path = str(result_file_path)
            record.result_summary = result.get_overall_summary()
            
            # 保存图表文件路径（如果有）
            # TODO: 集成图表生成和保存
            
            saved_record = self.save_record(record)
            self.logger.info(f"分析完成，记录已保存: {record.analysis_id}")
            return saved_record
            
        except Exception as e:
            self.logger.error(f"保存分析结果失败: {str(e)}")
            # 标记为失败
            record.status = AnalysisStatus.FAILED
            record.error_message = str(e)
            record.completed_at = datetime.now()
            record.execution_time_ms = execution_time_ms
            return self.save_record(record)
    
    def fail_analysis(
        self,
        record: AnalysisHistoryRecord,
        error_message: str,
        execution_time_ms: Optional[int] = None
    ) -> AnalysisHistoryRecord:
        """标记分析失败"""
        record.status = AnalysisStatus.FAILED
        record.error_message = error_message
        record.completed_at = datetime.now()
        if execution_time_ms:
            record.execution_time_ms = execution_time_ms
        
        saved_record = self.save_record(record)
        self.logger.warning(f"分析失败: {record.analysis_id} - {error_message}")
        return saved_record
    
    def cancel_analysis(self, record: AnalysisHistoryRecord) -> AnalysisHistoryRecord:
        """取消分析"""
        record.status = AnalysisStatus.CANCELLED
        record.completed_at = datetime.now()
        
        saved_record = self.save_record(record)
        self.logger.info(f"分析已取消: {record.analysis_id}")
        return saved_record
    
    def load_analysis_result(self, record: AnalysisHistoryRecord) -> Optional[AnalysisResult]:
        """加载分析结果"""
        if not record.result_file_path:
            return None
        
        try:
            result_path = Path(record.result_file_path)
            if not result_path.exists():
                self.logger.warning(f"结果文件不存在: {result_path}")
                return None
            
            with open(result_path, 'rb') as f:
                result = pickle.load(f)
            
            self.logger.info(f"加载分析结果: {record.analysis_id}")
            return result
            
        except Exception as e:
            self.logger.error(f"加载分析结果失败: {str(e)}")
            return None
    
    def find_existing_analysis(
        self,
        file_path: str,
        analysis_config: AnalysisConfig,
        analysis_type: str = "comprehensive",
        time_column: Optional[str] = None
    ) -> Optional[AnalysisHistoryRecord]:
        """查找已存在的分析记录"""
        try:
            # 计算文件哈希
            file_hash = self._calculate_file_hash(Path(file_path))
            
            # 创建配置字符串用于匹配
            config_str = json.dumps({
                'correlation_method': analysis_config.correlation_method,
                'outlier_method': analysis_config.outlier_method,
                'outlier_threshold': analysis_config.outlier_threshold,
                'analysis_type': analysis_type,
                'time_column': time_column
            }, sort_keys=True)
            
            # 查询数据库中相同文件哈希和配置的已完成分析
            records = self.db.get_records(status=AnalysisStatus.COMPLETED, limit=100)
            
            for record in records:
                if (record.file_hash == file_hash and 
                    record.analysis_type == analysis_type and
                    record.time_column == time_column):
                    # 比较分析配置
                    if record.analysis_config:
                        record_config_str = json.dumps({
                            'correlation_method': record.analysis_config.get('correlation_method'),
                            'outlier_method': record.analysis_config.get('outlier_method'),
                            'outlier_threshold': record.analysis_config.get('outlier_threshold'),
                            'analysis_type': analysis_type,
                            'time_column': time_column
                        }, sort_keys=True)
                        
                        if config_str == record_config_str:
                            self.logger.info(f"找到已存在的分析: {record.analysis_id}")
                            return record
            
            return None
            
        except Exception as e:
            self.logger.error(f"查找已存在分析失败: {str(e)}")
            return None
    
    def get_recent_records(self, limit: int = 10) -> List[AnalysisHistoryRecord]:
        """获取最近的记录"""
        return self.db.get_records(limit=limit, order_by="created_at", order_desc=True)
    
    def get_completed_records(self, limit: int = 50) -> List[AnalysisHistoryRecord]:
        """获取已完成的记录"""
        return self.db.get_records(
            status=AnalysisStatus.COMPLETED,
            limit=limit,
            order_by="completed_at",
            order_desc=True
        )
    
    def search_records(self, search_text: str, limit: int = 50) -> List[AnalysisHistoryRecord]:
        """搜索记录"""
        return self.db.search_records(search_text, limit=limit)
    
    def delete_record(self, record_id: int) -> bool:
        """删除记录及其关联文件"""
        try:
            # 获取记录
            record = self.db.get_record_by_id(record_id)
            if not record:
                return False
            
            # 删除结果文件
            if record.result_file_path:
                result_path = Path(record.result_file_path)
                if result_path.exists():
                    result_path.unlink()
            
            # 删除图表文件
            for chart_file in record.chart_files:
                chart_path = Path(chart_file)
                if chart_path.exists():
                    chart_path.unlink()
            
            # 删除数据库记录
            success = self.db.delete_record(record_id)
            
            if success:
                self.logger.info(f"删除记录: {record.analysis_id}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"删除记录失败: {str(e)}")
            return False
    
    def export_record(self, record: AnalysisHistoryRecord, export_path: str) -> bool:
        """导出记录和结果"""
        try:
            export_path_obj = Path(export_path)
            export_path_obj.mkdir(parents=True, exist_ok=True)
            
            # 导出记录信息
            record_file = export_path_obj / f"{record.analysis_id}_record.json"
            with open(record_file, 'w', encoding='utf-8') as f:
                json.dump(record.to_dict(), f, ensure_ascii=False, indent=2)
            
            # 复制结果文件
            if record.result_file_path:
                result_path = Path(record.result_file_path)
                if result_path.exists():
                    target_result = export_path_obj / f"{record.analysis_id}_result.pkl"
                    import shutil
                    shutil.copy2(result_path, target_result)
            
            # 复制图表文件
            for chart_file in record.chart_files:
                chart_path = Path(chart_file)
                if chart_path.exists():
                    target_chart = export_path_obj / chart_path.name
                    import shutil
                    shutil.copy2(chart_path, target_chart)
            
            self.logger.info(f"导出记录: {record.analysis_id} 到 {export_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"导出记录失败: {str(e)}")
            return False
    
    def cleanup_old_records(self, days: int = 30) -> int:
        """清理旧记录"""
        try:
            # 获取要删除的记录
            cutoff_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            cutoff_date = cutoff_date.replace(day=cutoff_date.day - days)
            
            # 清理文件
            deleted_files = 0
            for result_file in self.results_dir.glob("*.pkl"):
                if result_file.stat().st_mtime < cutoff_date.timestamp():
                    result_file.unlink()
                    deleted_files += 1
            
            # 清理数据库记录
            deleted_records = self.db.cleanup_old_records(days)
            
            self.logger.info(f"清理完成: {deleted_records} 条记录, {deleted_files} 个文件")
            return deleted_records
            
        except Exception as e:
            self.logger.error(f"清理旧记录失败: {str(e)}")
            return 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self.db.get_statistics()
        
        # 添加文件统计
        result_files = list(self.results_dir.glob("*.pkl"))
        stats['result_files_count'] = len(result_files)
        
        total_size = sum(f.stat().st_size for f in result_files)
        stats['total_result_size_mb'] = total_size / (1024 * 1024)
        
        return stats
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """计算文件哈希"""
        if not file_path.exists():
            return ""
        
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            # 只读取文件的前1MB来计算哈希，提高性能
            chunk = f.read(1024 * 1024)
            hash_md5.update(chunk)
        
        return hash_md5.hexdigest()
    
    def _generate_analysis_id(
        self,
        file_hash: str,
        analysis_config: AnalysisConfig,
        analysis_type: str,
        time_column: Optional[str]
    ) -> str:
        """生成分析ID"""
        # 创建配置字符串
        config_str = json.dumps({
            'correlation_method': analysis_config.correlation_method,
            'outlier_method': analysis_config.outlier_method,
            'outlier_threshold': analysis_config.outlier_threshold,
            'analysis_type': analysis_type,
            'time_column': time_column
        }, sort_keys=True)
        
        # 添加时间戳确保唯一性，避免重复分析时的ID冲突
        timestamp = datetime.now().isoformat()
        
        # 组合文件哈希、配置和时间戳
        combined = f"{file_hash}:{config_str}:{timestamp}"
        
        # 生成SHA256哈希
        analysis_hash = hashlib.sha256(combined.encode()).hexdigest()
        
        return analysis_hash
    
    def _config_to_dict(self, config: AnalysisConfig) -> Dict[str, Any]:
        """将分析配置转换为字典"""
        return {
            'correlation_method': config.correlation_method,
            'outlier_method': config.outlier_method,
            'outlier_threshold': config.outlier_threshold,
            'iqr_multiplier': config.iqr_multiplier,
            'min_correlation_threshold': config.min_correlation_threshold,
            'stationarity_tests': config.stationarity_tests,
            'max_lags': config.max_lags,
            'alpha': config.alpha,
            'n_threads': config.n_threads,
            'enable_parallel': config.enable_parallel
        }
    
    def _save_analysis_result(self, analysis_id: str, result: AnalysisResult) -> Path:
        """保存分析结果到文件"""
        result_file = self.results_dir / f"{analysis_id}_result.pkl"
        
        with open(result_file, 'wb') as f:
            pickle.dump(result, f)
        
        return result_file


# 全局历史管理器实例
_history_manager: Optional[HistoryManager] = None


def get_history_manager() -> HistoryManager:
    """获取全局历史管理器实例"""
    global _history_manager
    if _history_manager is None:
        _history_manager = HistoryManager()
    return _history_manager


def set_history_manager(manager: HistoryManager):
    """设置全局历史管理器实例"""
    global _history_manager
    _history_manager = manager