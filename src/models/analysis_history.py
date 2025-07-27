"""
分析历史数据模型
定义分析历史记录的数据结构和数据库操作
"""

import sqlite3
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from enum import Enum

from ..utils.basic_logging import LoggerMixin
from ..utils.exceptions import AnalysisError


class AnalysisStatus(str, Enum):
    """分析状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AnalysisHistoryRecord:
    """分析历史记录"""
    # 基本信息
    id: Optional[int] = None
    analysis_id: str = ""
    file_name: str = ""
    file_path: str = ""
    file_size: int = 0
    file_hash: str = ""
    
    # 分析配置
    analysis_config: Dict[str, Any] = None
    analysis_type: str = "comprehensive"  # comprehensive, descriptive, correlation, anomaly, timeseries
    time_column: Optional[str] = None
    
    # 分析结果元数据
    status: AnalysisStatus = AnalysisStatus.PENDING
    error_message: Optional[str] = None
    execution_time_ms: Optional[int] = None
    result_summary: Optional[Dict[str, Any]] = None
    
    # 时间戳
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # 结果存储路径
    result_file_path: Optional[str] = None
    chart_files: List[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.analysis_config is None:
            self.analysis_config = {}
        if self.chart_files is None:
            self.chart_files = []
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        data = asdict(self)
        # 转换datetime为字符串
        if data['created_at']:
            data['created_at'] = data['created_at'].isoformat()
        if data['started_at']:
            data['started_at'] = data['started_at'].isoformat()
        if data['completed_at']:
            data['completed_at'] = data['completed_at'].isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnalysisHistoryRecord":
        """从字典创建实例"""
        # 转换字符串为datetime
        if data.get('created_at'):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if data.get('started_at'):
            data['started_at'] = datetime.fromisoformat(data['started_at'])
        if data.get('completed_at'):
            data['completed_at'] = datetime.fromisoformat(data['completed_at'])
        
        return cls(**data)
    
    def get_duration_text(self) -> str:
        """获取执行时长文本"""
        if self.execution_time_ms:
            if self.execution_time_ms < 1000:
                return f"{self.execution_time_ms}ms"
            elif self.execution_time_ms < 60000:
                return f"{self.execution_time_ms / 1000:.1f}s"
            else:
                return f"{self.execution_time_ms / 60000:.1f}min"
        return "未知"
    
    def get_status_text(self) -> str:
        """获取状态文本"""
        status_map = {
            AnalysisStatus.PENDING: "等待中",
            AnalysisStatus.RUNNING: "运行中",
            AnalysisStatus.COMPLETED: "已完成",
            AnalysisStatus.FAILED: "失败",
            AnalysisStatus.CANCELLED: "已取消"
        }
        return status_map.get(self.status, "未知")
    
    def get_file_size_text(self) -> str:
        """获取文件大小文本"""
        if self.file_size < 1024:
            return f"{self.file_size}B"
        elif self.file_size < 1024 * 1024:
            return f"{self.file_size / 1024:.1f}KB"
        elif self.file_size < 1024 * 1024 * 1024:
            return f"{self.file_size / (1024 * 1024):.1f}MB"
        else:
            return f"{self.file_size / (1024 * 1024 * 1024):.1f}GB"


class AnalysisHistoryDB(LoggerMixin):
    """分析历史数据库管理"""
    
    def __init__(self, db_path: Union[str, Path]):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        self.logger.info(f"AnalysisHistoryDB初始化完成: {self.db_path}")
    
    def _init_database(self):
        """初始化数据库表结构"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 创建分析历史表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS analysis_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        analysis_id TEXT UNIQUE NOT NULL,
                        file_name TEXT NOT NULL,
                        file_path TEXT NOT NULL,
                        file_size INTEGER NOT NULL,
                        file_hash TEXT NOT NULL,
                        analysis_config TEXT NOT NULL,
                        analysis_type TEXT NOT NULL DEFAULT 'comprehensive',
                        time_column TEXT,
                        status TEXT NOT NULL DEFAULT 'pending',
                        error_message TEXT,
                        execution_time_ms INTEGER,
                        result_summary TEXT,
                        created_at TEXT NOT NULL,
                        started_at TEXT,
                        completed_at TEXT,
                        result_file_path TEXT,
                        chart_files TEXT
                    )
                ''')
                
                # 创建索引
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_analysis_id ON analysis_history(analysis_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_file_hash ON analysis_history(file_hash)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_status ON analysis_history(status)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON analysis_history(created_at)')
                
                conn.commit()
                self.logger.info("数据库表结构初始化完成")
                
        except Exception as e:
            self.logger.error(f"数据库初始化失败: {str(e)}")
            raise AnalysisError(f"数据库初始化失败: {str(e)}")
    
    def save_record(self, record: AnalysisHistoryRecord) -> AnalysisHistoryRecord:
        """保存分析记录"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 序列化复杂字段
                analysis_config_json = json.dumps(record.analysis_config)
                result_summary_json = json.dumps(record.result_summary) if record.result_summary else None
                chart_files_json = json.dumps(record.chart_files)
                
                if record.id is None:
                    # 插入新记录
                    cursor.execute('''
                        INSERT INTO analysis_history (
                            analysis_id, file_name, file_path, file_size, file_hash,
                            analysis_config, analysis_type, time_column,
                            status, error_message, execution_time_ms, result_summary,
                            created_at, started_at, completed_at,
                            result_file_path, chart_files
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        record.analysis_id, record.file_name, record.file_path,
                        record.file_size, record.file_hash, analysis_config_json,
                        record.analysis_type, record.time_column,
                        record.status.value, record.error_message, record.execution_time_ms,
                        result_summary_json,
                        record.created_at.isoformat() if record.created_at else None,
                        record.started_at.isoformat() if record.started_at else None,
                        record.completed_at.isoformat() if record.completed_at else None,
                        record.result_file_path, chart_files_json
                    ))
                    record.id = cursor.lastrowid
                else:
                    # 更新现有记录
                    cursor.execute('''
                        UPDATE analysis_history SET
                            file_name=?, file_path=?, file_size=?, file_hash=?,
                            analysis_config=?, analysis_type=?, time_column=?,
                            status=?, error_message=?, execution_time_ms=?, result_summary=?,
                            started_at=?, completed_at=?, result_file_path=?, chart_files=?
                        WHERE id=?
                    ''', (
                        record.file_name, record.file_path, record.file_size, record.file_hash,
                        analysis_config_json, record.analysis_type, record.time_column,
                        record.status.value, record.error_message, record.execution_time_ms,
                        result_summary_json,
                        record.started_at.isoformat() if record.started_at else None,
                        record.completed_at.isoformat() if record.completed_at else None,
                        record.result_file_path, chart_files_json, record.id
                    ))
                
                conn.commit()
                self.logger.info(f"分析记录已保存: {record.analysis_id}")
                return record
                
        except Exception as e:
            self.logger.error(f"保存分析记录失败: {str(e)}")
            raise AnalysisError(f"保存分析记录失败: {str(e)}")
    
    def get_record_by_id(self, record_id: int) -> Optional[AnalysisHistoryRecord]:
        """根据ID获取记录"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM analysis_history WHERE id = ?', (record_id,))
                row = cursor.fetchone()
                
                if row:
                    return self._row_to_record(row)
                return None
                
        except Exception as e:
            self.logger.error(f"获取分析记录失败: {str(e)}")
            return None
    
    def get_record_by_analysis_id(self, analysis_id: str) -> Optional[AnalysisHistoryRecord]:
        """根据分析ID获取记录"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM analysis_history WHERE analysis_id = ?', (analysis_id,))
                row = cursor.fetchone()
                
                if row:
                    return self._row_to_record(row)
                return None
                
        except Exception as e:
            self.logger.error(f"获取分析记录失败: {str(e)}")
            return None
    
    def get_records(
        self,
        status: Optional[AnalysisStatus] = None,
        limit: int = 100,
        offset: int = 0,
        order_by: str = "created_at",
        order_desc: bool = True
    ) -> List[AnalysisHistoryRecord]:
        """获取分析记录列表"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 构建查询
                query = "SELECT * FROM analysis_history"
                params = []
                
                if status:
                    query += " WHERE status = ?"
                    params.append(status.value)
                
                query += f" ORDER BY {order_by}"
                if order_desc:
                    query += " DESC"
                
                query += " LIMIT ? OFFSET ?"
                params.extend([limit, offset])
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                return [self._row_to_record(row) for row in rows]
                
        except Exception as e:
            self.logger.error(f"获取分析记录列表失败: {str(e)}")
            return []
    
    def search_records(
        self,
        search_text: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[AnalysisHistoryRecord]:
        """搜索分析记录"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                query = '''
                    SELECT * FROM analysis_history 
                    WHERE file_name LIKE ? OR file_path LIKE ? OR analysis_id LIKE ?
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                '''
                
                search_pattern = f"%{search_text}%"
                params = [search_pattern, search_pattern, search_pattern, limit, offset]
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                return [self._row_to_record(row) for row in rows]
                
        except Exception as e:
            self.logger.error(f"搜索分析记录失败: {str(e)}")
            return []
    
    def delete_record(self, record_id: int) -> bool:
        """删除分析记录"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM analysis_history WHERE id = ?', (record_id,))
                
                if cursor.rowcount > 0:
                    conn.commit()
                    self.logger.info(f"分析记录已删除: {record_id}")
                    return True
                return False
                
        except Exception as e:
            self.logger.error(f"删除分析记录失败: {str(e)}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 总记录数
                cursor.execute('SELECT COUNT(*) FROM analysis_history')
                total_count = cursor.fetchone()[0]
                
                # 按状态统计
                cursor.execute('''
                    SELECT status, COUNT(*) 
                    FROM analysis_history 
                    GROUP BY status
                ''')
                status_counts = dict(cursor.fetchall())
                
                # 最近分析时间
                cursor.execute('SELECT MAX(created_at) FROM analysis_history')
                last_analysis = cursor.fetchone()[0]
                
                # 平均执行时间
                cursor.execute('''
                    SELECT AVG(execution_time_ms) 
                    FROM analysis_history 
                    WHERE execution_time_ms IS NOT NULL
                ''')
                avg_execution_time = cursor.fetchone()[0]
                
                return {
                    'total_count': total_count,
                    'status_counts': status_counts,
                    'last_analysis': last_analysis,
                    'avg_execution_time_ms': avg_execution_time
                }
                
        except Exception as e:
            self.logger.error(f"获取统计信息失败: {str(e)}")
            return {}
    
    def _row_to_record(self, row) -> AnalysisHistoryRecord:
        """将数据库行转换为记录对象"""
        (
            id, analysis_id, file_name, file_path, file_size, file_hash,
            analysis_config_json, analysis_type, time_column,
            status, error_message, execution_time_ms, result_summary_json,
            created_at_str, started_at_str, completed_at_str,
            result_file_path, chart_files_json
        ) = row
        
        # 反序列化JSON字段
        analysis_config = json.loads(analysis_config_json) if analysis_config_json else {}
        result_summary = json.loads(result_summary_json) if result_summary_json else None
        chart_files = json.loads(chart_files_json) if chart_files_json else []
        
        # 转换时间字段
        created_at = datetime.fromisoformat(created_at_str) if created_at_str else None
        started_at = datetime.fromisoformat(started_at_str) if started_at_str else None
        completed_at = datetime.fromisoformat(completed_at_str) if completed_at_str else None
        
        return AnalysisHistoryRecord(
            id=id,
            analysis_id=analysis_id,
            file_name=file_name,
            file_path=file_path,
            file_size=file_size,
            file_hash=file_hash,
            analysis_config=analysis_config,
            analysis_type=analysis_type,
            time_column=time_column,
            status=AnalysisStatus(status),
            error_message=error_message,
            execution_time_ms=execution_time_ms,
            result_summary=result_summary,
            created_at=created_at,
            started_at=started_at,
            completed_at=completed_at,
            result_file_path=result_file_path,
            chart_files=chart_files
        )
    
    def cleanup_old_records(self, days: int = 30) -> int:
        """清理旧记录"""
        try:
            cutoff_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            cutoff_date = cutoff_date.replace(day=cutoff_date.day - days)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    'DELETE FROM analysis_history WHERE created_at < ?',
                    (cutoff_date.isoformat(),)
                )
                
                deleted_count = cursor.rowcount
                conn.commit()
                
                self.logger.info(f"清理了 {deleted_count} 条旧记录")
                return deleted_count
                
        except Exception as e:
            self.logger.error(f"清理旧记录失败: {str(e)}")
            return 0