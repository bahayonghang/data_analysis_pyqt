# -*- coding: utf-8 -*-
"""
分析结果管理器

统一管理分析结果的存储、检索和生命周期，解决数据流混乱问题
"""

import threading
from typing import Dict, Optional, Any
from pathlib import Path
from datetime import datetime, timedelta

from ..models.extended_analysis_result import AnalysisResult
from ..models.file_info import FileInfo
from ..utils.basic_logging import LoggerMixin


class AnalysisResultManager(LoggerMixin):
    """
    分析结果管理器
    
    使用单例模式确保全局唯一，负责：
    1. 分析结果的存储和检索
    2. 内存管理和生命周期控制
    3. 结果的有效性验证
    4. 线程安全的访问控制
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """单例模式实现"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化管理器"""
        if hasattr(self, '_initialized'):
            return
            
        super().__init__()
        
        # 存储分析结果的字典，key为文件路径，value为结果信息
        self._results: Dict[str, Dict[str, Any]] = {}
        
        # 访问锁，确保线程安全
        self._access_lock = threading.RLock()
        
        # 配置参数
        self._max_results = 10  # 最大缓存结果数量
        self._max_age_hours = 24  # 结果最大保存时间（小时）
        
        self._initialized = True
        self.logger.info("分析结果管理器初始化完成")
    
    def store_result(self, file_path: str, result: AnalysisResult, file_info: FileInfo = None) -> bool:
        """
        存储分析结果
        
        Args:
            file_path: 文件路径，作为唯一标识
            result: 分析结果对象
            file_info: 文件信息对象（可选）
            
        Returns:
            bool: 存储是否成功
        """
        try:
            with self._access_lock:
                # 标准化文件路径
                normalized_path = str(Path(file_path).resolve())
                
                # 清理过期结果
                self._cleanup_expired_results()
                
                # 如果超过最大数量，删除最旧的结果
                if len(self._results) >= self._max_results:
                    self._remove_oldest_result()
                
                # 存储结果
                self._results[normalized_path] = {
                    'result': result,
                    'file_info': file_info,
                    'timestamp': datetime.now(),
                    'access_count': 0
                }
                
                self.logger.info(f"分析结果已存储: {normalized_path}")
                return True
                
        except Exception as e:
            self.logger.error(f"存储分析结果失败: {str(e)}")
            return False
    
    def get_result(self, file_path: str) -> Optional[AnalysisResult]:
        """
        获取分析结果
        
        Args:
            file_path: 文件路径
            
        Returns:
            AnalysisResult: 分析结果对象，如果不存在则返回None
        """
        try:
            with self._access_lock:
                normalized_path = str(Path(file_path).resolve())
                
                if normalized_path not in self._results:
                    return None
                
                result_info = self._results[normalized_path]
                
                # 检查结果是否过期
                if self._is_expired(result_info['timestamp']):
                    del self._results[normalized_path]
                    self.logger.info(f"删除过期结果: {normalized_path}")
                    return None
                
                # 更新访问计数
                result_info['access_count'] += 1
                
                self.logger.debug(f"获取分析结果: {normalized_path}")
                return result_info['result']
                
        except Exception as e:
            self.logger.error(f"获取分析结果失败: {str(e)}")
            return None
    
    def get_file_info(self, file_path: str) -> Optional[FileInfo]:
        """
        获取文件信息
        
        Args:
            file_path: 文件路径
            
        Returns:
            FileInfo: 文件信息对象，如果不存在则返回None
        """
        try:
            with self._access_lock:
                normalized_path = str(Path(file_path).resolve())
                
                if normalized_path not in self._results:
                    return None
                
                result_info = self._results[normalized_path]
                
                # 检查结果是否过期
                if self._is_expired(result_info['timestamp']):
                    del self._results[normalized_path]
                    return None
                
                return result_info['file_info']
                
        except Exception as e:
            self.logger.error(f"获取文件信息失败: {str(e)}")
            return None
    
    def has_result(self, file_path: str) -> bool:
        """
        检查是否存在分析结果
        
        Args:
            file_path: 文件路径
            
        Returns:
            bool: 是否存在有效的分析结果
        """
        return self.get_result(file_path) is not None
    
    def clear_result(self, file_path: str) -> bool:
        """
        清除指定文件的分析结果
        
        Args:
            file_path: 文件路径
            
        Returns:
            bool: 清除是否成功
        """
        try:
            with self._access_lock:
                normalized_path = str(Path(file_path).resolve())
                
                if normalized_path in self._results:
                    del self._results[normalized_path]
                    self.logger.info(f"清除分析结果: {normalized_path}")
                    return True
                
                return False
                
        except Exception as e:
            self.logger.error(f"清除分析结果失败: {str(e)}")
            return False
    
    def clear_all_results(self) -> bool:
        """
        清除所有分析结果
        
        Returns:
            bool: 清除是否成功
        """
        try:
            with self._access_lock:
                count = len(self._results)
                self._results.clear()
                self.logger.info(f"清除所有分析结果，共 {count} 个")
                return True
                
        except Exception as e:
            self.logger.error(f"清除所有分析结果失败: {str(e)}")
            return False
    
    def get_result_info(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        获取结果的详细信息
        
        Args:
            file_path: 文件路径
            
        Returns:
            dict: 包含时间戳、访问次数等信息的字典
        """
        try:
            with self._access_lock:
                normalized_path = str(Path(file_path).resolve())
                
                if normalized_path not in self._results:
                    return None
                
                result_info = self._results[normalized_path]
                
                return {
                    'timestamp': result_info['timestamp'],
                    'access_count': result_info['access_count'],
                    'age_minutes': (datetime.now() - result_info['timestamp']).total_seconds() / 60,
                    'has_file_info': result_info['file_info'] is not None
                }
                
        except Exception as e:
            self.logger.error(f"获取结果信息失败: {str(e)}")
            return None
    
    def get_all_results_summary(self) -> Dict[str, Any]:
        """
        获取所有结果的摘要信息
        
        Returns:
            dict: 包含总数、最旧结果等信息的摘要
        """
        try:
            with self._access_lock:
                if not self._results:
                    return {
                        'total_count': 0,
                        'oldest_timestamp': None,
                        'newest_timestamp': None,
                        'total_access_count': 0
                    }
                
                timestamps = [info['timestamp'] for info in self._results.values()]
                access_counts = [info['access_count'] for info in self._results.values()]
                
                return {
                    'total_count': len(self._results),
                    'oldest_timestamp': min(timestamps),
                    'newest_timestamp': max(timestamps),
                    'total_access_count': sum(access_counts)
                }
                
        except Exception as e:
            self.logger.error(f"获取结果摘要失败: {str(e)}")
            return {}
    
    def _cleanup_expired_results(self):
        """
        清理过期的分析结果
        """
        try:
            expired_paths = []
            
            for path, result_info in self._results.items():
                if self._is_expired(result_info['timestamp']):
                    expired_paths.append(path)
            
            for path in expired_paths:
                del self._results[path]
                self.logger.debug(f"删除过期结果: {path}")
            
            if expired_paths:
                self.logger.info(f"清理过期结果 {len(expired_paths)} 个")
                
        except Exception as e:
            self.logger.error(f"清理过期结果失败: {str(e)}")
    
    def _remove_oldest_result(self):
        """
        删除最旧的分析结果
        """
        try:
            if not self._results:
                return
            
            # 找到最旧的结果
            oldest_path = min(self._results.keys(), 
                            key=lambda k: self._results[k]['timestamp'])
            
            del self._results[oldest_path]
            self.logger.info(f"删除最旧结果: {oldest_path}")
            
        except Exception as e:
            self.logger.error(f"删除最旧结果失败: {str(e)}")
    
    def _is_expired(self, timestamp: datetime) -> bool:
        """
        检查时间戳是否过期
        
        Args:
            timestamp: 时间戳
            
        Returns:
            bool: 是否过期
        """
        age = datetime.now() - timestamp
        return age > timedelta(hours=self._max_age_hours)


# 全局实例
_result_manager = None


def get_analysis_result_manager() -> AnalysisResultManager:
    """
    获取分析结果管理器的全局实例
    
    Returns:
        AnalysisResultManager: 管理器实例
    """
    global _result_manager
    if _result_manager is None:
        _result_manager = AnalysisResultManager()
    return _result_manager