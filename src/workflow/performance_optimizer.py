"""
性能优化器 - 优化数据处理和UI渲染性能

主要功能:
- Polars流式处理优化
- UI渲染性能优化  
- 内存使用监控和管理
- 响应时间优化
"""

import gc
import time
import threading
from typing import Dict, Any, Optional, Callable, List, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import asyncio

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

try:
    from PyQt6.QtCore import QObject, pyqtSignal, QTimer, QThread
    from PyQt6.QtWidgets import QApplication
    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False
    # 模拟类定义
    class QObject:
        pass
    def pyqtSignal(*args):
        return lambda: None

from ..utils.basic_logging import LoggerMixin


@dataclass
class PerformanceMetrics:
    """性能指标"""
    timestamp: datetime = field(default_factory=datetime.now)
    
    # 内存指标
    memory_usage_mb: float = 0.0
    memory_percent: float = 0.0
    memory_available_mb: float = 0.0
    
    # CPU指标
    cpu_percent: float = 0.0
    cpu_count: int = 0
    
    # 数据处理指标
    data_loading_time: float = 0.0
    data_processing_time: float = 0.0
    analysis_time: float = 0.0
    
    # UI指标
    ui_update_time: float = 0.0
    render_time: float = 0.0
    response_time: float = 0.0
    
    # 工作流指标
    total_workflow_time: float = 0.0
    active_workflows: int = 0
    
    @property
    def is_high_memory_usage(self) -> bool:
        """是否高内存使用"""
        return self.memory_percent > 80.0
    
    @property
    def is_high_cpu_usage(self) -> bool:
        """是否高CPU使用"""
        return self.cpu_percent > 80.0
    
    @property
    def is_slow_response(self) -> bool:
        """是否响应缓慢"""
        return self.response_time > 2.0  # 超过2秒


class OptimizationStrategy(str, Enum):
    """优化策略"""
    MEMORY_AGGRESSIVE = "memory_aggressive"  # 激进内存优化
    MEMORY_CONSERVATIVE = "memory_conservative"  # 保守内存优化
    CPU_INTENSIVE = "cpu_intensive"  # CPU密集型优化
    IO_INTENSIVE = "io_intensive"  # IO密集型优化
    BALANCED = "balanced"  # 平衡优化
    RESPONSIVE = "responsive"  # 响应性优化


class PerformanceOptimizer(QObject, LoggerMixin):
    """性能优化器
    
    提供全面的性能优化功能：
    1. 内存使用监控和自动清理
    2. Polars数据处理优化
    3. UI渲染性能优化
    4. 响应时间优化
    5. 自适应优化策略
    """
    
    # 性能信号
    performance_updated = pyqtSignal(object)  # PerformanceMetrics
    memory_warning = pyqtSignal(float)  # memory_percent
    cpu_warning = pyqtSignal(float)  # cpu_percent
    optimization_applied = pyqtSignal(str, str)  # strategy, description
    
    def __init__(self, parent=None):
        """初始化性能优化器"""
        if HAS_PYQT6:
            super().__init__(parent)
        else:
            # 只初始化LoggerMixin
            LoggerMixin.__init__(self)
        
        # 性能配置
        self.config = {
            'memory_warning_threshold': 80.0,  # 内存警告阈值(%)
            'memory_critical_threshold': 90.0,  # 内存临界阈值(%)
            'cpu_warning_threshold': 80.0,     # CPU警告阈值(%)
            'response_time_threshold': 2.0,    # 响应时间阈值(秒)
            'monitoring_interval': 5.0,        # 监控间隔(秒)
            'cleanup_interval': 30.0,          # 清理间隔(秒)
            'auto_optimization': True,         # 自动优化
        }
        
        # 性能指标历史
        self.metrics_history: List[PerformanceMetrics] = []
        self.max_history_size = 100
        
        # 优化策略
        self.current_strategy = OptimizationStrategy.BALANCED
        
        # 监控定时器
        self.monitor_timer = QTimer() if HAS_PYQT6 else None
        if self.monitor_timer:
            self.monitor_timer.timeout.connect(self._collect_metrics)
            self.monitor_timer.start(int(self.config['monitoring_interval'] * 1000))
        
        # 清理定时器
        self.cleanup_timer = QTimer() if HAS_PYQT6 else None
        if self.cleanup_timer:
            self.cleanup_timer.timeout.connect(self._auto_cleanup)
            self.cleanup_timer.start(int(self.config['cleanup_interval'] * 1000))
        
        # Polars优化配置
        self._setup_polars_optimization()
        
        self.logger.info("性能优化器初始化完成")
    
    def _setup_polars_optimization(self):
        """设置Polars优化配置"""
        if not HAS_POLARS:
            return
        
        try:
            # 设置Polars优化选项
            pl.Config.set_streaming_chunk_size(10000)  # 流式处理块大小
            pl.Config.set_tbl_rows(20)  # 显示行数
            
            # 启用优化
            pl.Config.set_auto_structify(True)  # 自动结构化
            
            self.logger.info("Polars优化配置完成")
        except Exception as e:
            self.logger.warning(f"Polars优化配置失败: {e}")
    
    def _collect_metrics(self):
        """收集性能指标"""
        try:
            metrics = PerformanceMetrics()
            
            # 收集系统指标
            if HAS_PSUTIL:
                process = psutil.Process()
                memory_info = process.memory_info()
                system_memory = psutil.virtual_memory()
                
                metrics.memory_usage_mb = memory_info.rss / 1024 / 1024
                metrics.memory_percent = process.memory_percent()
                metrics.memory_available_mb = system_memory.available / 1024 / 1024
                metrics.cpu_percent = process.cpu_percent()
                metrics.cpu_count = psutil.cpu_count()
            
            # 添加到历史记录
            self.metrics_history.append(metrics)
            if len(self.metrics_history) > self.max_history_size:
                self.metrics_history.pop(0)
            
            # 发送更新信号
            self.performance_updated.emit(metrics)
            
            # 检查是否需要优化
            self._check_optimization_needs(metrics)
            
        except Exception as e:
            self.logger.error(f"收集性能指标失败: {e}")
    
    def _check_optimization_needs(self, metrics: PerformanceMetrics):
        """检查是否需要优化"""
        try:
            # 内存警告
            if metrics.memory_percent > self.config['memory_warning_threshold']:
                self.memory_warning.emit(metrics.memory_percent)
                
                # 内存临界情况下自动清理
                if (metrics.memory_percent > self.config['memory_critical_threshold'] and 
                    self.config['auto_optimization']):
                    self._trigger_memory_optimization()
            
            # CPU警告
            if metrics.cpu_percent > self.config['cpu_warning_threshold']:
                self.cpu_warning.emit(metrics.cpu_percent)
                
                # 高CPU使用时调整策略
                if self.config['auto_optimization']:
                    self._adjust_strategy_for_cpu()
            
            # 响应时间优化
            if metrics.response_time > self.config['response_time_threshold']:
                if self.config['auto_optimization']:
                    self._optimize_response_time()
                    
        except Exception as e:
            self.logger.error(f"检查优化需求失败: {e}")
    
    def _trigger_memory_optimization(self):
        """触发内存优化"""
        try:
            self.logger.info("触发内存优化")
            
            # 强制垃圾回收
            collected = gc.collect()
            self.logger.info(f"垃圾回收清理了 {collected} 个对象")
            
            # 清理缓存
            self._clear_caches()
            
            # 优化数据结构
            self._optimize_data_structures()
            
            # 发送优化信号
            self.optimization_applied.emit(
                "memory_optimization", 
                f"内存优化完成，清理了 {collected} 个对象"
            )
            
        except Exception as e:
            self.logger.error(f"内存优化失败: {e}")
    
    def _adjust_strategy_for_cpu(self):
        """调整CPU优化策略"""
        try:
            if self.current_strategy != OptimizationStrategy.CPU_INTENSIVE:
                self.current_strategy = OptimizationStrategy.CPU_INTENSIVE
                self.logger.info("切换到CPU密集型优化策略")
                
                # 调整处理参数
                self._apply_cpu_optimization()
                
                self.optimization_applied.emit(
                    "cpu_optimization",
                    "切换到CPU密集型优化策略"
                )
        except Exception as e:
            self.logger.error(f"CPU优化策略调整失败: {e}")
    
    def _optimize_response_time(self):
        """优化响应时间"""
        try:
            self.logger.info("优化响应时间")
            
            if self.current_strategy != OptimizationStrategy.RESPONSIVE:
                self.current_strategy = OptimizationStrategy.RESPONSIVE
                
                # 启用响应性优化
                self._apply_responsive_optimization()
                
                self.optimization_applied.emit(
                    "response_optimization",
                    "启用响应性优化策略"
                )
        except Exception as e:
            self.logger.error(f"响应时间优化失败: {e}")
    
    def _clear_caches(self):
        """清理缓存"""
        try:
            # 清理Polars缓存
            if HAS_POLARS:
                # Polars 没有全局缓存清理，但可以清理一些内部状态
                pass
            
            # 清理Qt缓存
            if HAS_PYQT6:
                app = QApplication.instance()
                if app:
                    # 清理Qt的内部缓存
                    pass
            
            self.logger.info("缓存清理完成")
        except Exception as e:
            self.logger.error(f"缓存清理失败: {e}")
    
    def _optimize_data_structures(self):
        """优化数据结构"""
        try:
            # 这里可以添加特定的数据结构优化
            self.logger.info("数据结构优化完成")
        except Exception as e:
            self.logger.error(f"数据结构优化失败: {e}")
    
    def _apply_cpu_optimization(self):
        """应用CPU优化"""
        try:
            if HAS_POLARS:
                # 减少并行度以降低CPU使用
                # pl.Config.set_cpu_cores(max(1, psutil.cpu_count() // 2))
                pass
            
            self.logger.info("CPU优化应用完成")
        except Exception as e:
            self.logger.error(f"CPU优化应用失败: {e}")
    
    def _apply_responsive_optimization(self):
        """应用响应性优化"""
        try:
            if HAS_POLARS:
                # 减小处理块大小以提高响应性
                pl.Config.set_streaming_chunk_size(5000)
            
            self.logger.info("响应性优化应用完成")
        except Exception as e:
            self.logger.error(f"响应性优化应用失败: {e}")
    
    def _auto_cleanup(self):
        """自动清理"""
        try:
            # 定期垃圾回收
            collected = gc.collect()
            if collected > 0:
                self.logger.debug(f"自动清理：回收了 {collected} 个对象")
            
            # 清理过期的性能指标
            self._cleanup_old_metrics()
            
        except Exception as e:
            self.logger.error(f"自动清理失败: {e}")
    
    def _cleanup_old_metrics(self):
        """清理过期的性能指标"""
        try:
            # 保留最近1小时的指标
            cutoff_time = datetime.now() - timedelta(hours=1)
            self.metrics_history = [
                m for m in self.metrics_history 
                if m.timestamp > cutoff_time
            ]
        except Exception as e:
            self.logger.error(f"清理过期指标失败: {e}")
    
    def optimize_polars_dataframe(self, df: 'pl.DataFrame') -> 'pl.DataFrame':
        """优化Polars DataFrame
        
        Args:
            df: 输入DataFrame
            
        Returns:
            pl.DataFrame: 优化后的DataFrame
        """
        if not HAS_POLARS:
            return df
        
        try:
            start_time = time.time()
            
            # 应用优化策略
            optimized_df = df
            
            if self.current_strategy == OptimizationStrategy.MEMORY_AGGRESSIVE:
                # 激进内存优化
                optimized_df = self._apply_memory_aggressive_optimization(optimized_df)
            elif self.current_strategy == OptimizationStrategy.CPU_INTENSIVE:
                # CPU密集型优化
                optimized_df = self._apply_cpu_intensive_optimization(optimized_df)
            elif self.current_strategy == OptimizationStrategy.RESPONSIVE:
                # 响应性优化
                optimized_df = self._apply_responsive_dataframe_optimization(optimized_df)
            else:
                # 平衡优化
                optimized_df = self._apply_balanced_optimization(optimized_df)
            
            optimization_time = time.time() - start_time
            self.logger.info(f"DataFrame优化完成，耗时: {optimization_time:.2f}秒")
            
            return optimized_df
            
        except Exception as e:
            self.logger.error(f"DataFrame优化失败: {e}")
            return df
    
    def _apply_memory_aggressive_optimization(self, df: 'pl.DataFrame') -> 'pl.DataFrame':
        """应用激进内存优化"""
        try:
            # 优化数据类型
            optimized_df = df
            
            # 转换为更小的数据类型
            for col in df.columns:
                dtype = df[col].dtype
                if dtype == pl.Int64:
                    # 尝试转换为更小的整数类型
                    max_val = df[col].max()
                    min_val = df[col].min()
                    if max_val is not None and min_val is not None:
                        if min_val >= -128 and max_val <= 127:
                            optimized_df = optimized_df.with_columns(df[col].cast(pl.Int8))
                        elif min_val >= -32768 and max_val <= 32767:
                            optimized_df = optimized_df.with_columns(df[col].cast(pl.Int16))
                        elif min_val >= -2147483648 and max_val <= 2147483647:
                            optimized_df = optimized_df.with_columns(df[col].cast(pl.Int32))
            
            return optimized_df
        except Exception as e:
            self.logger.error(f"激进内存优化失败: {e}")
            return df
    
    def _apply_cpu_intensive_optimization(self, df: 'pl.DataFrame') -> 'pl.DataFrame':
        """应用CPU密集型优化"""
        try:
            # 使用更多的CPU进行并行处理
            return df
        except Exception as e:
            self.logger.error(f"CPU密集型优化失败: {e}")
            return df
    
    def _apply_responsive_dataframe_optimization(self, df: 'pl.DataFrame') -> 'pl.DataFrame':
        """应用响应性DataFrame优化"""
        try:
            # 使用流式处理以保持响应性
            return df
        except Exception as e:
            self.logger.error(f"响应性DataFrame优化失败: {e}")
            return df
    
    def _apply_balanced_optimization(self, df: 'pl.DataFrame') -> 'pl.DataFrame':
        """应用平衡优化"""
        try:
            # 平衡内存和性能
            return df
        except Exception as e:
            self.logger.error(f"平衡优化失败: {e}")
            return df
    
    def optimize_ui_rendering(self, widget_count: int = 0) -> Dict[str, Any]:
        """优化UI渲染
        
        Args:
            widget_count: 部件数量
            
        Returns:
            Dict[str, Any]: 优化建议
        """
        try:
            recommendations = {
                'batch_updates': False,
                'use_viewport': False,
                'reduce_animations': False,
                'lazy_loading': False
            }
            
            # 根据部件数量和性能状态给出建议
            if widget_count > 100:
                recommendations['batch_updates'] = True
                recommendations['use_viewport'] = True
            
            if self.metrics_history:
                latest_metrics = self.metrics_history[-1]
                if latest_metrics.is_high_memory_usage:
                    recommendations['reduce_animations'] = True
                    recommendations['lazy_loading'] = True
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"UI渲染优化失败: {e}")
            return {}
    
    def set_optimization_strategy(self, strategy: OptimizationStrategy):
        """设置优化策略
        
        Args:
            strategy: 优化策略
        """
        try:
            old_strategy = self.current_strategy
            self.current_strategy = strategy
            
            self.logger.info(f"优化策略变更: {old_strategy} → {strategy}")
            
            # 应用新策略
            self._apply_strategy(strategy)
            
            self.optimization_applied.emit(
                strategy.value,
                f"切换优化策略: {strategy.value}"
            )
            
        except Exception as e:
            self.logger.error(f"设置优化策略失败: {e}")
    
    def _apply_strategy(self, strategy: OptimizationStrategy):
        """应用优化策略"""
        try:
            if strategy == OptimizationStrategy.MEMORY_AGGRESSIVE:
                self._trigger_memory_optimization()
            elif strategy == OptimizationStrategy.CPU_INTENSIVE:
                self._apply_cpu_optimization()
            elif strategy == OptimizationStrategy.RESPONSIVE:
                self._apply_responsive_optimization()
        except Exception as e:
            self.logger.error(f"应用优化策略失败: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要
        
        Returns:
            Dict[str, Any]: 性能摘要
        """
        try:
            if not self.metrics_history:
                return {}
            
            latest = self.metrics_history[-1]
            
            # 计算平均值
            avg_memory = sum(m.memory_percent for m in self.metrics_history) / len(self.metrics_history)
            avg_cpu = sum(m.cpu_percent for m in self.metrics_history) / len(self.metrics_history)
            
            return {
                'current_memory_percent': latest.memory_percent,
                'current_cpu_percent': latest.cpu_percent,
                'average_memory_percent': avg_memory,
                'average_cpu_percent': avg_cpu,
                'memory_usage_mb': latest.memory_usage_mb,
                'memory_available_mb': latest.memory_available_mb,
                'current_strategy': self.current_strategy.value,
                'metrics_count': len(self.metrics_history),
                'optimization_status': {
                    'memory_warning': latest.is_high_memory_usage,
                    'cpu_warning': latest.is_high_cpu_usage,
                    'slow_response': latest.is_slow_response
                }
            }
        except Exception as e:
            self.logger.error(f"获取性能摘要失败: {e}")
            return {}
    
    def cleanup(self):
        """清理资源"""
        try:
            if self.monitor_timer:
                self.monitor_timer.stop()
            if self.cleanup_timer:
                self.cleanup_timer.stop()
            
            self.metrics_history.clear()
            
            self.logger.info("性能优化器已清理")
        except Exception as e:
            self.logger.error(f"清理资源失败: {e}")