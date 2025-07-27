"""
工作流集成器 - 管理整个应用的端到端工作流

提供完整的数据分析工作流管理：
- 数据上传 → 分析 → 结果展示 → 导出
- 异步处理协调
- 状态管理和进度跟踪
- 自动保存和历史记录
"""

import asyncio
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import uuid

try:
    from PyQt6.QtCore import QObject, pyqtSignal, QTimer
    from PyQt6.QtWidgets import QMessageBox
    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False
    # 模拟类定义
    class QObject:
        pass
    def pyqtSignal(*args):
        return lambda: None

from ..utils.basic_logging import LoggerMixin
from ..models.file_info import FileInfo
from ..models.extended_analysis_result import AnalysisResult
from ..models.analysis_history import AnalysisHistoryRecord, AnalysisStatus
from ..core.history_manager import get_history_manager
from .performance_optimizer import PerformanceOptimizer


class WorkflowState(str, Enum):
    """工作流状态"""
    IDLE = "idle"
    UPLOADING = "uploading"
    VALIDATING = "validating" 
    ANALYZING = "analyzing"
    RENDERING = "rendering"
    SAVING = "saving"
    EXPORTING = "exporting"
    COMPLETED = "completed"
    ERROR = "error"


class WorkflowStep(str, Enum):
    """工作流步骤"""
    FILE_UPLOAD = "file_upload"
    DATA_VALIDATION = "data_validation"
    DATA_LOADING = "data_loading"
    ANALYSIS_EXECUTION = "analysis_execution"
    RESULT_RENDERING = "result_rendering"
    HISTORY_SAVING = "history_saving"
    REPORT_GENERATION = "report_generation"


@dataclass
class WorkflowContext:
    """工作流上下文"""
    workflow_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    current_state: WorkflowState = WorkflowState.IDLE
    current_step: Optional[WorkflowStep] = None
    progress: float = 0.0
    
    # 数据
    file_info: Optional[FileInfo] = None
    raw_data: Optional[Any] = None
    analysis_result: Optional[AnalysisResult] = None
    history_record: Optional[AnalysisHistoryRecord] = None
    
    # 配置
    analysis_config: Dict[str, Any] = field(default_factory=dict)
    export_config: Dict[str, Any] = field(default_factory=dict)
    
    # 错误信息
    error_message: Optional[str] = None
    error_step: Optional[WorkflowStep] = None
    
    # 时间追踪
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    @property
    def duration(self) -> Optional[float]:
        """工作流持续时间（秒）"""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    @property
    def is_active(self) -> bool:
        """是否处于活动状态"""
        return self.current_state not in [WorkflowState.IDLE, WorkflowState.COMPLETED, WorkflowState.ERROR]


class WorkflowIntegrator(QObject, LoggerMixin):
    """工作流集成器
    
    统一管理整个应用的数据分析工作流：
    1. 文件上传和验证
    2. 数据加载和预处理
    3. 分析执行和结果生成
    4. UI更新和用户反馈
    5. 历史记录保存
    6. 报告导出
    """
    
    # 工作流信号
    workflow_started = pyqtSignal(str)  # workflow_id
    workflow_completed = pyqtSignal(str, object)  # workflow_id, result
    workflow_failed = pyqtSignal(str, str)  # workflow_id, error_message
    workflow_cancelled = pyqtSignal(str)  # workflow_id
    
    # 状态变更信号
    state_changed = pyqtSignal(str, str)  # workflow_id, new_state
    step_changed = pyqtSignal(str, str)  # workflow_id, new_step
    progress_updated = pyqtSignal(str, float, str)  # workflow_id, progress, message
    
    # 数据流信号
    data_loaded = pyqtSignal(str, object, object)  # workflow_id, data, file_info
    analysis_completed = pyqtSignal(str, object)  # workflow_id, analysis_result
    history_saved = pyqtSignal(str, object)  # workflow_id, history_record
    
    def __init__(self, parent=None):
        """初始化工作流集成器"""
        if HAS_PYQT6:
            super().__init__(parent)
        else:
            # 只初始化LoggerMixin
            LoggerMixin.__init__(self)
            
        self.history_manager = get_history_manager()
        
        # 初始化性能优化器
        self.performance_optimizer = PerformanceOptimizer(self)
        
        # 活动工作流
        self.active_workflows: Dict[str, WorkflowContext] = {}
        
        # 自动保存定时器
        self.auto_save_timer = QTimer() if HAS_PYQT6 else None
        if self.auto_save_timer:
            self.auto_save_timer.timeout.connect(self._auto_save_check)
            self.auto_save_timer.start(30000)  # 30秒检查一次
        
        # 内存监控定时器  
        self.memory_monitor_timer = QTimer() if HAS_PYQT6 else None
        if self.memory_monitor_timer:
            self.memory_monitor_timer.timeout.connect(self._memory_monitor_check)
            self.memory_monitor_timer.start(60000)  # 60秒检查一次
        
        # 连接性能优化器信号
        if HAS_PYQT6:
            self.performance_optimizer.memory_warning.connect(self._handle_memory_warning)
            self.performance_optimizer.cpu_warning.connect(self._handle_cpu_warning)
            self.performance_optimizer.optimization_applied.connect(self._handle_optimization_applied)
        
        self.logger.info("工作流集成器初始化完成")
    
    def start_workflow(
        self,
        file_path: str,
        analysis_config: Optional[Dict[str, Any]] = None,
        export_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """启动新的工作流
        
        Args:
            file_path: 数据文件路径
            analysis_config: 分析配置
            export_config: 导出配置
            
        Returns:
            str: 工作流ID
        """
        try:
            # 创建工作流上下文
            context = WorkflowContext(
                analysis_config=analysis_config or {},
                export_config=export_config or {},
                started_at=datetime.now()
            )
            
            # 注册工作流
            self.active_workflows[context.workflow_id] = context
            
            # 更新状态
            self._update_state(context.workflow_id, WorkflowState.UPLOADING)
            self._update_step(context.workflow_id, WorkflowStep.FILE_UPLOAD)
            self._update_progress(context.workflow_id, 0.0, "开始工作流...")
            
            # 发送启动信号
            self.workflow_started.emit(context.workflow_id)
            
            # 异步执行工作流
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self._execute_workflow(context.workflow_id, file_path))
                else:
                    # 如果没有事件循环，使用同步方式
                    self._execute_workflow_sync(context.workflow_id, file_path)
            except RuntimeError:
                # 没有事件循环，使用同步方式
                self._execute_workflow_sync(context.workflow_id, file_path)
            
            self.logger.info(f"启动工作流: {context.workflow_id}")
            return context.workflow_id
            
        except Exception as e:
            self.logger.error(f"启动工作流失败: {e}")
            raise
    
    async def _execute_workflow(self, workflow_id: str, file_path: str):
        """异步执行工作流"""
        try:
            context = self.active_workflows[workflow_id]
            
            # 步骤1: 文件验证和信息提取
            await self._step_file_validation(workflow_id, file_path)
            
            # 步骤2: 数据加载
            await self._step_data_loading(workflow_id)
            
            # 步骤3: 分析执行
            await self._step_analysis_execution(workflow_id)
            
            # 步骤4: 结果渲染
            await self._step_result_rendering(workflow_id)
            
            # 步骤5: 历史记录保存
            await self._step_history_saving(workflow_id)
            
            # 完成工作流
            self._complete_workflow(workflow_id)
            
        except Exception as e:
            self._fail_workflow(workflow_id, str(e))
    
    def _execute_workflow_sync(self, workflow_id: str, file_path: str):
        """同步执行工作流（兼容模式）"""
        try:
            context = self.active_workflows[workflow_id]
            
            # 使用同步版本的步骤
            self._step_file_validation_sync(workflow_id, file_path)
            self._step_data_loading_sync(workflow_id)
            self._step_analysis_execution_sync(workflow_id)
            self._step_result_rendering_sync(workflow_id)
            self._step_history_saving_sync(workflow_id)
            
            self._complete_workflow(workflow_id)
            
        except Exception as e:
            self._fail_workflow(workflow_id, str(e))
    
    async def _step_file_validation(self, workflow_id: str, file_path: str):
        """步骤1: 文件验证和信息提取"""
        context = self.active_workflows[workflow_id]
        self._update_step(workflow_id, WorkflowStep.DATA_VALIDATION)
        self._update_progress(workflow_id, 10.0, "验证文件...")
        
        # 模拟异步文件验证
        await asyncio.sleep(0.1)
        
        try:
            from ..models.file_info import FileInfo
            context.file_info = FileInfo.create_from_file(file_path)
            self.logger.info(f"文件验证完成: {context.file_info.file_name}")
        except Exception as e:
            raise Exception(f"文件验证失败: {e}")
    
    def _step_file_validation_sync(self, workflow_id: str, file_path: str):
        """步骤1: 文件验证和信息提取（同步版本）"""
        context = self.active_workflows[workflow_id]
        self._update_step(workflow_id, WorkflowStep.DATA_VALIDATION)
        self._update_progress(workflow_id, 10.0, "验证文件...")
        
        try:
            from ..models.file_info import FileInfo
            context.file_info = FileInfo.create_from_file(file_path)
            self.logger.info(f"文件验证完成: {context.file_info.file_name}")
        except Exception as e:
            raise Exception(f"文件验证失败: {e}")
    
    async def _step_data_loading(self, workflow_id: str):
        """步骤2: 数据加载"""
        context = self.active_workflows[workflow_id]
        self._update_step(workflow_id, WorkflowStep.DATA_LOADING)
        self._update_progress(workflow_id, 30.0, "加载数据...")
        
        await asyncio.sleep(0.2)
        
        try:
            from ..data.data_loader import DataLoader
            loader = DataLoader()
            context.raw_data = loader.load_file(context.file_info.file_path)
            
            # 应用Polars优化
            if hasattr(context.raw_data, 'lazy'):  # 检查是否为Polars DataFrame
                context.raw_data = self.performance_optimizer.optimize_polars_dataframe(context.raw_data)
            
            # 发送数据加载完成信号
            self.data_loaded.emit(workflow_id, context.raw_data, context.file_info)
            self.logger.info(f"数据加载完成: {len(context.raw_data)} 行")
        except Exception as e:
            raise Exception(f"数据加载失败: {e}")
    
    def _step_data_loading_sync(self, workflow_id: str):
        """步骤2: 数据加载（同步版本）"""
        context = self.active_workflows[workflow_id]
        self._update_step(workflow_id, WorkflowStep.DATA_LOADING)
        self._update_progress(workflow_id, 30.0, "加载数据...")
        
        try:
            from ..data.data_loader import DataLoader
            loader = DataLoader()
            context.raw_data = loader.load_file(context.file_info.file_path)
            
            # 应用Polars优化
            if hasattr(context.raw_data, 'lazy'):  # 检查是否为Polars DataFrame
                context.raw_data = self.performance_optimizer.optimize_polars_dataframe(context.raw_data)
            
            # 发送数据加载完成信号
            self.data_loaded.emit(workflow_id, context.raw_data, context.file_info)
            self.logger.info(f"数据加载完成: {len(context.raw_data)} 行")
        except Exception as e:
            raise Exception(f"数据加载失败: {e}")
    
    async def _step_analysis_execution(self, workflow_id: str):
        """步骤3: 分析执行"""
        context = self.active_workflows[workflow_id]
        self._update_state(workflow_id, WorkflowState.ANALYZING)
        self._update_step(workflow_id, WorkflowStep.ANALYSIS_EXECUTION)
        self._update_progress(workflow_id, 50.0, "执行分析...")
        
        await asyncio.sleep(0.5)
        
        try:
            from ..core.analysis_engine import AnalysisEngine, AnalysisConfig
            
            # 创建分析引擎
            analysis_config = AnalysisConfig(**context.analysis_config)
            engine = AnalysisEngine(analysis_config)
            
            # 执行分析
            context.analysis_result = engine.analyze(context.raw_data)
            
            # 发送分析完成信号
            self.analysis_completed.emit(workflow_id, context.analysis_result)
            self.logger.info(f"分析执行完成")
        except Exception as e:
            raise Exception(f"分析执行失败: {e}")
    
    def _step_analysis_execution_sync(self, workflow_id: str):
        """步骤3: 分析执行（同步版本）"""
        context = self.active_workflows[workflow_id]
        self._update_state(workflow_id, WorkflowState.ANALYZING)
        self._update_step(workflow_id, WorkflowStep.ANALYSIS_EXECUTION)
        self._update_progress(workflow_id, 50.0, "执行分析...")
        
        try:
            from ..core.analysis_engine import AnalysisEngine, AnalysisConfig
            
            # 创建分析引擎
            analysis_config = AnalysisConfig(**context.analysis_config)
            engine = AnalysisEngine(analysis_config)
            
            # 执行分析
            context.analysis_result = engine.analyze(context.raw_data)
            
            # 发送分析完成信号
            self.analysis_completed.emit(workflow_id, context.analysis_result)
            self.logger.info(f"分析执行完成")
        except Exception as e:
            raise Exception(f"分析执行失败: {e}")
    
    async def _step_result_rendering(self, workflow_id: str):
        """步骤4: 结果渲染"""
        context = self.active_workflows[workflow_id]
        self._update_state(workflow_id, WorkflowState.RENDERING)
        self._update_step(workflow_id, WorkflowStep.RESULT_RENDERING)
        self._update_progress(workflow_id, 70.0, "渲染结果...")
        
        await asyncio.sleep(0.3)
        
        try:
            # 这里可以添加图表渲染等操作
            self.logger.info("结果渲染完成")
        except Exception as e:
            raise Exception(f"结果渲染失败: {e}")
    
    def _step_result_rendering_sync(self, workflow_id: str):
        """步骤4: 结果渲染（同步版本）"""
        context = self.active_workflows[workflow_id]
        self._update_state(workflow_id, WorkflowState.RENDERING)
        self._update_step(workflow_id, WorkflowStep.RESULT_RENDERING)
        self._update_progress(workflow_id, 70.0, "渲染结果...")
        
        try:
            # 这里可以添加图表渲染等操作
            self.logger.info("结果渲染完成")
        except Exception as e:
            raise Exception(f"结果渲染失败: {e}")
    
    async def _step_history_saving(self, workflow_id: str):
        """步骤5: 历史记录保存"""
        context = self.active_workflows[workflow_id]
        self._update_state(workflow_id, WorkflowState.SAVING)
        self._update_step(workflow_id, WorkflowStep.HISTORY_SAVING)
        self._update_progress(workflow_id, 90.0, "保存历史记录...")
        
        await asyncio.sleep(0.1)
        
        try:
            # 创建历史记录
            context.history_record = AnalysisHistoryRecord(
                file_info=context.file_info,
                analysis_result=context.analysis_result,
                analysis_config=context.analysis_config,
                status=AnalysisStatus.COMPLETED
            )
            
            # 保存到数据库
            saved_record = self.history_manager.save_record(context.history_record)
            context.history_record = saved_record
            
            # 发送历史保存完成信号
            self.history_saved.emit(workflow_id, context.history_record)
            self.logger.info(f"历史记录保存完成: {saved_record.analysis_id}")
        except Exception as e:
            raise Exception(f"历史记录保存失败: {e}")
    
    def _step_history_saving_sync(self, workflow_id: str):
        """步骤5: 历史记录保存（同步版本）"""
        context = self.active_workflows[workflow_id]
        self._update_state(workflow_id, WorkflowState.SAVING)
        self._update_step(workflow_id, WorkflowStep.HISTORY_SAVING)
        self._update_progress(workflow_id, 90.0, "保存历史记录...")
        
        try:
            # 创建历史记录
            context.history_record = AnalysisHistoryRecord(
                file_info=context.file_info,
                analysis_result=context.analysis_result,
                analysis_config=context.analysis_config,
                status=AnalysisStatus.COMPLETED
            )
            
            # 保存到数据库
            saved_record = self.history_manager.save_record(context.history_record)
            context.history_record = saved_record
            
            # 发送历史保存完成信号
            self.history_saved.emit(workflow_id, context.history_record)
            self.logger.info(f"历史记录保存完成: {saved_record.analysis_id}")
        except Exception as e:
            raise Exception(f"历史记录保存失败: {e}")
    
    def _complete_workflow(self, workflow_id: str):
        """完成工作流"""
        context = self.active_workflows[workflow_id]
        context.completed_at = datetime.now()
        
        self._update_state(workflow_id, WorkflowState.COMPLETED)
        self._update_progress(workflow_id, 100.0, "工作流完成")
        
        # 发送完成信号
        self.workflow_completed.emit(workflow_id, context.analysis_result)
        
        self.logger.info(f"工作流完成: {workflow_id}, 耗时: {context.duration:.2f}秒")
    
    def _fail_workflow(self, workflow_id: str, error_message: str):
        """工作流失败"""
        context = self.active_workflows[workflow_id]
        context.error_message = error_message
        context.completed_at = datetime.now()
        
        self._update_state(workflow_id, WorkflowState.ERROR)
        
        # 发送失败信号
        self.workflow_failed.emit(workflow_id, error_message)
        
        self.logger.error(f"工作流失败: {workflow_id}, 错误: {error_message}")
    
    def cancel_workflow(self, workflow_id: str) -> bool:
        """取消工作流
        
        Args:
            workflow_id: 工作流ID
            
        Returns:
            bool: 是否成功取消
        """
        if workflow_id not in self.active_workflows:
            return False
        
        context = self.active_workflows[workflow_id]
        if not context.is_active:
            return False
        
        # 更新状态
        context.completed_at = datetime.now()
        self._update_state(workflow_id, WorkflowState.IDLE)
        
        # 发送取消信号
        self.workflow_cancelled.emit(workflow_id)
        
        self.logger.info(f"工作流已取消: {workflow_id}")
        return True
    
    def get_workflow_context(self, workflow_id: str) -> Optional[WorkflowContext]:
        """获取工作流上下文"""
        return self.active_workflows.get(workflow_id)
    
    def get_active_workflows(self) -> List[str]:
        """获取活动的工作流ID列表"""
        return [wf_id for wf_id, ctx in self.active_workflows.items() if ctx.is_active]
    
    def _update_state(self, workflow_id: str, new_state: WorkflowState):
        """更新工作流状态"""
        if workflow_id in self.active_workflows:
            old_state = self.active_workflows[workflow_id].current_state
            self.active_workflows[workflow_id].current_state = new_state
            self.state_changed.emit(workflow_id, new_state.value)
            self.logger.debug(f"工作流状态变更: {workflow_id} {old_state} → {new_state}")
    
    def _update_step(self, workflow_id: str, new_step: WorkflowStep):
        """更新工作流步骤"""
        if workflow_id in self.active_workflows:
            self.active_workflows[workflow_id].current_step = new_step
            self.step_changed.emit(workflow_id, new_step.value)
            self.logger.debug(f"工作流步骤变更: {workflow_id} → {new_step}")
    
    def _update_progress(self, workflow_id: str, progress: float, message: str):
        """更新工作流进度"""
        if workflow_id in self.active_workflows:
            self.active_workflows[workflow_id].progress = progress
            self.progress_updated.emit(workflow_id, progress, message)
            self.logger.debug(f"工作流进度: {workflow_id} {progress:.1f}% - {message}")
    
    def _auto_save_check(self):
        """自动保存检查"""
        try:
            # 检查是否有需要自动保存的工作流
            for workflow_id, context in self.active_workflows.items():
                if (context.current_state == WorkflowState.COMPLETED and 
                    context.analysis_result and 
                    not context.history_record):
                    # 触发自动保存
                    self.logger.info(f"触发自动保存: {workflow_id}")
                    self._perform_auto_save(workflow_id, context)
                    
                # 检查长时间运行的工作流，触发临时保存
                elif (context.is_active and 
                      context.started_at and 
                      (datetime.now() - context.started_at).total_seconds() > 300):  # 5分钟
                    self._perform_temp_save(workflow_id, context)
                    
        except Exception as e:
            self.logger.error(f"自动保存检查失败: {e}")
    
    def _perform_auto_save(self, workflow_id: str, context: WorkflowContext):
        """执行自动保存"""
        try:
            if context.analysis_result and context.file_info:
                # 创建历史记录
                history_record = AnalysisHistoryRecord(
                    file_info=context.file_info,
                    analysis_result=context.analysis_result,
                    analysis_config=context.analysis_config,
                    status=AnalysisStatus.COMPLETED
                )
                
                # 保存到数据库
                saved_record = self.history_manager.save_record(history_record)
                context.history_record = saved_record
                
                self.logger.info(f"自动保存完成: {workflow_id} -> {saved_record.analysis_id}")
                
                # 发送保存完成信号
                self.history_saved.emit(workflow_id, saved_record)
                
        except Exception as e:
            self.logger.error(f"自动保存失败: {workflow_id} - {e}")
    
    def _perform_temp_save(self, workflow_id: str, context: WorkflowContext):
        """执行临时保存"""
        try:
            if context.raw_data and context.file_info:
                # 创建临时历史记录
                temp_record = AnalysisHistoryRecord(
                    file_info=context.file_info,
                    analysis_result=context.analysis_result,  # 可能为None
                    analysis_config=context.analysis_config,
                    status=AnalysisStatus.IN_PROGRESS
                )
                
                # 保存临时记录
                saved_record = self.history_manager.save_record(temp_record)
                self.logger.info(f"临时保存完成: {workflow_id} -> {saved_record.analysis_id}")
                
        except Exception as e:
            self.logger.error(f"临时保存失败: {workflow_id} - {e}")
    
    def _memory_monitor_check(self):
        """内存监控检查"""
        try:
            import psutil
            process = psutil.Process()
            memory_percent = process.memory_percent()
            
            if memory_percent > 80:  # 内存使用超过80%
                self.logger.warning(f"内存使用率高: {memory_percent:.1f}%")
                # 触发内存清理
                self._trigger_memory_cleanup()
            
        except ImportError:
            # psutil不可用，跳过内存监控
            pass
        except Exception as e:
            self.logger.error(f"内存监控检查失败: {e}")
    
    def _trigger_memory_cleanup(self):
        """触发内存清理"""
        try:
            # 清理已完成的工作流
            completed_workflows = []
            for workflow_id, context in self.active_workflows.items():
                if context.current_state in [WorkflowState.COMPLETED, WorkflowState.ERROR]:
                    # 保留最近的5个工作流，清理其他的
                    completed_workflows.append((workflow_id, context.completed_at))
            
            # 按完成时间排序，保留最新的5个
            completed_workflows.sort(key=lambda x: x[1] or datetime.min, reverse=True)
            workflows_to_remove = completed_workflows[5:]
            
            for workflow_id, _ in workflows_to_remove:
                del self.active_workflows[workflow_id]
                self.logger.info(f"清理工作流: {workflow_id}")
            
            # 强制垃圾回收
            import gc
            gc.collect()
            
        except Exception as e:
            self.logger.error(f"内存清理失败: {e}")
    
    def _handle_memory_warning(self, memory_percent: float):
        """处理内存警告"""
        self.logger.warning(f"收到内存警告: {memory_percent:.1f}%")
        # 可以在这里实现特定的内存优化策略
        self._trigger_memory_cleanup()
    
    def _handle_cpu_warning(self, cpu_percent: float):
        """处理CPU警告"""
        self.logger.warning(f"收到CPU警告: {cpu_percent:.1f}%")
        # 可以在这里实现CPU优化策略，比如降低并发度
    
    def _handle_optimization_applied(self, strategy: str, description: str):
        """处理优化应用通知"""
        self.logger.info(f"优化策略应用: {strategy} - {description}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        return self.performance_optimizer.get_performance_summary()
    
    def set_optimization_strategy(self, strategy):
        """设置优化策略"""
        self.performance_optimizer.set_optimization_strategy(strategy)
    
    def optimize_ui_rendering(self, widget_count: int = 0) -> Dict[str, Any]:
        """优化UI渲染"""
        return self.performance_optimizer.optimize_ui_rendering(widget_count)
    
    def cleanup(self):
        """清理资源"""
        if self.auto_save_timer:
            self.auto_save_timer.stop()
        if self.memory_monitor_timer:
            self.memory_monitor_timer.stop()
        
        # 清理性能优化器
        if hasattr(self, 'performance_optimizer'):
            self.performance_optimizer.cleanup()
        
        self.active_workflows.clear()
        self.logger.info("工作流集成器已清理")