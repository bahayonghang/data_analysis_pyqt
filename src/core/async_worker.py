"""
异步工作流管理
支持任务进度回调、队列管理和并发控制
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union, Coroutine
from datetime import datetime
import time
import threading
import queue
import uuid
import traceback
from contextlib import asynccontextmanager

from ..utils.basic_logging import LoggerMixin
from ..utils.exceptions import WorkflowError


class TaskStatus(str, Enum):
    """任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class TaskPriority(int, Enum):
    """任务优先级"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class TaskProgress:
    """任务进度信息"""
    current: int = 0
    total: int = 100
    message: str = ""
    percentage: float = 0.0
    estimated_remaining_seconds: Optional[float] = None
    
    def __post_init__(self):
        self.update_percentage()
    
    def update(self, current: int, total: Optional[int] = None, message: str = ""):
        """更新进度"""
        self.current = current
        if total is not None:
            self.total = total
        if message:
            self.message = message
        self.update_percentage()
    
    def update_percentage(self):
        """更新百分比"""
        if self.total > 0:
            self.percentage = min(100.0, (self.current / self.total) * 100.0)
        else:
            self.percentage = 0.0
    
    def increment(self, amount: int = 1, message: str = ""):
        """递增进度"""
        self.update(self.current + amount, message=message)


@dataclass
class AsyncTask:
    """异步任务"""
    task_id: str
    name: str
    func: Union[Callable, Coroutine]
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 0
    
    # 状态管理
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # 进度和结果
    progress: TaskProgress = field(default_factory=TaskProgress)
    result: Any = None
    error: Optional[Exception] = None
    error_traceback: Optional[str] = None
    
    # 回调函数
    progress_callback: Optional[Callable[[TaskProgress], None]] = None
    completion_callback: Optional[Callable[["AsyncTask"], None]] = None
    error_callback: Optional[Callable[["AsyncTask", Exception], None]] = None
    
    # 依赖管理
    dependencies: List[str] = field(default_factory=list)
    dependents: List[str] = field(default_factory=list)
    
    # 取消支持
    _cancel_event: Optional[threading.Event] = field(default=None, init=False)
    
    def __post_init__(self):
        if self.task_id is None:
            self.task_id = str(uuid.uuid4())[:8]
        self._cancel_event = threading.Event()
    
    def mark_started(self):
        """标记任务开始"""
        self.status = TaskStatus.RUNNING
        self.started_at = datetime.now()
    
    def mark_completed(self, result: Any = None):
        """标记任务完成"""
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.now()
        self.result = result
        self.progress.update(self.progress.total, message="完成")
        
        if self.completion_callback:
            try:
                self.completion_callback(self)
            except Exception as e:
                # 不让回调错误影响主任务
                pass
    
    def mark_failed(self, error: Exception):
        """标记任务失败"""
        self.status = TaskStatus.FAILED
        self.completed_at = datetime.now()
        self.error = error
        self.error_traceback = traceback.format_exc()
        
        if self.error_callback:
            try:
                self.error_callback(self, error)
            except Exception:
                # 不让回调错误影响主任务
                pass
    
    def mark_cancelled(self):
        """标记任务取消"""
        self.status = TaskStatus.CANCELLED
        self.completed_at = datetime.now()
        self._cancel_event.set()
    
    def is_cancelled(self) -> bool:
        """检查是否被取消"""
        return self._cancel_event.is_set()
    
    def update_progress(self, current: int, total: Optional[int] = None, message: str = ""):
        """更新任务进度"""
        self.progress.update(current, total, message)
        
        if self.progress_callback:
            try:
                self.progress_callback(self.progress)
            except Exception:
                # 不让回调错误影响主任务
                pass
    
    def can_run(self) -> bool:
        """检查任务是否可以运行"""
        return (self.status == TaskStatus.PENDING and 
                not self.is_cancelled())
    
    def should_retry(self) -> bool:
        """检查是否应该重试"""
        return (self.status == TaskStatus.FAILED and 
                self.retry_count < self.max_retries)
    
    def prepare_retry(self):
        """准备重试"""
        self.retry_count += 1
        self.status = TaskStatus.PENDING
        self.started_at = None
        self.completed_at = None
        self.error = None
        self.error_traceback = None
        self.progress = TaskProgress()
    
    @property
    def execution_time(self) -> Optional[float]:
        """获取执行时间（秒）"""
        if self.started_at:
            end_time = self.completed_at or datetime.now()
            return (end_time - self.started_at).total_seconds()
        return None
    
    @property
    def is_finished(self) -> bool:
        """检查任务是否已完成（包括成功、失败、取消）"""
        return self.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]


@dataclass
class WorkflowConfig:
    """工作流配置"""
    # 并发控制
    max_concurrent_tasks: int = 4
    max_queue_size: int = 100
    
    # 超时控制
    default_task_timeout: float = 300.0  # 5分钟
    queue_timeout: float = 10.0
    
    # 重试控制
    default_max_retries: int = 2
    retry_delay: float = 1.0
    
    # 性能监控
    enable_performance_monitoring: bool = True
    max_memory_mb: int = 1000
    
    # 错误处理
    stop_on_first_error: bool = False
    log_task_details: bool = True


class AsyncWorker(LoggerMixin):
    """异步工作流管理器"""
    
    def __init__(self, config: Optional[WorkflowConfig] = None):
        self.config = config or WorkflowConfig()
        
        # 任务管理
        self.tasks: Dict[str, AsyncTask] = {}
        self.task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue(
            maxsize=self.config.max_queue_size
        )
        
        # 线程池
        self.executor = ThreadPoolExecutor(
            max_workers=self.config.max_concurrent_tasks,
            thread_name_prefix="AsyncWorker"
        )
        
        # 状态管理
        self.is_running = False
        self.is_shutting_down = False
        self._worker_tasks: List[asyncio.Task] = []
        self._lock = asyncio.Lock()
        
        # 性能监控
        self.stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'cancelled_tasks': 0,
            'average_execution_time': 0.0,
            'total_execution_time': 0.0
        }
    
    async def start(self):
        """启动异步工作器"""
        if self.is_running:
            return
        
        self.logger.info("启动异步工作器")
        self.is_running = True
        self.is_shutting_down = False
        
        # 启动工作线程
        for i in range(self.config.max_concurrent_tasks):
            worker_task = asyncio.create_task(self._worker_loop(f"worker-{i}"))
            self._worker_tasks.append(worker_task)
    
    async def stop(self, timeout: float = 30.0):
        """停止异步工作器"""
        if not self.is_running:
            return
        
        self.logger.info("停止异步工作器")
        self.is_shutting_down = True
        
        # 取消所有待处理任务
        await self._cancel_pending_tasks()
        
        # 等待工作线程完成
        try:
            await asyncio.wait_for(
                asyncio.gather(*self._worker_tasks, return_exceptions=True),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            self.logger.warning("工作线程未在指定时间内完成，强制取消")
            for task in self._worker_tasks:
                if not task.done():
                    task.cancel()
        
        self.is_running = False
        self._worker_tasks.clear()
        
        # 关闭线程池
        self.executor.shutdown(wait=True)
        
        self.logger.info("异步工作器已停止")
    
    async def submit_task(
        self,
        name: str,
        func: Union[Callable, Coroutine],
        args: tuple = (),
        kwargs: dict = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout: Optional[float] = None,
        max_retries: int = None,
        task_id: str = None,
        progress_callback: Optional[Callable[[TaskProgress], None]] = None,
        completion_callback: Optional[Callable[[AsyncTask], None]] = None,
        error_callback: Optional[Callable[[AsyncTask, Exception], None]] = None,
        dependencies: List[str] = None
    ) -> str:
        """提交任务"""
        if not self.is_running:
            raise WorkflowError("异步工作器未启动")
        
        kwargs = kwargs or {}
        dependencies = dependencies or []
        
        task = AsyncTask(
            task_id=task_id or str(uuid.uuid4())[:8],
            name=name,
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            timeout=timeout or self.config.default_task_timeout,
            max_retries=max_retries or self.config.default_max_retries,
            progress_callback=progress_callback,
            completion_callback=completion_callback,
            error_callback=error_callback,
            dependencies=dependencies
        )
        
        async with self._lock:
            self.tasks[task.task_id] = task
            self.stats['total_tasks'] += 1
        
        # 检查依赖关系
        if await self._check_dependencies(task):
            await self._enqueue_task(task)
        
        self.logger.info(f"任务已提交: {task.name} (ID: {task.task_id})")
        return task.task_id
    
    async def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        async with self._lock:
            task = self.tasks.get(task_id)
            if not task:
                return False
            
            if task.status == TaskStatus.RUNNING:
                task.mark_cancelled()
                self.stats['cancelled_tasks'] += 1
                self.logger.info(f"任务已取消: {task.name} (ID: {task_id})")
                return True
            elif task.status == TaskStatus.PENDING:
                task.mark_cancelled()
                self.stats['cancelled_tasks'] += 1
                self.logger.info(f"等待中的任务已取消: {task.name} (ID: {task_id})")
                return True
            
            return False
    
    async def pause_task(self, task_id: str) -> bool:
        """暂停任务"""
        async with self._lock:
            task = self.tasks.get(task_id)
            if task and task.status == TaskStatus.RUNNING:
                task.status = TaskStatus.PAUSED
                self.logger.info(f"任务已暂停: {task.name} (ID: {task_id})")
                return True
            return False
    
    async def resume_task(self, task_id: str) -> bool:
        """恢复任务"""
        async with self._lock:
            task = self.tasks.get(task_id)
            if task and task.status == TaskStatus.PAUSED:
                task.status = TaskStatus.RUNNING
                await self._enqueue_task(task)
                self.logger.info(f"任务已恢复: {task.name} (ID: {task_id})")
                return True
            return False
    
    def get_task_status(self, task_id: str) -> Optional[AsyncTask]:
        """获取任务状态"""
        return self.tasks.get(task_id)
    
    def get_all_tasks(self) -> Dict[str, AsyncTask]:
        """获取所有任务"""
        return self.tasks.copy()
    
    def get_running_tasks(self) -> List[AsyncTask]:
        """获取正在运行的任务"""
        return [task for task in self.tasks.values() if task.status == TaskStatus.RUNNING]
    
    def get_pending_tasks(self) -> List[AsyncTask]:
        """获取等待中的任务"""
        return [task for task in self.tasks.values() if task.status == TaskStatus.PENDING]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self.stats.copy()
        
        # 计算平均执行时间
        if stats['completed_tasks'] > 0:
            stats['average_execution_time'] = stats['total_execution_time'] / stats['completed_tasks']
        
        # 添加当前状态
        stats.update({
            'is_running': self.is_running,
            'current_running_tasks': len(self.get_running_tasks()),
            'current_pending_tasks': len(self.get_pending_tasks()),
            'queue_size': self.task_queue.qsize() if hasattr(self.task_queue, 'qsize') else 0
        })
        
        return stats
    
    async def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        """等待所有任务完成"""
        start_time = time.time()
        
        while self.is_running:
            running_tasks = self.get_running_tasks()
            pending_tasks = self.get_pending_tasks()
            
            if not running_tasks and not pending_tasks:
                return True
            
            if timeout and (time.time() - start_time) > timeout:
                return False
            
            await asyncio.sleep(0.1)
        
        return True
    
    async def _worker_loop(self, worker_name: str):
        """工作线程循环"""
        self.logger.debug(f"启动工作线程: {worker_name}")
        
        while self.is_running and not self.is_shutting_down:
            try:
                # 获取任务（带超时）
                try:
                    priority_item = await asyncio.wait_for(
                        self.task_queue.get(),
                        timeout=self.config.queue_timeout
                    )
                    priority, task_id = priority_item
                except asyncio.TimeoutError:
                    continue
                
                task = self.tasks.get(task_id)
                if not task or not task.can_run():
                    continue
                
                # 执行任务
                await self._execute_task(task, worker_name)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"工作线程 {worker_name} 发生错误: {str(e)}")
                if self.config.stop_on_first_error:
                    break
        
        self.logger.debug(f"工作线程 {worker_name} 已停止")
    
    async def _execute_task(self, task: AsyncTask, worker_name: str):
        """执行单个任务"""
        task.mark_started()
        
        if self.config.log_task_details:
            self.logger.info(f"[{worker_name}] 开始执行任务: {task.name} (ID: {task.task_id})")
        
        try:
            # 检查取消状态
            if task.is_cancelled():
                task.mark_cancelled()
                return
            
            # 执行任务
            if asyncio.iscoroutinefunction(task.func):
                # 异步函数
                result = await asyncio.wait_for(
                    task.func(*task.args, **task.kwargs),
                    timeout=task.timeout
                )
            else:
                # 同步函数，在线程池中执行
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(
                        self.executor,
                        lambda: task.func(*task.args, **task.kwargs)
                    ),
                    timeout=task.timeout
                )
            
            # 任务完成
            task.mark_completed(result)
            
            async with self._lock:
                self.stats['completed_tasks'] += 1
                if task.execution_time:
                    self.stats['total_execution_time'] += task.execution_time
            
            if self.config.log_task_details:
                self.logger.info(
                    f"[{worker_name}] 任务完成: {task.name} "
                    f"(耗时: {task.execution_time:.2f}s)"
                )
            
            # 处理依赖任务
            await self._process_dependents(task)
            
        except asyncio.TimeoutError:
            error = WorkflowError(f"任务执行超时: {task.timeout}秒")
            task.mark_failed(error)
            async with self._lock:
                self.stats['failed_tasks'] += 1
            
            self.logger.warning(f"[{worker_name}] 任务超时: {task.name}")
            
            # 重试逻辑
            if task.should_retry():
                await self._retry_task(task)
        
        except asyncio.CancelledError:
            task.mark_cancelled()
            async with self._lock:
                self.stats['cancelled_tasks'] += 1
            self.logger.info(f"[{worker_name}] 任务被取消: {task.name}")
        
        except Exception as e:
            task.mark_failed(e)
            async with self._lock:
                self.stats['failed_tasks'] += 1
            
            self.logger.error(f"[{worker_name}] 任务失败: {task.name}, 错误: {str(e)}")
            
            # 重试逻辑
            if task.should_retry():
                await self._retry_task(task)
    
    async def _retry_task(self, task: AsyncTask):
        """重试任务"""
        self.logger.info(f"重试任务: {task.name} (第{task.retry_count + 1}次重试)")
        
        # 等待重试延迟
        await asyncio.sleep(self.config.retry_delay)
        
        # 准备重试
        task.prepare_retry()
        
        # 重新入队
        await self._enqueue_task(task)
    
    async def _enqueue_task(self, task: AsyncTask):
        """将任务加入队列"""
        # 使用负优先级，使高优先级任务先执行
        priority = -task.priority.value
        await self.task_queue.put((priority, task.task_id))
    
    async def _check_dependencies(self, task: AsyncTask) -> bool:
        """检查任务依赖是否满足"""
        for dep_id in task.dependencies:
            dep_task = self.tasks.get(dep_id)
            if not dep_task or dep_task.status != TaskStatus.COMPLETED:
                return False
        return True
    
    async def _process_dependents(self, completed_task: AsyncTask):
        """处理依赖已完成任务的其他任务"""
        for task in self.tasks.values():
            if (completed_task.task_id in task.dependencies and 
                task.status == TaskStatus.PENDING and
                await self._check_dependencies(task)):
                await self._enqueue_task(task)
    
    async def _cancel_pending_tasks(self):
        """取消所有待处理任务"""
        pending_tasks = self.get_pending_tasks()
        for task in pending_tasks:
            task.mark_cancelled()
        
        async with self._lock:
            self.stats['cancelled_tasks'] += len(pending_tasks)
    
    @asynccontextmanager
    async def task_context(self):
        """任务上下文管理器"""
        await self.start()
        try:
            yield self
        finally:
            await self.stop()
    
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)