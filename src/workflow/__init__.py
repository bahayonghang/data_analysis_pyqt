"""
工作流管理模块 - 整合和优化系统工作流

主要组件:
- workflow_integrator: 完整工作流集成器
- performance_optimizer: 性能优化器
- memory_manager: 内存管理器
"""

from .workflow_integrator import WorkflowIntegrator, WorkflowState, WorkflowStep, WorkflowContext
from .performance_optimizer import PerformanceOptimizer, PerformanceMetrics, OptimizationStrategy

__all__ = [
    'WorkflowIntegrator',
    'WorkflowState', 
    'WorkflowStep',
    'WorkflowContext',
    'PerformanceOptimizer',
    'PerformanceMetrics',
    'OptimizationStrategy'
]