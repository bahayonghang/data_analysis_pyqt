"""
Exception handler utilities for centralized error handling and logging.
"""

import sys
import traceback
from typing import Any, Callable, Optional, Type

try:
    from PyQt6.QtWidgets import QMessageBox, QWidget
    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False
    QMessageBox = None
    QWidget = None

from .exceptions import DataAnalysisError
from .basic_logging import get_logger

logger = get_logger(__name__)


class ExceptionHandler:
    """
    全局异常处理器，提供统一的异常处理和用户通知机制
    """
    
    def __init__(self, parent_widget: Optional[QWidget] = None):
        self.parent_widget = parent_widget
        self.error_callbacks: dict[Type[Exception], Callable] = {}
    
    def register_error_callback(
        self,
        exception_type: Type[Exception],
        callback: Callable[[Exception], None]
    ) -> None:
        """
        注册特定异常类型的回调函数
        
        Args:
            exception_type: 异常类型
            callback: 回调函数
        """
        self.error_callbacks[exception_type] = callback
    
    def handle_exception(
        self,
        exception: Exception,
        show_user_message: bool = True,
        log_error: bool = True
    ) -> None:
        """
        处理异常
        
        Args:
            exception: 要处理的异常
            show_user_message: 是否显示用户消息
            log_error: 是否记录错误日志
        """
        if log_error:
            self._log_exception(exception)
        
        # 查找并执行注册的回调函数
        for exc_type, callback in self.error_callbacks.items():
            if isinstance(exception, exc_type):
                try:
                    callback(exception)
                except Exception as callback_error:
                    logger.error(f"Error in exception callback: {callback_error}")
                break
        
        if show_user_message:
            self._show_user_message(exception)
    
    def _log_exception(self, exception: Exception) -> None:
        """记录异常信息"""
        if isinstance(exception, DataAnalysisError):
            logger.error(
                f"Application error: {exception.message}",
                extra={
                    "error_code": exception.error_code,
                    "details": exception.details,
                    "original_exception": str(exception.original_exception)
                }
            )
        else:
            logger.error(
                f"Unhandled exception: {str(exception)}",
                extra={
                    "exception_type": type(exception).__name__,
                    "traceback": traceback.format_exc()
                }
            )
    
    def _show_user_message(self, exception: Exception) -> None:
        """显示用户友好的错误消息"""
        if not HAS_PYQT or not self.parent_widget:
            # 如果没有PyQt6或没有父窗口，只输出到控制台
            print(f"错误: {str(exception)}")
            return
            
        if isinstance(exception, DataAnalysisError):
            title = "操作错误"
            message = exception.message
            icon = QMessageBox.Icon.Warning
        else:
            title = "系统错误"
            message = f"发生了意外错误：{str(exception)}\n\n请检查日志文件获取详细信息。"
            icon = QMessageBox.Icon.Critical
        
        msg_box = QMessageBox(self.parent_widget)
        msg_box.setIcon(icon)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.exec()


def exception_handler(
    exc_type: Type[BaseException],
    exc_value: BaseException,
    exc_traceback: Any
) -> None:
    """
    全局异常处理函数，用于sys.excepthook
    
    Args:
        exc_type: 异常类型
        exc_value: 异常值
        exc_traceback: 异常回溯
    """
    if issubclass(exc_type, KeyboardInterrupt):
        # 允许KeyboardInterrupt正常处理
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    
    logger.critical(
        f"Uncaught exception: {exc_value}",
        extra={
            "exception_type": exc_type.__name__,
            "traceback": "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        }
    )
    
    # 在GUI应用中，显示错误对话框
    if HAS_PYQT:
        try:
            from PyQt6.QtWidgets import QApplication
            app = QApplication.instance()
            if app:
                handler = ExceptionHandler()
                handler.handle_exception(exc_value, show_user_message=True, log_error=False)
        except Exception as gui_error:
            logger.error(f"Failed to show GUI error message: {gui_error}")
    else:
        print(f"错误: {exc_value}")


def setup_exception_handling() -> None:
    """设置全局异常处理"""
    sys.excepthook = exception_handler


def safe_execute(
    func: Callable,
    *args,
    exception_handler: Optional[ExceptionHandler] = None,
    default_return: Any = None,
    **kwargs
) -> Any:
    """
    安全执行函数，捕获并处理异常
    
    Args:
        func: 要执行的函数
        *args: 函数参数
        exception_handler: 异常处理器
        default_return: 发生异常时的默认返回值
        **kwargs: 函数关键字参数
        
    Returns:
        函数执行结果或默认返回值
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if exception_handler:
            exception_handler.handle_exception(e)
        else:
            logger.exception(f"Error executing {func.__name__}: {str(e)}")
        return default_return


def async_exception_handler(loop, context):
    """
    异步异常处理器
    
    Args:
        loop: 事件循环
        context: 异常上下文
    """
    exception = context.get("exception")
    if exception:
        logger.error(
            f"Async exception: {str(exception)}",
            extra={
                "context": context,
                "exception_type": type(exception).__name__
            }
        )
    else:
        logger.error(f"Async error: {context['message']}", extra={"context": context})