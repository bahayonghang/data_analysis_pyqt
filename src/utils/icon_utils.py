# -*- coding: utf-8 -*-
"""
图标工具模块
提供安全的图标设置功能，防止类型错误导致的运行时异常
"""

import logging
from typing import Union, Optional
try:
    from PyQt6.QtGui import QIcon
    from PyQt6.QtWidgets import QWidget
except ImportError:
    try:
        from PyQt5.QtGui import QIcon
        from PyQt5.QtWidgets import QWidget
    except ImportError:
        # 模拟类定义以防止导入错误
        class QIcon:
            pass
        class QWidget:
            pass
try:
    from qfluentwidgets import FluentIcon
    HAS_FLUENT_WIDGETS = True
except ImportError:
    HAS_FLUENT_WIDGETS = False
    # 模拟FluentIcon类
    from enum import Enum
    class FluentIcon(Enum):
        HISTORY = "history"
        SEARCH = "search"
        FILTER = "filter"
        DELETE = "delete"
        DOWNLOAD = "download"
        SYNC = "sync"
        SETTING = "setting"
        FOLDER = "folder"
        HOME = "home"
        DOCUMENT = "document"

# 设置日志记录器
logger = logging.getLogger(__name__)


def safe_set_icon(widget: QWidget, icon: Union[FluentIcon, QIcon, None]) -> bool:
    """
    安全地为控件设置图标
    
    Args:
        widget: 需要设置图标的控件（如按钮）
        icon: 图标对象，可以是FluentIcon枚举、QIcon对象或None
        
    Returns:
        bool: 设置是否成功
        
    Examples:
        # 使用FluentIcon枚举（推荐）
        safe_set_icon(button, FluentIcon.FOLDER)
        
        # 使用QIcon对象
        safe_set_icon(button, QIcon("path/to/icon.png"))
        
        # 清除图标
        safe_set_icon(button, None)
    """
    try:
        # 检查控件是否有setIcon方法
        if not hasattr(widget, 'setIcon'):
            logger.warning(f"控件 {type(widget).__name__} 没有setIcon方法")
            return False
            
        # 处理None值（清除图标）
        if icon is None:
            widget.setIcon(QIcon())
            logger.debug(f"已清除 {type(widget).__name__} 的图标")
            return True
            
        # 处理FluentIcon枚举值（主要用法）
        if isinstance(icon, FluentIcon):
            # 直接传递FluentIcon枚举值给setIcon方法
            # qfluentwidgets的控件可以直接接受FluentIcon枚举
            widget.setIcon(icon)
            # 安全地获取图标名称，避免模拟类没有name属性的问题
            icon_name = getattr(icon, 'name', str(icon))
            logger.debug(f"已为 {type(widget).__name__} 设置FluentIcon: {icon_name}")
            return True
            
        # 处理QIcon对象
        if isinstance(icon, QIcon):
            widget.setIcon(icon)
            logger.debug(f"已为 {type(widget).__name__} 设置QIcon对象")
            return True
            
        # 处理无效类型
        logger.error(
            f"无效的图标类型: {type(icon).__name__}，期望FluentIcon、QIcon或None。"
            f"控件: {type(widget).__name__}，图标值: {icon}"
        )
        
        # 记录详细的调试信息
        if callable(icon):
            logger.error(f"检测到函数对象被传递给setIcon: {icon}")
            logger.error(f"函数名称: {getattr(icon, '__name__', 'unknown')}")
            logger.error(f"函数模块: {getattr(icon, '__module__', 'unknown')}")
            
        return False
        
    except Exception as e:
        logger.error(f"设置图标时发生异常: {e}")
        logger.error(f"控件类型: {type(widget).__name__}")
        logger.error(f"图标类型: {type(icon).__name__}")
        logger.error(f"图标值: {icon}")
        return False


def get_fluent_icon(icon_name: str) -> Optional[FluentIcon]:
    """
    根据名称获取FluentIcon枚举值
    
    Args:
        icon_name: FluentIcon的名称（如'FOLDER'、'CANCEL'等）
        
    Returns:
        FluentIcon: 对应的枚举值，如果不存在则返回None
        
    Examples:
        folder_icon = get_fluent_icon('FOLDER')
        if folder_icon:
            safe_set_icon(button, folder_icon)
    """
    try:
        return getattr(FluentIcon, icon_name.upper())
    except AttributeError:
        logger.warning(f"未找到FluentIcon: {icon_name}")
        return None


def validate_icon_setup(widget: QWidget) -> bool:
    """
    验证控件的图标设置是否正确
    
    Args:
        widget: 要验证的控件
        
    Returns:
        bool: 图标设置是否有效
    """
    try:
        if not hasattr(widget, 'icon'):
            return True  # 没有icon方法的控件认为是有效的
            
        icon = widget.icon()
        
        # 检查icon()返回的是否是QIcon对象
        if not isinstance(icon, QIcon):
            logger.error(
                f"控件 {type(widget).__name__} 的icon()方法返回了 {type(icon).__name__} "
                f"而不是QIcon对象。返回值: {icon}"
            )
            return False
            
        # 检查QIcon是否有isNull方法
        if not hasattr(icon, 'isNull'):
            logger.error(
                f"控件 {type(widget).__name__} 的icon()返回的对象没有isNull方法。"
                f"对象类型: {type(icon).__name__}"
            )
            return False
            
        # 尝试调用isNull方法
        try:
            is_null = icon.isNull()
            logger.debug(f"控件 {type(widget).__name__} 图标验证成功，isNull: {is_null}")
            return True
        except Exception as e:
            logger.error(
                f"调用 {type(widget).__name__} 图标的isNull()方法时出错: {e}"
            )
            return False
            
    except Exception as e:
        logger.error(f"验证控件 {type(widget).__name__} 图标时发生异常: {e}")
        return False