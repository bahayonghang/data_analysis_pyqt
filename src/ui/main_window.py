"""
主窗口和导航系统
基于PyQt6和PyQt-Fluent-Widgets的现代化UI框架
"""

from enum import Enum
from typing import Any

try:
    from PyQt6.QtCore import (
        QEasingCurve,
        QObject,
        QPropertyAnimation,
        QRect,
        QSize,
        Qt,
        QTimer,
        pyqtSignal,
    )
    from PyQt6.QtGui import QColor, QIcon
    from PyQt6.QtWidgets import (
        QApplication,
        QGraphicsOpacityEffect,
        QStackedWidget,
        QWidget,
    )
    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False
    # 提供模拟类，避免导入错误
    class QObject:
        pass
    class QStackedWidget:
        pass
    class QWidget:
        pass
    class QColor:
        def __init__(self, r=0, g=0, b=0, a=255):
            self.r = r
            self.g = g
            self.b = b
            self.a = a
    class QSize:
        def __init__(self, width=0, height=0):
            self._width = width
            self._height = height
        def width(self):
            return self._width
        def height(self):
            return self._height
    def pyqtSignal(*args):
        return lambda *a, **k: None

try:
    from qfluentwidgets import (
        MSFluentWindow,
        NavigationItemPosition,
        Theme,
        setTheme,
        setThemeColor,
    )
    HAS_FLUENT_WIDGETS = True
except ImportError:
    HAS_FLUENT_WIDGETS = False
    # 提供模拟类，避免导入错误
    class MSFluentWindow:
        def __init__(self):
            self.navigationInterface = None
            self.stackedWidget = None
    class NavigationInterface:
        pass
    class NavigationItemPosition:
        TOP = "top"
        BOTTOM = "bottom"

from ..utils.basic_logging import LoggerMixin
from ..utils.exceptions import ComponentInitializationError


class NavigationPage(str, Enum):
    """导航页面枚举"""
    HOME = "home"
    UPLOAD = "upload"
    ANALYSIS = "analysis"
    HISTORY = "history"
    SETTINGS = "settings"


class ThemeMode(str, Enum):
    """主题模式"""
    LIGHT = "light"
    DARK = "dark"
    AUTO = "auto"


class WindowState(str, Enum):
    """窗口状态"""
    NORMAL = "normal"
    MAXIMIZED = "maximized"
    MINIMIZED = "minimized"
    FULLSCREEN = "fullscreen"


class UIConfig:
    """UI配置"""

    def __init__(self):
        # 窗口配置
        self.window_title = "数据分析工具"
        self.window_icon = None
        self.min_window_size = (1000, 700)
        self.default_window_size = (1400, 900)

        # 主题配置
        self.theme_mode = ThemeMode.LIGHT
        self.theme_color = QColor(0, 120, 212)  # Windows蓝
        self.enable_mica_effect = True

        # 导航配置
        self.navigation_width = 250
        self.navigation_compact_width = 48
        self.enable_navigation_animation = True

        # 页面配置
        self.enable_page_animation = True
        self.page_transition_duration = 300
        self.enable_loading_animation = True

        # 性能配置
        self.enable_hardware_acceleration = True
        self.vsync_enabled = True


class NavigationManager(QObject, LoggerMixin):
    """导航管理器"""

    # 信号
    pageChanged = pyqtSignal(str, str)  # (old_page, new_page)
    navigationToggled = pyqtSignal(bool)  # (is_expanded)

    def __init__(self, navigation_interface: 'NavigationInterface'):
        super().__init__()
        self.navigation = navigation_interface
        self.current_page = NavigationPage.HOME
        self.page_widgets: dict[str, Any] = {}  # 使用Any避免QWidget导入问题
        self.page_routes: dict[str, str] = {}
        self.page_container = None  # 页面容器引用，用于onClick处理

        # 连接信号（仅在PyQt6可用时）
        if HAS_PYQT6 and hasattr(self.navigation, 'currentItemChanged'):
            self.navigation.currentItemChanged.connect(self._on_current_item_changed)

    def set_page_container(self, page_container):
        """设置页面容器引用"""
        self.page_container = page_container
        self.logger.info("页面容器引用已设置")

    def add_page(
        self,
        page_id: str,
        widget: Any,  # 使用Any避免QWidget导入问题
        text: str,
        icon: Any = None,
        position: Any = None  # NavigationItemPosition可能不可用
    ):
        """添加页面"""
        try:
            # 添加到导航界面（仅在Fluent Widgets可用时）
            # NavigationInterface使用addItem方法添加导航项
            if HAS_FLUENT_WIDGETS and hasattr(self.navigation, 'addItem'):
                # 创建点击事件处理函数，用于切换页面
                def on_page_click():
                    """页面点击处理函数"""
                    try:
                        # 更新导航状态
                        old_page = self.current_page.value if isinstance(self.current_page, NavigationPage) else self.current_page
                        self.current_page = page_id
                        
                        # 切换页面内容
                        if self.page_container:
                            self.page_container.show_page(page_id)
                        
                        # 发出信号
                        self.pageChanged.emit(old_page, page_id)
                        
                        self.logger.info(f"通过导航点击切换到页面: {page_id}")
                    except Exception as e:
                        self.logger.error(f"页面点击处理失败: {page_id}, 错误: {str(e)}")
                
                # 根据qfluentwidgets文档，addItem的正确参数顺序
                # NavigationInterface.addItem(routeKey, icon, text, onClick, position)
                self.navigation.addItem(
                    routeKey=page_id,  # routeKey
                    icon=icon,         # icon (FluentIcon枚举值或None)
                    text=text,         # text
                    onClick=on_page_click,  # onClick函数
                    position=position or NavigationItemPosition.TOP  # position
                )
            elif HAS_FLUENT_WIDGETS:
                # 如果没有addItem方法，记录警告
                self.logger.warning(f"NavigationInterface没有addItem方法，页面 {page_id} 无法添加到导航")

            # 注册页面
            self.page_widgets[page_id] = widget
            self.page_routes[page_id] = page_id

            self.logger.info(f"页面已添加: {page_id} - {text}")

        except Exception as e:
            self.logger.error(f"添加页面失败: {page_id}, 错误: {str(e)}")
            raise ComponentInitializationError(f"添加页面失败: {str(e)}") from e

    def navigate_to(self, page_id: str):
        """导航到指定页面"""
        try:
            if page_id not in self.page_widgets:
                raise ValueError(f"页面不存在: {page_id}")

            old_page = self.current_page.value if isinstance(self.current_page, NavigationPage) else self.current_page
            self.current_page = page_id

            # 切换页面内容
            if self.page_container:
                self.page_container.show_page(page_id)

            # 更新导航状态（不触发onClick）
            if hasattr(self.navigation, 'setCurrentItem'):
                self.navigation.setCurrentItem(page_id)

            # 发出信号
            self.pageChanged.emit(old_page, page_id)

            self.logger.info(f"导航到页面: {page_id}")

        except Exception as e:
            self.logger.error(f"导航失败: {page_id}, 错误: {str(e)}")

    def _on_current_item_changed(self, route_key: str):
        """当前项改变处理"""
        if route_key in self.page_widgets:
            old_page = self.current_page.value if isinstance(self.current_page, NavigationPage) else self.current_page
            self.current_page = route_key
            self.pageChanged.emit(old_page, route_key)

    def get_current_page(self) -> str:
        """获取当前页面"""
        return self.current_page.value if isinstance(self.current_page, NavigationPage) else self.current_page

    def get_page_widget(self, page_id: str) -> QWidget | None:
        """获取页面widget"""
        return self.page_widgets.get(page_id)


class PageContainer(QStackedWidget, LoggerMixin):
    """页面容器 - 带动画的页面切换和响应式布局"""

    # 信号
    pageTransitionStarted = pyqtSignal(str, str)  # (old_page, new_page)
    pageTransitionFinished = pyqtSignal(str)  # (current_page)
    layoutChanged = pyqtSignal(QSize)  # (new_size)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.page_widgets: dict[str, QWidget] = {}
        self.current_page_id: str | None = None

        # 动画配置
        self.animation_enabled = True
        self.animation_duration = 300
        self.current_animation: QPropertyAnimation | None = None
        self.animation_type = "fade"  # fade, slide_horizontal, slide_vertical

        # 响应式布局配置
        self.breakpoints = {
            'mobile': 768,
            'tablet': 1024,
            'desktop': 1200
        }
        self.current_layout_mode = 'desktop'

        # 页面历史
        self.page_history: list[str] = []
        self.max_history = 10

        # 设置样式
        self.setObjectName("PageContainer")
        self._setup_styles()

        # 监听窗口大小变化
        self._last_size = self.size()
        self._setup_resize_timer()

    def _setup_styles(self):
        """设置样式"""
        self.setStyleSheet("""
            #PageContainer {
                background-color: transparent;
                border: none;
            }
            QWidget {
                background-color: transparent;
            }
        """)

    def _setup_resize_timer(self):
        """设置大小变化检测定时器"""
        self.resize_timer = QTimer()
        self.resize_timer.setSingleShot(True)
        self.resize_timer.timeout.connect(self._handle_resize)
        self.resize_timer.setInterval(150)  # 150ms延迟，避免频繁触发

    def _detect_layout_mode(self, width: int) -> str:
        """检测当前布局模式"""
        if width < self.breakpoints['mobile']:
            return 'mobile'
        elif width < self.breakpoints['tablet']:
            return 'tablet'
        elif width < self.breakpoints['desktop']:
            return 'tablet'
        else:
            return 'desktop'

    def _handle_resize(self):
        """处理窗口大小变化"""
        try:
            current_size = self.size()
            new_layout_mode = self._detect_layout_mode(current_size.width())

            if new_layout_mode != self.current_layout_mode:
                self.current_layout_mode = new_layout_mode
                self._apply_responsive_layout()
                self.logger.info(f"布局模式切换到: {new_layout_mode}")

            self.layoutChanged.emit(current_size)
            self._last_size = current_size

        except Exception as e:
            self.logger.error(f"处理窗口大小变化失败: {str(e)}")

    def _apply_responsive_layout(self):
        """应用响应式布局"""
        try:
            for _, widget in self.page_widgets.items():
                if hasattr(widget, 'apply_responsive_layout'):
                    widget.apply_responsive_layout(self.current_layout_mode)

                # 根据布局模式调整基本属性
                if self.current_layout_mode == 'mobile':
                    widget.setContentsMargins(8, 8, 8, 8)
                elif self.current_layout_mode == 'tablet':
                    widget.setContentsMargins(12, 12, 12, 12)
                else:  # desktop
                    widget.setContentsMargins(16, 16, 16, 16)

        except Exception as e:
            self.logger.warning(f"应用响应式布局失败: {str(e)}")

    def resizeEvent(self, event):
        """重写resize事件"""
        super().resizeEvent(event)

        # 延迟处理resize，避免频繁调用
        if hasattr(self, 'resize_timer'):
            self.resize_timer.stop()
            self.resize_timer.start()

    def set_animation_type(self, animation_type: str):
        """设置动画类型"""
        if animation_type in ['fade', 'slide_horizontal', 'slide_vertical']:
            self.animation_type = animation_type
            self.logger.info(f"动画类型设置为: {animation_type}")
        else:
            self.logger.warning(f"不支持的动画类型: {animation_type}")

    def set_breakpoints(self, **breakpoints):
        """自定义断点"""
        self.breakpoints.update(breakpoints)
        self.logger.info(f"断点已更新: {self.breakpoints}")

    def add_page(self, page_id: str, widget: Any):  # 使用Any避免QWidget导入问题
        """添加页面"""
        try:
            if page_id in self.page_widgets:
                self.logger.warning(f"页面已存在，将替换: {page_id}")
                self.remove_page(page_id)

            # 添加到堆栈
            index = self.addWidget(widget)

            self.page_widgets[page_id] = widget

            # 设置页面属性
            widget.setProperty("pageId", page_id)
            widget.setProperty("pageIndex", index)

            # 应用当前布局模式
            if hasattr(widget, 'apply_responsive_layout'):
                widget.apply_responsive_layout(self.current_layout_mode)

            # 设置初始边距
            if self.current_layout_mode == 'mobile':
                widget.setContentsMargins(8, 8, 8, 8)
            elif self.current_layout_mode == 'tablet':
                widget.setContentsMargins(12, 12, 12, 12)
            else:  # desktop
                widget.setContentsMargins(16, 16, 16, 16)

            self.logger.info(f"页面容器中添加页面: {page_id}")

        except Exception as e:
            self.logger.error(f"添加页面到容器失败: {page_id}, 错误: {str(e)}")
            raise

    def remove_page(self, page_id: str):
        """移除页面"""
        # try:
        #     # 从页面容器移除
        #     if self.page_container and page_id in self.page_widgets:
        #         self.page_container.remove_page(page_id)
        #
        #     # 从导航管理器移除
        #     if self.navigation_manager:
        #         self.navigation_manager.remove_page(page_id)
        #
        #     # 从页面字典移除
        #     if page_id in self.page_widgets:
        #         del self.page_widgets[page_id]
        #
        #     self.logger.info(f"页面已移除: {page_id}")
        #
        # except Exception as e:
        #     self.logger.error(f"移除页面失败: {page_id}, 错误: {str(e)}")
        pass

    def show_page(self, page_id: str, animated: bool = None, add_to_history: bool = True):
        """显示指定页面，并处理动画和历史记录"""
        if page_id not in self.page_widgets:
            self.logger.warning(f"尝试显示的页面不存在: {page_id}")
            return

        # 如果页面已经是当前页，则不执行任何操作
        if page_id == self.current_page_id:
            return

        old_page_id = self.current_page_id
        self.current_page_id = page_id

        # 添加到历史记录
        if add_to_history:
            self._add_to_history(page_id)

        # 确定是否使用动画
        use_animation = self.animation_enabled if animated is None else animated

        # 发出切换开始信号
        self.pageTransitionStarted.emit(old_page_id, page_id)

        if use_animation and old_page_id is not None:
            self._animate_page_transition(old_page_id, page_id)
        else:
            # 无动画切换
            new_widget = self.page_widgets.get(page_id)
            if new_widget:
                self.setCurrentWidget(new_widget)
                self.pageTransitionFinished.emit(page_id)
            else:
                self.logger.error(f"页面 '{page_id}' 对应的widget未找到")

    def navigate_to(self, page_id: str):
        """导航到指定页面"""
        try:
            if self.navigation_manager:
                self.navigation_manager.navigate_to(page_id)
            else:
                self.show_page(page_id)

            self.logger.info(f"导航到页面: {page_id}")

        except Exception as e:
            self.logger.error(f"导航失败: {page_id}, 错误: {str(e)}")
            raise

    def _add_to_history(self, page_id: str):
        """添加页面到历史记录"""
        if page_id in self.page_history:
            self.page_history.remove(page_id)

        self.page_history.append(page_id)

        # 限制历史记录长度
        if len(self.page_history) > self.max_history:
            self.page_history.pop(0)

    def go_back(self) -> bool:
        """返回上一页"""
        if len(self.page_history) > 0:
            previous_page = self.page_history.pop()
            self.show_page(previous_page, add_to_history=False)
            return True
        return False

    def get_page_history(self) -> list[str]:
        """获取页面历史"""
        return self.page_history.copy()

    def clear_history(self):
        """清空页面历史"""
        self.page_history.clear()
        self.logger.info("页面历史已清空")

    def _animate_page_transition(self, old_page_id: str, new_page_id: str):
        """页面切换动画"""
        try:
            # 停止当前动画
            if self.current_animation:
                self.current_animation.stop()

            old_widget = self.page_widgets[old_page_id]
            new_widget = self.page_widgets[new_page_id]

            # 设置新页面为当前页面
            self.setCurrentWidget(new_widget)
            self.current_page_id = new_page_id

            # 根据动画类型执行不同的动画
            if self.animation_type == "fade":
                self._animate_fade(new_widget)
            elif self.animation_type == "slide_horizontal":
                self._animate_slide_horizontal(old_widget, new_widget)
            elif self.animation_type == "slide_vertical":
                self._animate_slide_vertical(old_widget, new_widget)
            else:
                # 默认淡入动画
                self._animate_fade(new_widget)

        except Exception as e:
            self.logger.error(f"页面切换动画失败: {str(e)}")
            # 降级到无动画切换
            self.setCurrentWidget(self.page_widgets[new_page_id])
            self.current_page_id = new_page_id
            self.pageTransitionFinished.emit(new_page_id)

    def _animate_fade(self, new_widget: QWidget):
        """淡入动画"""
        try:
            # 创建透明度动画
            self.opacity_effect = QGraphicsOpacityEffect()
            new_widget.setGraphicsEffect(self.opacity_effect)

            self.current_animation = QPropertyAnimation(self.opacity_effect, b"opacity")
            self.current_animation.setDuration(self.animation_duration)
            self.current_animation.setStartValue(0.0)
            self.current_animation.setEndValue(1.0)
            self.current_animation.setEasingCurve(QEasingCurve.Type.OutCubic)

            # 动画完成后清理
            self.current_animation.finished.connect(
                lambda: self._cleanup_animation(new_widget)
            )

            self.current_animation.start()

        except Exception as e:
            self.logger.error(f"淡入动画失败: {str(e)}")
            self._cleanup_animation(new_widget)

    def _animate_slide_horizontal(self, old_widget: QWidget, new_widget: QWidget):
        """水平滑动动画"""
        try:
            # 设置新widget的初始位置（从右边滑入）
            widget_width = self.width()
            new_widget.move(widget_width, 0)
            new_widget.show()

            # 创建位置动画
            self.current_animation = QPropertyAnimation(new_widget, b"pos")
            self.current_animation.setDuration(self.animation_duration)
            self.current_animation.setStartValue(new_widget.pos())
            self.current_animation.setEndValue(QRect(0, 0, widget_width, self.height()).topLeft())
            self.current_animation.setEasingCurve(QEasingCurve.Type.OutCubic)

            # 动画完成后清理
            self.current_animation.finished.connect(
                lambda: self._cleanup_slide_animation(new_widget)
            )

            self.current_animation.start()

        except Exception as e:
            self.logger.error(f"水平滑动动画失败: {str(e)}")
            self._cleanup_slide_animation(new_widget)

    def _animate_slide_vertical(self, old_widget: QWidget, new_widget: QWidget):
        """垂直滑动动画"""
        try:
            # 设置新widget的初始位置（从下方滑入）
            widget_height = self.height()
            new_widget.move(0, widget_height)
            new_widget.show()

            # 创建位置动画
            self.current_animation = QPropertyAnimation(new_widget, b"pos")
            self.current_animation.setDuration(self.animation_duration)
            self.current_animation.setStartValue(new_widget.pos())
            self.current_animation.setEndValue(QRect(0, 0, self.width(), widget_height).topLeft())
            self.current_animation.setEasingCurve(QEasingCurve.Type.OutCubic)

            # 动画完成后清理
            self.current_animation.finished.connect(
                lambda: self._cleanup_slide_animation(new_widget)
            )

            self.current_animation.start()

        except Exception as e:
            self.logger.error(f"垂直滑动动画失败: {str(e)}")
            self._cleanup_slide_animation(new_widget)

    def _cleanup_animation(self, widget: QWidget):
        """清理淡入动画资源"""
        try:
            widget.setGraphicsEffect(None)
            self.current_animation = None
            self.pageTransitionFinished.emit(self.current_page_id)
        except Exception as e:
            self.logger.warning(f"清理动画资源失败: {str(e)}")

    def _cleanup_slide_animation(self, widget: QWidget):
        """清理滑动动画资源"""
        try:
            widget.move(0, 0)  # 确保位置正确
            self.current_animation = None
            self.pageTransitionFinished.emit(self.current_page_id)
        except Exception as e:
            self.logger.warning(f"清理滑动动画资源失败: {str(e)}")

    def get_current_page_id(self) -> str | None:
        """获取当前页面ID"""
        return self.current_page_id

    def get_page_widget(self, page_id: str) -> QWidget | None:
        """获取页面widget"""
        return self.page_widgets.get(page_id)

    def get_layout_mode(self) -> str:
        """获取当前布局模式"""
        return self.current_layout_mode

    def get_available_pages(self) -> list[str]:
        """获取所有可用页面ID"""
        return list(self.page_widgets.keys())

    def has_page(self, page_id: str) -> bool:
        """检查页面是否存在"""
        return page_id in self.page_widgets

    def get_page_count(self) -> int:
        """获取页面数量"""
        return len(self.page_widgets)

    def set_animation_duration(self, duration: int):
        """设置动画持续时间"""
        if 100 <= duration <= 2000:
            self.animation_duration = duration
            self.logger.info(f"动画持续时间设置为: {duration}ms")
        else:
            self.logger.warning(f"动画持续时间超出范围(100-2000ms): {duration}")

    def enable_animation(self, enabled: bool = True):
        """启用/禁用动画"""
        self.animation_enabled = enabled
        self.logger.info(f"页面切换动画: {'启用' if enabled else '禁用'}")

    def is_transitioning(self) -> bool:
        """检查是否正在进行页面切换"""
        return self.current_animation is not None and self.current_animation.state() == QPropertyAnimation.State.Running


class MainWindow(MSFluentWindow, LoggerMixin):
    """主窗口"""

    # 信号
    windowStateChanged = pyqtSignal(str)  # 窗口状态改变
    pageChanged = pyqtSignal(str, str)  # 页面改变 (old_page, new_page)

    def __init__(self, config: UIConfig | None = None):
        super().__init__()

        if not HAS_PYQT6:
            raise ComponentInitializationError("PyQt6未安装")
        if not HAS_FLUENT_WIDGETS:
            raise ComponentInitializationError("PyQt-Fluent-Widgets未安装")

        self.config = config or UIConfig()

        # 初始化组件
        self.navigation_manager: NavigationManager | None = None
        self.page_container: PageContainer | None = None
        self.current_window_state = WindowState.NORMAL

        # 初始化页面管理相关属性
        self.page_widgets: dict[str, Any] = {}  # 页面widget字典
        self.current_page_id: str | None = None  # 当前页面ID

        # 初始化界面
        self._init_window()
        self._init_theme()
        self._init_navigation()
        self._init_page_container()
        self._setup_connections()

        self.logger.info("主窗口初始化完成")

    def _init_window(self):
        """初始化窗口"""
        try:
            # 设置窗口基本属性
            self.setWindowTitle(self.config.window_title)

            # 设置窗口大小
            self.resize(*self.config.default_window_size)
            self.setMinimumSize(*self.config.min_window_size)

            # 设置窗口图标
            if self.config.window_icon:
                self.setWindowIcon(QIcon(self.config.window_icon))

            # 居中显示
            self._center_window()

            # 设置窗口属性
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)

        except Exception as e:
            self.logger.error(f"初始化窗口失败: {str(e)}")
            raise ComponentInitializationError(f"初始化窗口失败: {str(e)}") from e

    def _init_theme(self):
        """初始化主题"""
        try:
            # 设置主题颜色
            setThemeColor(self.config.theme_color)

            # 设置主题模式
            if self.config.theme_mode == ThemeMode.DARK:
                setTheme(Theme.DARK)
            elif self.config.theme_mode == ThemeMode.LIGHT:
                setTheme(Theme.LIGHT)
            else:
                setTheme(Theme.AUTO)

            self.logger.info(f"主题设置完成: {self.config.theme_mode.value}")

        except Exception as e:
            self.logger.error(f"初始化主题失败: {str(e)}")

    def _init_navigation(self):
        """初始化导航"""
        try:
            # 创建导航管理器
            self.navigation_manager = NavigationManager(self.navigationInterface)

            # 设置导航宽度 - 临时注释掉，PyQt6-Fluent-Widgets可能不支持此方法
            # self.navigationInterface.setExpandWidth(self.config.navigation_width)
            # TODO: 查找PyQt6-Fluent-Widgets中设置导航宽度的正确方法

            # PyQt6-Fluent-Widgets使用NavigationInterface.setDefaultRouteKey替代qrouter
            if HAS_FLUENT_WIDGETS and hasattr(self.navigationInterface, 'setDefaultRouteKey'):
                self.navigationInterface.setDefaultRouteKey(NavigationPage.HOME.value)

            self.logger.info("导航系统初始化完成")

        except Exception as e:
            self.logger.error(f"初始化导航失败: {str(e)}")
            raise ComponentInitializationError(f"初始化导航失败: {str(e)}") from e

    def _init_page_container(self):
        """初始化页面容器"""
        try:
            # 替换默认的stacked widget
            self.page_container = PageContainer()

            self.page_container.animation_enabled = self.config.enable_page_animation
            self.page_container.animation_duration = self.config.page_transition_duration

            # 替换现有的stackedWidget
            old_stacked = self.stackedWidget
            layout = old_stacked.parent().layout() if old_stacked.parent() else None

            if layout:
                layout.replaceWidget(old_stacked, self.page_container)
                old_stacked.deleteLater()
                self.stackedWidget = self.page_container
            else:
                self.stackedWidget = self.page_container

            # 设置导航管理器的页面容器引用
            if self.navigation_manager:
                self.navigation_manager.set_page_container(self.page_container)

            self.logger.info("页面容器初始化完成")

        except Exception as e:
            self.logger.error(f"初始化页面容器失败: {str(e)}")
            raise ComponentInitializationError(f"初始化页面容器失败: {str(e)}") from e

    def _setup_connections(self):
        """设置信号连接"""
        try:
            # 连接导航管理器信号
            if self.navigation_manager:
                self.navigation_manager.pageChanged.connect(self._on_page_changed)

            # 连接页面容器信号
            if self.page_container is not None:
                self.page_container.pageTransitionStarted.connect(self._on_page_transition_started)
                self.page_container.pageTransitionFinished.connect(self._on_page_transition_finished)
                self.page_container.layoutChanged.connect(self._on_layout_changed)

            # 连接窗口状态信号
            self.windowStateChanged.connect(self._on_window_state_changed)

        except Exception as e:
            self.logger.error(f"设置信号连接失败: {str(e)}")

    def _on_page_transition_started(self, old_page: str, new_page: str):
        """页面切换开始处理"""
        self.logger.debug(f"页面切换开始: {old_page} -> {new_page}")

    def _on_page_transition_finished(self, page_id: str):
        """页面切换完成处理"""
        self.logger.debug(f"页面切换完成: {page_id}")

    def _on_layout_changed(self, new_size: QSize):
        """布局变化处理"""
        if self.page_container is not None:
            layout_mode = self.page_container.get_layout_mode()
            self.logger.debug(f"布局模式: {layout_mode}, 窗口大小: {new_size.width()}x{new_size.height()}")

    def _on_page_changed(self, old_page: str, new_page: str):
        """页面改变处理"""
        try:
            # 显示新页面
            if self.page_container is not None:
                self.page_container.show_page(new_page)

            # 更新当前页面ID
            self.current_page_id = new_page

            # 发出信号
            self.pageChanged.emit(old_page, new_page)

            self.logger.info(f"页面切换: {old_page} -> {new_page}")

        except Exception as e:
            self.logger.error(f"页面切换处理失败: {str(e)}")

    def _on_window_state_changed(self, state: str):
        """窗口状态改变处理"""
        try:
            self.current_window_state = WindowState(state)
            self.logger.debug(f"窗口状态改变: {state}")
        except Exception as e:
            self.logger.error(f"窗口状态改变处理失败: {str(e)}")

    def _center_window(self):
        """窗口居中"""
        try:
            if QApplication.instance():
                screen = QApplication.instance().primaryScreen()
                if screen:
                    screen_geometry = screen.availableGeometry()
                    window_geometry = self.frameGeometry()
                    center_point = screen_geometry.center()
                    window_geometry.moveCenter(center_point)
                    self.move(window_geometry.topLeft())
        except Exception as e:
            self.logger.warning(f"窗口居中失败: {str(e)}")

    def add_page(
        self,
        page_id: str,
        widget: Any,  # 使用Any避免QWidget导入问题
        text: str,
        icon: Any = None,
        position: Any = None  # NavigationItemPosition可能不可用
    ):
        """添加页面"""
        try:
            # 添加到页面容器
            if self.page_container is not None:
                self.page_container.add_page(page_id, widget)

            # 添加到导航管理器
            if self.navigation_manager:
                self.navigation_manager.add_page(page_id, widget, text, icon, position)

            # 记录到页面字典
            self.page_widgets[page_id] = widget

            self.logger.info(f"主窗口中添加页面: {page_id}")

        except Exception as e:
            self.logger.error(f"添加页面失败: {page_id}, 错误: {str(e)}")
            raise

    def navigate_to(self, page_id: str):
        """导航到指定页面"""
        try:
            if self.navigation_manager:
                self.navigation_manager.navigate_to(page_id)
            else:
                self.show_page(page_id)
            self.logger.info(f"导航到页面: {page_id}")
        except Exception as e:
            self.logger.error(f"导航失败: {page_id}, 错误: {str(e)}")
            raise

    def show_page(self, page_id: str):
        """显示指定页面"""
        try:
            if self.page_container is not None:
                self.page_container.show_page(page_id)
            self.current_page_id = page_id
            self.logger.info(f"显示页面: {page_id}")
        except Exception as e:
            self.logger.error(f"显示页面失败: {page_id}, 错误: {str(e)}")
            raise

    def remove_page(self, page_id: str):
        """移除页面"""
        try:
            # 从页面容器移除
            if self.page_container is not None:
                self.page_container.remove_page(page_id)

            # 从导航管理器移除
            if self.navigation_manager:
                self.navigation_manager.remove_page(page_id)

            # 从页面字典移除
            if page_id in self.page_widgets:
                del self.page_widgets[page_id]

            self.logger.info(f"页面已移除: {page_id}")

        except Exception as e:
            self.logger.error(f"移除页面失败: {page_id}, 错误: {str(e)}")
            raise

    def _add_to_history(self, page_id: str):
        """添加页面到历史记录"""
        if self.page_container is not None:
            self.page_container._add_to_history(page_id)

    def go_back(self) -> bool:
        """返回上一页"""
        if self.page_container is not None:
            return self.page_container.go_back()
        return False

    def get_page_history(self) -> list[str]:
        """获取页面历史"""
        if self.page_container is not None:
            return self.page_container.get_page_history()
        return []

    def get_layout_mode(self) -> str:
        """获取当前布局模式"""
        if self.page_container is not None:
            return self.page_container.get_layout_mode()
        return 'desktop'

    def set_responsive_breakpoints(self, **breakpoints):
        """设置响应式断点"""
        if self.page_container is not None:
            self.page_container.set_breakpoints(**breakpoints)

    def closeEvent(self, event):
        """关闭事件"""
        try:
            self.logger.info("主窗口关闭")
            super().closeEvent(event)
        except Exception as e:
            self.logger.error(f"关闭窗口时出错: {str(e)}")
            event.accept()


def create_main_window(config: UIConfig | None = None) -> MainWindow:
    """创建主窗口的工厂函数"""
    try:
        return MainWindow(config)
    except Exception as e:
        print(f"创建主窗口失败: {str(e)}")
