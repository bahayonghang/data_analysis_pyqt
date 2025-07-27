"""
交互式图表组件
基于PyQt6和matplotlib实现可缩放、可拖拽的图表
"""

from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

try:
    from PyQt6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QToolBar,
        QPushButton, QLabel, QComboBox, QCheckBox,
        QSlider, QSpinBox, QGroupBox, QSplitter,
        QAction, QActionGroup, QMenu, QSizePolicy
    )
    from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QObject
    from PyQt6.QtGui import QIcon, QPixmap, QPainter, QFont
    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False

try:
    import matplotlib
    matplotlib.use('Qt6Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qt6agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt6agg import NavigationToolbar2QT as NavigationToolbar
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    import matplotlib.dates as mdates
    from matplotlib.widgets import RectangleSelector, SpanSelector
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from ..utils.basic_logging import LoggerMixin
from ..utils.exceptions import UIError


class PlotType(str, Enum):
    """图表类型"""
    LINE = "line"
    BAR = "bar"
    SCATTER = "scatter"
    HISTOGRAM = "histogram"
    BOX_PLOT = "box_plot"
    VIOLIN_PLOT = "violin_plot"
    HEATMAP = "heatmap"
    CONTOUR = "contour"
    PIE = "pie"
    AREA = "area"


class PlotTheme(str, Enum):
    """图表主题"""
    DEFAULT = "default"
    SEABORN = "seaborn-v0_8"
    CLASSIC = "classic"
    DARK = "dark_background"
    MODERN = "seaborn-v0_8-whitegrid"
    MINIMAL = "seaborn-v0_8-white"


class NavigationMode(str, Enum):
    """导航模式"""
    PAN = "pan"
    ZOOM = "zoom"
    SELECT = "select"
    MEASURE = "measure"


class ChartInteraction(str, Enum):
    """图表交互类型"""
    CLICK = "click"
    HOVER = "hover"
    SELECT = "select"
    ZOOM = "zoom"
    PAN = "pan"


@dataclass
class ChartConfig:
    """图表配置"""
    # 基本设置
    width: float = 10.0
    height: float = 6.0
    dpi: int = 100
    
    # 样式设置
    theme: PlotTheme = PlotTheme.DEFAULT
    background_color: str = "white"
    grid: bool = True
    
    # 交互设置
    enable_zoom: bool = True
    enable_pan: bool = True
    enable_selection: bool = True
    
    # 动画设置
    enable_animation: bool = True
    animation_duration: int = 300
    
    # 性能设置
    use_blit: bool = True
    max_points: int = 10000
    
    # 工具栏设置
    show_toolbar: bool = True
    toolbar_position: str = "top"  # top, bottom, left, right


@dataclass  
class PlotData:
    """图表数据"""
    x: Union[np.ndarray, List, pd.Series] = None
    y: Union[np.ndarray, List, pd.Series] = None
    z: Union[np.ndarray, List, pd.Series] = None  # 用于3D或热力图
    
    # 样式属性
    color: Union[str, List[str]] = "blue"
    marker: str = "o"
    linestyle: str = "-"
    linewidth: float = 1.0
    alpha: float = 1.0
    
    # 标签
    label: Optional[str] = None
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None
    title: Optional[str] = None
    
    # 数据信息
    data_source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class InteractiveChartWidget(QWidget, LoggerMixin):
    """交互式图表组件"""
    
    # 信号定义
    data_selected = pyqtSignal(object)  # 数据选择信号
    chart_clicked = pyqtSignal(float, float)  # 图表点击信号
    zoom_changed = pyqtSignal(tuple)  # 缩放变化信号
    theme_changed = pyqtSignal(str)  # 主题变化信号
    
    def __init__(self, config: Optional[ChartConfig] = None, parent=None):
        super().__init__(parent)
        
        if not HAS_PYQT6:
            raise ImportError("PyQt6未安装，无法使用交互式图表功能")
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib未安装，无法绘制图表")
        if not HAS_PANDAS:
            raise ImportError("pandas未安装，无法处理数据")
        
        self.config = config or ChartConfig()
        self.plots = {}  # 存储多个绘图对象
        self.selectors = {}  # 存储选择器
        self.current_plot_type = PlotType.LINE
        self.navigation_mode = NavigationMode.PAN
        
        # 设置matplotlib主题
        self._apply_theme()
        
        # 初始化UI
        self._init_ui()
        self._setup_figure()
        self._connect_events()
        
        self.logger.info("交互式图表组件初始化完成")
    
    def _init_ui(self):
        """初始化用户界面"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 创建工具栏
        if self.config.show_toolbar:
            self.toolbar = self._create_toolbar()
            if self.config.toolbar_position == "top":
                layout.addWidget(self.toolbar)
        
        # 创建图表区域
        self.chart_area = QWidget()
        chart_layout = QVBoxLayout(self.chart_area)
        chart_layout.setContentsMargins(0, 0, 0, 0)
        
        layout.addWidget(self.chart_area)
        
        # 创建状态栏
        self.status_label = QLabel("就绪")
        layout.addWidget(self.status_label)
        
        if self.config.show_toolbar and self.config.toolbar_position == "bottom":
            layout.addWidget(self.toolbar)
    
    def _create_toolbar(self) -> QToolBar:
        """创建工具栏"""
        toolbar = QToolBar("图表工具栏")
        toolbar.setOrientation(Qt.Orientation.Horizontal)
        
        # 导航模式按钮组
        nav_group = QActionGroup(self)
        
        # 平移模式
        pan_action = QAction("平移", self)
        pan_action.setCheckable(True)
        pan_action.setChecked(True)
        pan_action.triggered.connect(lambda: self._set_navigation_mode(NavigationMode.PAN))
        nav_group.addAction(pan_action)
        toolbar.addAction(pan_action)
        
        # 缩放模式
        zoom_action = QAction("缩放", self)
        zoom_action.setCheckable(True)
        zoom_action.triggered.connect(lambda: self._set_navigation_mode(NavigationMode.ZOOM))
        nav_group.addAction(zoom_action)
        toolbar.addAction(zoom_action)
        
        # 选择模式
        select_action = QAction("选择", self)
        select_action.setCheckable(True)
        select_action.triggered.connect(lambda: self._set_navigation_mode(NavigationMode.SELECT))
        nav_group.addAction(select_action)
        toolbar.addAction(select_action)
        
        toolbar.addSeparator()
        
        # 重置视图
        reset_action = QAction("重置视图", self)
        reset_action.triggered.connect(self.reset_view)
        toolbar.addAction(reset_action)
        
        # 保存图表
        save_action = QAction("保存", self)
        save_action.triggered.connect(self.save_chart)
        toolbar.addAction(save_action)
        
        toolbar.addSeparator()
        
        # 主题选择
        theme_combo = QComboBox()
        theme_combo.addItems([theme.value for theme in PlotTheme])
        theme_combo.setCurrentText(self.config.theme.value)
        theme_combo.currentTextChanged.connect(self._change_theme)
        toolbar.addWidget(QLabel("主题:"))
        toolbar.addWidget(theme_combo)
        
        # 网格切换
        grid_checkbox = QCheckBox("网格")
        grid_checkbox.setChecked(self.config.grid)
        grid_checkbox.toggled.connect(self._toggle_grid)
        toolbar.addWidget(grid_checkbox)
        
        return toolbar
    
    def _setup_figure(self):
        """设置matplotlib图形"""
        # 创建图形和画布
        self.figure = Figure(
            figsize=(self.config.width, self.config.height),
            dpi=self.config.dpi,
            facecolor=self.config.background_color
        )
        
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # 创建默认轴
        self.axes = self.figure.add_subplot(111)
        self.axes.grid(self.config.grid)
        
        # 创建导航工具栏
        self.nav_toolbar = NavigationToolbar(self.canvas, self)
        
        # 添加到布局
        chart_layout = self.chart_area.layout()
        chart_layout.addWidget(self.canvas)
        chart_layout.addWidget(self.nav_toolbar)
        
        # 设置画布更新策略
        if self.config.use_blit:
            self.canvas.draw()
            self.background = self.canvas.copy_from_bbox(self.axes.bbox)
    
    def _connect_events(self):
        """连接事件处理器"""
        # 鼠标事件
        self.canvas.mpl_connect('button_press_event', self._on_click)
        self.canvas.mpl_connect('motion_notify_event', self._on_hover)
        self.canvas.mpl_connect('button_release_event', self._on_release)
        
        # 键盘事件
        self.canvas.mpl_connect('key_press_event', self._on_key_press)
        
        # 缩放事件
        self.canvas.mpl_connect('xlim_changed', self._on_xlim_changed)
        self.canvas.mpl_connect('ylim_changed', self._on_ylim_changed)
        
        # 设置焦点策略
        self.canvas.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
    
    def _apply_theme(self):
        """应用图表主题"""
        try:
            plt.style.use(self.config.theme.value)
        except OSError:
            # 如果主题不存在，使用默认主题
            plt.style.use('default')
            self.logger.warning(f"主题 {self.config.theme.value} 不存在，使用默认主题")
    
    def _set_navigation_mode(self, mode: NavigationMode):
        """设置导航模式"""
        self.navigation_mode = mode
        
        # 清除现有选择器
        self._clear_selectors()
        
        if mode == NavigationMode.SELECT:
            # 创建矩形选择器
            self.selectors['rect'] = RectangleSelector(
                self.axes,
                self._on_rectangle_select,
                useblit=self.config.use_blit,
                button=[1],  # 仅左键
                minspanx=5,
                minspany=5,
                spancoords='pixels',
                interactive=True
            )
        elif mode == NavigationMode.MEASURE:
            # 创建跨度选择器
            self.selectors['span'] = SpanSelector(
                self.axes,
                self._on_span_select,
                'horizontal',
                useblit=self.config.use_blit,
                button=[1]
            )
        
        self.status_label.setText(f"导航模式: {mode.value}")
        self.logger.info(f"导航模式切换为: {mode.value}")
    
    def _clear_selectors(self):
        """清除所有选择器"""
        for selector in self.selectors.values():
            if hasattr(selector, 'set_active'):
                selector.set_active(False)
        self.selectors.clear()
    
    def plot(self, data: PlotData, plot_type: PlotType = PlotType.LINE, **kwargs) -> str:
        """绘制图表"""
        try:
            # 数据验证
            if data.x is None or data.y is None:
                raise ValueError("x和y数据不能为空")
            
            # 数据转换
            x_data = self._convert_data(data.x)
            y_data = self._convert_data(data.y)
            
            # 数据量检查
            if len(x_data) > self.config.max_points:
                self.logger.warning(f"数据点数量 ({len(x_data)}) 超过最大限制 ({self.config.max_points})，将进行采样")
                indices = np.linspace(0, len(x_data) - 1, self.config.max_points, dtype=int)
                x_data = x_data[indices]
                y_data = y_data[indices]
            
            # 生成唯一的plot ID
            plot_id = f"{plot_type.value}_{len(self.plots)}"
            
            # 绘制图表
            if plot_type == PlotType.LINE:
                line, = self.axes.plot(
                    x_data, y_data,
                    color=data.color,
                    marker=data.marker,
                    linestyle=data.linestyle,
                    linewidth=data.linewidth,
                    alpha=data.alpha,
                    label=data.label,
                    **kwargs
                )
                plot_obj = line
                
            elif plot_type == PlotType.SCATTER:
                scatter = self.axes.scatter(
                    x_data, y_data,
                    c=data.color,
                    marker=data.marker,
                    alpha=data.alpha,
                    label=data.label,
                    **kwargs
                )
                plot_obj = scatter
                
            elif plot_type == PlotType.BAR:
                bars = self.axes.bar(
                    x_data, y_data,
                    color=data.color,
                    alpha=data.alpha,
                    label=data.label,
                    **kwargs
                )
                plot_obj = bars
                
            elif plot_type == PlotType.HISTOGRAM:
                n, bins, patches = self.axes.hist(
                    y_data,  # 直方图只需要y数据
                    bins=kwargs.get('bins', 30),
                    color=data.color,
                    alpha=data.alpha,
                    label=data.label,
                    **{k: v for k, v in kwargs.items() if k != 'bins'}
                )
                plot_obj = patches
                
            elif plot_type == PlotType.BOX_PLOT:
                box_plot = self.axes.boxplot(
                    [y_data],
                    patch_artist=True,
                    labels=[data.label or "数据"],
                    **kwargs
                )
                # 设置颜色
                for patch in box_plot['boxes']:
                    patch.set_facecolor(data.color)
                    patch.set_alpha(data.alpha)
                plot_obj = box_plot
                
            else:
                raise ValueError(f"不支持的图表类型: {plot_type}")
            
            # 设置标签
            if data.xlabel:
                self.axes.set_xlabel(data.xlabel)
            if data.ylabel:
                self.axes.set_ylabel(data.ylabel)
            if data.title:
                self.axes.set_title(data.title)
            
            # 添加图例
            if data.label:
                self.axes.legend()
            
            # 存储绘图对象
            self.plots[plot_id] = {
                'data': data,
                'type': plot_type,
                'object': plot_obj,
                'visible': True
            }
            
            # 更新显示
            self._update_display()
            
            self.logger.info(f"绘制图表完成: {plot_type.value}, ID: {plot_id}")
            return plot_id
            
        except Exception as e:
            self.logger.error(f"绘制图表失败: {str(e)}")
            raise UIError(f"绘制图表失败: {str(e)}") from e
    
    def _convert_data(self, data: Union[np.ndarray, List, pd.Series]) -> np.ndarray:
        """转换数据为numpy数组"""
        if isinstance(data, pd.Series):
            return data.values
        elif isinstance(data, list):
            return np.array(data)
        elif isinstance(data, np.ndarray):
            return data
        else:
            raise ValueError(f"不支持的数据类型: {type(data)}")
    
    def _update_display(self):
        """更新显示"""
        if self.config.use_blit:
            self.canvas.restore_region(self.background)
            for plot_info in self.plots.values():
                if plot_info['visible']:
                    self.axes.draw_artist(plot_info['object'])
            self.canvas.blit(self.axes.bbox)
        else:
            self.canvas.draw()
    
    def clear(self):
        """清除所有图表"""
        self.axes.clear()
        self.plots.clear()
        self._clear_selectors()
        self.axes.grid(self.config.grid)
        self.canvas.draw()
        self.logger.info("清除所有图表")
    
    def remove_plot(self, plot_id: str):
        """移除指定的图表"""
        if plot_id in self.plots:
            plot_obj = self.plots[plot_id]['object']
            if hasattr(plot_obj, 'remove'):
                plot_obj.remove()
            del self.plots[plot_id]
            self._update_display()
            self.logger.info(f"移除图表: {plot_id}")
    
    def set_plot_visibility(self, plot_id: str, visible: bool):
        """设置图表可见性"""
        if plot_id in self.plots:
            self.plots[plot_id]['visible'] = visible
            plot_obj = self.plots[plot_id]['object']
            if hasattr(plot_obj, 'set_visible'):
                plot_obj.set_visible(visible)
            self._update_display()
    
    def reset_view(self):
        """重置视图"""
        self.axes.relim()
        self.axes.autoscale()
        self.canvas.draw()
        self.status_label.setText("视图已重置")
    
    def save_chart(self, filename: Optional[str] = None):
        """保存图表"""
        try:
            if filename is None:
                from PyQt6.QtWidgets import QFileDialog
                filename, _ = QFileDialog.getSaveFileName(
                    self, "保存图表", "", 
                    "PNG文件 (*.png);;SVG文件 (*.svg);;PDF文件 (*.pdf)"
                )
            
            if filename:
                self.figure.savefig(filename, dpi=300, bbox_inches='tight')
                self.status_label.setText(f"图表已保存: {filename}")
                self.logger.info(f"图表保存成功: {filename}")
        except Exception as e:
            self.logger.error(f"保存图表失败: {str(e)}")
    
    def _change_theme(self, theme_name: str):
        """更改主题"""
        try:
            old_theme = self.config.theme
            self.config.theme = PlotTheme(theme_name)
            self._apply_theme()
            
            # 重新绘制现有图表
            self._redraw_all_plots()
            
            self.theme_changed.emit(theme_name)
            self.status_label.setText(f"主题已更改: {theme_name}")
            self.logger.info(f"主题从 {old_theme} 更改为 {theme_name}")
            
        except Exception as e:
            self.logger.error(f"更改主题失败: {str(e)}")
    
    def _toggle_grid(self, checked: bool):
        """切换网格显示"""
        self.config.grid = checked
        self.axes.grid(checked)
        self.canvas.draw()
        self.status_label.setText(f"网格: {'开启' if checked else '关闭'}")
    
    def _redraw_all_plots(self):
        """重新绘制所有图表"""
        if not self.plots:
            return
        
        # 保存当前图表数据
        plots_backup = self.plots.copy()
        
        # 清除图表
        self.clear()
        
        # 重新绘制
        for plot_id, plot_info in plots_backup.items():
            self.plot(
                plot_info['data'],
                plot_info['type']
            )
    
    # 事件处理器
    def _on_click(self, event):
        """鼠标点击事件"""
        if event.inaxes == self.axes:
            self.chart_clicked.emit(event.xdata, event.ydata)
            self.status_label.setText(f"点击位置: ({event.xdata:.2f}, {event.ydata:.2f})")
    
    def _on_hover(self, event):
        """鼠标悬停事件"""
        if event.inaxes == self.axes:
            self.status_label.setText(f"位置: ({event.xdata:.2f}, {event.ydata:.2f})")
    
    def _on_release(self, event):
        """鼠标释放事件"""
        pass
    
    def _on_key_press(self, event):
        """键盘按键事件"""
        if event.key == 'r':
            self.reset_view()
        elif event.key == 'g':
            self._toggle_grid(not self.config.grid)
        elif event.key == 's':
            self.save_chart()
    
    def _on_xlim_changed(self, axes):
        """X轴范围变化事件"""
        xlim = axes.get_xlim()
        self.zoom_changed.emit(('x', xlim))
    
    def _on_ylim_changed(self, axes):
        """Y轴范围变化事件"""
        ylim = axes.get_ylim()
        self.zoom_changed.emit(('y', ylim))
    
    def _on_rectangle_select(self, eclick, erelease):
        """矩形选择事件"""
        x1, x2 = sorted([eclick.xdata, erelease.xdata])
        y1, y2 = sorted([eclick.ydata, erelease.ydata])
        
        selected_data = {
            'type': 'rectangle',
            'bounds': (x1, y1, x2, y2),
            'area': (x2 - x1) * (y2 - y1)
        }
        
        self.data_selected.emit(selected_data)
        self.status_label.setText(f"选择区域: ({x1:.2f}, {y1:.2f}) to ({x2:.2f}, {y2:.2f})")
    
    def _on_span_select(self, xmin, xmax):
        """跨度选择事件"""
        selected_data = {
            'type': 'span',
            'range': (xmin, xmax),
            'width': xmax - xmin
        }
        
        self.data_selected.emit(selected_data)
        self.status_label.setText(f"选择范围: {xmin:.2f} to {xmax:.2f}")


def create_interactive_chart(config: Optional[ChartConfig] = None) -> InteractiveChartWidget:
    """创建交互式图表组件的工厂函数"""
    try:
        return InteractiveChartWidget(config)
    except Exception as e:
        raise UIError(f"创建交互式图表组件失败: {str(e)}") from e