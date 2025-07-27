"""
图表工厂模块
提供统一的图表创建接口
"""

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from .interactive_chart import (
    InteractiveChartWidget, 
    PlotData, 
    PlotType, 
    ChartConfig, 
    PlotTheme,
    create_interactive_chart
)
from ..utils.basic_logging import LoggerMixin
from ..utils.exceptions import UIError


class ChartPreset(str, Enum):
    """图表预设"""
    # 科学研究
    SCIENTIFIC = "scientific"
    PUBLICATION = "publication"
    
    # 商业报告
    BUSINESS = "business"
    DASHBOARD = "dashboard"
    
    # 数据探索
    EXPLORATION = "exploration"
    ANALYSIS = "analysis"
    
    # 演示展示
    PRESENTATION = "presentation"
    INTERACTIVE = "interactive"


class ChartFactory(LoggerMixin):
    """图表工厂类"""
    
    # 预设配置
    PRESET_CONFIGS = {
        ChartPreset.SCIENTIFIC: ChartConfig(
            theme=PlotTheme.SEABORN,
            width=12, height=8,
            grid=True,
            enable_animation=False,
            use_blit=False
        ),
        
        ChartPreset.PUBLICATION: ChartConfig(
            theme=PlotTheme.CLASSIC,
            width=10, height=6,
            dpi=300,
            background_color="white",
            grid=False,
            enable_animation=False
        ),
        
        ChartPreset.BUSINESS: ChartConfig(
            theme=PlotTheme.MODERN,
            width=14, height=8,
            grid=True,
            show_toolbar=True,
            enable_animation=True
        ),
        
        ChartPreset.DASHBOARD: ChartConfig(
            theme=PlotTheme.DARK,
            width=12, height=6,
            background_color="black",
            grid=True,
            enable_animation=True,
            animation_duration=200
        ),
        
        ChartPreset.EXPLORATION: ChartConfig(
            theme=PlotTheme.DEFAULT,
            width=10, height=8,
            grid=True,
            enable_zoom=True,
            enable_pan=True,
            enable_selection=True,
            show_toolbar=True
        ),
        
        ChartPreset.ANALYSIS: ChartConfig(
            theme=PlotTheme.SEABORN,
            width=14, height=10,
            grid=True,
            enable_zoom=True,
            enable_pan=True,
            enable_selection=True,
            max_points=50000
        ),
        
        ChartPreset.PRESENTATION: ChartConfig(
            theme=PlotTheme.MINIMAL,
            width=16, height=9,
            dpi=150,
            background_color="white",
            grid=False,
            enable_animation=True,
            animation_duration=500
        ),
        
        ChartPreset.INTERACTIVE: ChartConfig(
            theme=PlotTheme.MODERN,
            width=12, height=8,
            grid=True,
            enable_zoom=True,
            enable_pan=True,
            enable_selection=True,
            enable_animation=True,
            show_toolbar=True,
            use_blit=True
        )
    }
    
    @classmethod
    def create_chart(
        cls,
        preset: Optional[ChartPreset] = None,
        config: Optional[ChartConfig] = None,
        **kwargs
    ) -> InteractiveChartWidget:
        """创建图表组件
        
        Args:
            preset: 预设配置类型
            config: 自定义配置（会覆盖预设配置）
            **kwargs: 其他配置参数
            
        Returns:
            InteractiveChartWidget: 图表组件实例
        """
        try:
            # 获取基础配置
            if preset:
                base_config = cls.PRESET_CONFIGS.get(preset, ChartConfig())
            else:
                base_config = ChartConfig()
            
            # 合并自定义配置
            if config:
                # 更新配置属性
                for attr, value in config.__dict__.items():
                    setattr(base_config, attr, value)
            
            # 应用额外参数
            for key, value in kwargs.items():
                if hasattr(base_config, key):
                    setattr(base_config, key, value)
            
            # 创建图表组件
            chart = create_interactive_chart(base_config)
            
            logger = cls._get_logger()
            logger.info(f"创建图表成功，预设: {preset}, 配置: {base_config}")
            
            return chart
            
        except Exception as e:
            logger = cls._get_logger()
            logger.error(f"创建图表失败: {str(e)}")
            raise UIError(f"创建图表失败: {str(e)}") from e
    
    @classmethod
    def create_line_chart(
        cls,
        x_data: Union[np.ndarray, List, pd.Series],
        y_data: Union[np.ndarray, List, pd.Series],
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        color: str = "blue",
        preset: ChartPreset = ChartPreset.INTERACTIVE,
        **kwargs
    ) -> InteractiveChartWidget:
        """创建折线图
        
        Args:
            x_data: X轴数据
            y_data: Y轴数据
            title: 图表标题
            xlabel: X轴标签
            ylabel: Y轴标签
            color: 线条颜色
            preset: 预设配置
            **kwargs: 其他参数
            
        Returns:
            InteractiveChartWidget: 图表组件实例
        """
        # 创建图表组件
        chart = cls.create_chart(preset=preset, **kwargs)
        
        # 准备数据
        plot_data = PlotData(
            x=x_data,
            y=y_data,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            color=color,
            label=kwargs.get('label', '数据系列')
        )
        
        # 绘制图表
        chart.plot(plot_data, PlotType.LINE)
        
        return chart
    
    @classmethod
    def create_scatter_chart(
        cls,
        x_data: Union[np.ndarray, List, pd.Series],
        y_data: Union[np.ndarray, List, pd.Series],
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        color: str = "red",
        marker: str = "o",
        preset: ChartPreset = ChartPreset.ANALYSIS,
        **kwargs
    ) -> InteractiveChartWidget:
        """创建散点图"""
        chart = cls.create_chart(preset=preset, **kwargs)
        
        plot_data = PlotData(
            x=x_data,
            y=y_data,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            color=color,
            marker=marker,
            label=kwargs.get('label', '数据点')
        )
        
        chart.plot(plot_data, PlotType.SCATTER)
        return chart
    
    @classmethod
    def create_bar_chart(
        cls,
        categories: Union[List, np.ndarray, pd.Series],
        values: Union[List, np.ndarray, pd.Series],
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        color: str = "green",
        preset: ChartPreset = ChartPreset.BUSINESS,
        **kwargs
    ) -> InteractiveChartWidget:
        """创建柱状图"""
        chart = cls.create_chart(preset=preset, **kwargs)
        
        plot_data = PlotData(
            x=categories,
            y=values,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            color=color,
            label=kwargs.get('label', '数据')
        )
        
        chart.plot(plot_data, PlotType.BAR)
        return chart
    
    @classmethod
    def create_histogram(
        cls,
        data: Union[List, np.ndarray, pd.Series],
        bins: int = 30,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: str = "频数",
        color: str = "skyblue",
        preset: ChartPreset = ChartPreset.ANALYSIS,
        **kwargs
    ) -> InteractiveChartWidget:
        """创建直方图"""
        chart = cls.create_chart(preset=preset, **kwargs)
        
        plot_data = PlotData(
            y=data,  # 直方图只需要y数据
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            color=color,
            label=kwargs.get('label', '分布')
        )
        
        chart.plot(plot_data, PlotType.HISTOGRAM, bins=bins)
        return chart
    
    @classmethod
    def create_box_plot(
        cls,
        data: Union[List, np.ndarray, pd.Series],
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        color: str = "orange",
        preset: ChartPreset = ChartPreset.SCIENTIFIC,
        **kwargs
    ) -> InteractiveChartWidget:
        """创建箱线图"""
        chart = cls.create_chart(preset=preset, **kwargs)
        
        plot_data = PlotData(
            y=data,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            color=color,
            label=kwargs.get('label', '数据分布')
        )
        
        chart.plot(plot_data, PlotType.BOX_PLOT)
        return chart
    
    @classmethod
    def create_multi_series_chart(
        cls,
        data_series: List[Dict[str, Any]],
        plot_type: PlotType = PlotType.LINE,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        preset: ChartPreset = ChartPreset.ANALYSIS,
        **kwargs
    ) -> InteractiveChartWidget:
        """创建多系列图表
        
        Args:
            data_series: 数据系列列表，每个元素包含 x, y, label, color 等
            plot_type: 图表类型
            title: 图表标题
            xlabel: X轴标签
            ylabel: Y轴标签
            preset: 预设配置
            **kwargs: 其他参数
            
        Returns:
            InteractiveChartWidget: 图表组件实例
        """
        chart = cls.create_chart(preset=preset, **kwargs)
        
        # 默认颜色序列
        default_colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        for i, series in enumerate(data_series):
            plot_data = PlotData(
                x=series.get('x'),
                y=series.get('y'),
                title=title if i == 0 else None,  # 只在第一个系列设置标题
                xlabel=xlabel if i == 0 else None,
                ylabel=ylabel if i == 0 else None,
                color=series.get('color', default_colors[i % len(default_colors)]),
                marker=series.get('marker', 'o'),
                linestyle=series.get('linestyle', '-'),
                linewidth=series.get('linewidth', 1.0),
                alpha=series.get('alpha', 1.0),
                label=series.get('label', f'系列 {i+1}')
            )
            
            chart.plot(plot_data, plot_type)
        
        return chart
    
    @classmethod
    def create_time_series_chart(
        cls,
        dates: Union[List, np.ndarray, pd.DatetimeIndex],
        values: Union[List, np.ndarray, pd.Series],
        title: Optional[str] = None,
        ylabel: Optional[str] = None,
        color: str = "blue",
        preset: ChartPreset = ChartPreset.ANALYSIS,
        **kwargs
    ) -> InteractiveChartWidget:
        """创建时间序列图表"""
        chart = cls.create_chart(preset=preset, **kwargs)
        
        plot_data = PlotData(
            x=dates,
            y=values,
            title=title,
            xlabel="时间",
            ylabel=ylabel,
            color=color,
            label=kwargs.get('label', '时间序列')
        )
        
        chart.plot(plot_data, PlotType.LINE)
        
        # 设置时间轴格式
        import matplotlib.dates as mdates
        chart.axes.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        chart.axes.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        chart.figure.autofmt_xdate()  # 自动格式化日期
        
        return chart
    
    @classmethod
    def create_comparison_chart(
        cls,
        categories: List[str],
        series_data: Dict[str, List],
        title: Optional[str] = None,
        ylabel: Optional[str] = None,
        plot_type: PlotType = PlotType.BAR,
        preset: ChartPreset = ChartPreset.BUSINESS,
        **kwargs
    ) -> InteractiveChartWidget:
        """创建对比图表
        
        Args:
            categories: 类别列表
            series_data: 系列数据字典，键为系列名，值为数据列表
            title: 图表标题
            ylabel: Y轴标签
            plot_type: 图表类型
            preset: 预设配置
            **kwargs: 其他参数
        """
        chart = cls.create_chart(preset=preset, **kwargs)
        
        default_colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, (series_name, values) in enumerate(series_data.items()):
            plot_data = PlotData(
                x=categories,
                y=values,
                title=title if i == 0 else None,
                xlabel="类别",
                ylabel=ylabel,
                color=default_colors[i % len(default_colors)],
                label=series_name
            )
            
            chart.plot(plot_data, plot_type)
        
        return chart
    
    @classmethod
    def _get_logger(cls):
        """获取日志记录器"""
        logger_mixin = LoggerMixin()
        return logger_mixin.logger
    
    @classmethod
    def get_preset_config(cls, preset: ChartPreset) -> ChartConfig:
        """获取预设配置"""
        return cls.PRESET_CONFIGS.get(preset, ChartConfig())
    
    @classmethod
    def list_presets(cls) -> List[str]:
        """列出所有可用的预设"""
        return [preset.value for preset in ChartPreset]


# 便捷函数
def quick_line_chart(x, y, **kwargs) -> InteractiveChartWidget:
    """快速创建折线图"""
    return ChartFactory.create_line_chart(x, y, **kwargs)


def quick_scatter_chart(x, y, **kwargs) -> InteractiveChartWidget:
    """快速创建散点图"""
    return ChartFactory.create_scatter_chart(x, y, **kwargs)


def quick_bar_chart(categories, values, **kwargs) -> InteractiveChartWidget:
    """快速创建柱状图"""
    return ChartFactory.create_bar_chart(categories, values, **kwargs)


def quick_histogram(data, **kwargs) -> InteractiveChartWidget:
    """快速创建直方图"""
    return ChartFactory.create_histogram(data, **kwargs)