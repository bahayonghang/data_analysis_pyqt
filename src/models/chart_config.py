"""
图表配置数据模型
包含图表样式、布局和导出设置
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, validator


class ChartType(str, Enum):
    """图表类型"""
    LINE = "line"  # 折线图
    SCATTER = "scatter"  # 散点图
    BAR = "bar"  # 柱状图
    HISTOGRAM = "histogram"  # 直方图
    HEATMAP = "heatmap"  # 热力图
    BOX = "box"  # 箱线图
    VIOLIN = "violin"  # 小提琴图
    PAIR = "pair"  # 配对图
    CORRELATION = "correlation"  # 相关性图
    DISTRIBUTION = "distribution"  # 分布图


class PlotStyle(str, Enum):
    """绘图样式"""
    NATURE = "nature"  # Nature杂志风格
    SEABORN = "seaborn"  # Seaborn默认风格
    MATPLOTLIB = "matplotlib"  # Matplotlib默认风格
    PLOTLY = "plotly"  # Plotly风格
    CUSTOM = "custom"  # 自定义风格


class ColorPalette(str, Enum):
    """色彩方案"""
    VIRIDIS = "viridis"
    PLASMA = "plasma"
    INFERNO = "inferno"
    MAGMA = "magma"
    BLUES = "blues"
    GREENS = "greens"
    REDS = "reds"
    CUSTOM = "custom"


class ExportFormat(str, Enum):
    """导出格式"""
    PNG = "png"
    JPG = "jpg"
    SVG = "svg"
    PDF = "pdf"
    HTML = "html"


class AxisConfig(BaseModel):
    """坐标轴配置"""
    label: str = Field(default="", description="轴标签")
    scale: str = Field(default="linear", description="轴刻度类型 (linear, log, symlog)")
    min_value: Optional[float] = Field(None, description="最小值")
    max_value: Optional[float] = Field(None, description="最大值")
    show_grid: bool = Field(default=True, description="是否显示网格")
    tick_rotation: float = Field(default=0, description="刻度标签旋转角度")
    
    @validator('scale')
    def validate_scale(cls, v):
        """验证轴刻度类型"""
        allowed_scales = ['linear', 'log', 'symlog']
        if v not in allowed_scales:
            raise ValueError(f"不支持的轴刻度类型: {v}")
        return v


class LegendConfig(BaseModel):
    """图例配置"""
    show: bool = Field(default=True, description="是否显示图例")
    position: str = Field(default="best", description="图例位置")
    font_size: int = Field(default=10, ge=6, le=20, description="字体大小")
    title: str = Field(default="", description="图例标题")
    
    @validator('position')
    def validate_position(cls, v):
        """验证图例位置"""
        allowed_positions = ['best', 'upper right', 'upper left', 'lower left', 
                           'lower right', 'right', 'center left', 'center right',
                           'lower center', 'upper center', 'center']
        if v not in allowed_positions:
            raise ValueError(f"不支持的图例位置: {v}")
        return v


class TextConfig(BaseModel):
    """文本配置"""
    title: str = Field(default="", description="图表标题")
    title_font_size: int = Field(default=14, ge=8, le=24, description="标题字体大小")
    subtitle: str = Field(default="", description="副标题")
    font_family: str = Field(default="sans-serif", description="字体族")
    font_size: int = Field(default=10, ge=6, le=20, description="默认字体大小")


class ChartConfig(BaseModel):
    """图表配置模型"""
    
    # 基本信息
    chart_id: str = Field(..., description="图表ID")
    chart_type: ChartType = Field(..., description="图表类型")
    title: str = Field(default="", description="图表标题")
    
    # 数据配置
    x_column: Optional[str] = Field(None, description="X轴数据列")
    y_columns: List[str] = Field(default_factory=list, description="Y轴数据列")
    color_column: Optional[str] = Field(None, description="颜色分组列")
    size_column: Optional[str] = Field(None, description="大小列")
    
    # 样式配置
    plot_style: PlotStyle = Field(default=PlotStyle.NATURE, description="绘图样式")
    color_palette: ColorPalette = Field(default=ColorPalette.VIRIDIS, description="色彩方案")
    custom_colors: List[str] = Field(default_factory=list, description="自定义颜色列表")
    
    # 尺寸配置
    figure_width: float = Field(default=10.0, ge=2.0, le=20.0, description="图表宽度(英寸)")
    figure_height: float = Field(default=6.0, ge=2.0, le=20.0, description="图表高度(英寸)")
    dpi: int = Field(default=300, ge=72, le=600, description="分辨率")
    
    # 坐标轴配置
    x_axis: AxisConfig = Field(default_factory=AxisConfig, description="X轴配置")
    y_axis: AxisConfig = Field(default_factory=AxisConfig, description="Y轴配置")
    
    # 图例和文本配置
    legend: LegendConfig = Field(default_factory=LegendConfig, description="图例配置")
    text: TextConfig = Field(default_factory=TextConfig, description="文本配置")
    
    # 特殊配置
    show_annotations: bool = Field(default=False, description="是否显示注释")
    alpha: float = Field(default=0.8, ge=0.0, le=1.0, description="透明度")
    line_width: float = Field(default=2.0, ge=0.5, le=5.0, description="线宽")
    marker_size: float = Field(default=50.0, ge=10.0, le=200.0, description="标记大小")
    
    # 交互配置
    interactive: bool = Field(default=False, description="是否交互式图表")
    zoom_enabled: bool = Field(default=True, description="是否启用缩放")
    pan_enabled: bool = Field(default=True, description="是否启用平移")
    
    # 导出配置
    export_formats: List[ExportFormat] = Field(
        default_factory=lambda: [ExportFormat.PNG], 
        description="支持的导出格式"
    )
    
    # 自定义参数
    custom_params: Dict[str, Any] = Field(default_factory=dict, description="自定义参数")
    
    @validator('chart_id')
    def validate_chart_id(cls, v):
        """验证图表ID"""
        if not v or len(v) < 4:
            raise ValueError("图表ID必须至少4个字符")
        return v
    
    @validator('custom_colors')
    def validate_custom_colors(cls, v):
        """验证自定义颜色"""
        for color in v:
            if not color.startswith('#') or len(color) != 7:
                raise ValueError(f"无效的颜色格式: {color}")
        return v
    
    def get_figure_size(self) -> Tuple[float, float]:
        """获取图表尺寸"""
        return (self.figure_width, self.figure_height)
    
    def get_color_list(self) -> List[str]:
        """获取颜色列表"""
        if self.custom_colors:
            return self.custom_colors
        
        # 返回预定义调色板的默认颜色
        palette_colors = {
            ColorPalette.VIRIDIS: ['#440154', '#31688e', '#35b779', '#fde725'],
            ColorPalette.PLASMA: ['#0d0887', '#7201a8', '#bd3786', '#f0f921'],
            ColorPalette.BLUES: ['#08519c', '#3182bd', '#6baed6', '#c6dbef'],
            ColorPalette.GREENS: ['#00441b', '#238b45', '#74c476', '#c7e9c0'],
            ColorPalette.REDS: ['#67000d', '#cb181d', '#fb6a4a', '#fcbba1'],
        }
        
        return palette_colors.get(self.color_palette, ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    
    def update_axis_config(self, axis: str, config: AxisConfig) -> None:
        """更新坐标轴配置"""
        if axis.lower() == 'x':
            self.x_axis = config
        elif axis.lower() == 'y':
            self.y_axis = config
        else:
            raise ValueError(f"无效的坐标轴: {axis}")
    
    def set_style_preset(self, preset: str) -> None:
        """设置样式预设"""
        presets = {
            "nature": {
                "plot_style": PlotStyle.NATURE,
                "color_palette": ColorPalette.VIRIDIS,
                "figure_width": 8.5,
                "figure_height": 6.0,
                "dpi": 300,
                "text": TextConfig(
                    title_font_size=12,
                    font_family="Arial",
                    font_size=10
                )
            },
            "presentation": {
                "plot_style": PlotStyle.SEABORN,
                "color_palette": ColorPalette.BLUES,
                "figure_width": 12.0,
                "figure_height": 8.0,
                "dpi": 150,
                "text": TextConfig(
                    title_font_size=16,
                    font_family="sans-serif",
                    font_size=12
                )
            }
        }
        
        if preset in presets:
            preset_config = presets[preset]
            for key, value in preset_config.items():
                if hasattr(self, key):
                    setattr(self, key, value)
    
    def get_matplotlib_params(self) -> Dict[str, Any]:
        """获取matplotlib参数"""
        params = {
            'figure.figsize': self.get_figure_size(),
            'figure.dpi': self.dpi,
            'font.family': self.text.font_family,
            'font.size': self.text.font_size,
            'axes.titlesize': self.text.title_font_size,
            'axes.labelsize': self.text.font_size,
            'legend.fontsize': self.legend.font_size,
            'lines.linewidth': self.line_width,
            'lines.markersize': self.marker_size / 10,  # matplotlib使用不同的单位
        }
        return params
    
    def get_plotly_layout(self) -> Dict[str, Any]:
        """获取plotly布局配置"""
        layout = {
            'title': {
                'text': self.title or self.text.title,
                'font': {'size': self.text.title_font_size}
            },
            'width': int(self.figure_width * 100),
            'height': int(self.figure_height * 100),
            'showlegend': self.legend.show,
            'xaxis': {
                'title': self.x_axis.label,
                'type': self.x_axis.scale,
                'showgrid': self.x_axis.show_grid,
            },
            'yaxis': {
                'title': self.y_axis.label,
                'type': self.y_axis.scale,
                'showgrid': self.y_axis.show_grid,
            }
        }
        
        if self.x_axis.min_value is not None or self.x_axis.max_value is not None:
            layout['xaxis']['range'] = [self.x_axis.min_value, self.x_axis.max_value]
        
        if self.y_axis.min_value is not None or self.y_axis.max_value is not None:
            layout['yaxis']['range'] = [self.y_axis.min_value, self.y_axis.max_value]
        
        return layout
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return self.dict()
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ChartConfig":
        """从字典创建实例"""
        return cls.parse_obj(data)
    
    def clone(self) -> "ChartConfig":
        """克隆配置"""
        return self.copy(deep=True)
    
    def __str__(self) -> str:
        return f"ChartConfig({self.chart_type.value}, {self.figure_width}x{self.figure_height})"
    
    def __repr__(self) -> str:
        return (f"ChartConfig(id='{self.chart_id}', type='{self.chart_type.value}', "
                f"style='{self.plot_style.value}')")