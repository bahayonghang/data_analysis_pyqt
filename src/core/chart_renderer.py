"""
图表渲染器
支持matplotlib和plotly的高质量图表生成
"""

import io
import base64
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import warnings

# 图表库导入
try:
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
    import matplotlib.pyplot as plt
    import matplotlib.style as mplstyle
    from matplotlib.colors import LinearSegmentedColormap
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None
    sns = None

try:
    import plotly.graph_objects as go
    import plotly.express as px
    import plotly.figure_factory as ff
    from plotly.subplots import make_subplots
    import plotly.io as pio
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    go = None
    px = None

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from ..utils.basic_logging import LoggerMixin
from ..utils.exceptions import ChartGenerationError


class ChartBackend(str, Enum):
    """图表后端"""
    MATPLOTLIB = "matplotlib"
    PLOTLY = "plotly"
    AUTO = "auto"


class ChartFormat(str, Enum):
    """图表格式"""
    PNG = "png"
    SVG = "svg"
    PDF = "pdf"
    HTML = "html"
    JSON = "json"


class ChartType(str, Enum):
    """图表类型"""
    CORRELATION_HEATMAP = "correlation_heatmap"
    LINE_PLOT = "line_plot"
    BOX_PLOT = "box_plot"
    HISTOGRAM = "histogram"
    SCATTER_PLOT = "scatter_plot"
    BAR_PLOT = "bar_plot"
    VIOLIN_PLOT = "violin_plot"
    DISTRIBUTION_PLOT = "distribution_plot"


@dataclass
class ChartStyle:
    """图表样式配置"""
    # 基础样式
    theme: str = "nature"  # nature, scientific, minimal, dark
    color_palette: str = "viridis"
    figure_size: Tuple[int, int] = (10, 8)
    dpi: int = 300
    
    # 字体配置
    font_family: str = "Arial"
    font_size: int = 12
    title_font_size: int = 16
    label_font_size: int = 14
    
    # 颜色配置
    background_color: str = "#ffffff"
    grid_color: str = "#e0e0e0"
    text_color: str = "#333333"
    
    # 边距和间距
    margins: Dict[str, float] = field(default_factory=lambda: {
        'left': 0.1, 'right': 0.9, 'top': 0.9, 'bottom': 0.1
    })
    
    # 特定图表样式
    heatmap_colormap: str = "RdYlBu_r"
    line_width: float = 2.0
    marker_size: float = 6.0
    alpha: float = 0.7
    
    # 交互性
    interactive: bool = True
    show_legend: bool = True
    show_grid: bool = True


@dataclass
class ChartConfig:
    """图表配置"""
    chart_type: ChartType
    backend: ChartBackend = ChartBackend.AUTO
    style: ChartStyle = field(default_factory=ChartStyle)
    title: Optional[str] = None
    subtitle: Optional[str] = None
    x_label: Optional[str] = None
    y_label: Optional[str] = None
    
    # 数据配置
    x_column: Optional[str] = None
    y_column: Optional[str] = None
    color_column: Optional[str] = None
    size_column: Optional[str] = None
    
    # 特定配置
    custom_options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChartResult:
    """图表结果"""
    chart_type: ChartType
    backend: ChartBackend
    format: ChartFormat
    
    # 图表数据
    figure: Any = None  # matplotlib Figure 或 plotly Figure
    image_data: Optional[bytes] = None
    html_data: Optional[str] = None
    
    # 元数据
    file_path: Optional[str] = None
    size_bytes: int = 0
    creation_time: Optional[str] = None
    
    # 配置信息
    config: Optional[ChartConfig] = None
    
    def save(self, file_path: str, format: Optional[ChartFormat] = None) -> bool:
        """保存图表到文件"""
        try:
            target_format = format or self.format
            path = Path(file_path)
            
            if self.backend == ChartBackend.MATPLOTLIB and self.figure:
                if target_format == ChartFormat.PNG:
                    self.figure.savefig(path, dpi=self.config.style.dpi, 
                                      bbox_inches='tight', facecolor='white')
                elif target_format == ChartFormat.SVG:
                    self.figure.savefig(path, format='svg', bbox_inches='tight')
                elif target_format == ChartFormat.PDF:
                    self.figure.savefig(path, format='pdf', bbox_inches='tight')
                else:
                    return False
                    
            elif self.backend == ChartBackend.PLOTLY and self.figure:
                if target_format == ChartFormat.PNG:
                    self.figure.write_image(path)
                elif target_format == ChartFormat.HTML:
                    self.figure.write_html(path)
                elif target_format == ChartFormat.JSON:
                    self.figure.write_json(path)
                else:
                    return False
            else:
                return False
            
            self.file_path = str(path)
            return True
            
        except Exception:
            return False
    
    def to_base64(self) -> Optional[str]:
        """转换为base64字符串"""
        if self.image_data:
            return base64.b64encode(self.image_data).decode('utf-8')
        return None


class ChartRenderer(LoggerMixin):
    """图表渲染器"""
    
    def __init__(self, default_style: Optional[ChartStyle] = None):
        self.default_style = default_style or ChartStyle()
        
        # 检查可用的后端
        self.has_matplotlib = HAS_MATPLOTLIB
        self.has_plotly = HAS_PLOTLY
        self.has_numpy = HAS_NUMPY
        self.has_pandas = HAS_PANDAS
        
        if not self.has_matplotlib and not self.has_plotly:
            raise ChartGenerationError("至少需要matplotlib或plotly中的一个")
        
        # 初始化样式
        self._setup_matplotlib_styles()
        self._setup_plotly_styles()
        
        self.logger.info(f"ChartRenderer初始化完成 (matplotlib: {self.has_matplotlib}, plotly: {self.has_plotly})")
    
    def _setup_matplotlib_styles(self):
        """设置matplotlib样式"""
        if not self.has_matplotlib:
            return
            
        # 自定义nature风格
        nature_style = {
            'figure.figsize': (10, 8),
            'figure.dpi': 300,
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'axes.edgecolor': '#333333',
            'axes.linewidth': 1.0,
            'axes.grid': True,
            'axes.grid.which': 'major',
            'axes.labelcolor': '#333333',
            'axes.axisbelow': True,
            'grid.color': '#e0e0e0',
            'grid.linestyle': '-',
            'grid.linewidth': 0.5,
            'xtick.color': '#333333',
            'ytick.color': '#333333',
            'text.color': '#333333',
            'font.family': ['Arial', 'DejaVu Sans'],
            'font.size': 12,
            'legend.frameon': True,
            'legend.fancybox': True,
            'legend.shadow': True,
            'legend.framealpha': 0.9,
        }
        
        # 注册自定义样式
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                plt.style.use('default')
                for key, value in nature_style.items():
                    plt.rcParams[key] = value
            except Exception as e:
                self.logger.warning(f"设置matplotlib样式失败: {e}")
    
    def _setup_plotly_styles(self):
        """设置plotly样式"""
        if not self.has_plotly:
            return
            
        # 设置默认模板
        try:
            pio.templates.default = "plotly_white"
        except Exception as e:
            self.logger.warning(f"设置plotly样式失败: {e}")
    
    def _select_backend(self, preferred: ChartBackend, chart_type: ChartType) -> ChartBackend:
        """选择图表后端"""
        if preferred == ChartBackend.AUTO:
            # 根据图表类型自动选择
            if chart_type in [ChartType.CORRELATION_HEATMAP]:
                return ChartBackend.MATPLOTLIB if self.has_matplotlib else ChartBackend.PLOTLY
            elif chart_type in [ChartType.LINE_PLOT, ChartType.SCATTER_PLOT]:
                return ChartBackend.PLOTLY if self.has_plotly else ChartBackend.MATPLOTLIB
            else:
                return ChartBackend.MATPLOTLIB if self.has_matplotlib else ChartBackend.PLOTLY
        
        # 检查后端可用性
        if preferred == ChartBackend.MATPLOTLIB and not self.has_matplotlib:
            if self.has_plotly:
                self.logger.warning("matplotlib不可用，切换到plotly")
                return ChartBackend.PLOTLY
            else:
                raise ChartGenerationError("matplotlib不可用且没有备用后端")
        
        if preferred == ChartBackend.PLOTLY and not self.has_plotly:
            if self.has_matplotlib:
                self.logger.warning("plotly不可用，切换到matplotlib")
                return ChartBackend.MATPLOTLIB
            else:
                raise ChartGenerationError("plotly不可用且没有备用后端")
        
        return preferred
    
    def create_correlation_heatmap(
        self, 
        correlation_matrix: Any,
        columns: List[str],
        config: Optional[ChartConfig] = None
    ) -> ChartResult:
        """创建关联热力图"""
        config = config or ChartConfig(chart_type=ChartType.CORRELATION_HEATMAP)
        backend = self._select_backend(config.backend, ChartType.CORRELATION_HEATMAP)
        
        try:
            if backend == ChartBackend.MATPLOTLIB:
                return self._create_heatmap_matplotlib(correlation_matrix, columns, config)
            else:
                return self._create_heatmap_plotly(correlation_matrix, columns, config)
                
        except Exception as e:
            self.logger.error(f"创建关联热力图失败: {str(e)}")
            raise ChartGenerationError(f"创建关联热力图失败: {str(e)}") from e
    
    def _create_heatmap_matplotlib(
        self, 
        correlation_matrix: Any, 
        columns: List[str], 
        config: ChartConfig
    ) -> ChartResult:
        """使用matplotlib创建热力图"""
        if not self.has_matplotlib:
            raise ChartGenerationError("matplotlib不可用")
        
        # 转换数据格式
        if HAS_NUMPY and isinstance(correlation_matrix, list):
            correlation_matrix = np.array(correlation_matrix)
        
        # 创建图表
        fig, ax = plt.subplots(figsize=config.style.figure_size, dpi=config.style.dpi)
        
        # 使用seaborn创建热力图（如果可用）
        if self.has_matplotlib and sns is not None:
            # 创建DataFrame用于seaborn
            if HAS_PANDAS:
                df = pd.DataFrame(correlation_matrix, index=columns, columns=columns)
                sns.heatmap(
                    df,
                    annot=True,
                    cmap=config.style.heatmap_colormap,
                    center=0,
                    square=True,
                    fmt='.2f',
                    cbar_kws={'shrink': 0.8},
                    ax=ax
                )
            else:
                # 不用pandas，直接使用matplotlib
                im = ax.imshow(correlation_matrix, cmap=config.style.heatmap_colormap, 
                             aspect='equal', vmin=-1, vmax=1)
                
                # 添加颜色条
                cbar = plt.colorbar(im, ax=ax, shrink=0.8)
                cbar.set_label('相关系数', fontsize=config.style.label_font_size)
                
                # 设置刻度标签
                ax.set_xticks(range(len(columns)))
                ax.set_yticks(range(len(columns)))
                ax.set_xticklabels(columns, rotation=45, ha='right')
                ax.set_yticklabels(columns)
                
                # 添加数值标注
                for i in range(len(columns)):
                    for j in range(len(columns)):
                        value = correlation_matrix[i][j]
                        ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                               color='white' if abs(value) > 0.5 else 'black')
        else:
            # 基础matplotlib实现
            im = ax.imshow(correlation_matrix, cmap=config.style.heatmap_colormap, 
                         aspect='equal', vmin=-1, vmax=1)
            
            # 添加颜色条
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('相关系数', fontsize=config.style.label_font_size)
            
            # 设置刻度标签
            ax.set_xticks(range(len(columns)))
            ax.set_yticks(range(len(columns)))
            ax.set_xticklabels(columns, rotation=45, ha='right')
            ax.set_yticklabels(columns)
        
        # 设置标题和标签
        if config.title:
            ax.set_title(config.title, fontsize=config.style.title_font_size, 
                        color=config.style.text_color, pad=20)
        else:
            ax.set_title('变量关联热力图', fontsize=config.style.title_font_size, 
                        color=config.style.text_color, pad=20)
        
        # 调整布局
        plt.tight_layout()
        
        # 生成图像数据
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=config.style.dpi, 
                   bbox_inches='tight', facecolor='white')
        buffer.seek(0)
        image_data = buffer.getvalue()
        buffer.close()
        
        return ChartResult(
            chart_type=ChartType.CORRELATION_HEATMAP,
            backend=ChartBackend.MATPLOTLIB,
            format=ChartFormat.PNG,
            figure=fig,
            image_data=image_data,
            config=config
        )
    
    def _create_heatmap_plotly(
        self, 
        correlation_matrix: Any, 
        columns: List[str], 
        config: ChartConfig
    ) -> ChartResult:
        """使用plotly创建热力图"""
        if not self.has_plotly:
            raise ChartGenerationError("plotly不可用")
        
        # 转换数据格式
        if isinstance(correlation_matrix, list):
            matrix_data = correlation_matrix
        else:
            matrix_data = correlation_matrix.tolist() if hasattr(correlation_matrix, 'tolist') else correlation_matrix
        
        # 创建热力图
        fig = go.Figure(data=go.Heatmap(
            z=matrix_data,
            x=columns,
            y=columns,
            colorscale='RdBu',
            zmid=0,
            zmin=-1,
            zmax=1,
            text=[[f'{val:.2f}' for val in row] for row in matrix_data],
            texttemplate='%{text}',
            textfont={'size': 12},
            colorbar=dict(
                title='相关系数',
                titleside='right'
            )
        ))
        
        # 设置布局
        title = config.title or '变量关联热力图'
        fig.update_layout(
            title={
                'text': title,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': config.style.title_font_size}
            },
            width=config.style.figure_size[0] * 100,
            height=config.style.figure_size[1] * 100,
            xaxis={'side': 'bottom'},
            yaxis={'autorange': 'reversed'},
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # 生成HTML数据
        html_data = fig.to_html(include_plotlyjs=True)
        
        return ChartResult(
            chart_type=ChartType.CORRELATION_HEATMAP,
            backend=ChartBackend.PLOTLY,
            format=ChartFormat.HTML,
            figure=fig,
            html_data=html_data,
            config=config
        )
    
    def create_line_plot(
        self,
        data: Any,
        config: Optional[ChartConfig] = None
    ) -> ChartResult:
        """创建曲线图"""
        config = config or ChartConfig(chart_type=ChartType.LINE_PLOT)
        backend = self._select_backend(config.backend, ChartType.LINE_PLOT)
        
        try:
            if backend == ChartBackend.MATPLOTLIB:
                return self._create_line_plot_matplotlib(data, config)
            else:
                return self._create_line_plot_plotly(data, config)
                
        except Exception as e:
            self.logger.error(f"创建曲线图失败: {str(e)}")
            raise ChartGenerationError(f"创建曲线图失败: {str(e)}") from e
    
    def create_box_plot(
        self,
        data: Any,
        config: Optional[ChartConfig] = None
    ) -> ChartResult:
        """创建箱线图"""
        config = config or ChartConfig(chart_type=ChartType.BOX_PLOT)
        backend = self._select_backend(config.backend, ChartType.BOX_PLOT)
        
        try:
            if backend == ChartBackend.MATPLOTLIB:
                return self._create_box_plot_matplotlib(data, config)
            else:
                return self._create_box_plot_plotly(data, config)
                
        except Exception as e:
            self.logger.error(f"创建箱线图失败: {str(e)}")
            raise ChartGenerationError(f"创建箱线图失败: {str(e)}") from e
    
    def create_histogram(
        self,
        data: Any,
        config: Optional[ChartConfig] = None
    ) -> ChartResult:
        """创建分布直方图"""
        config = config or ChartConfig(chart_type=ChartType.HISTOGRAM)
        backend = self._select_backend(config.backend, ChartType.HISTOGRAM)
        
        try:
            if backend == ChartBackend.MATPLOTLIB:
                return self._create_histogram_matplotlib(data, config)
            else:
                return self._create_histogram_plotly(data, config)
                
        except Exception as e:
            self.logger.error(f"创建直方图失败: {str(e)}")
            raise ChartGenerationError(f"创建直方图失败: {str(e)}") from e
    
    def _create_line_plot_matplotlib(self, data: Any, config: ChartConfig) -> ChartResult:
        """使用matplotlib创建曲线图"""
        if not self.has_matplotlib:
            raise ChartGenerationError("matplotlib不可用")
        
        fig, ax = plt.subplots(figsize=config.style.figure_size, dpi=config.style.dpi)
        
        if HAS_PANDAS and hasattr(data, 'columns'):
            # DataFrame数据
            if config.x_column and config.y_column:
                x_data = data[config.x_column]
                y_data = data[config.y_column]
                ax.plot(x_data, y_data, linewidth=config.style.line_width, 
                       alpha=config.style.alpha, marker='o', markersize=config.style.marker_size)
            else:
                # 绘制所有数值列
                numeric_columns = data.select_dtypes(include=[np.number]).columns
                for i, col in enumerate(numeric_columns):
                    ax.plot(data.index, data[col], label=col, linewidth=config.style.line_width,
                           alpha=config.style.alpha, marker='o', markersize=config.style.marker_size)
                if config.style.show_legend:
                    ax.legend()
        else:
            # 假设是简单的x, y数据
            if isinstance(data, dict) and 'x' in data and 'y' in data:
                ax.plot(data['x'], data['y'], linewidth=config.style.line_width, 
                       alpha=config.style.alpha, marker='o', markersize=config.style.marker_size)
            else:
                # 假设是单一序列
                ax.plot(data, linewidth=config.style.line_width, 
                       alpha=config.style.alpha, marker='o', markersize=config.style.marker_size)
        
        # 设置标题和标签
        if config.title:
            ax.set_title(config.title, fontsize=config.style.title_font_size)
        if config.x_label:
            ax.set_xlabel(config.x_label, fontsize=config.style.label_font_size)
        if config.y_label:
            ax.set_ylabel(config.y_label, fontsize=config.style.label_font_size)
        
        if config.style.show_grid:
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 生成图像数据
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=config.style.dpi, 
                   bbox_inches='tight', facecolor='white')
        buffer.seek(0)
        image_data = buffer.getvalue()
        buffer.close()
        
        return ChartResult(
            chart_type=ChartType.LINE_PLOT,
            backend=ChartBackend.MATPLOTLIB,
            format=ChartFormat.PNG,
            figure=fig,
            image_data=image_data,
            config=config
        )
    
    def _create_line_plot_plotly(self, data: Any, config: ChartConfig) -> ChartResult:
        """使用plotly创建曲线图"""
        if not self.has_plotly:
            raise ChartGenerationError("plotly不可用")
        
        fig = go.Figure()
        
        if HAS_PANDAS and hasattr(data, 'columns'):
            # DataFrame数据
            if config.x_column and config.y_column:
                fig.add_trace(go.Scatter(
                    x=data[config.x_column],
                    y=data[config.y_column],
                    mode='lines+markers',
                    line=dict(width=config.style.line_width),
                    marker=dict(size=config.style.marker_size),
                    opacity=config.style.alpha
                ))
            else:
                # 绘制所有数值列
                numeric_columns = data.select_dtypes(include=[np.number]).columns
                for col in numeric_columns:
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=data[col],
                        mode='lines+markers',
                        name=col,
                        line=dict(width=config.style.line_width),
                        marker=dict(size=config.style.marker_size),
                        opacity=config.style.alpha
                    ))
        else:
            # 简单数据
            if isinstance(data, dict) and 'x' in data and 'y' in data:
                fig.add_trace(go.Scatter(
                    x=data['x'],
                    y=data['y'],
                    mode='lines+markers',
                    line=dict(width=config.style.line_width),
                    marker=dict(size=config.style.marker_size),
                    opacity=config.style.alpha
                ))
            else:
                fig.add_trace(go.Scatter(
                    y=data,
                    mode='lines+markers',
                    line=dict(width=config.style.line_width),
                    marker=dict(size=config.style.marker_size),
                    opacity=config.style.alpha
                ))
        
        # 设置布局
        fig.update_layout(
            title=config.title or '曲线图',
            xaxis_title=config.x_label or 'X轴',
            yaxis_title=config.y_label or 'Y轴',
            width=config.style.figure_size[0] * 100,
            height=config.style.figure_size[1] * 100,
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=config.style.show_legend,
            font=dict(size=config.style.font_size)
        )
        
        if config.style.show_grid:
            fig.update_xaxes(showgrid=True, gridcolor=config.style.grid_color)
            fig.update_yaxes(showgrid=True, gridcolor=config.style.grid_color)
        
        html_data = fig.to_html(include_plotlyjs=True)
        
        return ChartResult(
            chart_type=ChartType.LINE_PLOT,
            backend=ChartBackend.PLOTLY,
            format=ChartFormat.HTML,
            figure=fig,
            html_data=html_data,
            config=config
        )
    
    def _create_box_plot_matplotlib(self, data: Any, config: ChartConfig) -> ChartResult:
        """使用matplotlib创建箱线图"""
        if not self.has_matplotlib:
            raise ChartGenerationError("matplotlib不可用")
        
        fig, ax = plt.subplots(figsize=config.style.figure_size, dpi=config.style.dpi)
        
        if HAS_PANDAS and hasattr(data, 'columns'):
            # DataFrame数据
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            box_data = [data[col].dropna() for col in numeric_columns]
            box_plot = ax.boxplot(box_data, labels=numeric_columns, patch_artist=True)
            
            # 设置颜色
            colors = plt.cm.viridis(np.linspace(0, 1, len(box_data)))
            for patch, color in zip(box_plot['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(config.style.alpha)
        else:
            # 简单数据
            if isinstance(data, list):
                ax.boxplot(data)
            else:
                ax.boxplot([data])
        
        # 设置标题和标签
        if config.title:
            ax.set_title(config.title, fontsize=config.style.title_font_size)
        if config.x_label:
            ax.set_xlabel(config.x_label, fontsize=config.style.label_font_size)
        if config.y_label:
            ax.set_ylabel(config.y_label, fontsize=config.style.label_font_size)
        
        if config.style.show_grid:
            ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # 生成图像数据
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=config.style.dpi, 
                   bbox_inches='tight', facecolor='white')
        buffer.seek(0)
        image_data = buffer.getvalue()
        buffer.close()
        
        return ChartResult(
            chart_type=ChartType.BOX_PLOT,
            backend=ChartBackend.MATPLOTLIB,
            format=ChartFormat.PNG,
            figure=fig,
            image_data=image_data,
            config=config
        )
    
    def _create_box_plot_plotly(self, data: Any, config: ChartConfig) -> ChartResult:
        """使用plotly创建箱线图"""
        if not self.has_plotly:
            raise ChartGenerationError("plotly不可用")
        
        fig = go.Figure()
        
        if HAS_PANDAS and hasattr(data, 'columns'):
            # DataFrame数据
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                fig.add_trace(go.Box(
                    y=data[col].dropna(),
                    name=col,
                    boxpoints='outliers',
                    jitter=0.3,
                    pointpos=-1.8
                ))
        else:
            # 简单数据
            if isinstance(data, dict):
                for key, values in data.items():
                    fig.add_trace(go.Box(y=values, name=key))
            else:
                fig.add_trace(go.Box(y=data, name='数据'))
        
        # 设置布局
        fig.update_layout(
            title=config.title or '箱线图',
            xaxis_title=config.x_label or '变量',
            yaxis_title=config.y_label or '数值',
            width=config.style.figure_size[0] * 100,
            height=config.style.figure_size[1] * 100,
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=config.style.show_legend,
            font=dict(size=config.style.font_size)
        )
        
        if config.style.show_grid:
            fig.update_xaxes(showgrid=True, gridcolor=config.style.grid_color)
            fig.update_yaxes(showgrid=True, gridcolor=config.style.grid_color)
        
        html_data = fig.to_html(include_plotlyjs=True)
        
        return ChartResult(
            chart_type=ChartType.BOX_PLOT,
            backend=ChartBackend.PLOTLY,
            format=ChartFormat.HTML,
            figure=fig,
            html_data=html_data,
            config=config
        )
    
    def _create_histogram_matplotlib(self, data: Any, config: ChartConfig) -> ChartResult:
        """使用matplotlib创建直方图"""
        if not self.has_matplotlib:
            raise ChartGenerationError("matplotlib不可用")
        
        fig, ax = plt.subplots(figsize=config.style.figure_size, dpi=config.style.dpi)
        
        if HAS_PANDAS and hasattr(data, 'columns'):
            # DataFrame数据
            if config.y_column:
                # 单列直方图
                data_values = data[config.y_column].dropna()
                ax.hist(data_values, bins=30, alpha=config.style.alpha, 
                       edgecolor='black', linewidth=0.5)
                ax.set_xlabel(config.y_column)
            else:
                # 多列直方图
                numeric_columns = data.select_dtypes(include=[np.number]).columns
                for i, col in enumerate(numeric_columns[:3]):  # 最多3列
                    data_values = data[col].dropna()
                    ax.hist(data_values, bins=30, alpha=config.style.alpha, 
                           label=col, edgecolor='black', linewidth=0.5)
                if len(numeric_columns) > 1 and config.style.show_legend:
                    ax.legend()
        else:
            # 简单数据
            if isinstance(data, dict) and 'values' in data:
                ax.hist(data['values'], bins=30, alpha=config.style.alpha, 
                       edgecolor='black', linewidth=0.5)
            else:
                ax.hist(data, bins=30, alpha=config.style.alpha, 
                       edgecolor='black', linewidth=0.5)
        
        # 设置标题和标签
        if config.title:
            ax.set_title(config.title, fontsize=config.style.title_font_size)
        if config.x_label:
            ax.set_xlabel(config.x_label, fontsize=config.style.label_font_size)
        else:
            ax.set_xlabel('数值', fontsize=config.style.label_font_size)
        if config.y_label:
            ax.set_ylabel(config.y_label, fontsize=config.style.label_font_size)
        else:
            ax.set_ylabel('频次', fontsize=config.style.label_font_size)
        
        if config.style.show_grid:
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 生成图像数据
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=config.style.dpi, 
                   bbox_inches='tight', facecolor='white')
        buffer.seek(0)
        image_data = buffer.getvalue()
        buffer.close()
        
        return ChartResult(
            chart_type=ChartType.HISTOGRAM,
            backend=ChartBackend.MATPLOTLIB,
            format=ChartFormat.PNG,
            figure=fig,
            image_data=image_data,
            config=config
        )
    
    def _create_histogram_plotly(self, data: Any, config: ChartConfig) -> ChartResult:
        """使用plotly创建直方图"""
        if not self.has_plotly:
            raise ChartGenerationError("plotly不可用")
        
        fig = go.Figure()
        
        if HAS_PANDAS and hasattr(data, 'columns'):
            # DataFrame数据
            if config.y_column:
                # 单列直方图
                data_values = data[config.y_column].dropna()
                fig.add_trace(go.Histogram(
                    x=data_values,
                    name=config.y_column,
                    opacity=config.style.alpha,
                    nbinsx=30
                ))
            else:
                # 多列直方图
                numeric_columns = data.select_dtypes(include=[np.number]).columns
                for col in numeric_columns[:3]:  # 最多3列
                    data_values = data[col].dropna()
                    fig.add_trace(go.Histogram(
                        x=data_values,
                        name=col,
                        opacity=config.style.alpha,
                        nbinsx=30
                    ))
        else:
            # 简单数据
            if isinstance(data, dict) and 'values' in data:
                fig.add_trace(go.Histogram(
                    x=data['values'],
                    opacity=config.style.alpha,
                    nbinsx=30
                ))
            else:
                fig.add_trace(go.Histogram(
                    x=data,
                    opacity=config.style.alpha,
                    nbinsx=30
                ))
        
        # 设置布局
        fig.update_layout(
            title=config.title or '分布直方图',
            xaxis_title=config.x_label or '数值',
            yaxis_title=config.y_label or '频次',
            width=config.style.figure_size[0] * 100,
            height=config.style.figure_size[1] * 100,
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=config.style.show_legend,
            font=dict(size=config.style.font_size),
            barmode='overlay'
        )
        
        if config.style.show_grid:
            fig.update_xaxes(showgrid=True, gridcolor=config.style.grid_color)
            fig.update_yaxes(showgrid=True, gridcolor=config.style.grid_color)
        
        html_data = fig.to_html(include_plotlyjs=True)
        
        return ChartResult(
            chart_type=ChartType.HISTOGRAM,
            backend=ChartBackend.PLOTLY,
            format=ChartFormat.HTML,
            figure=fig,
            html_data=html_data,
            config=config
        )

    def create_chart(
        self,
        chart_type: ChartType,
        data: Any,
        title: Optional[str] = None,
        export_format: ChartFormat = ChartFormat.PNG,
        **kwargs
    ) -> bytes:
        """创建图表并返回字节数据"""
        try:
            # 创建配置
            config = ChartConfig(
                chart_type=chart_type,
                style=self.default_style,
                title=title,
                **kwargs
            )
            
            # 渲染图表
            if chart_type == ChartType.CORRELATION_HEATMAP:
                # 期望 data 是包含 correlation_matrix 和 columns 的字典
                if isinstance(data, dict) and 'correlation_matrix' in data and 'columns' in data:
                    result = self.create_correlation_heatmap(
                        data['correlation_matrix'],
                        data['columns'],
                        config
                    )
                else:
                    raise ChartGenerationError("关联热力图需要包含 correlation_matrix 和 columns 的字典数据")
            else:
                result = self.render_chart(data, config)
            
            # 返回图像数据
            if result.image_data:
                return result.image_data
            elif result.html_data and export_format == ChartFormat.HTML:
                return result.html_data.encode('utf-8')
            else:
                raise ChartGenerationError("无法生成图表数据")
                
        except Exception as e:
            self.logger.error(f"创建图表失败: {str(e)}")
            raise ChartGenerationError(f"创建图表失败: {str(e)}") from e

    def render_chart(self, data: Any, config: ChartConfig) -> ChartResult:
        """通用图表渲染方法"""
        try:
            if config.chart_type == ChartType.CORRELATION_HEATMAP:
                # 假设data是(correlation_matrix, columns)的元组
                if isinstance(data, tuple) and len(data) == 2:
                    return self.create_correlation_heatmap(data[0], data[1], config)
                else:
                    raise ChartGenerationError("关联热力图需要(correlation_matrix, columns)格式的数据")
            elif config.chart_type == ChartType.LINE_PLOT:
                return self.create_line_plot(data, config)
            elif config.chart_type == ChartType.BOX_PLOT:
                return self.create_box_plot(data, config)
            elif config.chart_type == ChartType.HISTOGRAM:
                return self.create_histogram(data, config)
            else:
                raise ChartGenerationError(f"暂不支持的图表类型: {config.chart_type}")
                
        except Exception as e:
            self.logger.error(f"图表渲染失败: {str(e)}")
            raise ChartGenerationError(f"图表渲染失败: {str(e)}") from e
    
    def get_supported_formats(self, backend: ChartBackend) -> List[ChartFormat]:
        """获取支持的输出格式"""
        if backend == ChartBackend.MATPLOTLIB:
            return [ChartFormat.PNG, ChartFormat.SVG, ChartFormat.PDF]
        elif backend == ChartBackend.PLOTLY:
            return [ChartFormat.PNG, ChartFormat.HTML, ChartFormat.JSON]
        else:
            return []
    
    def get_available_backends(self) -> List[ChartBackend]:
        """获取可用的后端"""
        backends = []
        if self.has_matplotlib:
            backends.append(ChartBackend.MATPLOTLIB)
        if self.has_plotly:
            backends.append(ChartBackend.PLOTLY)
        return backends
    
    def __del__(self):
        """清理资源"""
        if self.has_matplotlib:
            plt.close('all')