"""
图表导出器 - 支持多种格式的图表导出

提供图表的多格式导出功能：
- PNG格式：高质量位图
- SVG格式：矢量图形
- PDF格式：文档格式
- HTML格式：交互式图表
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, Union, List
import base64
import io

try:
    import matplotlib
    matplotlib.use('Agg')  # 无GUI后端
    import matplotlib.pyplot as plt
    import matplotlib.figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.io as pio
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from ..utils.basic_logging import LoggerMixin


class ChartExporter(LoggerMixin):
    """图表导出器
    
    支持将matplotlib和plotly图表导出为多种格式：
    - PNG: 高质量位图，适合打印和文档
    - SVG: 矢量图形，可缩放
    - PDF: 文档格式，适合报告
    - HTML: 交互式图表，适合网页
    """
    
    def __init__(self):
        """初始化图表导出器"""
        super().__init__()
        self.supported_formats = ['png', 'svg', 'pdf', 'html', 'jpg', 'jpeg']
        self.default_dpi = 300
        self.default_size = (10, 6)  # 英寸
    
    def export_matplotlib_chart(
        self,
        figure: 'matplotlib.figure.Figure',
        output_path: str,
        format_type: str = 'png',
        options: Optional[Dict[str, Any]] = None
    ) -> bool:
        """导出matplotlib图表
        
        Args:
            figure: matplotlib图形对象
            output_path: 输出文件路径
            format_type: 导出格式
            options: 导出选项
            
        Returns:
            bool: 导出是否成功
        """
        if not MATPLOTLIB_AVAILABLE:
            self.logger.error("matplotlib未安装，无法导出图表")
            return False
        
        if format_type not in self.supported_formats:
            self.logger.error(f"不支持的导出格式: {format_type}")
            return False
        
        try:
            options = options or {}
            
            # 确保输出目录存在
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # 设置导出参数
            save_params = {
                'dpi': options.get('dpi', self.default_dpi),
                'bbox_inches': options.get('bbox_inches', 'tight'),
                'pad_inches': options.get('pad_inches', 0.1),
                'facecolor': options.get('facecolor', 'white'),
                'edgecolor': options.get('edgecolor', 'none'),
                'transparent': options.get('transparent', False)
            }
            
            # 根据格式调整参数
            if format_type in ['jpg', 'jpeg']:
                save_params['format'] = 'jpeg'
                save_params['transparent'] = False  # JPEG不支持透明度
            elif format_type == 'png':
                save_params['format'] = 'png'
            elif format_type == 'svg':
                save_params['format'] = 'svg'
                save_params.pop('dpi', None)  # SVG不需要DPI
            elif format_type == 'pdf':
                save_params['format'] = 'pdf'
            
            # 保存图表
            figure.savefig(output_path, **save_params)
            
            self.logger.info(f"matplotlib图表导出成功: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"matplotlib图表导出失败: {e}")
            return False
    
    def export_plotly_chart(
        self,
        figure: 'go.Figure',
        output_path: str,
        format_type: str = 'png',
        options: Optional[Dict[str, Any]] = None
    ) -> bool:
        """导出plotly图表
        
        Args:
            figure: plotly图形对象
            output_path: 输出文件路径  
            format_type: 导出格式
            options: 导出选项
            
        Returns:
            bool: 导出是否成功
        """
        if not PLOTLY_AVAILABLE:
            self.logger.error("plotly未安装，无法导出图表")
            return False
        
        if format_type not in ['png', 'jpg', 'jpeg', 'svg', 'pdf', 'html']:
            self.logger.error(f"plotly不支持的导出格式: {format_type}")
            return False
        
        try:
            options = options or {}
            
            # 确保输出目录存在
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            if format_type == 'html':
                # 导出HTML
                html_config = {
                    'include_plotlyjs': options.get('include_plotlyjs', True),
                    'config': options.get('config', {'displayModeBar': True}),
                    'div_id': options.get('div_id', None)
                }
                
                figure.write_html(output_path, **html_config)
                
            else:
                # 导出静态图片
                image_params = {
                    'width': options.get('width', 1200),
                    'height': options.get('height', 800),
                    'scale': options.get('scale', 2)  # 高分辨率
                }
                
                if format_type in ['jpg', 'jpeg']:
                    figure.write_image(output_path, format='jpeg', **image_params)
                else:
                    figure.write_image(output_path, format=format_type, **image_params)
            
            self.logger.info(f"plotly图表导出成功: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"plotly图表导出失败: {e}")
            return False
    
    def export_chart_collection(
        self,
        charts: List[Dict[str, Any]],
        output_dir: str,
        format_type: str = 'png',
        options: Optional[Dict[str, Any]] = None
    ) -> bool:
        """批量导出图表集合
        
        Args:
            charts: 图表列表，每个元素包含 {'figure': figure, 'name': name, 'type': 'matplotlib'|'plotly'}
            output_dir: 输出目录
            format_type: 导出格式
            options: 导出选项
            
        Returns:
            bool: 是否全部导出成功
        """
        try:
            # 确保输出目录存在
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            success_count = 0
            total_count = len(charts)
            
            for i, chart_info in enumerate(charts):
                figure = chart_info.get('figure')
                chart_name = chart_info.get('name', f'chart_{i+1}')
                chart_type = chart_info.get('type', 'matplotlib')
                
                # 构建输出路径
                output_path = os.path.join(output_dir, f"{chart_name}.{format_type}")
                
                # 根据图表类型选择导出方法
                if chart_type == 'matplotlib' and figure:
                    success = self.export_matplotlib_chart(figure, output_path, format_type, options)
                elif chart_type == 'plotly' and figure:
                    success = self.export_plotly_chart(figure, output_path, format_type, options)
                else:
                    self.logger.warning(f"未知图表类型或图表为空: {chart_type}")
                    success = False
                
                if success:
                    success_count += 1
            
            self.logger.info(f"图表批量导出完成: {success_count}/{total_count} 成功")
            return success_count == total_count
            
        except Exception as e:
            self.logger.error(f"图表批量导出失败: {e}")
            return False
    
    def create_chart_gallery(
        self,
        charts: List[Dict[str, Any]],
        output_path: str,
        title: str = "图表集合",
        options: Optional[Dict[str, Any]] = None
    ) -> bool:
        """创建图表画廊HTML页面
        
        Args:
            charts: 图表列表
            output_path: 输出HTML文件路径
            title: 页面标题
            options: 选项
            
        Returns:
            bool: 创建是否成功
        """
        try:
            options = options or {}
            
            html_content = f"""
            <!DOCTYPE html>
            <html lang="zh-CN">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>{title}</title>
                <style>
                    body {{
                        font-family: 'Microsoft YaHei', Arial, sans-serif;
                        margin: 0;
                        padding: 20px;
                        background-color: #f5f5f5;
                    }}
                    .container {{
                        max-width: 1200px;
                        margin: 0 auto;
                        background: white;
                        padding: 20px;
                        border-radius: 8px;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    }}
                    h1 {{
                        text-align: center;
                        color: #333;
                        margin-bottom: 30px;
                    }}
                    .chart-grid {{
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                        gap: 20px;
                        margin-top: 20px;
                    }}
                    .chart-item {{
                        border: 1px solid #ddd;
                        border-radius: 8px;
                        padding: 15px;
                        background: #fafafa;
                    }}
                    .chart-title {{
                        font-weight: bold;
                        margin-bottom: 10px;
                        color: #555;
                    }}
                    .chart-image {{
                        width: 100%;
                        height: auto;
                        border-radius: 4px;
                    }}
                    .timestamp {{
                        text-align: center;
                        color: #888;
                        font-size: 14px;
                        margin-top: 30px;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>{title}</h1>
                    <div class="chart-grid">
            """
            
            # 为每个图表生成HTML
            for i, chart_info in enumerate(charts):
                chart_name = chart_info.get('name', f'图表 {i+1}')
                chart_description = chart_info.get('description', '')
                
                # 如果是matplotlib图表，转换为base64
                if chart_info.get('type') == 'matplotlib' and chart_info.get('figure'):
                    img_data = self._matplotlib_to_base64(chart_info['figure'])
                    if img_data:
                        html_content += f"""
                        <div class="chart-item">
                            <div class="chart-title">{chart_name}</div>
                            {f'<p>{chart_description}</p>' if chart_description else ''}
                            <img src="data:image/png;base64,{img_data}" class="chart-image" alt="{chart_name}">
                        </div>
                        """
                
                # 如果是plotly图表，嵌入HTML
                elif chart_info.get('type') == 'plotly' and chart_info.get('figure'):
                    plotly_html = chart_info['figure'].to_html(
                        include_plotlyjs='inline',
                        div_id=f"plotly-div-{i}"
                    )
                    html_content += f"""
                    <div class="chart-item">
                        <div class="chart-title">{chart_name}</div>
                        {f'<p>{chart_description}</p>' if chart_description else ''}
                        {plotly_html}
                    </div>
                    """
            
            html_content += f"""
                    </div>
                    <div class="timestamp">
                        生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                    </div>
                </div>
            </body>
            </html>
            """
            
            # 写入文件
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.info(f"图表画廊创建成功: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"图表画廊创建失败: {e}")
            return False
    
    def _matplotlib_to_base64(self, figure) -> Optional[str]:
        """将matplotlib图表转换为base64字符串"""
        try:
            buffer = io.BytesIO()
            figure.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            buffer.close()
            return image_base64
        except Exception as e:
            self.logger.error(f"matplotlib图表转base64失败: {e}")
            return None
    
    def get_supported_formats(self) -> List[str]:
        """获取支持的导出格式"""
        return self.supported_formats.copy()
    
    def check_dependencies(self) -> Dict[str, bool]:
        """检查依赖是否可用
        
        Returns:
            Dict[str, bool]: 依赖可用性状态
        """
        return {
            'matplotlib': MATPLOTLIB_AVAILABLE,
            'plotly': PLOTLY_AVAILABLE
        }


# 导入datetime用于时间戳
from datetime import datetime