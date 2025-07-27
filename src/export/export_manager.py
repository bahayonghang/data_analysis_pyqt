"""
导出管理器 - 统一管理所有导出功能

提供一站式的导出解决方案：
- PDF报告生成
- 数据导出
- 图表导出
- 批量导出
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from datetime import datetime

from .pdf_generator import PDFReportGenerator
from .data_exporter import DataExporter
from .chart_exporter import ChartExporter
from ..models.analysis_result import AnalysisResult
from ..models.file_info import FileInfo
from ..utils.basic_logging import LoggerMixin


class ExportManager(LoggerMixin):
    """导出管理器
    
    统一管理所有导出功能，提供简化的接口：
    - 一键导出完整报告
    - 选择性导出特定内容
    - 批量导出操作
    - 导出进度跟踪
    """
    
    def __init__(self):
        """初始化导出管理器"""
        super().__init__()
        self.pdf_generator = PDFReportGenerator()
        self.data_exporter = DataExporter()
        self.chart_exporter = ChartExporter()
        
        # 默认导出配置
        self.default_config = {
            'pdf': {
                'include_cover': True,
                'include_toc': True,
                'include_summary': True,
                'include_charts': True,
                'include_appendix': True,
                'page_size': 'A4'
            },
            'data': {
                'include_metadata': True,
                'formats': ['csv', 'xlsx']
            },
            'charts': {
                'format': 'png',
                'dpi': 300,
                'size': (10, 6)
            }
        }
    
    def export_complete_report(
        self,
        analysis_result: AnalysisResult,
        file_info: FileInfo,
        output_dir: str,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, bool]:
        """导出完整分析报告
        
        Args:
            analysis_result: 分析结果
            file_info: 文件信息
            output_dir: 输出目录
            config: 导出配置
            
        Returns:
            Dict[str, bool]: 各部分导出结果
        """
        results = {}
        config = config or self.default_config
        
        try:
            # 确保输出目录存在
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # 生成文件名前缀
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_prefix = f"{Path(file_info.filename).stem}_{timestamp}"
            
            # 1. 导出PDF报告
            if config.get('export_pdf', True):
                pdf_path = os.path.join(output_dir, f"{file_prefix}_report.pdf")
                results['pdf'] = self.pdf_generator.generate_report(
                    analysis_result, file_info, pdf_path, config.get('pdf', {})
                )
                if results['pdf']:
                    self.logger.info(f"PDF报告导出成功: {pdf_path}")
            
            # 2. 导出数据文件
            if config.get('export_data', True) and hasattr(analysis_result, 'data'):
                data_formats = config.get('data', {}).get('formats', ['csv'])
                results['data'] = {}
                
                for fmt in data_formats:
                    data_path = os.path.join(output_dir, f"{file_prefix}_data.{fmt}")
                    success = self.data_exporter.export_data(
                        analysis_result.data, data_path, fmt, config.get('data', {})
                    )
                    results['data'][fmt] = success
                    if success:
                        self.logger.info(f"数据导出成功 ({fmt}): {data_path}")
            
            # 3. 导出图表
            if config.get('export_charts', True):
                charts_dir = os.path.join(output_dir, f"{file_prefix}_charts")
                charts = self._collect_charts(analysis_result)
                
                if charts:
                    results['charts'] = self.chart_exporter.export_chart_collection(
                        charts, charts_dir, 
                        config.get('charts', {}).get('format', 'png'),
                        config.get('charts', {})
                    )
                    if results['charts']:
                        self.logger.info(f"图表导出成功: {charts_dir}")
                        
                        # 创建图表画廊
                        gallery_path = os.path.join(output_dir, f"{file_prefix}_gallery.html")
                        self.chart_exporter.create_chart_gallery(
                            charts, gallery_path, f"{file_info.filename} - 图表集合"
                        )
            
            # 4. 导出分析摘要
            if config.get('export_summary', True):
                summary_data = self._create_analysis_summary(analysis_result, file_info)
                summary_path = os.path.join(output_dir, f"{file_prefix}_summary.json")
                results['summary'] = self.data_exporter.export_analysis_summary(
                    summary_data, summary_path, 'json'
                )
                if results['summary']:
                    self.logger.info(f"分析摘要导出成功: {summary_path}")
            
            # 5. 创建导出清单
            manifest_path = os.path.join(output_dir, f"{file_prefix}_manifest.txt")
            self._create_export_manifest(output_dir, manifest_path, results)
            
            self.logger.info("完整报告导出完成")
            return results
            
        except Exception as e:
            self.logger.error(f"完整报告导出失败: {e}")
            return {'error': str(e)}
    
    def export_pdf_only(
        self,
        analysis_result: AnalysisResult,
        file_info: FileInfo,
        output_path: str,
        config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """仅导出PDF报告
        
        Args:
            analysis_result: 分析结果
            file_info: 文件信息
            output_path: 输出路径
            config: PDF配置
            
        Returns:
            bool: 导出是否成功
        """
        pdf_config = config or self.default_config.get('pdf', {})
        return self.pdf_generator.generate_report(
            analysis_result, file_info, output_path, pdf_config
        )
    
    def export_data_only(
        self,
        data,
        output_path: str,
        format_type: str = 'csv',
        config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """仅导出数据
        
        Args:
            data: 数据框
            output_path: 输出路径
            format_type: 导出格式
            config: 数据导出配置
            
        Returns:
            bool: 导出是否成功
        """
        data_config = config or self.default_config.get('data', {})
        return self.data_exporter.export_data(
            data, output_path, format_type, data_config
        )
    
    def export_charts_only(
        self,
        charts: List[Dict[str, Any]],
        output_dir: str,
        format_type: str = 'png',
        config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """仅导出图表
        
        Args:
            charts: 图表列表
            output_dir: 输出目录
            format_type: 导出格式
            config: 图表导出配置
            
        Returns:
            bool: 导出是否成功
        """
        chart_config = config or self.default_config.get('charts', {})
        return self.chart_exporter.export_chart_collection(
            charts, output_dir, format_type, chart_config
        )
    
    def _collect_charts(self, analysis_result: AnalysisResult) -> List[Dict[str, Any]]:
        """收集分析结果中的所有图表
        
        Args:
            analysis_result: 分析结果
            
        Returns:
            List[Dict[str, Any]]: 图表列表
        """
        charts = []
        
        # 从分析结果中提取图表
        # 这里需要根据实际的图表存储方式来实现
        
        # 示例：如果图表存储在chart_data属性中
        if hasattr(analysis_result, 'charts') and analysis_result.charts:
            for chart_name, chart_data in analysis_result.charts.items():
                if 'figure' in chart_data:
                    charts.append({
                        'name': chart_name,
                        'figure': chart_data['figure'],
                        'type': chart_data.get('type', 'matplotlib'),
                        'description': chart_data.get('description', '')
                    })
        
        return charts
    
    def _create_analysis_summary(
        self, 
        analysis_result: AnalysisResult, 
        file_info: FileInfo
    ) -> Dict[str, Any]:
        """创建分析摘要数据
        
        Args:
            analysis_result: 分析结果
            file_info: 文件信息
            
        Returns:
            Dict[str, Any]: 摘要数据
        """
        summary = {
            'export_info': {
                'export_time': datetime.now().isoformat(),
                'version': '1.0'
            },
            'file_info': {
                'filename': file_info.filename,
                'filepath': file_info.filepath,
                'size_mb': file_info.size_mb,
                'row_count': file_info.row_count,
                'column_count': file_info.column_count,
                'upload_time': file_info.upload_time.isoformat() if file_info.upload_time else None
            },
            'analysis_results': {}
        }
        
        # 添加各种分析结果
        if hasattr(analysis_result, 'descriptive_stats') and analysis_result.descriptive_stats:
            summary['analysis_results']['descriptive_stats'] = analysis_result.descriptive_stats
        
        if hasattr(analysis_result, 'outliers') and analysis_result.outliers:
            summary['analysis_results']['outlier_count'] = len(analysis_result.outliers)
        
        if hasattr(analysis_result, 'stationarity_test') and analysis_result.stationarity_test:
            summary['analysis_results']['stationarity_test'] = analysis_result.stationarity_test
        
        return summary
    
    def _create_export_manifest(
        self, 
        output_dir: str, 
        manifest_path: str, 
        results: Dict[str, Any]
    ) -> None:
        """创建导出清单文件
        
        Args:
            output_dir: 输出目录
            manifest_path: 清单文件路径
            results: 导出结果
        """
        try:
            with open(manifest_path, 'w', encoding='utf-8') as f:
                f.write("数据分析报告导出清单\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"导出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"输出目录: {output_dir}\n\n")
                
                f.write("导出文件:\n")
                for category, result in results.items():
                    if isinstance(result, bool):
                        status = "✓" if result else "✗"
                        f.write(f"{status} {category}\n")
                    elif isinstance(result, dict):
                        f.write(f"+ {category}:\n")
                        for sub_category, sub_result in result.items():
                            status = "✓" if sub_result else "✗"
                            f.write(f"  {status} {sub_category}\n")
                
                f.write(f"\n清单文件: {manifest_path}\n")
                
        except Exception as e:
            self.logger.error(f"创建导出清单失败: {e}")
    
    def check_dependencies(self) -> Dict[str, bool]:
        """检查所有导出依赖
        
        Returns:
            Dict[str, bool]: 依赖可用性状态
        """
        return {
            'pdf_generator': self.pdf_generator.check_dependencies(),
            'data_exporter': True,  # 基于Polars，应该总是可用
            'chart_exporter': self.chart_exporter.check_dependencies()
        }
    
    def get_default_config(self) -> Dict[str, Any]:
        """获取默认配置
        
        Returns:
            Dict[str, Any]: 默认配置
        """
        return self.default_config.copy()