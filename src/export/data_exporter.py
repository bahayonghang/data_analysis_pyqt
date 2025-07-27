"""
数据导出器 - 支持多种格式的数据导出

提供数据集的多格式导出功能：
- CSV格式
- Excel格式  
- JSON格式
- Parquet格式
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any, Union
import polars as pl
from datetime import datetime

from ..utils.basic_logging import LoggerMixin


class DataExporter(LoggerMixin):
    """数据导出器
    
    支持将处理后的数据导出为多种格式，包括：
    - CSV: 通用表格格式
    - Excel: Microsoft Excel格式
    - JSON: 结构化数据格式
    - Parquet: 高效列式存储格式
    """
    
    def __init__(self):
        """初始化数据导出器"""
        super().__init__()
        self.supported_formats = ['csv', 'xlsx', 'json', 'parquet']
    
    def export_data(
        self,
        data: pl.DataFrame,
        output_path: str,
        format_type: str = 'csv',
        options: Optional[Dict[str, Any]] = None
    ) -> bool:
        """导出数据到指定格式
        
        Args:
            data: 要导出的数据框
            output_path: 输出文件路径
            format_type: 导出格式 ('csv', 'xlsx', 'json', 'parquet')
            options: 导出选项
            
        Returns:
            bool: 导出是否成功
        """
        if format_type not in self.supported_formats:
            self.logger.error(f"不支持的导出格式: {format_type}")
            return False
        
        try:
            options = options or {}
            
            # 确保输出目录存在
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            if format_type == 'csv':
                return self._export_csv(data, output_path, options)
            elif format_type == 'xlsx':
                return self._export_excel(data, output_path, options)
            elif format_type == 'json':
                return self._export_json(data, output_path, options)
            elif format_type == 'parquet':
                return self._export_parquet(data, output_path, options)
                
        except Exception as e:
            self.logger.error(f"数据导出失败: {e}")
            return False
    
    def _export_csv(
        self, 
        data: pl.DataFrame, 
        output_path: str, 
        options: Dict[str, Any]
    ) -> bool:
        """导出为CSV格式"""
        try:
            data.write_csv(
                output_path,
                separator=options.get('separator', ','),
                include_header=options.get('include_header', True),
                quote_char=options.get('quote_char', '"'),
                null_value=options.get('null_value', '')
            )
            self.logger.info(f"CSV导出成功: {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"CSV导出失败: {e}")
            return False
    
    def _export_excel(
        self, 
        data: pl.DataFrame, 
        output_path: str, 
        options: Dict[str, Any]
    ) -> bool:
        """导出为Excel格式"""
        try:
            # 转换为pandas DataFrame用于Excel导出
            pandas_df = data.to_pandas()
            
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                pandas_df.to_excel(
                    writer,
                    sheet_name=options.get('sheet_name', 'Data'),
                    index=options.get('include_index', False),
                    header=options.get('include_header', True)
                )
                
                # 添加元数据工作表
                if options.get('include_metadata', True):
                    metadata = {
                        '属性': ['导出时间', '数据行数', '数据列数', '文件大小'],
                        '值': [
                            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            len(data),
                            len(data.columns),
                            f"{len(data) * len(data.columns)} 单元格"
                        ]
                    }
                    metadata_df = pd.DataFrame(metadata)
                    metadata_df.to_excel(
                        writer, 
                        sheet_name='元数据',
                        index=False
                    )
            
            self.logger.info(f"Excel导出成功: {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"Excel导出失败: {e}")
            return False
    
    def _export_json(
        self, 
        data: pl.DataFrame, 
        output_path: str, 
        options: Dict[str, Any]
    ) -> bool:
        """导出为JSON格式"""
        try:
            # 根据选项决定JSON结构
            orient = options.get('orient', 'records')  # records, index, values, table
            
            if orient == 'records':
                # 每行作为一个对象
                json_data = data.to_dicts()
            elif orient == 'index':
                # 以行索引为键的对象
                json_data = {i: row for i, row in enumerate(data.to_dicts())}
            elif orient == 'values':
                # 仅包含值的二维数组
                json_data = data.to_numpy().tolist()
            elif orient == 'table':
                # 包含schema和data的表格格式
                json_data = {
                    'schema': {
                        'fields': [{'name': col, 'type': str(data[col].dtype)} for col in data.columns]
                    },
                    'data': data.to_dicts()
                }
            else:
                json_data = data.to_dicts()
            
            # 添加元数据
            if options.get('include_metadata', True):
                output_data = {
                    'metadata': {
                        'export_time': datetime.now().isoformat(),
                        'row_count': len(data),
                        'column_count': len(data.columns),
                        'columns': data.columns
                    },
                    'data': json_data
                }
            else:
                output_data = json_data
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(
                    output_data,
                    f,
                    ensure_ascii=False,
                    indent=options.get('indent', 2),
                    default=str  # 处理日期等特殊类型
                )
            
            self.logger.info(f"JSON导出成功: {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"JSON导出失败: {e}")
            return False
    
    def _export_parquet(
        self, 
        data: pl.DataFrame, 
        output_path: str, 
        options: Dict[str, Any]
    ) -> bool:
        """导出为Parquet格式"""
        try:
            data.write_parquet(
                output_path,
                compression=options.get('compression', 'snappy'),
                statistics=options.get('statistics', True),
                use_pyarrow=options.get('use_pyarrow', True)
            )
            self.logger.info(f"Parquet导出成功: {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"Parquet导出失败: {e}")
            return False
    
    def export_analysis_summary(
        self,
        analysis_data: Dict[str, Any],
        output_path: str,
        format_type: str = 'json'
    ) -> bool:
        """导出分析摘要
        
        Args:
            analysis_data: 分析结果数据
            output_path: 输出路径
            format_type: 导出格式
            
        Returns:
            bool: 导出是否成功
        """
        try:
            if format_type == 'json':
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(analysis_data, f, ensure_ascii=False, indent=2, default=str)
            elif format_type == 'csv':
                # 将嵌套字典展平为CSV
                flattened = self._flatten_dict(analysis_data)
                summary_df = pl.DataFrame([flattened])
                summary_df.write_csv(output_path)
            
            self.logger.info(f"分析摘要导出成功: {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"分析摘要导出失败: {e}")
            return False
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        """展平嵌套字典"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def get_supported_formats(self) -> list:
        """获取支持的导出格式"""
        return self.supported_formats.copy()


# 为了兼容性，导入pandas
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False