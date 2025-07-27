"""
数据加载器
支持CSV、Parquet等格式的高性能数据加载
"""

import asyncio
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import time

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False
    import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

from ..models import FileInfo, FileType, DataQualityInfo
from ..utils.exceptions import DataLoadingError
from ..utils.basic_logging import LoggerMixin


@dataclass
class LoaderConfig:
    """数据加载配置"""
    # 性能配置
    chunk_size: int = 10000
    max_memory_mb: int = 500
    use_polars: bool = True
    n_threads: int = 4
    
    # CSV配置
    csv_separator: str = ","
    csv_encoding: str = "utf-8"
    csv_quote_char: str = '"'
    csv_skip_rows: int = 0
    csv_has_header: bool = True
    
    # Parquet配置
    parquet_use_pandas_metadata: bool = True
    
    # 数据类型推断
    infer_schema_length: int = 1000
    try_parse_dates: bool = True
    
    # 内存管理
    low_memory: bool = True
    sample_size: int = 50000  # 大文件采样大小


class DataLoader(LoggerMixin):
    """数据加载器"""
    
    def __init__(self, config: Optional[LoaderConfig] = None):
        self.config = config or LoaderConfig()
        self.executor = ThreadPoolExecutor(max_workers=self.config.n_threads)
        
        # 检查可用的库
        self.use_polars = HAS_POLARS and self.config.use_polars
        if not self.use_polars and not HAS_POLARS:
            self.logger.warning("Polars not available, falling back to pandas")
    
    async def load_file_async(self, file_info: FileInfo) -> Tuple[Any, DataQualityInfo]:
        """异步加载文件"""
        loop = asyncio.get_event_loop()
        
        # 在线程池中执行数据加载
        return await loop.run_in_executor(
            self.executor,
            self._load_file_sync,
            file_info
        )
    
    def load_file(self, file_info: FileInfo) -> Tuple[Any, DataQualityInfo]:
        """同步加载文件"""
        return self._load_file_sync(file_info)
    
    def _load_file_sync(self, file_info: FileInfo) -> Tuple[Any, DataQualityInfo]:
        """同步加载文件的内部实现"""
        start_time = time.time()
        
        try:
            file_path = Path(file_info.file_path)
            
            # 根据文件类型选择加载方法
            if file_info.file_type == FileType.CSV:
                df, quality_info = self._load_csv(file_path)
            elif file_info.file_type == FileType.PARQUET:
                df, quality_info = self._load_parquet(file_path)
            elif file_info.file_type == FileType.EXCEL:
                df, quality_info = self._load_excel(file_path)
            else:
                raise DataLoadingError(f"不支持的文件类型: {file_info.file_type}")
            
            load_time = time.time() - start_time
            self.logger.info(f"文件加载完成: {file_path.name}, 耗时: {load_time:.2f}秒")
            
            return df, quality_info
            
        except Exception as e:
            self.logger.error(f"文件加载失败: {file_info.file_path}, 错误: {str(e)}")
            raise DataLoadingError(f"加载文件失败: {str(e)}") from e
    
    def _load_csv(self, file_path: Path) -> Tuple[Any, DataQualityInfo]:
        """加载CSV文件"""
        self.logger.info(f"开始加载CSV文件: {file_path}")
        
        try:
            if self.use_polars:
                return self._load_csv_polars(file_path)
            else:
                return self._load_csv_pandas(file_path)
        except Exception as e:
            # 如果Polars失败，尝试pandas
            if self.use_polars:
                self.logger.warning(f"Polars加载失败，切换到pandas: {str(e)}")
                return self._load_csv_pandas(file_path)
            else:
                raise
    
    def _load_csv_polars(self, file_path: Path) -> Tuple["pl.DataFrame", DataQualityInfo]:
        """使用Polars加载CSV"""
        try:
            # 先尝试推断schema
            schema_sample = pl.read_csv(
                file_path,
                separator=self.config.csv_separator,
                encoding=self.config.csv_encoding,
                quote_char=self.config.csv_quote_char,
                skip_rows=self.config.csv_skip_rows,
                has_header=self.config.csv_has_header,
                n_rows=self.config.infer_schema_length,
                try_parse_dates=self.config.try_parse_dates
            )
            
            # 检查文件大小，决定是否分块加载
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            
            if file_size_mb > self.config.max_memory_mb:
                # 大文件采样
                self.logger.info(f"文件较大({file_size_mb:.1f}MB)，使用采样模式")
                df = self._sample_large_csv_polars(file_path, schema_sample.schema)
            else:
                # 直接加载
                df = pl.read_csv(
                    file_path,
                    separator=self.config.csv_separator,
                    encoding=self.config.csv_encoding,
                    quote_char=self.config.csv_quote_char,
                    skip_rows=self.config.csv_skip_rows,
                    has_header=self.config.csv_has_header,
                    try_parse_dates=self.config.try_parse_dates
                )
            
            quality_info = self._calculate_quality_info_polars(df)
            return df, quality_info
            
        except Exception as e:
            raise DataLoadingError(f"Polars CSV加载失败: {str(e)}") from e
    
    def _load_csv_pandas(self, file_path: Path) -> Tuple[Any, DataQualityInfo]:
        """使用Pandas加载CSV"""
        try:
            # 检查文件大小
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            
            if file_size_mb > self.config.max_memory_mb:
                # 大文件采样
                self.logger.info(f"文件较大({file_size_mb:.1f}MB)，使用采样模式")
                df = self._sample_large_csv_pandas(file_path)
            else:
                # 直接加载
                df = pd.read_csv(
                    file_path,
                    sep=self.config.csv_separator,
                    encoding=self.config.csv_encoding,
                    quotechar=self.config.csv_quote_char,
                    skiprows=self.config.csv_skip_rows,
                    low_memory=self.config.low_memory,
                    parse_dates=self.config.try_parse_dates
                )
            
            quality_info = self._calculate_quality_info_pandas(df)
            return df, quality_info
            
        except Exception as e:
            raise DataLoadingError(f"Pandas CSV加载失败: {str(e)}") from e
    
    def _load_parquet(self, file_path: Path) -> Tuple[Any, DataQualityInfo]:
        """加载Parquet文件"""
        self.logger.info(f"开始加载Parquet文件: {file_path}")
        
        if not HAS_PYARROW:
            raise DataLoadingError("PyArrow not available for Parquet files")
        
        try:
            if self.use_polars:
                return self._load_parquet_polars(file_path)
            else:
                return self._load_parquet_pandas(file_path)
        except Exception as e:
            if self.use_polars:
                self.logger.warning(f"Polars加载失败，切换到pandas: {str(e)}")
                return self._load_parquet_pandas(file_path)
            else:
                raise
    
    def _load_parquet_polars(self, file_path: Path) -> Tuple["pl.DataFrame", DataQualityInfo]:
        """使用Polars加载Parquet"""
        try:
            # 检查文件大小
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            
            if file_size_mb > self.config.max_memory_mb:
                # 大文件采样
                self.logger.info(f"Parquet文件较大({file_size_mb:.1f}MB)，使用采样模式")
                df = self._sample_large_parquet_polars(file_path)
            else:
                df = pl.read_parquet(file_path)
            
            quality_info = self._calculate_quality_info_polars(df)
            return df, quality_info
            
        except Exception as e:
            raise DataLoadingError(f"Polars Parquet加载失败: {str(e)}") from e
    
    def _load_parquet_pandas(self, file_path: Path) -> Tuple[Any, DataQualityInfo]:
        """使用Pandas加载Parquet"""
        try:
            # 检查文件大小
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            
            if file_size_mb > self.config.max_memory_mb:
                # 大文件采样
                self.logger.info(f"Parquet文件较大({file_size_mb:.1f}MB)，使用采样模式")
                df = self._sample_large_parquet_pandas(file_path)
            else:
                df = pd.read_parquet(file_path)
            
            quality_info = self._calculate_quality_info_pandas(df)
            return df, quality_info
            
        except Exception as e:
            raise DataLoadingError(f"Pandas Parquet加载失败: {str(e)}") from e
    
    def _load_excel(self, file_path: Path) -> Tuple[Any, DataQualityInfo]:
        """加载Excel文件"""
        self.logger.info(f"开始加载Excel文件: {file_path}")
        
        try:
            df = pd.read_excel(file_path)
            quality_info = self._calculate_quality_info_pandas(df)
            return df, quality_info
            
        except Exception as e:
            raise DataLoadingError(f"Excel加载失败: {str(e)}") from e
    
    def _sample_large_csv_polars(self, file_path: Path, schema: Dict) -> "pl.DataFrame":
        """Polars大文件采样"""
        try:
            # 获取总行数估计
            with open(file_path, 'r', encoding=self.config.csv_encoding) as f:
                line_count = sum(1 for _ in f)
            
            # 计算采样间隔
            if line_count <= self.config.sample_size:
                return pl.read_csv(file_path, schema=schema)
            
            skip_interval = max(1, line_count // self.config.sample_size)
            
            # 分块读取并采样
            chunks = []
            for chunk in pl.read_csv_batched(
                file_path,
                batch_size=self.config.chunk_size,
                schema=schema
            ):
                # 采样
                sampled = chunk.sample(
                    min(len(chunk), self.config.chunk_size // skip_interval)
                )
                chunks.append(sampled)
                
                if sum(len(c) for c in chunks) >= self.config.sample_size:
                    break
            
            if HAS_POLARS:
                import polars as pl
            return pl.concat(chunks) if chunks else pl.DataFrame(schema=schema)
            
        except Exception as e:
            raise DataLoadingError(f"Polars大文件采样失败: {str(e)}") from e
    
    def _sample_large_csv_pandas(self, file_path: Path) -> Any:
        """Pandas大文件采样"""
        try:
            # 估算总行数
            with open(file_path, 'r', encoding=self.config.csv_encoding) as f:
                line_count = sum(1 for _ in f)
            
            if line_count <= self.config.sample_size:
                return pd.read_csv(file_path)
            
            # 随机采样行号
            import random
            skip_rows = sorted(random.sample(
                range(1, line_count), 
                line_count - self.config.sample_size - 1
            ))
            
            return pd.read_csv(file_path, skiprows=skip_rows)
            
        except Exception as e:
            raise DataLoadingError(f"Pandas大文件采样失败: {str(e)}") from e
    
    def _sample_large_parquet_polars(self, file_path: Path) -> "pl.DataFrame":
        """Polars Parquet大文件采样"""
        try:
            # 读取前几行获取schema
            schema_df = pl.read_parquet(file_path, n_rows=100)
            
            # 使用lazy evaluation进行采样
            lazy_df = pl.scan_parquet(file_path)
            total_rows = lazy_df.select(pl.count()).collect().item()
            
            if total_rows <= self.config.sample_size:
                return lazy_df.collect()
            
            # 随机采样
            sample_fraction = self.config.sample_size / total_rows
            return lazy_df.sample(fraction=sample_fraction).collect()
            
        except Exception as e:
            raise DataLoadingError(f"Polars Parquet采样失败: {str(e)}") from e
    
    def _sample_large_parquet_pandas(self, file_path: Path) -> Any:
        """Pandas Parquet大文件采样"""
        try:
            # 读取parquet文件信息
            parquet_file = pq.ParquetFile(file_path)
            total_rows = parquet_file.metadata.num_rows
            
            if total_rows <= self.config.sample_size:
                return pd.read_parquet(file_path)
            
            # 分批读取并采样
            chunks = []
            rows_read = 0
            
            for batch in parquet_file.iter_batches(batch_size=self.config.chunk_size):
                df_chunk = batch.to_pandas()
                
                # 计算当前批次的采样率
                remaining_samples = self.config.sample_size - sum(len(c) for c in chunks)
                if remaining_samples <= 0:
                    break
                
                sample_size = min(remaining_samples, len(df_chunk))
                if sample_size < len(df_chunk):
                    df_chunk = df_chunk.sample(n=sample_size)
                
                chunks.append(df_chunk)
                rows_read += len(df_chunk)
            
            return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
            
        except Exception as e:
            raise DataLoadingError(f"Pandas Parquet采样失败: {str(e)}") from e
    
    def _calculate_quality_info_polars(self, df: "pl.DataFrame") -> DataQualityInfo:
        """计算Polars DataFrame的数据质量信息"""
        try:
            total_rows = len(df)
            total_columns = len(df.columns)
            
            # 计算缺失值
            missing_count = 0
            for col in df.columns:
                missing_count += df[col].null_count()
            
            # 计算重复行数
            duplicate_count = total_rows - df.unique().height if total_rows > 0 else 0
            
            # 估算内存使用
            memory_mb = df.estimated_size() / (1024 * 1024)
            
            return DataQualityInfo(
                total_rows=total_rows,
                total_columns=total_columns,
                missing_values_count=missing_count,
                duplicate_rows_count=duplicate_count,
                memory_usage_mb=memory_mb
            )
            
        except Exception as e:
            self.logger.error(f"计算Polars数据质量信息失败: {str(e)}")
            # 返回基础信息
            return DataQualityInfo(
                total_rows=len(df) if df is not None else 0,
                total_columns=len(df.columns) if df is not None else 0,
                missing_values_count=0,
                duplicate_rows_count=0,
                memory_usage_mb=0.0
            )
    
    def _calculate_quality_info_pandas(self, df: Any) -> DataQualityInfo:
        """计算Pandas DataFrame的数据质量信息"""
        try:
            total_rows = len(df)
            total_columns = len(df.columns)
            
            # 计算缺失值
            missing_count = df.isnull().sum().sum()
            
            # 计算重复行数
            duplicate_count = df.duplicated().sum()
            
            # 计算内存使用
            memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
            
            return DataQualityInfo(
                total_rows=total_rows,
                total_columns=total_columns,
                missing_values_count=int(missing_count),
                duplicate_rows_count=int(duplicate_count),
                memory_usage_mb=float(memory_mb)
            )
            
        except Exception as e:
            self.logger.error(f"计算Pandas数据质量信息失败: {str(e)}")
            # 返回基础信息
            return DataQualityInfo(
                total_rows=len(df) if df is not None else 0,
                total_columns=len(df.columns) if df is not None else 0,
                missing_values_count=0,
                duplicate_rows_count=0,
                memory_usage_mb=0.0
            )
    
    def get_column_info(self, df: Any) -> Tuple[List[str], Dict[str, str]]:
        """获取列信息"""
        try:
            if self.use_polars and isinstance(df, pl.DataFrame):
                column_names = df.columns
                column_types = {col: str(df[col].dtype) for col in column_names}
            else:
                column_names = list(df.columns)
                column_types = {col: str(df[col].dtype) for col in column_names}
            
            return column_names, column_types
            
        except Exception as e:
            self.logger.error(f"获取列信息失败: {str(e)}")
            return [], {}
    
    def get_sample_data(self, df: Any, n_rows: int = 10) -> List[List[Any]]:
        """获取样本数据"""
        try:
            if self.use_polars and isinstance(df, pl.DataFrame):
                sample_df = df.head(n_rows)
                return sample_df.to_numpy().tolist()
            else:
                sample_df = df.head(n_rows)
                return sample_df.values.tolist()
                
        except Exception as e:
            self.logger.error(f"获取样本数据失败: {str(e)}")
            return []
    
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)