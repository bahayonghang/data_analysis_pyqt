"""
数据预览组件
提供数据表格预览、基本统计信息显示和时间列检测结果
"""

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QColor
    from PyQt6.QtWidgets import (
        QAbstractItemView,
        QFrame,
        QGridLayout,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QScrollArea,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )
    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False
    # 模拟类定义
    class QWidget:
        pass
    class QVBoxLayout:
        pass
    class QHBoxLayout:
        pass
    class QTableWidget:
        pass
    class QLabel:
        pass
    class QFrame:
        pass

try:
    from qfluentwidgets import (
        BodyLabel,
        CaptionLabel,
        CardWidget,
        HeaderCardWidget,
        ScrollArea,
        SimpleCardWidget,
        StrongBodyLabel,
        TableWidget,
    )
    HAS_FLUENT_WIDGETS = True
except ImportError:
    HAS_FLUENT_WIDGETS = False
    # 模拟类定义
    class CardWidget:
        pass
    class TableWidget:
        pass
    class BodyLabel:
        pass
    class ScrollArea:
        pass

try:
    import numpy as np
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

from ..utils.basic_logging import LoggerMixin
from ..utils.exceptions import DataProcessingError


class FileType(str, Enum):
    """支持的文件类型"""
    CSV = "csv"
    PARQUET = "parquet"
    EXCEL = "excel"
    JSON = "json"
    UNKNOWN = "unknown"


@dataclass
class FileInfo:
    """文件信息"""
    file_path: str
    file_name: str
    file_size: int
    file_type: FileType
    mime_type: str
    is_valid: bool = False
    error_message: str = ""

    # 数据信息（加载后填充）
    row_count: int | None = None
    column_count: int | None = None
    columns: list[str] | None = None
    time_columns: list[str] | None = None
    memory_usage: int | None = None


@dataclass
class PreviewConfig:
    """预览配置"""
    # 表格配置
    max_preview_rows: int = 100
    max_preview_columns: int = 20
    table_row_height: int = 25

    # 统计配置
    show_basic_stats: bool = True
    show_column_info: bool = True
    show_time_detection: bool = True

    # 性能配置
    lazy_loading: bool = True
    chunk_size: int = 1000


@dataclass
class ColumnInfo:
    """列信息"""
    name: str
    data_type: str
    null_count: int
    null_percentage: float
    unique_count: int | None = None
    sample_values: list[str] | None = None
    is_time_column: bool = False
    time_format: str | None = None
    memory_usage: int | None = None


@dataclass
class BasicStats:
    """基本统计信息"""
    total_rows: int
    total_columns: int
    memory_usage: int
    file_size: int
    numeric_columns: int
    text_columns: int
    time_columns: int
    missing_values: int
    missing_percentage: float


class DataPreviewTable(QTableWidget if HAS_PYQT6 else QWidget, LoggerMixin):
    """数据预览表格"""

    def __init__(self, config: PreviewConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self.data: Any | None = None  # pandas DataFrame 或 polars DataFrame
        self.column_info: list[ColumnInfo] = []

        if HAS_PYQT6:
            self._setup_table()

    def _setup_table(self):
        """设置表格"""
        if not HAS_PYQT6:
            return

        # 表格属性
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setSortingEnabled(False)

        # 设置表格样式
        self.setObjectName("dataPreviewTable")

        # 设置行高
        self.verticalHeader().setDefaultSectionSize(self.config.table_row_height)

        # 设置列宽自适应
        self.horizontalHeader().setStretchLastSection(True)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)

        # 样式表
        self.setStyleSheet("""
            #dataPreviewTable {
                gridline-color: #e0e0e0;
                background-color: white;
                alternate-background-color: #f8f9fa;
            }
            #dataPreviewTable::item {
                padding: 4px;
                border: none;
            }
            #dataPreviewTable::item:selected {
                background-color: #e3f2fd;
            }
        """)

    def load_data(self, data: Any, column_info: list[ColumnInfo]):
        """加载数据"""
        if not HAS_PYQT6:
            return

        try:
            self.data = data
            self.column_info = column_info

            # 获取预览数据
            preview_data = self._get_preview_data()

            if preview_data is None:
                self.logger.warning("无法获取预览数据")
                return

            # 设置表格大小
            rows, cols = preview_data.shape
            display_rows = min(rows, self.config.max_preview_rows)
            display_cols = min(cols, self.config.max_preview_columns)

            self.setRowCount(display_rows)
            self.setColumnCount(display_cols)

            # 设置表头
            if hasattr(preview_data, 'columns'):
                headers = list(preview_data.columns[:display_cols])
            else:
                headers = [f"Column_{i+1}" for i in range(display_cols)]

            self.setHorizontalHeaderLabels(headers)

            # 填充数据
            for row in range(display_rows):
                for col in range(display_cols):
                    try:
                        if HAS_PANDAS and isinstance(preview_data, pd.DataFrame):
                            value = preview_data.iloc[row, col]
                        elif HAS_POLARS and hasattr(preview_data, 'item'):
                            value = preview_data.item(row, col)
                        else:
                            value = str(preview_data[row][col]) if hasattr(preview_data, '__getitem__') else ""

                        # 处理特殊值
                        if pd.isna(value) if HAS_PANDAS else value is None:
                            display_value = "<NULL>"
                            item = QTableWidgetItem(display_value)
                            item.setForeground(QColor("#999999"))
                        else:
                            display_value = str(value)
                            # 限制显示长度
                            if len(display_value) > 100:
                                display_value = display_value[:97] + "..."
                            item = QTableWidgetItem(display_value)

                        # 设置列类型样式
                        col_info = self.column_info[col] if col < len(self.column_info) else None
                        if col_info:
                            if col_info.is_time_column:
                                item.setBackground(QColor("#fff3cd"))  # 黄色背景表示时间列
                            elif "int" in col_info.data_type.lower() or "float" in col_info.data_type.lower():
                                item.setForeground(QColor("#0066cc"))  # 蓝色表示数值

                        self.setItem(row, col, item)

                    except Exception as e:
                        self.logger.warning(f"设置表格项失败 ({row}, {col}): {str(e)}")
                        self.setItem(row, col, QTableWidgetItem("<ERROR>"))

            # 调整列宽
            self.resizeColumnsToContents()

            # 如果数据被截断，显示提示
            if rows > self.config.max_preview_rows or cols > self.config.max_preview_columns:
                self.logger.info(f"数据预览已截断: 显示 {display_rows}/{rows} 行, {display_cols}/{cols} 列")

        except Exception as e:
            self.logger.error(f"加载数据到表格失败: {str(e)}")
            raise DataProcessingError(f"加载数据到表格失败: {str(e)}") from e

    def _get_preview_data(self) -> Any | None:
        """获取预览数据"""
        if self.data is None:
            return None

        try:
            # 处理 pandas DataFrame
            if HAS_PANDAS and isinstance(self.data, pd.DataFrame):
                return self.data.head(self.config.max_preview_rows)

            # 处理 polars DataFrame
            elif HAS_POLARS and hasattr(self.data, 'head'):
                return self.data.head(self.config.max_preview_rows)

            # 处理其他类型的数据
            elif hasattr(self.data, '__len__') and hasattr(self.data, '__getitem__'):
                # 假设是二维数组或类似结构
                return self.data[:self.config.max_preview_rows]

            else:
                self.logger.warning(f"不支持的数据类型: {type(self.data)}")
                return None

        except Exception as e:
            self.logger.error(f"获取预览数据失败: {str(e)}")
            return None


class StatisticsWidget(QWidget, LoggerMixin):
    """统计信息组件"""

    def __init__(self, config: PreviewConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self._setup_ui()

    def _setup_ui(self):
        """设置UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        # 基本统计信息卡片
        if HAS_FLUENT_WIDGETS:
            self.basic_stats_card = HeaderCardWidget()
            self.basic_stats_card.setTitle("基本统计")
        else:
            self.basic_stats_card = QFrame()
            self.basic_stats_card.setFrameStyle(QFrame.Shape.Box)

        self.basic_stats_layout = QGridLayout(self.basic_stats_card)
        layout.addWidget(self.basic_stats_card)

        # 列信息卡片
        if HAS_FLUENT_WIDGETS:
            self.column_info_card = HeaderCardWidget()
            self.column_info_card.setTitle("列信息")
        else:
            self.column_info_card = QFrame()
            self.column_info_card.setFrameStyle(QFrame.Shape.Box)

        self.column_info_layout = QVBoxLayout(self.column_info_card)
        layout.addWidget(self.column_info_card)

        # 时间检测结果卡片
        if HAS_FLUENT_WIDGETS:
            self.time_detection_card = HeaderCardWidget()
            self.time_detection_card.setTitle("时间列检测")
        else:
            self.time_detection_card = QFrame()
            self.time_detection_card.setFrameStyle(QFrame.Shape.Box)

        self.time_detection_layout = QVBoxLayout(self.time_detection_card)
        layout.addWidget(self.time_detection_card)

    def update_statistics(self, basic_stats: BasicStats, column_info: list[ColumnInfo]):
        """更新统计信息"""
        try:
            # 更新基本统计
            self._update_basic_stats(basic_stats)

            # 更新列信息
            if self.config.show_column_info:
                self._update_column_info(column_info)

            # 更新时间检测结果
            if self.config.show_time_detection:
                self._update_time_detection(column_info)

        except Exception as e:
            self.logger.error(f"更新统计信息失败: {str(e)}")

    def _update_basic_stats(self, stats: BasicStats):
        """更新基本统计信息"""
        # 清除现有内容
        self._clear_layout(self.basic_stats_layout)

        # 统计项目
        stat_items = [
            ("总行数", f"{stats.total_rows:,}"),
            ("总列数", f"{stats.total_columns}"),
            ("内存占用", self._format_size(stats.memory_usage)),
            ("文件大小", self._format_size(stats.file_size)),
            ("数值列", f"{stats.numeric_columns}"),
            ("文本列", f"{stats.text_columns}"),
            ("时间列", f"{stats.time_columns}"),
            ("缺失值", f"{stats.missing_values:,} ({stats.missing_percentage:.1f}%)")
        ]

        # 创建网格布局
        row = 0
        for i, (label, value) in enumerate(stat_items):
            col = (i % 2) * 2  # 每行两列，每列占用2个网格位置

            if HAS_FLUENT_WIDGETS:
                label_widget = BodyLabel(f"{label}:")
                value_widget = StrongBodyLabel(value)
            else:
                label_widget = QLabel(f"{label}:")
                value_widget = QLabel(value)
                value_widget.setStyleSheet("font-weight: bold;")

            self.basic_stats_layout.addWidget(label_widget, row, col)
            self.basic_stats_layout.addWidget(value_widget, row, col + 1)

            if i % 2 == 1:  # 每两个项目换行
                row += 1

        # 如果有奇数个项目，最后一行只有一个项目
        if len(stat_items) % 2 == 1:
            row += 1

    def _update_column_info(self, column_info: list[ColumnInfo]):
        """更新列信息"""
        # 清除现有内容
        self._clear_layout(self.column_info_layout)

        if not column_info:
            if HAS_FLUENT_WIDGETS:
                no_data_label = CaptionLabel("无列信息")
            else:
                no_data_label = QLabel("无列信息")
            self.column_info_layout.addWidget(no_data_label)
            return

        # 创建滚动区域（如果列很多）
        if HAS_FLUENT_WIDGETS:
            scroll_area = ScrollArea()
        else:
            scroll_area = QScrollArea()

        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)

        # 限制显示的列数
        display_columns = column_info[:20]  # 最多显示20列

        for col_info in display_columns:
            # 创建列信息项
            if HAS_FLUENT_WIDGETS:
                col_card = SimpleCardWidget()
            else:
                col_card = QFrame()
                col_card.setFrameStyle(QFrame.Shape.StyledPanel)

            col_layout = QVBoxLayout(col_card)
            col_layout.setSpacing(5)

            # 列名和类型
            header_layout = QHBoxLayout()

            if HAS_FLUENT_WIDGETS:
                name_label = StrongBodyLabel(col_info.name)
                type_label = CaptionLabel(f"({col_info.data_type})")
            else:
                name_label = QLabel(col_info.name)
                name_label.setStyleSheet("font-weight: bold;")
                type_label = QLabel(f"({col_info.data_type})")
                type_label.setStyleSheet("color: #666666;")

            header_layout.addWidget(name_label)
            header_layout.addWidget(type_label)
            header_layout.addStretch()

            # 时间列标识
            if col_info.is_time_column:
                if HAS_FLUENT_WIDGETS:
                    time_label = CaptionLabel("🕒 时间列")
                else:
                    time_label = QLabel("🕒 时间列")
                time_label.setStyleSheet("color: #ff9800;")
                header_layout.addWidget(time_label)

            col_layout.addLayout(header_layout)

            # 统计信息

            info_items = []
            if col_info.null_count > 0:
                info_items.append(f"缺失: {col_info.null_percentage:.1f}%")

            if col_info.unique_count is not None:
                info_items.append(f"唯一值: {col_info.unique_count}")

            if col_info.memory_usage:
                info_items.append(f"内存: {self._format_size(col_info.memory_usage)}")

            if info_items:
                if HAS_FLUENT_WIDGETS:
                    info_label = CaptionLabel(" | ".join(info_items))
                else:
                    info_label = QLabel(" | ".join(info_items))
                    info_label.setStyleSheet("color: #666666; font-size: 11px;")
                col_layout.addWidget(info_label)

            # 示例值
            if col_info.sample_values:
                sample_text = ", ".join(str(v) for v in col_info.sample_values[:3])
                if len(sample_text) > 100:
                    sample_text = sample_text[:97] + "..."

                if HAS_FLUENT_WIDGETS:
                    sample_label = CaptionLabel(f"示例: {sample_text}")
                else:
                    sample_label = QLabel(f"示例: {sample_text}")
                    sample_label.setStyleSheet("color: #888888; font-size: 10px;")
                sample_label.setWordWrap(True)
                col_layout.addWidget(sample_label)

            scroll_layout.addWidget(col_card)

        # 如果有更多列未显示
        if len(column_info) > 20:
            if HAS_FLUENT_WIDGETS:
                more_label = CaptionLabel(f"... 还有 {len(column_info) - 20} 列未显示")
            else:
                more_label = QLabel(f"... 还有 {len(column_info) - 20} 列未显示")
            more_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            scroll_layout.addWidget(more_label)

        scroll_area.setWidget(scroll_widget)
        scroll_area.setMaximumHeight(300)  # 限制高度
        self.column_info_layout.addWidget(scroll_area)

    def _update_time_detection(self, column_info: list[ColumnInfo]):
        """更新时间检测结果"""
        # 清除现有内容
        self._clear_layout(self.time_detection_layout)

        # 查找时间列
        time_columns = [col for col in column_info if col.is_time_column]

        if not time_columns:
            if HAS_FLUENT_WIDGETS:
                no_time_label = CaptionLabel("未检测到时间列")
            else:
                no_time_label = QLabel("未检测到时间列")
            no_time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.time_detection_layout.addWidget(no_time_label)
            return

        # 显示检测到的时间列
        for time_col in time_columns:
            item_layout = QHBoxLayout()

            if HAS_FLUENT_WIDGETS:
                col_label = BodyLabel(f"🕒 {time_col.name}")
                format_label = CaptionLabel(f"格式: {time_col.time_format or '自动检测'}")
            else:
                col_label = QLabel(f"🕒 {time_col.name}")
                format_label = QLabel(f"格式: {time_col.time_format or '自动检测'}")
                format_label.setStyleSheet("color: #666666;")

            item_layout.addWidget(col_label)
            item_layout.addStretch()
            item_layout.addWidget(format_label)

            self.time_detection_layout.addLayout(item_layout)

        # 提示信息
        if HAS_FLUENT_WIDGETS:
            tip_label = CaptionLabel("💡 时间列将在分析时自动排除")
        else:
            tip_label = QLabel("💡 时间列将在分析时自动排除")
        tip_label.setStyleSheet("color: #ff9800; font-style: italic;")
        self.time_detection_layout.addWidget(tip_label)

    def _clear_layout(self, layout):
        """清除布局中的所有项目"""
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                self._clear_layout(item.layout())

    def _format_size(self, size_bytes: int) -> str:
        """格式化字节大小"""
        if size_bytes == 0:
            return "0 B"

        size_names = ["B", "KB", "MB", "GB", "TB"]
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return f"{s} {size_names[i]}"


class DataPreviewWidget(QWidget, LoggerMixin):
    """数据预览主组件"""

    def __init__(self, config: PreviewConfig | None = None, parent=None):
        super().__init__(parent)
        self.config = config or PreviewConfig()
        self.file_info: FileInfo | None = None
        self.data: Any | None = None
        self.column_info: list[ColumnInfo] = []
        self.basic_stats: BasicStats | None = None

        self._setup_ui()
        self.logger.info("数据预览组件初始化完成")

    def _setup_ui(self):
        """设置UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(0, 0, 0, 0)

        # 数据表格区域
        if HAS_FLUENT_WIDGETS:
            table_card = HeaderCardWidget()
            table_card.setTitle("数据预览")
        else:
            table_card = QFrame()
            table_card.setFrameStyle(QFrame.Shape.Box)

        table_layout = QVBoxLayout(table_card)

        # 创建表格
        self.preview_table = DataPreviewTable(self.config)
        table_layout.addWidget(self.preview_table)

        layout.addWidget(table_card, 2)  # 表格占2/3空间

        # 统计信息区域
        self.statistics_widget = StatisticsWidget(self.config)
        layout.addWidget(self.statistics_widget, 1)  # 统计信息占1/3空间

    def load_file_data(self, file_info: FileInfo, data: Any):
        """加载文件数据"""
        try:
            self.file_info = file_info
            self.data = data

            # 分析数据并生成列信息
            self.column_info = self._analyze_columns(data)

            # 生成基本统计信息
            self.basic_stats = self._generate_basic_stats(data, self.column_info)

            # 更新UI
            self.preview_table.load_data(data, self.column_info)
            self.statistics_widget.update_statistics(self.basic_stats, self.column_info)

            self.logger.info(f"数据预览加载完成: {file_info.file_name}")

        except Exception as e:
            self.logger.error(f"加载文件数据失败: {str(e)}")
            raise DataProcessingError(f"加载文件数据失败: {str(e)}") from e

    def _analyze_columns(self, data: Any) -> list[ColumnInfo]:
        """分析列信息"""
        try:
            column_info = []

            if HAS_PANDAS and isinstance(data, pd.DataFrame):
                # 处理 pandas DataFrame
                for col_name in data.columns:
                    col_data = data[col_name]

                    info = ColumnInfo(
                        name=str(col_name),
                        data_type=str(col_data.dtype),
                        null_count=int(col_data.isnull().sum()),
                        null_percentage=float(col_data.isnull().sum() / len(col_data) * 100),
                        unique_count=int(col_data.nunique()),
                        sample_values=col_data.dropna().head(3).tolist(),
                        memory_usage=int(col_data.memory_usage(deep=True))
                    )

                    # 检测时间列
                    info.is_time_column = self._is_time_column(col_name, col_data)
                    if info.is_time_column:
                        info.time_format = self._detect_time_format(col_data)

                    column_info.append(info)

            elif HAS_POLARS and hasattr(data, 'schema'):
                # 处理 polars DataFrame
                for col_name, col_type in data.schema.items():
                    col_data = data[col_name]

                    info = ColumnInfo(
                        name=str(col_name),
                        data_type=str(col_type),
                        null_count=int(col_data.null_count()),
                        null_percentage=float(col_data.null_count() / len(data) * 100),
                        unique_count=int(col_data.n_unique()),
                        sample_values=col_data.drop_nulls().head(3).to_list()
                    )

                    # 检测时间列
                    info.is_time_column = self._is_time_column_polars(col_name, col_data)
                    if info.is_time_column:
                        info.time_format = self._detect_time_format_polars(col_data)

                    column_info.append(info)

            return column_info

        except Exception as e:
            self.logger.error(f"分析列信息失败: {str(e)}")
            return []

    def _generate_basic_stats(self, data: Any, column_info: list[ColumnInfo]) -> BasicStats:
        """生成基本统计信息"""
        try:
            if HAS_PANDAS and isinstance(data, pd.DataFrame):
                total_rows = len(data)
                total_columns = len(data.columns)
                memory_usage = int(data.memory_usage(deep=True).sum())

                # 统计不同类型的列
                numeric_cols = len(data.select_dtypes(include=[np.number]).columns)
                text_cols = len(data.select_dtypes(include=['object']).columns)

            elif HAS_POLARS and hasattr(data, 'shape'):
                total_rows, total_columns = data.shape
                memory_usage = data.estimated_size() if hasattr(data, 'estimated_size') else 0

                # 统计不同类型的列
                numeric_cols = len([col for col in data.columns if data[col].dtype in [pl.Int32, pl.Int64, pl.Float32, pl.Float64]])
                text_cols = len([col for col in data.columns if data[col].dtype == pl.Utf8])

            else:
                # 默认值
                total_rows = 0
                total_columns = 0
                memory_usage = 0
                numeric_cols = 0
                text_cols = 0

            # 计算时间列数量
            time_cols = len([col for col in column_info if col.is_time_column])

            # 计算总缺失值
            total_missing = sum(col.null_count for col in column_info)
            missing_percentage = (total_missing / (total_rows * total_columns)) * 100 if total_rows > 0 and total_columns > 0 else 0

            return BasicStats(
                total_rows=total_rows,
                total_columns=total_columns,
                memory_usage=memory_usage,
                file_size=self.file_info.file_size if self.file_info else 0,
                numeric_columns=numeric_cols,
                text_columns=text_cols,
                time_columns=time_cols,
                missing_values=total_missing,
                missing_percentage=missing_percentage
            )

        except Exception as e:
            self.logger.error(f"生成基本统计失败: {str(e)}")
            return BasicStats(0, 0, 0, 0, 0, 0, 0, 0, 0.0)

    def _is_time_column(self, col_name: str, col_data: Any) -> bool:
        """检测是否为时间列（pandas）"""
        try:
            # 检查列名
            time_names = ['datetime', 'timestamp', 'time', 'date', 'tagtime']
            if any(name in col_name.lower() for name in time_names):
                return True

            # 检查数据类型
            if pd.api.types.is_datetime64_any_dtype(col_data):
                return True

            # 尝试解析字符串为时间
            if col_data.dtype == 'object':
                sample = col_data.dropna().head(10)
                if len(sample) > 0:
                    try:
                        pd.to_datetime(sample.iloc[0])
                        return True
                    except Exception:
                        pass

            return False

        except Exception:
            return False

    def _is_time_column_polars(self, col_name: str, col_data: Any) -> bool:
        """检测是否为时间列（polars）"""
        try:
            # 检查列名
            time_names = ['datetime', 'timestamp', 'time', 'date', 'tagtime']
            if any(name in col_name.lower() for name in time_names):
                return True

            # 检查数据类型
            if col_data.dtype in [pl.Date, pl.Datetime, pl.Time]:
                return True

            return False

        except Exception:
            return False

    def _detect_time_format(self, col_data: Any) -> str | None:
        """检测时间格式（pandas）"""
        try:
            if col_data.dtype == 'object':
                sample = col_data.dropna().head(5)
                if len(sample) > 0:
                    # 常见的时间格式
                    formats = [
                        '%Y-%m-%d %H:%M:%S',
                        '%Y-%m-%d',
                        '%d/%m/%Y',
                        '%m/%d/%Y',
                        '%Y/%m/%d',
                        '%d-%m-%Y',
                        '%m-%d-%Y'
                    ]

                    for fmt in formats:
                        try:
                            pd.to_datetime(sample.iloc[0], format=fmt)
                            return fmt
                        except Exception:
                            continue

            return None

        except Exception:
            return None

    def _detect_time_format_polars(self, col_data: Any) -> str | None:
        """检测时间格式（polars）"""
        # polars 的时间格式检测相对简单
        try:
            if col_data.dtype in [pl.Date, pl.Datetime]:
                return "自动检测"
            return None
        except Exception:
            return None

    def clear_preview(self):
        """清除预览"""
        self.file_info = None
        self.data = None
        self.column_info = []
        self.basic_stats = None

        # 清除表格
        if hasattr(self.preview_table, 'clear'):
            self.preview_table.clear()

        self.logger.info("数据预览已清除")

    def get_column_info(self) -> list[ColumnInfo]:
        """获取列信息"""
        return self.column_info

    def get_basic_stats(self) -> BasicStats | None:
        """获取基本统计信息"""
        return self.basic_stats

    def apply_responsive_layout(self, layout_mode: str):
        """应用响应式布局"""
        if layout_mode == 'mobile':
            # 移动端：隐藏部分列信息，简化显示
            self.config.max_preview_columns = 5
            self.config.show_column_info = False
        elif layout_mode == 'tablet':
            # 平板：适中显示
            self.config.max_preview_columns = 10
            self.config.show_column_info = True
        else:  # desktop
            # 桌面：完整显示
            self.config.max_preview_columns = 20
            self.config.show_column_info = True

        self.logger.debug(f"应用响应式布局: {layout_mode}")


def create_data_preview_widget(config: PreviewConfig | None = None) -> DataPreviewWidget:
    """创建数据预览组件的工厂函数"""
    try:
        return DataPreviewWidget(config)
    except Exception as e:
        print(f"创建数据预览组件失败: {str(e)}")
        raise
