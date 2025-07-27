"""
历史管理页面
提供分析历史记录的查看、搜索、筛选和管理功能
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

try:
    from PyQt6.QtCore import (
        QAbstractTableModel,
        QModelIndex,
        Qt,
        QTimer,
        pyqtSignal,
    )
    from PyQt6.QtWidgets import (
        QAbstractItemView,
        QDialog,
        QDialogButtonBox,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QLineEdit,
        QMessageBox,
        QPushButton,
        QSplitter,
        QTableWidget,
        QTableWidgetItem,
        QTextEdit,
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
    class QLabel:
        pass
    class QPushButton:
        pass
    class QGroupBox:
        pass
    class QTableWidget:
        pass
    class QLineEdit:
        pass
    class QComboBox:
        pass
    class QThread:
        pass
    def pyqtSignal(*args):
        return lambda *a, **k: None

try:
    from qfluentwidgets import (
        BodyLabel,
        CaptionLabel,
        CardWidget,
        ComboBox,
        FluentIcon,
        HeaderCardWidget,
        InfoBar,
        InfoBarPosition,
        LineEdit,
        PlainTextEdit,
        PrimaryPushButton,
        PushButton,
        SearchLineEdit,
        SimpleCardWidget,
        StrongBodyLabel,
        TableWidget,
        TitleLabel,
        ToolButton,
    )
    HAS_FLUENT_WIDGETS = True
except ImportError:
    HAS_FLUENT_WIDGETS = False
    # 模拟类定义
    class CardWidget:
        pass
    class HeaderCardWidget:
        def setTitle(self, title):
            pass
        @property
        def viewLayout(self):
            return QVBoxLayout()
    class PrimaryPushButton:
        pass
    class PushButton:
        pass
    class ToolButton:
        pass
    class BodyLabel:
        pass
    class StrongBodyLabel:
        pass
    class CaptionLabel:
        pass
    class TitleLabel:
        pass
    class LineEdit:
        pass
    class ComboBox:
        def addItem(self, text, data=None):
            pass
        def currentData(self):
            return None
    class TableWidget:
        pass
    class SearchLineEdit:
        def setPlaceholderText(self, text):
            pass
    class SimpleCardWidget:
        pass
    class PlainTextEdit:
        def setPlainText(self, text):
            pass
        def setMaximumHeight(self, height):
            pass
    class InfoBar:
        @staticmethod
        def error(*args, **kwargs):
            pass
        @staticmethod
        def success(*args, **kwargs):
            pass
    class InfoBarPosition:
        TOP = "top"
    class FluentIcon(Enum):
        HISTORY = "history"
        SEARCH = "search"
        FILTER = "filter"
        DELETE = "delete"
        DOWNLOAD = "download"
        SYNC = "sync"
        SETTING = "setting"

from ..models.analysis_history import (
    AnalysisHistoryDB,
    AnalysisHistoryRecord,
    AnalysisStatus,
)
from ..utils.basic_logging import LoggerMixin
from ..utils.icon_utils import safe_set_icon


class HistoryFilterType(str, Enum):
    """历史记录筛选类型"""
    ALL = "all"
    COMPLETED = "completed"
    FAILED = "failed"
    RECENT = "recent"
    TODAY = "today"
    THIS_WEEK = "this_week"
    THIS_MONTH = "this_month"


@dataclass
class HistoryPageConfig:
    """历史页面配置"""
    # 数据库配置
    db_path: str = "data/analysis_history.db"

    # 显示配置
    records_per_page: int = 50
    auto_refresh_interval: int = 30  # 秒
    show_thumbnails: bool = True

    # 筛选配置
    default_filter: HistoryFilterType = HistoryFilterType.ALL
    enable_date_filter: bool = True
    enable_status_filter: bool = True

    # 操作配置
    enable_delete: bool = True
    enable_export: bool = True
    enable_reload: bool = True
    confirm_delete: bool = True


class HistoryTableModel(QAbstractTableModel, LoggerMixin):
    """历史记录表格模型"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.records: list[AnalysisHistoryRecord] = []
        self.headers = [
            "ID", "文件名", "分析类型", "状态", "创建时间",
            "执行时长", "文件大小", "分析ID"
        ]

    def rowCount(self, parent=QModelIndex()):
        return len(self.records)

    def columnCount(self, parent=QModelIndex()):
        return len(self.headers)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or index.row() >= len(self.records):
            return None

        record = self.records[index.row()]
        col = index.column()

        if role == Qt.ItemDataRole.DisplayRole:
            if col == 0:  # ID
                return str(record.id) if record.id else ""
            elif col == 1:  # 文件名
                return record.file_name
            elif col == 2:  # 分析类型
                type_map = {
                    "comprehensive": "综合分析",
                    "descriptive": "描述性统计",
                    "correlation": "关联分析",
                    "anomaly": "异常检测",
                    "timeseries": "时间序列"
                }
                return type_map.get(record.analysis_type, record.analysis_type)
            elif col == 3:  # 状态
                return record.get_status_text()
            elif col == 4:  # 创建时间
                return record.created_at.strftime("%Y-%m-%d %H:%M") if record.created_at else ""
            elif col == 5:  # 执行时长
                return record.get_duration_text()
            elif col == 6:  # 文件大小
                return record.get_file_size_text()
            elif col == 7:  # 分析ID
                return record.analysis_id[:8] + "..." if len(record.analysis_id) > 8 else record.analysis_id

        elif role == Qt.ItemDataRole.UserRole:
            return record

        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return self.headers[section]
        return None

    def set_records(self, records: list[AnalysisHistoryRecord]):
        """设置记录数据"""
        self.beginResetModel()
        self.records = records
        self.endResetModel()

    def get_record(self, row: int) -> AnalysisHistoryRecord | None:
        """获取指定行的记录"""
        if 0 <= row < len(self.records):
            return self.records[row]
        return None


class HistoryDetailDialog(QDialog, LoggerMixin):
    """历史记录详情对话框"""

    def __init__(self, record: AnalysisHistoryRecord, parent=None):
        super().__init__(parent)
        self.record = record
        self._setup_ui()
        self._load_record_details()

    def _setup_ui(self):
        """设置UI"""
        self.setWindowTitle(f"分析记录详情 - {self.record.file_name}")
        self.setMinimumSize(600, 500)

        layout = QVBoxLayout(self)

        # 基本信息
        basic_group = self._create_basic_info_group()
        layout.addWidget(basic_group)

        # 分析配置
        config_group = self._create_config_info_group()
        layout.addWidget(config_group)

        # 结果摘要
        if self.record.result_summary:
            result_group = self._create_result_info_group()
            layout.addWidget(result_group)

        # 按钮
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _create_basic_info_group(self):
        """创建基本信息组"""
        if HAS_FLUENT_WIDGETS:
            group = HeaderCardWidget()
            group.setTitle("基本信息")
        else:
            group = QGroupBox("基本信息")

        layout = QGridLayout(group)
        row = 0

        info_items = [
            ("分析ID", self.record.analysis_id),
            ("文件名", self.record.file_name),
            ("文件路径", self.record.file_path),
            ("文件大小", self.record.get_file_size_text()),
            ("文件哈希", self.record.file_hash[:16] + "..." if len(self.record.file_hash) > 16 else self.record.file_hash),
            ("分析类型", self.record.analysis_type),
            ("时间列", self.record.time_column or "无"),
            ("状态", self.record.get_status_text()),
            ("创建时间", self.record.created_at.strftime("%Y-%m-%d %H:%M:%S") if self.record.created_at else ""),
            ("开始时间", self.record.started_at.strftime("%Y-%m-%d %H:%M:%S") if self.record.started_at else ""),
            ("完成时间", self.record.completed_at.strftime("%Y-%m-%d %H:%M:%S") if self.record.completed_at else ""),
            ("执行时长", self.record.get_duration_text())
        ]

        for label_text, value_text in info_items:
            if HAS_FLUENT_WIDGETS:
                label = BodyLabel(f"{label_text}:")
                value = CaptionLabel(str(value_text))
            else:
                label = QLabel(f"{label_text}:")
                value = QLabel(str(value_text))
                value.setWordWrap(True)

            layout.addWidget(label, row, 0)
            layout.addWidget(value, row, 1)
            row += 1

        return group

    def _create_config_info_group(self):
        """创建配置信息组"""
        if HAS_FLUENT_WIDGETS:
            group = HeaderCardWidget()
            group.setTitle("分析配置")
        else:
            group = QGroupBox("分析配置")

        layout = QVBoxLayout(group)

        if HAS_FLUENT_WIDGETS:
            config_text = PlainTextEdit()
        else:
            config_text = QTextEdit()

        # 格式化配置信息
        config_str = ""
        if self.record.analysis_config:
            for key, value in self.record.analysis_config.items():
                config_str += f"{key}: {value}\n"
        else:
            config_str = "无配置信息"

        config_text.setPlainText(config_str)
        config_text.setMaximumHeight(150)
        layout.addWidget(config_text)

        return group

    def _create_result_info_group(self):
        """创建结果信息组"""
        if HAS_FLUENT_WIDGETS:
            group = HeaderCardWidget()
            group.setTitle("结果摘要")
            # 创建结果信息组的内容布局
            layout = QVBoxLayout()
            layout.setContentsMargins(0, 0, 0, 0)
        else:
            group = QGroupBox("结果摘要")
            layout = QVBoxLayout(group)

        if HAS_FLUENT_WIDGETS:
            result_text = PlainTextEdit()
        else:
            result_text = QTextEdit()

        # 格式化结果摘要
        result_str = ""
        if self.record.result_summary:
            for key, value in self.record.result_summary.items():
                result_str += f"{key}: {value}\n"
        else:
            result_str = "无结果摘要"

        result_text.setPlainText(result_str)
        result_text.setMaximumHeight(200)
        layout.addWidget(result_text)

        # 将内容布局添加到HeaderCardWidget（仅在使用Fluent Widgets时）
        if HAS_FLUENT_WIDGETS:
            group.viewLayout.addLayout(layout)

        return group

    def _load_record_details(self):
        """加载记录详情"""
        # 这里可以添加更多详情加载逻辑
        pass


class HistoryPage(QWidget, LoggerMixin):
    """历史管理页面"""

    # 信号
    record_selected = pyqtSignal(object)  # AnalysisHistoryRecord
    record_reload_requested = pyqtSignal(object)  # AnalysisHistoryRecord

    def __init__(self, config: HistoryPageConfig | None = None, parent=None):
        super().__init__(parent)
        self.config = config or HistoryPageConfig()

        # 核心组件
        self.db: AnalysisHistoryDB | None = None
        self.table_model: HistoryTableModel | None = None
        self.current_records: list[AnalysisHistoryRecord] = []
        self.current_filter = HistoryFilterType.ALL
        self.search_text = ""

        self._setup_ui()
        self._setup_connections()
        self._init_database()
        self._refresh_records()

        # 自动刷新定时器
        if self.config.auto_refresh_interval > 0:
            self.refresh_timer = QTimer()
            self.refresh_timer.timeout.connect(self._refresh_records)
            self.refresh_timer.start(self.config.auto_refresh_interval * 1000)

        self.logger.info("历史管理页面初始化完成")

    def _setup_ui(self):
        """设置UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # 页面标题
        title_layout = QHBoxLayout()

        if HAS_FLUENT_WIDGETS:
            title_label = TitleLabel("分析历史")
        else:
            title_label = QLabel("分析历史")
            title_label.setStyleSheet("font-size: 20px; font-weight: bold;")

        title_layout.addWidget(title_label)
        title_layout.addStretch()

        # 刷新按钮
        if HAS_FLUENT_WIDGETS:
            self.refresh_btn = ToolButton()
            safe_set_icon(self.refresh_btn, FluentIcon.SYNC)
        else:
            self.refresh_btn = QPushButton("刷新")
        self.refresh_btn.setToolTip("刷新记录列表")
        title_layout.addWidget(self.refresh_btn)

        layout.addLayout(title_layout)

        # 控制面板
        control_panel = self._create_control_panel()
        layout.addWidget(control_panel)

        # 主内容区域
        content_splitter = QSplitter(Qt.Orientation.Vertical)

        # 记录列表
        records_widget = self._create_records_widget()
        content_splitter.addWidget(records_widget)

        # 统计信息
        stats_widget = self._create_stats_widget()
        content_splitter.addWidget(stats_widget)

        # 设置分割比例
        content_splitter.setStretchFactor(0, 3)
        content_splitter.setStretchFactor(1, 1)

        layout.addWidget(content_splitter)

    def _create_control_panel(self):
        """创建控制面板"""
        if HAS_FLUENT_WIDGETS:
            panel = SimpleCardWidget()
        else:
            panel = QFrame()
            panel.setFrameStyle(QFrame.Shape.StyledPanel)

        layout = QHBoxLayout(panel)

        # 搜索框
        if HAS_FLUENT_WIDGETS:
            self.search_edit = SearchLineEdit()
            self.search_edit.setPlaceholderText("搜索文件名、路径或分析ID...")
        else:
            self.search_edit = QLineEdit()
            self.search_edit.setPlaceholderText("搜索文件名、路径或分析ID...")

        self.search_edit.setMaximumWidth(300)
        layout.addWidget(self.search_edit)

        # 状态筛选
        if HAS_FLUENT_WIDGETS:
            layout.addWidget(BodyLabel("筛选:"))
            self.filter_combo = ComboBox()
        else:
            layout.addWidget(QLabel("筛选:"))
            self.filter_combo = QComboBox()

        filter_items = [
            ("all", "全部"),
            ("completed", "已完成"),
            ("failed", "失败"),
            ("recent", "最近"),
            ("today", "今天"),
            ("this_week", "本周"),
            ("this_month", "本月")
        ]

        for value, text in filter_items:
            self.filter_combo.addItem(text, value)

        layout.addWidget(self.filter_combo)

        layout.addStretch()

        # 操作按钮
        if self.config.enable_export:
            if HAS_FLUENT_WIDGETS:
                self.export_btn = PushButton("导出")
                safe_set_icon(self.export_btn, FluentIcon.DOWNLOAD)
            else:
                self.export_btn = QPushButton("导出")
            layout.addWidget(self.export_btn)

        if self.config.enable_delete:
            if HAS_FLUENT_WIDGETS:
                self.delete_btn = PushButton("删除选中")
                safe_set_icon(self.delete_btn, FluentIcon.DELETE)
            else:
                self.delete_btn = QPushButton("删除选中")
            self.delete_btn.setEnabled(False)
            layout.addWidget(self.delete_btn)

        return panel

    def _create_records_widget(self):
        """创建记录列表组件"""
        if HAS_FLUENT_WIDGETS:
            widget = HeaderCardWidget()
            widget.setTitle("历史记录")
            layout = QVBoxLayout()
            layout.setContentsMargins(0, 0, 0, 0)
        else:
            widget = QGroupBox("历史记录")
            layout = QVBoxLayout(widget)

        # 创建表格
        if HAS_FLUENT_WIDGETS:
            self.records_table = TableWidget()
        else:
            self.records_table = QTableWidget()

        # 设置表格模型
        self.table_model = HistoryTableModel()

        if HAS_PYQT6:
            # 设置表格属性
            self.records_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
            self.records_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
            self.records_table.setAlternatingRowColors(True)
            self.records_table.setSortingEnabled(True)

            # 设置列数和标题
            headers = ["ID", "文件名", "分析类型", "状态", "创建时间", "执行时长", "文件大小", "分析ID"]
            self.records_table.setColumnCount(len(headers))
            self.records_table.setHorizontalHeaderLabels(headers)

            # 设置列宽
            header = self.records_table.horizontalHeader()
            header.setStretchLastSection(True)
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)  # ID
            header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)  # 文件名
            header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)  # 分析类型
            header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)  # 状态
            header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)  # 创建时间
            header.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)  # 执行时长
            header.setSectionResizeMode(6, QHeaderView.ResizeMode.ResizeToContents)  # 文件大小

        layout.addWidget(self.records_table)

        # 将内容布局添加到HeaderCardWidget（仅在使用Fluent Widgets时）
        if HAS_FLUENT_WIDGETS:
            widget.viewLayout.addLayout(layout)

        return widget

    def _create_stats_widget(self):
        """创建统计信息组件"""
        if HAS_FLUENT_WIDGETS:
            widget = HeaderCardWidget()
            widget.setTitle("统计信息")
            # 创建统计信息组件的内容布局
            layout = QHBoxLayout()
            layout.setContentsMargins(0, 0, 0, 0)
        else:
            widget = QGroupBox("统计信息")
            layout = QHBoxLayout(widget)

        # 总数统计
        if HAS_FLUENT_WIDGETS:
            self.total_label = StrongBodyLabel("总记录数: 0")
        else:
            self.total_label = QLabel("总记录数: 0")
            self.total_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.total_label)

        # 完成统计
        if HAS_FLUENT_WIDGETS:
            self.completed_label = BodyLabel("已完成: 0")
        else:
            self.completed_label = QLabel("已完成: 0")
        layout.addWidget(self.completed_label)

        # 失败统计
        if HAS_FLUENT_WIDGETS:
            self.failed_label = BodyLabel("失败: 0")
        else:
            self.failed_label = QLabel("失败: 0")
        layout.addWidget(self.failed_label)

        layout.addStretch()

        # 平均执行时间
        if HAS_FLUENT_WIDGETS:
            self.avg_time_label = CaptionLabel("平均执行时间: --")
        else:
            self.avg_time_label = QLabel("平均执行时间: --")
        layout.addWidget(self.avg_time_label)

        # 将内容布局添加到HeaderCardWidget（仅在使用Fluent Widgets时）
        if HAS_FLUENT_WIDGETS:
            widget.viewLayout.addLayout(layout)

        return widget

    def _setup_connections(self):
        """设置信号连接"""
        # 搜索
        self.search_edit.textChanged.connect(self._on_search_changed)

        # 筛选
        if hasattr(self.filter_combo, 'currentTextChanged'):
            self.filter_combo.currentTextChanged.connect(self._on_filter_changed)

        # 刷新
        self.refresh_btn.clicked.connect(self._refresh_records)

        # 表格事件
        if HAS_PYQT6:
            self.records_table.itemSelectionChanged.connect(self._on_selection_changed)
            self.records_table.itemDoubleClicked.connect(self._on_item_double_clicked)

        # 操作按钮
        if hasattr(self, 'export_btn'):
            self.export_btn.clicked.connect(self._export_records)

        if hasattr(self, 'delete_btn'):
            self.delete_btn.clicked.connect(self._delete_selected_records)

    def _init_database(self):
        """初始化数据库"""
        try:
            self.db = AnalysisHistoryDB(self.config.db_path)
            self.logger.info("历史数据库初始化完成")
        except Exception as e:
            self.logger.error(f"历史数据库初始化失败: {str(e)}")
            self._show_error(f"数据库初始化失败: {str(e)}")

    def _refresh_records(self):
        """刷新记录列表"""
        if not self.db:
            return

        try:
            # 根据当前筛选条件获取记录
            if self.search_text:
                records = self.db.search_records(
                    self.search_text,
                    limit=self.config.records_per_page
                )
            else:
                # 根据筛选类型获取记录
                status_filter = None
                if self.current_filter == HistoryFilterType.COMPLETED:
                    status_filter = AnalysisStatus.COMPLETED
                elif self.current_filter == HistoryFilterType.FAILED:
                    status_filter = AnalysisStatus.FAILED

                records = self.db.get_records(
                    status=status_filter,
                    limit=self.config.records_per_page
                )

                # 时间筛选
                if self.current_filter in [HistoryFilterType.TODAY, HistoryFilterType.THIS_WEEK, HistoryFilterType.THIS_MONTH]:
                    records = self._filter_by_date(records, self.current_filter)
                elif self.current_filter == HistoryFilterType.RECENT:
                    records = records[:10]  # 最近10条

            self.current_records = records
            self._update_table(records)
            self._update_statistics()

            self.logger.debug(f"刷新记录列表: {len(records)} 条记录")

        except Exception as e:
            self.logger.error(f"刷新记录失败: {str(e)}")
            self._show_error(f"刷新记录失败: {str(e)}")

    def _update_table(self, records: list[AnalysisHistoryRecord]):
        """更新表格数据"""
        if not HAS_PYQT6:
            return

        self.records_table.setRowCount(len(records))

        for row, record in enumerate(records):
            # ID
            self.records_table.setItem(row, 0, QTableWidgetItem(str(record.id) if record.id else ""))

            # 文件名
            self.records_table.setItem(row, 1, QTableWidgetItem(record.file_name))

            # 分析类型
            type_map = {
                "comprehensive": "综合分析",
                "descriptive": "描述性统计",
                "correlation": "关联分析",
                "anomaly": "异常检测",
                "timeseries": "时间序列"
            }
            analysis_type = type_map.get(record.analysis_type, record.analysis_type)
            self.records_table.setItem(row, 2, QTableWidgetItem(analysis_type))

            # 状态
            self.records_table.setItem(row, 3, QTableWidgetItem(record.get_status_text()))

            # 创建时间
            created_time = record.created_at.strftime("%Y-%m-%d %H:%M") if record.created_at else ""
            self.records_table.setItem(row, 4, QTableWidgetItem(created_time))

            # 执行时长
            self.records_table.setItem(row, 5, QTableWidgetItem(record.get_duration_text()))

            # 文件大小
            self.records_table.setItem(row, 6, QTableWidgetItem(record.get_file_size_text()))

            # 分析ID
            analysis_id = record.analysis_id[:8] + "..." if len(record.analysis_id) > 8 else record.analysis_id
            self.records_table.setItem(row, 7, QTableWidgetItem(analysis_id))

    def _update_statistics(self):
        """更新统计信息"""
        if not self.db:
            return

        try:
            stats = self.db.get_statistics()

            # 更新标签
            total_count = stats.get('total_count', 0)
            self.total_label.setText(f"总记录数: {total_count}")

            status_counts = stats.get('status_counts', {})
            completed_count = status_counts.get('completed', 0)
            failed_count = status_counts.get('failed', 0)

            self.completed_label.setText(f"已完成: {completed_count}")
            self.failed_label.setText(f"失败: {failed_count}")

            # 平均执行时间
            avg_time = stats.get('avg_execution_time_ms', 0)
            if avg_time:
                if avg_time < 1000:
                    avg_text = f"{avg_time:.0f}ms"
                elif avg_time < 60000:
                    avg_text = f"{avg_time / 1000:.1f}s"
                else:
                    avg_text = f"{avg_time / 60000:.1f}min"
                self.avg_time_label.setText(f"平均执行时间: {avg_text}")
            else:
                self.avg_time_label.setText("平均执行时间: --")

        except Exception as e:
            self.logger.error(f"更新统计信息失败: {str(e)}")

    def _filter_by_date(self, records: list[AnalysisHistoryRecord], filter_type: HistoryFilterType) -> list[AnalysisHistoryRecord]:
        """按日期筛选记录"""
        now = datetime.now()

        if filter_type == HistoryFilterType.TODAY:
            start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif filter_type == HistoryFilterType.THIS_WEEK:
            start_date = now - timedelta(days=now.weekday())
            start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        elif filter_type == HistoryFilterType.THIS_MONTH:
            start_date = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        else:
            return records

        filtered_records = []
        for record in records:
            if record.created_at and record.created_at >= start_date:
                filtered_records.append(record)

        return filtered_records

    def _on_search_changed(self, text: str):
        """搜索文本改变处理"""
        self.search_text = text.strip()
        # 延迟搜索，避免频繁查询
        if hasattr(self, 'search_timer'):
            self.search_timer.stop()

        self.search_timer = QTimer()
        self.search_timer.setSingleShot(True)
        self.search_timer.timeout.connect(self._refresh_records)
        self.search_timer.start(500)  # 500ms延迟

    def _on_filter_changed(self, text: str):
        """筛选改变处理"""
        # 获取当前选中的筛选值
        if hasattr(self.filter_combo, 'currentData'):
            filter_value = self.filter_combo.currentData()
        else:
            # 模拟获取
            filter_map = {
                "全部": "all",
                "已完成": "completed",
                "失败": "failed",
                "最近": "recent",
                "今天": "today",
                "本周": "this_week",
                "本月": "this_month"
            }
            filter_value = filter_map.get(text, "all")

        self.current_filter = HistoryFilterType(filter_value)
        self._refresh_records()

    def _on_selection_changed(self):
        """选择改变处理"""
        if not HAS_PYQT6:
            return

        selected_rows = set()
        for item in self.records_table.selectedItems():
            selected_rows.add(item.row())

        # 更新删除按钮状态
        if hasattr(self, 'delete_btn'):
            self.delete_btn.setEnabled(len(selected_rows) > 0)

        # 发出选择信号
        if len(selected_rows) == 1:
            row = list(selected_rows)[0]
            if row < len(self.current_records):
                record = self.current_records[row]
                self.record_selected.emit(record)

    def _on_item_double_clicked(self, item):
        """双击项目处理"""
        if not HAS_PYQT6:
            return

        row = item.row()
        if row < len(self.current_records):
            record = self.current_records[row]
            self._show_record_detail(record)

    def _show_record_detail(self, record: AnalysisHistoryRecord):
        """显示记录详情"""
        try:
            dialog = HistoryDetailDialog(record, self)
            dialog.exec()
        except Exception as e:
            self.logger.error(f"显示记录详情失败: {str(e)}")
            self._show_error(f"显示记录详情失败: {str(e)}")

    def _export_records(self):
        """导出记录"""
        # TODO: 实现导出功能
        self._show_info("导出功能开发中...")

    def _delete_selected_records(self):
        """删除选中的记录"""
        if not HAS_PYQT6 or not self.db:
            return

        selected_rows = set()
        for item in self.records_table.selectedItems():
            selected_rows.add(item.row())

        if not selected_rows:
            return

        # 确认删除
        if self.config.confirm_delete:
            reply = QMessageBox.question(
                self, "确认删除",
                f"确定要删除选中的 {len(selected_rows)} 条记录吗？\n此操作不可撤销。",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )

            if reply != QMessageBox.StandardButton.Yes:
                return

        # 执行删除
        deleted_count = 0
        for row in sorted(selected_rows, reverse=True):
            if row < len(self.current_records):
                record = self.current_records[row]
                if record.id and self.db.delete_record(record.id):
                    deleted_count += 1

        if deleted_count > 0:
            self._show_info(f"已删除 {deleted_count} 条记录")
            self._refresh_records()
        else:
            self._show_error("删除记录失败")

    def _show_error(self, message: str):
        """显示错误信息"""
        if HAS_FLUENT_WIDGETS:
            InfoBar.error(
                title="错误",
                content=message,
                orient=Qt.Orientation.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=5000,
                parent=self
            )
        else:
            QMessageBox.critical(self, "错误", message)

        self.logger.error(f"显示错误: {message}")

    def _show_info(self, message: str):
        """显示信息"""
        if HAS_FLUENT_WIDGETS:
            InfoBar.success(
                title="信息",
                content=message,
                orient=Qt.Orientation.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=3000,
                parent=self
            )
        else:
            QMessageBox.information(self, "信息", message)

        self.logger.info(f"显示信息: {message}")

    def add_record(self, record: AnalysisHistoryRecord):
        """添加新记录"""
        if self.db:
            try:
                self.db.save_record(record)
                self._refresh_records()
                self.logger.info(f"添加历史记录: {record.analysis_id}")
            except Exception as e:
                self.logger.error(f"添加历史记录失败: {str(e)}")

    def update_record(self, record: AnalysisHistoryRecord):
        """更新记录"""
        if self.db:
            try:
                self.db.save_record(record)
                self._refresh_records()
                self.logger.info(f"更新历史记录: {record.analysis_id}")
            except Exception as e:
                self.logger.error(f"更新历史记录失败: {str(e)}")

    def apply_responsive_layout(self, layout_mode: str):
        """应用响应式布局"""
        if layout_mode == 'mobile':
            self.setContentsMargins(10, 10, 10, 10)
        elif layout_mode == 'tablet':
            self.setContentsMargins(15, 15, 15, 15)
        else:  # desktop
            self.setContentsMargins(20, 20, 20, 20)

        self.logger.debug(f"应用响应式布局: {layout_mode}")


def create_history_page(config: HistoryPageConfig | None = None) -> HistoryPage:
    """创建历史页面的工厂函数"""
    try:
        print("开始创建历史页面...")
        page = HistoryPage(config)
        print("历史页面创建成功")
        return page
    except Exception as e:
        import traceback
        print(f"创建历史页面失败: {str(e)}")
        print(f"异常类型: {type(e).__name__}")
        print("完整堆栈跟踪:")
        traceback.print_exc()
        raise
