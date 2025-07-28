"""
数据分析页面
提供数据分析配置界面和结果展示功能
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

try:
    from PyQt6.QtCore import Qt, QThread, pyqtSignal
    from PyQt6.QtWidgets import (
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QMessageBox,
        QProgressBar,
        QPushButton,
        QScrollArea,
        QSpinBox,
        QSplitter,
        QTabWidget,
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
    class QCheckBox:
        pass
    class QComboBox:
        pass
    class QSpinBox:
        pass
    class QDoubleSpinBox:
        pass
    class QProgressBar:
        pass
    class QTextEdit:
        pass
    class QSplitter:
        pass
    class QTabWidget:
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
        CheckBox,
        ComboBox,
        DoubleSpinBox,
        FluentIcon,
        HeaderCardWidget,
        InfoBar,
        InfoBarPosition,
        Pivot,
        PivotItem,
        PrimaryPushButton,
        ProgressBar,
        PushButton,
        ScrollArea,
        SimpleCardWidget,
        SpinBox,
        StrongBodyLabel,
        SubtitleLabel,
        TextEdit,
        TitleLabel,
    )
    HAS_FLUENT_WIDGETS = True
except ImportError:
    HAS_FLUENT_WIDGETS = False
    # 模拟类定义
    class CardWidget:
        pass
    class HeaderCardWidget:
        pass
    class PrimaryPushButton:
        pass
    class BodyLabel:
        pass
    class CheckBox:
        pass
    class ComboBox:
        pass
    class SpinBox:
        pass
    class DoubleSpinBox:
        pass
    class ProgressBar:
        pass
    class TextEdit:
        pass
    class ScrollArea:
        pass
    class Pivot:
        pass
    class PivotItem:
        pass
    class FluentIcon:
        PLAY = "play"
        SAVE = "save"
        DOWNLOAD = "download"
        SETTING = "setting"
        CHART = "chart"
        DOWNLOAD = "download"

try:

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from ..core.analysis_engine import AnalysisConfig, AnalysisEngine
from ..core.analysis_result_manager import get_analysis_result_manager
from ..core.chart_renderer import ChartFormat, ChartRenderer, ChartStyle, ChartType
from ..core.history_manager import HistoryManager, get_history_manager
from ..export.export_manager import ExportManager
from ..models.analysis_history import AnalysisHistoryRecord, AnalysisStatus
from ..models.extended_analysis_result import AnalysisResult
from ..models.file_info import FileInfo
from ..utils.basic_logging import LoggerMixin
from ..utils.icon_utils import safe_set_icon


@dataclass
class AnalysisPageConfig:
    """分析页面配置"""
    # 分析配置
    enable_descriptive_stats: bool = True
    enable_correlation_analysis: bool = True
    enable_anomaly_detection: bool = True
    enable_time_series_analysis: bool = False

    # 默认分析参数
    default_correlation_method: str = "pearson"
    default_outlier_method: str = "iqr"
    default_outlier_threshold: float = 3.0

    # 图表配置
    enable_charts: bool = True
    default_chart_backend: str = "matplotlib"
    default_export_format: str = "png"

    # UI配置
    show_advanced_options: bool = False
    auto_run_analysis: bool = False
    max_result_rows: int = 1000


class AnalysisWorker(QThread):
    """分析工作线程"""

    # 信号
    progress_updated = pyqtSignal(int, str)  # (progress, message)
    analysis_completed = pyqtSignal(object)  # AnalysisResult
    analysis_failed = pyqtSignal(str)  # error_message
    status_changed = pyqtSignal(str)  # AnalysisStatus

    def __init__(self, engine: AnalysisEngine, data: Any, config: dict[str, Any]):
        super().__init__()
        self.engine = engine
        self.data = data
        self.config = config
        self._cancel_requested = False

    def run(self):
        """执行分析"""
        try:
            self.status_changed.emit(AnalysisStatus.RUNNING.value)
            self.progress_updated.emit(0, "开始数据分析...")

            # 准备分析参数
            time_column = self.config.get('time_column')
            exclude_columns = self.config.get('exclude_columns', [])

            self.progress_updated.emit(20, "准备数据...")

            if self._cancel_requested:
                self.status_changed.emit(AnalysisStatus.CANCELLED.value)
                return

            # 执行分析
            self.progress_updated.emit(40, "执行统计分析...")
            result = self.engine.analyze_dataset(
                self.data,
                time_column=time_column,
                exclude_columns=exclude_columns
            )

            if self._cancel_requested:
                self.status_changed.emit(AnalysisStatus.CANCELLED.value)
                return

            self.progress_updated.emit(100, "分析完成")
            self.status_changed.emit(AnalysisStatus.COMPLETED.value)
            self.analysis_completed.emit(result)

        except Exception as e:
            self.analysis_failed.emit(str(e))
            self.status_changed.emit(AnalysisStatus.FAILED.value)

    def cancel(self):
        """取消分析"""
        self._cancel_requested = True


class AnalysisConfigWidget(QWidget, LoggerMixin):
    """分析配置组件"""

    # 信号
    config_changed = pyqtSignal(dict)  # 配置改变

    def __init__(self, config: AnalysisPageConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self.analysis_config = AnalysisConfig()
        self._setup_ui()
        self._setup_connections()

    def _setup_ui(self):
        """设置UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        # 分析选项组
        analysis_group = self._create_analysis_options()
        layout.addWidget(analysis_group)

        # 参数配置组
        params_group = self._create_parameter_config()
        layout.addWidget(params_group)

        # 高级选项组
        if self.config.show_advanced_options:
            advanced_group = self._create_advanced_options()
            layout.addWidget(advanced_group)

        layout.addStretch()

    def _create_analysis_options(self):
        """创建分析选项组"""
        if HAS_FLUENT_WIDGETS:
            group = HeaderCardWidget()
            group.setTitle("分析选项")
        else:
            group = QGroupBox("分析选项")

        layout = QVBoxLayout(group)

        # 描述性统计
        if HAS_FLUENT_WIDGETS:
            self.descriptive_cb = CheckBox("描述性统计")
        else:
            self.descriptive_cb = QCheckBox("描述性统计")
        self.descriptive_cb.setChecked(self.config.enable_descriptive_stats)
        layout.addWidget(self.descriptive_cb)

        # 关联分析
        if HAS_FLUENT_WIDGETS:
            self.correlation_cb = CheckBox("关联分析")
        else:
            self.correlation_cb = QCheckBox("关联分析")
        self.correlation_cb.setChecked(self.config.enable_correlation_analysis)
        layout.addWidget(self.correlation_cb)

        # 异常值检测
        if HAS_FLUENT_WIDGETS:
            self.anomaly_cb = CheckBox("异常值检测")
        else:
            self.anomaly_cb = QCheckBox("异常值检测")
        self.anomaly_cb.setChecked(self.config.enable_anomaly_detection)
        layout.addWidget(self.anomaly_cb)

        # 时间序列分析
        if HAS_FLUENT_WIDGETS:
            self.timeseries_cb = CheckBox("时间序列分析")
        else:
            self.timeseries_cb = QCheckBox("时间序列分析")
        self.timeseries_cb.setChecked(self.config.enable_time_series_analysis)
        layout.addWidget(self.timeseries_cb)

        return group

    def _create_parameter_config(self):
        """创建参数配置组"""
        if HAS_FLUENT_WIDGETS:
            group = HeaderCardWidget()
            group.setTitle("参数配置")
        else:
            group = QGroupBox("参数配置")

        layout = QGridLayout(group)
        row = 0

        # 关联方法
        if HAS_FLUENT_WIDGETS:
            layout.addWidget(BodyLabel("关联方法:"), row, 0)
            self.correlation_method_combo = ComboBox()
        else:
            layout.addWidget(QLabel("关联方法:"), row, 0)
            self.correlation_method_combo = QComboBox()

        self.correlation_method_combo.addItems(["pearson", "spearman", "kendall"])
        self.correlation_method_combo.setCurrentText(self.config.default_correlation_method)
        layout.addWidget(self.correlation_method_combo, row, 1)
        row += 1

        # 异常值检测方法
        if HAS_FLUENT_WIDGETS:
            layout.addWidget(BodyLabel("异常值方法:"), row, 0)
            self.outlier_method_combo = ComboBox()
        else:
            layout.addWidget(QLabel("异常值方法:"), row, 0)
            self.outlier_method_combo = QComboBox()

        self.outlier_method_combo.addItems(["iqr", "zscore", "isolation_forest"])
        self.outlier_method_combo.setCurrentText(self.config.default_outlier_method)
        layout.addWidget(self.outlier_method_combo, row, 1)
        row += 1

        # 异常值阈值
        if HAS_FLUENT_WIDGETS:
            layout.addWidget(BodyLabel("异常值阈值:"), row, 0)
            self.outlier_threshold_spin = DoubleSpinBox()
        else:
            layout.addWidget(QLabel("异常值阈值:"), row, 0)
            self.outlier_threshold_spin = QDoubleSpinBox()

        self.outlier_threshold_spin.setRange(1.0, 5.0)
        self.outlier_threshold_spin.setSingleStep(0.1)
        self.outlier_threshold_spin.setValue(self.config.default_outlier_threshold)
        layout.addWidget(self.outlier_threshold_spin, row, 1)

        return group

    def _create_advanced_options(self):
        """创建高级选项组"""
        if HAS_FLUENT_WIDGETS:
            group = HeaderCardWidget()
            group.setTitle("高级选项")
        else:
            group = QGroupBox("高级选项")

        layout = QGridLayout(group)
        row = 0

        # 线程数
        if HAS_FLUENT_WIDGETS:
            layout.addWidget(BodyLabel("线程数:"), row, 0)
            self.threads_spin = SpinBox()
        else:
            layout.addWidget(QLabel("线程数:"), row, 0)
            self.threads_spin = QSpinBox()

        self.threads_spin.setRange(1, 8)
        self.threads_spin.setValue(4)
        layout.addWidget(self.threads_spin, row, 1)
        row += 1

        # 并行处理
        if HAS_FLUENT_WIDGETS:
            self.parallel_cb = CheckBox("启用并行处理")
        else:
            self.parallel_cb = QCheckBox("启用并行处理")
        self.parallel_cb.setChecked(True)
        layout.addWidget(self.parallel_cb, row, 0, 1, 2)

        return group

    def _setup_connections(self):
        """设置信号连接"""
        # 配置项改变时发出信号
        self.descriptive_cb.toggled.connect(self._emit_config_changed)
        self.correlation_cb.toggled.connect(self._emit_config_changed)
        self.anomaly_cb.toggled.connect(self._emit_config_changed)
        self.timeseries_cb.toggled.connect(self._emit_config_changed)

        if hasattr(self.correlation_method_combo, 'currentTextChanged'):
            self.correlation_method_combo.currentTextChanged.connect(self._emit_config_changed)
        if hasattr(self.outlier_method_combo, 'currentTextChanged'):
            self.outlier_method_combo.currentTextChanged.connect(self._emit_config_changed)
        if hasattr(self.outlier_threshold_spin, 'valueChanged'):
            self.outlier_threshold_spin.valueChanged.connect(self._emit_config_changed)

    def _emit_config_changed(self):
        """发出配置改变信号"""
        config = self.get_analysis_config()
        self.config_changed.emit(config)

    def get_analysis_config(self) -> dict[str, Any]:
        """获取分析配置"""
        config = {
            'enable_descriptive_stats': self.descriptive_cb.isChecked(),
            'enable_correlation_analysis': self.correlation_cb.isChecked(),
            'enable_anomaly_detection': self.anomaly_cb.isChecked(),
            'enable_time_series_analysis': self.timeseries_cb.isChecked(),
            'correlation_method': self.correlation_method_combo.currentText(),
            'outlier_method': self.outlier_method_combo.currentText(),
            'outlier_threshold': self.outlier_threshold_spin.value()
        }

        if hasattr(self, 'threads_spin'):
            config['n_threads'] = self.threads_spin.value()
        if hasattr(self, 'parallel_cb'):
            config['enable_parallel'] = self.parallel_cb.isChecked()

        return config

    def update_config(self, config: dict[str, Any]):
        """更新配置"""
        if 'enable_descriptive_stats' in config:
            self.descriptive_cb.setChecked(config['enable_descriptive_stats'])
        if 'enable_correlation_analysis' in config:
            self.correlation_cb.setChecked(config['enable_correlation_analysis'])
        if 'enable_anomaly_detection' in config:
            self.anomaly_cb.setChecked(config['enable_anomaly_detection'])
        if 'enable_time_series_analysis' in config:
            self.timeseries_cb.setChecked(config['enable_time_series_analysis'])


class AnalysisResultsWidget(QWidget, LoggerMixin):
    """分析结果展示组件"""

    def __init__(self, config: AnalysisPageConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self.current_result: AnalysisResult | None = None
        self.chart_renderer: ChartRenderer | None = None
        self._setup_ui()
        self._init_chart_renderer()

    def _setup_ui(self):
        """设置UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # 创建标签页
        if HAS_FLUENT_WIDGETS:
            self.tab_widget = Pivot()
        else:
            self.tab_widget = QTabWidget()

        layout.addWidget(self.tab_widget)

        # 初始状态显示
        self._show_empty_state()

    def _init_chart_renderer(self):
        """初始化图表渲染器"""
        try:
            chart_style = ChartStyle(
                theme="nature",
                color_palette="viridis",
                figure_size=(10, 8)
            )
            self.chart_renderer = ChartRenderer(chart_style)
        except Exception as e:
            self.logger.warning(f"图表渲染器初始化失败: {str(e)}")

    def _show_empty_state(self):
        """显示空状态"""
        self._clear_tabs()

        if HAS_FLUENT_WIDGETS:
            empty_widget = CardWidget()
            empty_layout = QVBoxLayout(empty_widget)

            empty_label = SubtitleLabel("尚未开始分析")
            empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty_layout.addWidget(empty_label)

            hint_label = CaptionLabel("请先上传数据并配置分析参数，然后点击\"开始分析\"")
            hint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty_layout.addWidget(hint_label)

        else:
            empty_widget = QWidget()
            empty_layout = QVBoxLayout(empty_widget)

            empty_label = QLabel("尚未开始分析")
            empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty_layout.addWidget(empty_label)

            hint_label = QLabel("请先上传数据并配置分析参数，然后点击\"开始分析\"")
            hint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty_layout.addWidget(hint_label)

        if HAS_FLUENT_WIDGETS:
            self.tab_widget.addItem("overview", "概览", None, lambda: empty_widget)
        else:
            self.tab_widget.addTab(empty_widget, "概览")

    def _clear_tabs(self):
        """清除所有标签页"""
        if HAS_FLUENT_WIDGETS:
            self.tab_widget.clear()
        else:
            self.tab_widget.clear()

    def display_results(self, result: AnalysisResult):
        """显示分析结果"""
        try:
            self.current_result = result
            self._clear_tabs()

            # 概览标签页
            self._create_overview_tab(result)

            # 描述性统计标签页
            if result.descriptive_stats:
                self._create_descriptive_stats_tab(result.descriptive_stats)

            # 关联分析标签页
            if result.correlation_matrix:
                self._create_correlation_tab(result.correlation_matrix)

            # 异常值检测标签页
            if result.anomaly_detection:
                self._create_anomaly_tab(result.anomaly_detection)

            # 时间序列分析标签页
            if result.time_series_analysis:
                self._create_time_series_tab(result.time_series_analysis)

            self.logger.info("分析结果显示完成")

        except Exception as e:
            self.logger.error(f"显示分析结果失败: {str(e)}")
            self._show_error(f"显示结果失败: {str(e)}")

    def _create_overview_tab(self, result: AnalysisResult):
        """创建概览标签页"""
        if HAS_FLUENT_WIDGETS:
            overview_widget = ScrollArea()
            content_widget = QWidget()
            overview_widget.setWidget(content_widget)
        else:
            overview_widget = QScrollArea()
            content_widget = QWidget()
            overview_widget.setWidget(content_widget)

        layout = QVBoxLayout(content_widget)
        layout.setSpacing(15)

        # 分析摘要
        summary_card = self._create_summary_card(result)
        layout.addWidget(summary_card)

        # 列信息
        column_card = self._create_column_info_card(result)
        layout.addWidget(column_card)

        layout.addStretch()

        if HAS_FLUENT_WIDGETS:
            self.tab_widget.addItem("overview", "概览", None, lambda: overview_widget)
        else:
            self.tab_widget.addTab(overview_widget, "概览")

    def _create_summary_card(self, result: AnalysisResult):
        """创建分析摘要卡片"""
        if HAS_FLUENT_WIDGETS:
            card = HeaderCardWidget()
            card.setTitle("分析摘要")
        else:
            card = QGroupBox("分析摘要")

        layout = QGridLayout(card)
        row = 0

        # 基本信息
        info_items = [
            ("分析时间", result.descriptive_stats.computed_at.strftime("%Y-%m-%d %H:%M:%S") if result.descriptive_stats else "未知"),
            ("分析列数", len(result.column_info.get('numeric_columns', [])) if result.column_info else 0),
            ("总列数", result.column_info.get('total_columns', 0) if result.column_info else 0)
        ]

        # 添加各种分析结果状态
        if result.descriptive_stats:
            info_items.append(("描述性统计", "✅ 已完成"))
        if result.correlation_matrix:
            info_items.append(("关联分析", "✅ 已完成"))
        if result.anomaly_detection:
            info_items.append(("异常值检测", "✅ 已完成"))
        if result.time_series_analysis:
            info_items.append(("时间序列分析", "✅ 已完成"))

        for label_text, value_text in info_items:
            if HAS_FLUENT_WIDGETS:
                label = BodyLabel(f"{label_text}:")
                value = StrongBodyLabel(str(value_text))
            else:
                label = QLabel(f"{label_text}:")
                value = QLabel(str(value_text))
                value.setStyleSheet("font-weight: bold;")

            layout.addWidget(label, row, 0)
            layout.addWidget(value, row, 1)
            row += 1

        return card

    def _create_column_info_card(self, result: AnalysisResult):
        """创建列信息卡片"""
        if HAS_FLUENT_WIDGETS:
            card = HeaderCardWidget()
            card.setTitle("列信息")
        else:
            card = QGroupBox("列信息")

        layout = QVBoxLayout(card)

        if result.column_info:
            # 数值列
            numeric_cols = result.column_info.get('numeric_columns', [])
            if numeric_cols:
                if HAS_FLUENT_WIDGETS:
                    label = BodyLabel(f"数值列 ({len(numeric_cols)}个):")
                    value = CaptionLabel(", ".join(numeric_cols[:10]) + ("..." if len(numeric_cols) > 10 else ""))
                else:
                    label = QLabel(f"数值列 ({len(numeric_cols)}个):")
                    value = QLabel(", ".join(numeric_cols[:10]) + ("..." if len(numeric_cols) > 10 else ""))

                layout.addWidget(label)
                layout.addWidget(value)

            # 排除的列
            excluded_cols = result.column_info.get('excluded_columns', [])
            if excluded_cols:
                if HAS_FLUENT_WIDGETS:
                    label = BodyLabel(f"排除的列 ({len(excluded_cols)}个):")
                    value = CaptionLabel(", ".join(excluded_cols[:10]) + ("..." if len(excluded_cols) > 10 else ""))
                else:
                    label = QLabel(f"排除的列 ({len(excluded_cols)}个):")
                    value = QLabel(", ".join(excluded_cols[:10]) + ("..." if len(excluded_cols) > 10 else ""))

                layout.addWidget(label)
                layout.addWidget(value)

        return card

    def _create_descriptive_stats_tab(self, descriptive_stats):
        """创建描述性统计标签页"""
        if HAS_FLUENT_WIDGETS:
            stats_widget = ScrollArea()
            content_widget = QWidget()
            stats_widget.setWidget(content_widget)
        else:
            stats_widget = QScrollArea()
            content_widget = QWidget()
            stats_widget.setWidget(content_widget)

        layout = QVBoxLayout(content_widget)

        # 为每个数值列创建统计卡片
        for column, stats in descriptive_stats.statistics.items():
            card = self._create_column_stats_card(column, stats)
            layout.addWidget(card)

        layout.addStretch()

        if HAS_FLUENT_WIDGETS:
            self.tab_widget.addItem("descriptive", "描述性统计", None, lambda: stats_widget)
        else:
            self.tab_widget.addTab(stats_widget, "描述性统计")

    def _create_column_stats_card(self, column: str, stats: dict[str, Any]):
        """创建列统计卡片"""
        if HAS_FLUENT_WIDGETS:
            card = HeaderCardWidget()
            card.setTitle(f"列: {column}")
        else:
            card = QGroupBox(f"列: {column}")

        layout = QGridLayout(card)

        # 主要统计指标
        main_stats = [
            ("数量", stats.get('count', 0)),
            ("均值", f"{stats.get('mean', 0):.4f}"),
            ("标准差", f"{stats.get('std', 0):.4f}"),
            ("最小值", f"{stats.get('min', 0):.4f}"),
            ("25%分位", f"{stats.get('percentile_25', 0):.4f}"),
            ("中位数", f"{stats.get('median', 0):.4f}"),
            ("75%分位", f"{stats.get('percentile_75', 0):.4f}"),
            ("最大值", f"{stats.get('max', 0):.4f}")
        ]

        # 可选统计指标
        if stats.get('skewness') is not None:
            main_stats.append(("偏度", f"{stats.get('skewness', 0):.4f}"))
        if stats.get('kurtosis') is not None:
            main_stats.append(("峰度", f"{stats.get('kurtosis', 0):.4f}"))

        # 排列统计指标
        row = 0
        col = 0
        for label_text, value_text in main_stats:
            if HAS_FLUENT_WIDGETS:
                label = CaptionLabel(f"{label_text}:")
                value = BodyLabel(str(value_text))
            else:
                label = QLabel(f"{label_text}:")
                value = QLabel(str(value_text))

            layout.addWidget(label, row, col * 2)
            layout.addWidget(value, row, col * 2 + 1)

            col += 1
            if col >= 2:  # 每行2列
                col = 0
                row += 1

        return card

    def _create_correlation_tab(self, correlation_matrix):
        """创建关联分析标签页"""
        if HAS_FLUENT_WIDGETS:
            corr_widget = ScrollArea()
            content_widget = QWidget()
            corr_widget.setWidget(content_widget)
        else:
            corr_widget = QScrollArea()
            content_widget = QWidget()
            corr_widget.setWidget(content_widget)

        layout = QVBoxLayout(content_widget)

        # 关联分析摘要
        summary_card = self._create_correlation_summary_card(correlation_matrix)
        layout.addWidget(summary_card)

        # 高关联性对
        high_corr_card = self._create_high_correlations_card(correlation_matrix)
        layout.addWidget(high_corr_card)

        # 图表展示（如果支持）
        if self.chart_renderer:
            chart_card = self._create_correlation_chart_card(correlation_matrix)
            if chart_card:
                layout.addWidget(chart_card)

        layout.addStretch()

        if HAS_FLUENT_WIDGETS:
            self.tab_widget.addItem("correlation", "关联分析", None, lambda: corr_widget)
        else:
            self.tab_widget.addTab(corr_widget, "关联分析")

    def _create_correlation_summary_card(self, correlation_matrix):
        """创建关联分析摘要卡片"""
        if HAS_FLUENT_WIDGETS:
            card = HeaderCardWidget()
            card.setTitle("关联分析摘要")
        else:
            card = QGroupBox("关联分析摘要")

        layout = QGridLayout(card)

        summary = correlation_matrix.get_summary()

        summary_items = [
            ("方法", summary.get('method', 'unknown')),
            ("矩阵大小", f"{summary.get('matrix_size', 0)} x {summary.get('matrix_size', 0)}"),
            ("变量对数", summary.get('total_pairs', 0)),
            ("最大关联系数", f"{summary.get('max_correlation', 0):.4f}"),
            ("平均关联系数", f"{summary.get('mean_correlation', 0):.4f}"),
            ("高关联对数", summary.get('high_correlation_count', 0))
        ]

        row = 0
        for label_text, value_text in summary_items:
            if HAS_FLUENT_WIDGETS:
                label = BodyLabel(f"{label_text}:")
                value = StrongBodyLabel(str(value_text))
            else:
                label = QLabel(f"{label_text}:")
                value = QLabel(str(value_text))
                value.setStyleSheet("font-weight: bold;")

            layout.addWidget(label, row, 0)
            layout.addWidget(value, row, 1)
            row += 1

        return card

    def _create_high_correlations_card(self, correlation_matrix):
        """创建高关联性对卡片"""
        if HAS_FLUENT_WIDGETS:
            card = HeaderCardWidget()
            card.setTitle("高关联性变量对")
        else:
            card = QGroupBox("高关联性变量对")

        layout = QVBoxLayout(card)

        high_corrs = correlation_matrix.get_high_correlations()

        if not high_corrs:
            if HAS_FLUENT_WIDGETS:
                no_data_label = CaptionLabel("没有发现高关联性变量对")
            else:
                no_data_label = QLabel("没有发现高关联性变量对")
            layout.addWidget(no_data_label)
        else:
            # 显示前10个高关联对
            for _, corr_info in enumerate(high_corrs[:10]):
                if HAS_FLUENT_WIDGETS:
                    item_text = f"{corr_info['column1']} ↔ {corr_info['column2']}: {corr_info['correlation']:.4f}"
                    item_label = BodyLabel(item_text)
                else:
                    item_text = f"{corr_info['column1']} ↔ {corr_info['column2']}: {corr_info['correlation']:.4f}"
                    item_label = QLabel(item_text)

                layout.addWidget(item_label)

            if len(high_corrs) > 10:
                if HAS_FLUENT_WIDGETS:
                    more_label = CaptionLabel(f"... 还有 {len(high_corrs) - 10} 个高关联对")
                else:
                    more_label = QLabel(f"... 还有 {len(high_corrs) - 10} 个高关联对")
                layout.addWidget(more_label)

        return card

    def _create_correlation_chart_card(self, correlation_matrix):
        """创建关联分析图表卡片"""
        try:
            if not self.chart_renderer:
                return None

            if HAS_FLUENT_WIDGETS:
                card = HeaderCardWidget()
                card.setTitle("关联热力图")
            else:
                card = QGroupBox("关联热力图")

            layout = QVBoxLayout(card)

            # 生成热力图
            chart_data = {
                'correlation_matrix': correlation_matrix.matrix,
                'columns': correlation_matrix.columns
            }

            chart_bytes = self.chart_renderer.create_chart(
                ChartType.CORRELATION_HEATMAP,
                chart_data,
                title="变量关联热力图",
                export_format=ChartFormat.PNG
            )

            if chart_bytes:
                # 这里可以显示图表，但需要更复杂的图像显示组件
                if HAS_FLUENT_WIDGETS:
                    chart_label = BodyLabel("✅ 热力图已生成")
                else:
                    chart_label = QLabel("✅ 热力图已生成")
                layout.addWidget(chart_label)

            return card

        except Exception as e:
            self.logger.error(f"创建关联分析图表失败: {str(e)}")
            return None

    def _create_anomaly_tab(self, anomaly_detection):
        """创建异常值检测标签页"""
        if HAS_FLUENT_WIDGETS:
            anomaly_widget = ScrollArea()
            content_widget = QWidget()
            anomaly_widget.setWidget(content_widget)
        else:
            anomaly_widget = QScrollArea()
            content_widget = QWidget()
            anomaly_widget.setWidget(content_widget)

        layout = QVBoxLayout(content_widget)

        # 异常值检测摘要
        summary_card = self._create_anomaly_summary_card(anomaly_detection)
        layout.addWidget(summary_card)

        # 各列异常值详情
        for column, anomaly_info in anomaly_detection.anomalies.items():
            if anomaly_info.get('outlier_count', 0) > 0:
                card = self._create_anomaly_detail_card(column, anomaly_info)
                layout.addWidget(card)

        layout.addStretch()

        if HAS_FLUENT_WIDGETS:
            self.tab_widget.addItem("anomaly", "异常值检测", None, lambda: anomaly_widget)
        else:
            self.tab_widget.addTab(anomaly_widget, "异常值检测")

    def _create_anomaly_summary_card(self, anomaly_detection):
        """创建异常值检测摘要卡片"""
        if HAS_FLUENT_WIDGETS:
            card = HeaderCardWidget()
            card.setTitle("异常值检测摘要")
        else:
            card = QGroupBox("异常值检测摘要")

        layout = QGridLayout(card)

        total_anomalies = anomaly_detection.get_total_anomalies()
        anomaly_columns = anomaly_detection.get_anomaly_columns()

        summary_items = [
            ("检测方法", anomaly_detection.method),
            ("阈值", str(anomaly_detection.threshold)),
            ("总异常值数", total_anomalies),
            ("异常值列数", len(anomaly_columns)),
            ("总分析列数", len(anomaly_detection.anomalies))
        ]

        row = 0
        for label_text, value_text in summary_items:
            if HAS_FLUENT_WIDGETS:
                label = BodyLabel(f"{label_text}:")
                value = StrongBodyLabel(str(value_text))
            else:
                label = QLabel(f"{label_text}:")
                value = QLabel(str(value_text))
                value.setStyleSheet("font-weight: bold;")

            layout.addWidget(label, row, 0)
            layout.addWidget(value, row, 1)
            row += 1

        return card

    def _create_anomaly_detail_card(self, column: str, anomaly_info: dict[str, Any]):
        """创建异常值详情卡片"""
        if HAS_FLUENT_WIDGETS:
            card = HeaderCardWidget()
            card.setTitle(f"列: {column}")
        else:
            card = QGroupBox(f"列: {column}")

        layout = QVBoxLayout(card)

        # 异常值统计
        outlier_count = anomaly_info.get('outlier_count', 0)
        outlier_percentage = anomaly_info.get('outlier_percentage', 0)

        if HAS_FLUENT_WIDGETS:
            stats_label = BodyLabel(f"异常值数量: {outlier_count} ({outlier_percentage:.2f}%)")
        else:
            stats_label = QLabel(f"异常值数量: {outlier_count} ({outlier_percentage:.2f}%)")
        layout.addWidget(stats_label)

        # 异常值样本（显示前5个）
        outlier_values = anomaly_info.get('outlier_values', [])
        if outlier_values:
            sample_values = outlier_values[:5]
            sample_text = ", ".join(f"{v:.4f}" for v in sample_values)
            if len(outlier_values) > 5:
                sample_text += f" ... (共{len(outlier_values)}个)"

            if HAS_FLUENT_WIDGETS:
                sample_label = CaptionLabel(f"异常值样本: {sample_text}")
            else:
                sample_label = QLabel(f"异常值样本: {sample_text}")
            layout.addWidget(sample_label)

        return card

    def _create_time_series_tab(self, time_series_analysis):
        """创建时间序列分析标签页"""
        if HAS_FLUENT_WIDGETS:
            ts_widget = ScrollArea()
            content_widget = QWidget()
            ts_widget.setWidget(content_widget)
        else:
            ts_widget = QScrollArea()
            content_widget = QWidget()
            ts_widget.setWidget(content_widget)

        layout = QVBoxLayout(content_widget)

        # 时间序列分析摘要
        if HAS_FLUENT_WIDGETS:
            summary_card = HeaderCardWidget()
            summary_card.setTitle("时间序列分析")
        else:
            summary_card = QGroupBox("时间序列分析")

        summary_layout = QVBoxLayout(summary_card)

        if HAS_FLUENT_WIDGETS:
            time_col_label = BodyLabel(f"时间列: {time_series_analysis.time_column}")
        else:
            time_col_label = QLabel(f"时间列: {time_series_analysis.time_column}")
        summary_layout.addWidget(time_col_label)

        # 平稳性检验结果
        if time_series_analysis.stationarity_tests:
            for column, tests in time_series_analysis.stationarity_tests.items():
                col_card = self._create_stationarity_card(column, tests)
                layout.addWidget(col_card)

        layout.addWidget(summary_card)
        layout.addStretch()

        if HAS_FLUENT_WIDGETS:
            self.tab_widget.addItem("timeseries", "时间序列", None, lambda: ts_widget)
        else:
            self.tab_widget.addTab(ts_widget, "时间序列")

    def _create_stationarity_card(self, column: str, tests: dict[str, Any]):
        """创建平稳性检验卡片"""
        if HAS_FLUENT_WIDGETS:
            card = HeaderCardWidget()
            card.setTitle(f"平稳性检验: {column}")
        else:
            card = QGroupBox(f"平稳性检验: {column}")

        layout = QVBoxLayout(card)

        for test_name, test_result in tests.items():
            if test_result:
                is_stationary = test_result.get('is_stationary', False)
                p_value = test_result.get('p_value', 0)
                statistic = test_result.get('statistic', 0)

                status_text = "平稳" if is_stationary else "非平稳"

                if HAS_FLUENT_WIDGETS:
                    test_label = BodyLabel(f"{test_name.upper()}检验: {status_text} (p={p_value:.4f}, 统计量={statistic:.4f})")
                else:
                    test_label = QLabel(f"{test_name.upper()}检验: {status_text} (p={p_value:.4f}, 统计量={statistic:.4f})")
                layout.addWidget(test_label)

        return card

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

    def clear_results(self):
        """清除结果"""
        self.current_result = None
        self._show_empty_state()


class AnalysisPage(QWidget, LoggerMixin):
    """数据分析页面"""

    # 信号
    analysis_started = pyqtSignal()
    analysis_completed = pyqtSignal(object)  # AnalysisResult
    analysis_failed = pyqtSignal(str)

    def __init__(self, config: AnalysisPageConfig | None = None, parent=None):
        super().__init__(parent)
        self.config = config or AnalysisPageConfig()

        # 核心组件
        self.analysis_engine: AnalysisEngine | None = None
        self.analysis_worker: AnalysisWorker | None = None
        self.history_manager: HistoryManager = get_history_manager()
        self.export_manager: ExportManager = ExportManager()
        self.current_data: Any | None = None
        self.current_file_path: str | None = None
        self.current_time_column: str | None = None
        self.current_result: AnalysisResult | None = None
        self.current_history_record: AnalysisHistoryRecord | None = None

        self._setup_ui()
        self._setup_connections()
        self._init_analysis_engine()

        self.logger.info("分析页面初始化完成")

    def _setup_ui(self):
        """设置UI"""
        layout = QHBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)

        # 创建分割器
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # 左侧：配置面板
        left_panel = self._create_left_panel()
        splitter.addWidget(left_panel)

        # 右侧：结果展示
        right_panel = self._create_right_panel()
        splitter.addWidget(right_panel)

        # 设置分割比例 (1:2)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

        layout.addWidget(splitter)

    def _create_left_panel(self):
        """创建左侧配置面板"""
        if HAS_FLUENT_WIDGETS:
            panel = CardWidget()
        else:
            panel = QFrame()
            panel.setFrameStyle(QFrame.Shape.StyledPanel)

        layout = QVBoxLayout(panel)
        layout.setSpacing(15)

        # 页面标题
        if HAS_FLUENT_WIDGETS:
            title_label = TitleLabel("数据分析")
        else:
            title_label = QLabel("数据分析")
            title_label.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(title_label)

        # 数据状态
        self.data_status_widget = self._create_data_status_widget()
        layout.addWidget(self.data_status_widget)

        # 分析配置
        self.config_widget = AnalysisConfigWidget(self.config)
        layout.addWidget(self.config_widget)

        # 控制按钮
        self.control_widget = self._create_control_widget()
        layout.addWidget(self.control_widget)

        # 进度显示
        self.progress_widget = self._create_progress_widget()
        layout.addWidget(self.progress_widget)

        layout.addStretch()

        return panel

    def _create_right_panel(self):
        """创建右侧结果面板"""
        # 结果展示组件
        self.results_widget = AnalysisResultsWidget(self.config)
        return self.results_widget

    def _create_data_status_widget(self):
        """创建数据状态组件"""
        if HAS_FLUENT_WIDGETS:
            widget = SimpleCardWidget()
        else:
            widget = QFrame()
            widget.setFrameStyle(QFrame.Shape.Box)

        layout = QVBoxLayout(widget)

        if HAS_FLUENT_WIDGETS:
            self.data_status_label = BodyLabel("尚未加载数据")
        else:
            self.data_status_label = QLabel("尚未加载数据")
        layout.addWidget(self.data_status_label)

        if HAS_FLUENT_WIDGETS:
            self.data_info_label = CaptionLabel("")
        else:
            self.data_info_label = QLabel("")
            self.data_info_label.setStyleSheet("color: #666666;")
        layout.addWidget(self.data_info_label)

        return widget

    def _create_control_widget(self):
        """创建控制按钮组件"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)

        # 开始分析按钮
        if HAS_FLUENT_WIDGETS:
            self.start_btn = PrimaryPushButton("开始分析")
            safe_set_icon(self.start_btn, FluentIcon.PLAY)
        else:
            self.start_btn = QPushButton("开始分析")
        self.start_btn.setEnabled(False)
        layout.addWidget(self.start_btn)

        # 停止分析按钮
        if HAS_FLUENT_WIDGETS:
            self.stop_btn = PushButton("停止分析")
        else:
            self.stop_btn = QPushButton("停止分析")
        self.stop_btn.setEnabled(False)
        layout.addWidget(self.stop_btn)

        # 清除结果按钮
        if HAS_FLUENT_WIDGETS:
            self.clear_btn = PushButton("清除结果")
        else:
            self.clear_btn = QPushButton("清除结果")
        self.clear_btn.setEnabled(False)
        layout.addWidget(self.clear_btn)

        # 导出按钮
        if HAS_FLUENT_WIDGETS:
            self.export_btn = PushButton("导出报告")
            safe_set_icon(self.export_btn, FluentIcon.DOWNLOAD)
        else:
            self.export_btn = QPushButton("导出报告")
        self.export_btn.setEnabled(False)
        layout.addWidget(self.export_btn)

        return widget

    def _create_progress_widget(self):
        """创建进度显示组件"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # 进度条
        if HAS_FLUENT_WIDGETS:
            self.progress_bar = ProgressBar()
        else:
            self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # 状态标签
        if HAS_FLUENT_WIDGETS:
            self.progress_label = CaptionLabel("")
        else:
            self.progress_label = QLabel("")
            self.progress_label.setStyleSheet("color: #666666;")
        self.progress_label.setVisible(False)
        layout.addWidget(self.progress_label)

        return widget

    def _setup_connections(self):
        """设置信号连接"""
        # 按钮连接
        self.start_btn.clicked.connect(self._start_analysis)
        self.stop_btn.clicked.connect(self._stop_analysis)
        self.clear_btn.clicked.connect(self._clear_results)
        self.export_btn.clicked.connect(self._export_report)

        # 配置改变
        self.config_widget.config_changed.connect(self._on_config_changed)

    def _init_analysis_engine(self):
        """初始化分析引擎"""
        try:
            analysis_config = AnalysisConfig()
            self.analysis_engine = AnalysisEngine(analysis_config)
            self.logger.info("分析引擎初始化完成")
        except Exception as e:
            self.logger.error(f"分析引擎初始化失败: {str(e)}")

    def _on_config_changed(self, config: dict[str, Any]):
        """配置改变处理"""
        if self.analysis_engine:
            # 更新分析引擎配置
            try:
                new_config = AnalysisConfig(
                    correlation_method=config.get('correlation_method', 'pearson'),
                    outlier_method=config.get('outlier_method', 'iqr'),
                    outlier_threshold=config.get('outlier_threshold', 3.0),
                    n_threads=config.get('n_threads', 4),
                    enable_parallel=config.get('enable_parallel', True)
                )
                self.analysis_engine.config = new_config
                self.logger.debug("分析配置已更新")
            except Exception as e:
                self.logger.error(f"更新分析配置失败: {str(e)}")

    def _start_analysis(self):
        """开始分析"""
        if not self.current_data:
            self._show_error("请先加载数据")
            return

        if not self.analysis_engine:
            self._show_error("分析引擎未初始化")
            return

        if not self.current_file_path:
            self._show_error("文件路径未设置")
            return

        try:
            # 获取分析配置
            analysis_config = self.config_widget.get_analysis_config()
            analysis_config['time_column'] = self.current_time_column

            # 创建分析引擎配置
            engine_config = AnalysisConfig(
                correlation_method=analysis_config.get('correlation_method', 'pearson'),
                outlier_method=analysis_config.get('outlier_method', 'iqr'),
                outlier_threshold=analysis_config.get('outlier_threshold', 3.0),
                n_threads=analysis_config.get('n_threads', 4),
                enable_parallel=analysis_config.get('enable_parallel', True)
            )

            # 检查是否已有相同的分析
            existing_record = self.history_manager.find_existing_analysis(
                self.current_file_path,
                engine_config,
                "comprehensive",  # 添加分析类型
                self.current_time_column
            )

            if existing_record:
                reply = QMessageBox.question(
                    self, "发现已有分析",
                    f"检测到相同配置的分析记录（{existing_record.created_at.strftime('%Y-%m-%d %H:%M')}）。\n"
                    f"是否加载已有结果？",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes
                )

                if reply == QMessageBox.StandardButton.Yes:
                    self._load_existing_result(existing_record)
                    return

            # 创建新的历史记录
            self.current_history_record = self.history_manager.create_record_from_data(
                self.current_file_path,
                engine_config,
                analysis_type="comprehensive",
                time_column=self.current_time_column
            )

            # 保存记录到数据库
            self.current_history_record = self.history_manager.save_record(self.current_history_record)

            # 标记分析开始
            self.current_history_record = self.history_manager.start_analysis(self.current_history_record)

            # 创建工作线程
            self.analysis_worker = AnalysisWorker(
                self.analysis_engine,
                self.current_data,
                analysis_config
            )

            # 连接信号
            self.analysis_worker.progress_updated.connect(self._on_progress_updated)
            self.analysis_worker.analysis_completed.connect(self._on_analysis_completed)
            self.analysis_worker.analysis_failed.connect(self._on_analysis_failed)
            self.analysis_worker.status_changed.connect(self._on_status_changed)

            # 更新UI状态
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.clear_btn.setEnabled(False)

            # 显示进度
            self._show_progress()

            # 记录开始时间
            self.analysis_start_time = datetime.now()

            # 开始分析
            self.analysis_worker.start()
            self.analysis_started.emit()

            self.logger.info(f"开始数据分析，历史记录ID: {self.current_history_record.analysis_id}")

        except Exception as e:
            self.logger.error(f"启动分析失败: {str(e)}")
            self._show_error(f"启动分析失败: {str(e)}")

            # 记录失败
            if self.current_history_record:
                self.history_manager.fail_analysis(
                    self.current_history_record,
                    str(e)
                )

    def _load_existing_result(self, existing_record: AnalysisHistoryRecord):
        """加载已有结果"""
        try:
            # 加载分析结果
            result = self.history_manager.load_analysis_result(existing_record)
            if result:
                self.current_result = result
                self.current_history_record = existing_record

                # 显示结果
                self.results_widget.display_results(result)
                self.clear_btn.setEnabled(True)

                # 更新状态
                self.data_status_label.setText("✅ 数据已加载（来自历史记录）")
                self.data_info_label.setText(
                    f"历史分析记录: {existing_record.created_at.strftime('%Y-%m-%d %H:%M')} | "
                    f"执行时间: {existing_record.get_duration_text()}"
                )

                self.logger.info(f"已加载历史分析结果: {existing_record.analysis_id}")
                self._show_info("已加载历史分析结果")
            else:
                self._show_error("无法加载历史分析结果")

        except Exception as e:
            self.logger.error(f"加载历史结果失败: {str(e)}")
            self._show_error(f"加载历史结果失败: {str(e)}")

    def _stop_analysis(self):
        """停止分析"""
        if self.analysis_worker and self.analysis_worker.isRunning():
            self.analysis_worker.cancel()
            self.analysis_worker.wait()

            # 记录取消状态
            if self.current_history_record:
                self.history_manager.cancel_analysis(self.current_history_record)

            self._hide_progress()
            self._reset_control_buttons()

            self.logger.info("分析已停止")

    def _clear_results(self):
        """清除结果"""
        self.current_result = None
        self.results_widget.clear_results()
        self.clear_btn.setEnabled(False)
        self.logger.info("分析结果已清除")

    def _on_progress_updated(self, progress: int, message: str):
        """进度更新处理"""
        self.progress_bar.setValue(progress)
        self.progress_label.setText(message)

    def _on_analysis_completed(self, result: AnalysisResult):
        """分析完成处理"""
        try:
            self.current_result = result
            self.results_widget.display_results(result)

            # 计算执行时间
            if hasattr(self, 'analysis_start_time'):
                execution_time = datetime.now() - self.analysis_start_time
                execution_time_ms = int(execution_time.total_seconds() * 1000)
            else:
                execution_time_ms = 0

            # 保存历史记录
            if self.current_history_record:
                self.current_history_record = self.history_manager.complete_analysis(
                    self.current_history_record,
                    result,
                    execution_time_ms
                )
                self.logger.info(f"分析历史记录已保存: {self.current_history_record.analysis_id}")

            self._hide_progress()
            self._reset_control_buttons()
            self.clear_btn.setEnabled(True)

            self.analysis_completed.emit(result)
            self.logger.info("数据分析完成")

        except Exception as e:
            self.logger.error(f"处理分析完成失败: {str(e)}")
            self._show_error(f"保存分析结果失败: {str(e)}")

            # 记录失败
            if self.current_history_record:
                self.history_manager.fail_analysis(
                    self.current_history_record,
                    f"保存结果失败: {str(e)}"
                )

    def _on_analysis_failed(self, error_message: str):
        """分析失败处理"""
        try:
            # 计算执行时间
            if hasattr(self, 'analysis_start_time'):
                execution_time = datetime.now() - self.analysis_start_time
                execution_time_ms = int(execution_time.total_seconds() * 1000)
            else:
                execution_time_ms = 0

            # 记录失败
            if self.current_history_record:
                self.history_manager.fail_analysis(
                    self.current_history_record,
                    error_message,
                    execution_time_ms
                )

            self._hide_progress()
            self._reset_control_buttons()
            self._show_error(f"分析失败: {error_message}")
            self.analysis_failed.emit(error_message)

        except Exception as e:
            self.logger.error(f"处理分析失败失败: {str(e)}")
            self._hide_progress()
            self._reset_control_buttons()
            self._show_error(f"分析失败: {error_message}")
            self.analysis_failed.emit(error_message)

    def _on_status_changed(self, status: str):
        """状态改变处理"""
        self.logger.debug(f"分析状态: {status}")

    def _show_progress(self):
        """显示进度"""
        self.progress_bar.setVisible(True)
        self.progress_label.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_label.setText("准备中...")

    def _hide_progress(self):
        """隐藏进度"""
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)

    def _reset_control_buttons(self):
        """重置控制按钮"""
        self.start_btn.setEnabled(self.current_data is not None)
        self.stop_btn.setEnabled(False)
        # 只有当有分析结果时才启用导出按钮
        self.export_btn.setEnabled(self.current_result is not None)

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

    def load_data(self, data: Any, file_path: str | None = None, time_column: str | None = None):
        """加载数据"""
        try:
            self.current_data = data
            self.current_file_path = file_path
            self.current_time_column = time_column

            # 更新数据状态显示
            if HAS_PANDAS and hasattr(data, 'shape'):
                rows, cols = data.shape
                self.data_status_label.setText("✅ 数据已加载")
                self.data_info_label.setText(f"数据形状: {rows} 行 x {cols} 列")
            else:
                self.data_status_label.setText("✅ 数据已加载")
                self.data_info_label.setText("数据信息不可用")

            # 启用开始分析按钮
            self.start_btn.setEnabled(True)

            # 如果有时间列，启用时间序列分析
            if time_column:
                # 这里可以自动启用时间序列分析选项
                pass

            self.logger.info(f"数据加载完成, 文件路径: {file_path}, 时间列: {time_column}")

        except Exception as e:
            self.logger.error(f"加载数据失败: {str(e)}")
            self._show_error(f"加载数据失败: {str(e)}")

    def get_current_result(self) -> AnalysisResult | None:
        """获取当前分析结果"""
        return self.current_result

    def load_from_manager(self, file_path: str) -> bool:
        """
        从分析结果管理器加载分析结果

        Args:
            file_path: 文件路径，用于从管理器中检索结果

        Returns:
            bool: 加载是否成功
        """
        try:
            # 获取分析结果管理器
            result_manager = get_analysis_result_manager()

            # 尝试从管理器获取分析结果
            analysis_result = result_manager.get_result(file_path)
            file_info = result_manager.get_file_info(file_path)

            if analysis_result is None:
                # 没有找到分析结果，显示提示信息
                self.logger.info(f"未找到文件的分析结果: {file_path}")
                self._show_info("未找到分析结果，请重新进行分析")

                # 清除当前结果显示
                self._clear_results()

                # 如果有文件信息，尝试加载数据以便重新分析
                if file_info and hasattr(file_info, 'data') and file_info.data is not None:
                    self.load_data(file_info.data, file_path)
                    self.logger.info(f"已加载数据，可以重新分析: {file_path}")
                else:
                    # 更新数据状态标签显示无数据
                    self.data_status_label.setText("无数据")
                    self.start_btn.setEnabled(False)
                    self.logger.warning(f"无法加载数据进行重新分析: {file_path}")

                return False

            # 成功获取到分析结果
            self.logger.info(f"从管理器加载分析结果成功: {file_path}")

            # 保存当前结果和相关信息
            self.current_result = analysis_result
            self.current_file_path = file_path

            # 如果有文件信息，也加载相关数据
            if file_info:
                if hasattr(file_info, 'data') and file_info.data is not None:
                    self.current_data = file_info.data
                    self.logger.debug("已加载关联的数据")

                # 更新数据状态标签
                file_name = getattr(file_info, 'file_name', 'unknown')
                data_shape = getattr(analysis_result, 'data_shape', 'unknown')
                self.data_status_label.setText(f"已加载: {file_name} ({data_shape})")
            else:
                # 没有文件信息，从分析结果中获取基本信息
                data_shape = getattr(analysis_result, 'data_shape', 'unknown')
                self.data_status_label.setText(f"已加载分析结果 ({data_shape})")

            # 显示分析结果
            self.results_widget.display_results(analysis_result)

            # 启用开始分析按钮（允许重新分析）
            self.start_btn.setEnabled(True)
            self.start_btn.setText("重新分析")

            # 显示成功信息
            result_info = result_manager.get_result_info(file_path)
            if result_info:
                age_minutes = result_info.get('age_minutes', 0)
                if age_minutes < 60:
                    age_text = f"{int(age_minutes)}分钟前"
                else:
                    age_text = f"{int(age_minutes/60)}小时前"
                self._show_info(f"已加载分析结果 (生成于{age_text})")
            else:
                self._show_info("已加载分析结果")

            return True

        except Exception as e:
            self.logger.error(f"从管理器加载分析结果失败: {str(e)}")
            self._show_error(f"加载分析结果失败: {str(e)}")

            # 清除当前结果显示
            self._clear_results()
            self.data_status_label.setText("加载失败")
            self.start_btn.setEnabled(False)

            return False

    def apply_responsive_layout(self, layout_mode: str):
        """应用响应式布局"""
        if layout_mode == 'mobile':
            # 移动端布局：可以考虑隐藏某些高级选项
            self.setContentsMargins(10, 10, 10, 10)
        elif layout_mode == 'tablet':
            # 平板布局
            self.setContentsMargins(15, 15, 15, 15)
        else:  # desktop
            # 桌面布局
            self.setContentsMargins(20, 20, 20, 20)

        self.logger.debug(f"应用响应式布局: {layout_mode}")

    def _export_report(self):
        """导出分析报告"""
        if not self.current_result or not self.current_data:
            self._show_error("没有可导出的分析结果")
            return

        try:
            from PyQt6.QtWidgets import QFileDialog, QMessageBox

            # 获取保存路径
            suggested_name = f"分析报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "保存分析报告",
                suggested_name,
                "PDF文件 (*.pdf);;所有文件 (*)"
            )

            if not file_path:
                return

            # 确保文件有正确的扩展名
            if not file_path.endswith('.pdf'):
                file_path += '.pdf'

            # 显示进度
            self._show_progress("正在生成PDF报告...")

            # 创建文件信息对象
            file_info = self._create_file_info_for_export()

            # 执行导出
            success = self.export_manager.export_pdf_only(
                self.current_result,
                file_info,
                file_path
            )

            self._hide_progress()

            if success:
                self._show_info(f"报告导出成功：{file_path}")

                # 询问是否打开文件
                reply = QMessageBox.question(
                    self, "导出成功",
                    f"报告已成功导出到：\n{file_path}\n\n是否现在打开文件？",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes
                )

                if reply == QMessageBox.StandardButton.Yes:
                    try:
                        import os
                        import platform
                        import subprocess

                        # 跨平台打开文件
                        if platform.system() == 'Darwin':  # macOS
                            subprocess.call(['open', file_path])
                        elif platform.system() == 'Windows':  # Windows
                            os.startfile(file_path)
                        else:  # Linux
                            subprocess.call(['xdg-open', file_path])
                    except Exception as e:
                        self.logger.warning(f"无法打开文件: {e}")
            else:
                self._show_error("报告导出失败，请检查日志获取详细信息")

        except Exception as e:
            self._hide_progress()
            self.logger.error(f"导出报告失败: {e}")
            self._show_error(f"导出报告失败: {str(e)}")

    def _create_file_info_for_export(self):
        """为导出创建文件信息对象"""
        from ..models.file_info import DataQualityInfo, FileType

        # 获取数据的基本信息
        row_count = len(self.current_data) if hasattr(self.current_data, '__len__') else 0
        column_count = len(self.current_data.columns) if hasattr(self.current_data, 'columns') else 0

        # 计算文件大小（估算）
        size_bytes = (row_count * column_count * 8) if row_count and column_count else 1000

        # 创建数据质量信息
        data_quality = DataQualityInfo(
            total_rows=row_count,
            total_columns=column_count,
            missing_values_count=0,  # 可以后续计算实际值
            duplicate_rows_count=0,  # 可以后续计算实际值
            memory_usage_mb=size_bytes / (1024 * 1024)
        )

        # 确定文件类型
        file_type = FileType.CSV
        if self.current_file_path:
            if self.current_file_path.endswith('.parquet'):
                file_type = FileType.PARQUET
            elif self.current_file_path.endswith(('.xlsx', '.xls')):
                file_type = FileType.EXCEL
            elif self.current_file_path.endswith('.json'):
                file_type = FileType.JSON

        return FileInfo(
            file_path=self.current_file_path or "/tmp/unknown.csv",
            file_name=self.current_file_path.split('/')[-1] if self.current_file_path else "unknown.csv",
            file_type=file_type,
            file_size_bytes=size_bytes,
            file_hash="d41d8cd98f00b204e9800998ecf8427e",  # 默认哈希值
            modified_at=self.current_history_record.created_at if self.current_history_record else datetime.now(),
            data_quality=data_quality
        )


def create_analysis_page(config: AnalysisPageConfig | None = None) -> AnalysisPage:
    """创建分析页面的工厂函数"""
    try:
        return AnalysisPage(config)
    except Exception as e:
        print(f"创建分析页面失败: {str(e)}")
        raise
