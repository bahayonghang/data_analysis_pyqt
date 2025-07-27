"""
应用程序集成器
将所有页面集成到主窗口，实现完整的数据分析应用
"""

from typing import Any

try:
    from PyQt6.QtCore import QObject, Qt, pyqtSignal
    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False
    class QObject:
        pass
    def pyqtSignal(*args):
        return lambda *a, **k: None

try:
    from qfluentwidgets import FluentIcon, NavigationItemPosition
    HAS_FLUENT_WIDGETS = True
except ImportError:
    HAS_FLUENT_WIDGETS = False
    class FluentIcon:
        HOME = "home"
        FOLDER = "folder"
        CHART = "chart"
        HISTORY = "history"
        SETTING = "setting"
    class NavigationItemPosition:
        TOP = "top"
        BOTTOM = "bottom"

from ..utils.basic_logging import LoggerMixin, get_logger
from ..utils.exceptions import ComponentInitializationError
from .analysis_page import AnalysisPageConfig, create_analysis_page
from .history_page import HistoryPageConfig, create_history_page
from .main_window import MainWindow, NavigationPage, UIConfig
from .upload_page import UploadConfig, create_upload_page


class HomePage(QObject, LoggerMixin):
    """主页 - 欢迎界面和快速导航"""

    def __init__(self, parent=None):
        super().__init__(parent)
        from PyQt6.QtWidgets import QVBoxLayout, QWidget
        from qfluentwidgets import (
            BodyLabel,
            FluentIcon,
            HeaderCardWidget,
            PrimaryPushButton,
            TitleLabel,
        )

        self.widget = QWidget()
        layout = QVBoxLayout(self.widget)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)

        # 标题
        title = TitleLabel("数据分析工具")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # 欢迎卡片
        welcome_card = HeaderCardWidget()
        welcome_card.setTitle("欢迎使用数据分析工具")

        # 创建欢迎卡片的内容布局
        welcome_layout = QVBoxLayout()
        welcome_layout.setContentsMargins(0, 0, 0, 0)

        welcome_text = BodyLabel(
            "这是一个基于PyQt6和Fluent Design的现代化数据分析工具。\n"
            "支持CSV和Parquet格式的数据文件，提供全面的统计分析功能。"
        )
        welcome_layout.addWidget(welcome_text)

        # 将布局添加到HeaderCardWidget的viewLayout
        welcome_card.viewLayout.addLayout(welcome_layout)
        layout.addWidget(welcome_card)

        # 快速操作卡片
        actions_card = HeaderCardWidget()
        actions_card.setTitle("快速开始")

        # 创建快速操作卡片的内容布局
        actions_layout = QVBoxLayout()
        actions_layout.setContentsMargins(0, 0, 0, 0)
        actions_layout.setSpacing(10)

        # 快速操作按钮
        upload_btn = PrimaryPushButton("上传数据文件")
        upload_btn.setIcon(FluentIcon.FOLDER)
        upload_btn.clicked.connect(lambda: self.navigate_requested.emit("upload"))
        actions_layout.addWidget(upload_btn)

        history_btn = PrimaryPushButton("查看历史记录")
        history_btn.setIcon(FluentIcon.HISTORY)
        history_btn.clicked.connect(lambda: self.navigate_requested.emit("history"))
        actions_layout.addWidget(history_btn)

        # 将布局添加到HeaderCardWidget的viewLayout
        actions_card.viewLayout.addLayout(actions_layout)
        layout.addWidget(actions_card)

        # 功能介绍卡片
        features_card = HeaderCardWidget()
        features_card.setTitle("主要功能")

        # 创建功能介绍卡片的内容布局
        features_layout = QVBoxLayout()
        features_layout.setContentsMargins(0, 0, 0, 0)
        features_layout.setSpacing(8)

        features = [
            "📊 描述性统计分析",
            "🔗 变量关联分析",
            "⚠️ 异常值检测",
            "📈 时间序列分析",
            "📋 分析历史管理",
            "📉 可视化图表生成"
        ]

        for feature in features:
            feature_label = BodyLabel(feature)
            features_layout.addWidget(feature_label)

        # 将布局添加到HeaderCardWidget的viewLayout
        features_card.viewLayout.addLayout(features_layout)
        layout.addWidget(features_card)
        layout.addStretch()

    # 信号
    navigate_requested = pyqtSignal(str)  # 导航请求

    def get_widget(self):
        """获取widget"""
        return self.widget

    def apply_responsive_layout(self, layout_mode: str):
        """应用响应式布局"""
        if layout_mode == 'mobile':
            self.widget.setContentsMargins(10, 10, 10, 10)
        elif layout_mode == 'tablet':
            self.widget.setContentsMargins(15, 15, 15, 15)
        else:  # desktop
            self.widget.setContentsMargins(20, 20, 20, 20)


class ApplicationIntegrator(QObject, LoggerMixin):
    """应用程序集成器"""

    # 信号
    data_loaded = pyqtSignal(object, str, str)  # (data, file_path, time_column)
    analysis_completed = pyqtSignal(object)  # AnalysisResult
    navigation_requested = pyqtSignal(str)  # page_id

    def __init__(self):
        super().__init__()

        # 检查依赖
        if not HAS_PYQT6:
            raise ComponentInitializationError("PyQt6未安装")
        if not HAS_FLUENT_WIDGETS:
            raise ComponentInitializationError("PyQt-Fluent-Widgets未安装")

        # 主窗口和页面
        self.main_window: MainWindow | None = None
        self.home_page: HomePage | None = None
        self.upload_page: Any | None = None
        self.analysis_page: Any | None = None
        self.history_page: Any | None = None

        # 应用数据
        self.current_data: Any | None = None
        self.current_file_path: str | None = None
        self.current_time_column: str | None = None

        self.logger.info("应用程序集成器初始化完成")

    def create_application(self) -> MainWindow:
        """创建完整应用程序"""
        try:
            # 创建主窗口
            ui_config = UIConfig()
            ui_config.window_title = "数据分析工具 - Data Analysis PyQt"
            ui_config.min_window_size = (1000, 700)
            ui_config.default_window_size = (1400, 900)

            self.main_window = MainWindow(ui_config)

            # 创建和集成页面
            self._create_pages()
            self._setup_page_connections()

            # 默认导航到主页
            self.main_window.navigate_to(NavigationPage.HOME.value)

            self.logger.info("完整应用程序创建成功")
            return self.main_window

        except Exception as e:
            self.logger.error(f"创建应用程序失败: {str(e)}")
            raise ComponentInitializationError(f"创建应用程序失败: {str(e)}") from e

    def _create_pages(self):
        """创建所有页面"""
        try:
            # 创建主页
            self.home_page = HomePage()
            self.main_window.add_page(
                NavigationPage.HOME.value,
                self.home_page.get_widget(),
                "主页",
                FluentIcon.HOME,
                NavigationItemPosition.TOP
            )

            # 创建上传页面
            upload_config = UploadConfig()
            upload_config.supported_formats = ['.csv', '.parquet']
            upload_config.max_file_size_mb = 500
            upload_config.enable_drag_drop = True

            self.upload_page = create_upload_page(upload_config)
            self.main_window.add_page(
                NavigationPage.UPLOAD.value,
                self.upload_page,
                "数据上传",
                FluentIcon.FOLDER,
                NavigationItemPosition.TOP
            )

            # 创建分析页面
            analysis_config = AnalysisPageConfig()
            analysis_config.enable_descriptive_stats = True
            analysis_config.enable_correlation_analysis = True
            analysis_config.enable_anomaly_detection = True
            analysis_config.enable_time_series_analysis = True
            analysis_config.enable_charts = True

            self.analysis_page = create_analysis_page(analysis_config)
            self.main_window.add_page(
                NavigationPage.ANALYSIS.value,
                self.analysis_page,
                "数据分析",
                FluentIcon.DOCUMENT,
                NavigationItemPosition.TOP
            )

            # 创建历史页面
            history_config = HistoryPageConfig()
            history_config.records_per_page = 50
            history_config.auto_refresh_interval = 30
            history_config.enable_delete = True
            history_config.enable_export = True

            self.history_page = create_history_page(history_config)
            self.main_window.add_page(
                NavigationPage.HISTORY.value,
                self.history_page,
                "历史记录",
                FluentIcon.FOLDER,
                NavigationItemPosition.TOP
            )

            self.logger.info("所有页面创建完成")

        except Exception as e:
            self.logger.error(f"创建页面失败: {str(e)}")
            raise

    def _setup_page_connections(self):
        """设置页面间的信号连接"""
        try:
            # 主页导航信号
            if self.home_page:
                self.home_page.navigate_requested.connect(self._on_navigate_requested)

            # 上传页面信号
            if self.upload_page:
                if hasattr(self.upload_page, 'data_loaded'):
                    self.upload_page.data_loaded.connect(self._on_data_loaded)
                if hasattr(self.upload_page, 'upload_completed'):
                    self.upload_page.upload_completed.connect(self._on_upload_completed)

            # 分析页面信号
            if self.analysis_page:
                if hasattr(self.analysis_page, 'analysis_completed'):
                    self.analysis_page.analysis_completed.connect(self._on_analysis_completed)
                if hasattr(self.analysis_page, 'analysis_started'):
                    self.analysis_page.analysis_started.connect(self._on_analysis_started)

            # 历史页面信号
            if self.history_page:
                if hasattr(self.history_page, 'record_selected'):
                    self.history_page.record_selected.connect(self._on_history_record_selected)
                if hasattr(self.history_page, 'record_reload_requested'):
                    self.history_page.record_reload_requested.connect(self._on_history_reload_requested)

            self.logger.info("页面信号连接设置完成")

        except Exception as e:
            self.logger.error(f"设置页面连接失败: {str(e)}")

    def _on_navigate_requested(self, page_id: str):
        """处理导航请求"""
        try:
            if self.main_window:
                self.main_window.navigate_to(page_id)
            self.navigation_requested.emit(page_id)
        except Exception as e:
            self.logger.error(f"导航处理失败: {str(e)}")

    def _on_data_loaded(self, data: Any, file_path: str, time_column: str | None = None):
        """处理数据加载"""
        try:
            self.current_data = data
            self.current_file_path = file_path
            self.current_time_column = time_column

            # 将数据传递给分析页面
            if self.analysis_page and hasattr(self.analysis_page, 'load_data'):
                self.analysis_page.load_data(data, file_path, time_column)

            # 发出信号
            self.data_loaded.emit(data, file_path, time_column or "")

            self.logger.info(f"数据已加载: {file_path}")

        except Exception as e:
            self.logger.error(f"数据加载处理失败: {str(e)}")

    def _on_upload_completed(self, file_path: str):
        """处理上传完成"""
        try:
            # 导航到分析页面
            if self.main_window:
                self.main_window.navigate_to(NavigationPage.ANALYSIS.value)

            self.logger.info(f"上传完成，已导航到分析页面: {file_path}")

        except Exception as e:
            self.logger.error(f"上传完成处理失败: {str(e)}")

    def _on_analysis_started(self):
        """处理分析开始"""
        try:
            self.logger.info("分析已开始")
        except Exception as e:
            self.logger.error(f"分析开始处理失败: {str(e)}")

    def _on_analysis_completed(self, result: Any):
        """处理分析完成"""
        try:
            # 发出信号
            self.analysis_completed.emit(result)

            self.logger.info("分析已完成")

        except Exception as e:
            self.logger.error(f"分析完成处理失败: {str(e)}")

    def _on_history_record_selected(self, record: Any):
        """处理历史记录选择"""
        try:
            self.logger.info(f"历史记录已选择: {record.analysis_id if hasattr(record, 'analysis_id') else 'unknown'}")
        except Exception as e:
            self.logger.error(f"历史记录选择处理失败: {str(e)}")

    def _on_history_reload_requested(self, record: Any):
        """处理历史记录重载请求"""
        try:
            # 可以在这里实现重新加载历史分析的逻辑
            self.logger.info(f"历史记录重载请求: {record.analysis_id if hasattr(record, 'analysis_id') else 'unknown'}")

            # 导航到分析页面
            if self.main_window:
                self.main_window.navigate_to(NavigationPage.ANALYSIS.value)

        except Exception as e:
            self.logger.error(f"历史记录重载处理失败: {str(e)}")

    def get_main_window(self) -> MainWindow | None:
        """获取主窗口"""
        return self.main_window

    def get_current_data(self) -> dict[str, Any]:
        """获取当前数据信息"""
        return {
            'data': self.current_data,
            'file_path': self.current_file_path,
            'time_column': self.current_time_column
        }


def create_application() -> ApplicationIntegrator:
    """创建应用程序的工厂函数"""
    try:
        integrator = ApplicationIntegrator()
        integrator.create_application()
        return integrator
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"创建应用程序失败: {str(e)}")
        raise
