"""
数据上传页面
支持拖拽上传、文件选择、格式验证和进度显示
"""

import os
import mimetypes
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum

try:
    from PyQt6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
        QLabel, QPushButton, QProgressBar, QFrame, QScrollArea,
        QFileDialog, QMessageBox, QGroupBox, QSizePolicy
    )
    from PyQt6.QtCore import (
        Qt, QThread, pyqtSignal, QTimer, QMimeData, QUrl,
        QPropertyAnimation, QEasingCurve, QRect
    )
    from PyQt6.QtGui import (
        QDragEnterEvent, QDropEvent, QPalette, QFont,
        QPixmap, QIcon
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
    class QProgressBar:
        pass
    class QFrame:
        pass
    class QThread:
        pass
    def pyqtSignal(*args):
        return lambda *a, **k: None

try:
    from qfluentwidgets import (
        CardWidget, HeaderCardWidget, SimpleCardWidget,
        PrimaryPushButton, PushButton, ToolButton,
        ProgressBar, IndeterminateProgressBar, ProgressRing,
        BodyLabel, CaptionLabel, StrongBodyLabel, SubtitleLabel,
        InfoBar, InfoBarPosition, InfoBarIcon,
        FluentIcon, Theme, isDarkTheme
    )
    HAS_FLUENT_WIDGETS = True
except ImportError:
    HAS_FLUENT_WIDGETS = False
    # 模拟类定义
    class CardWidget:
        pass
    class PrimaryPushButton:
        pass
    class ProgressBar:
        pass
    class BodyLabel:
        pass
    class FluentIcon:
        FOLDER = "folder"
        UPLOAD = "upload"
        ACCEPT = "accept"
        CANCEL = "cancel"

from ..utils.basic_logging import LoggerMixin
from ..utils.exceptions import DataProcessingError, FileValidationError

try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


class UploadStatus(str, Enum):
    """上传状态"""
    IDLE = "idle"
    VALIDATING = "validating" 
    UPLOADING = "uploading"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class UploadConfig:
    """上传配置"""
    # 支持的文件类型
    supported_extensions: List[str] = None
    max_file_size: int = 500 * 1024 * 1024  # 500MB
    
    # UI配置
    enable_drag_drop: bool = True
    show_preview: bool = True
    auto_detect_time_columns: bool = True
    
    # 进度配置
    show_detailed_progress: bool = True
    progress_update_interval: int = 100  # ms
    
    def __post_init__(self):
        if self.supported_extensions is None:
            self.supported_extensions = ['.csv', '.parquet', '.xlsx', '.json']


# 从data_preview导入共享的类，避免循环导入
try:
    from .data_preview import FileInfo, FileType
except ImportError:
    # 如果导入失败，提供最小的定义
    from enum import Enum
    @dataclass
    class FileInfo:
        file_path: str = ""
        file_name: str = ""
        file_size: int = 0
        file_type: str = ""
        mime_type: str = ""
        is_valid: bool = False
        error_message: str = ""
    
    class FileType(str, Enum):
        CSV = "csv"
        PARQUET = "parquet"
        EXCEL = "excel" 
        JSON = "json"
        UNKNOWN = "unknown"


class FileValidator(LoggerMixin):
    """文件验证器"""
    
    def __init__(self, config: UploadConfig):
        self.config = config
    
    def validate_file(self, file_path: str) -> FileInfo:
        """验证文件"""
        try:
            path = Path(file_path)
            
            # 基本信息
            file_info = FileInfo(
                file_path=str(path.absolute()),
                file_name=path.name,
                file_size=path.stat().st_size,
                file_type=self._detect_file_type(path),
                mime_type=mimetypes.guess_type(str(path))[0] or "unknown"
            )
            
            # 验证扩展名
            if path.suffix.lower() not in self.config.supported_extensions:
                file_info.error_message = f"不支持的文件类型: {path.suffix}"
                return file_info
            
            # 验证文件大小
            if file_info.file_size > self.config.max_file_size:
                size_mb = file_info.file_size / (1024 * 1024)
                max_mb = self.config.max_file_size / (1024 * 1024)
                file_info.error_message = f"文件过大: {size_mb:.1f}MB (最大 {max_mb:.1f}MB)"
                return file_info
            
            # 验证文件存在和可读
            if not path.exists():
                file_info.error_message = "文件不存在"
                return file_info
            
            if not os.access(path, os.R_OK):
                file_info.error_message = "文件无法读取"
                return file_info
            
            file_info.is_valid = True
            self.logger.info(f"文件验证通过: {file_info.file_name}")
            
            return file_info
            
        except Exception as e:
            self.logger.error(f"文件验证失败: {str(e)}")
            return FileInfo(
                file_path=file_path,
                file_name=Path(file_path).name if file_path else "unknown",
                file_size=0,
                file_type=FileType.UNKNOWN,
                mime_type="unknown",
                error_message=f"验证失败: {str(e)}"
            )
    
    def _detect_file_type(self, path: Path) -> FileType:
        """检测文件类型"""
        suffix = path.suffix.lower()
        
        if suffix == '.csv':
            return FileType.CSV
        elif suffix == '.parquet':
            return FileType.PARQUET
        elif suffix in ['.xlsx', '.xls']:
            return FileType.EXCEL
        elif suffix == '.json':
            return FileType.JSON
        else:
            return FileType.UNKNOWN


class UploadWorker(QThread):
    """上传工作线程"""
    
    # 信号
    progress_updated = pyqtSignal(int, str)  # (progress, message)
    status_changed = pyqtSignal(str)  # UploadStatus
    file_loaded = pyqtSignal(object)  # FileInfo
    error_occurred = pyqtSignal(str)  # error_message
    
    def __init__(self, file_info: FileInfo, config: UploadConfig):
        super().__init__()
        self.file_info = file_info
        self.config = config
        self._cancel_requested = False
    
    def run(self):
        """执行上传和加载"""
        try:
            self.status_changed.emit(UploadStatus.UPLOADING.value)
            self.progress_updated.emit(0, "开始处理文件...")
            
            # 模拟文件读取进度
            for i in range(0, 101, 10):
                if self._cancel_requested:
                    self.status_changed.emit(UploadStatus.CANCELLED.value)
                    return
                
                self.progress_updated.emit(i, f"读取文件... {i}%")
                self.msleep(50)  # 模拟处理时间
            
            # 这里应该集成实际的数据加载逻辑 (DataLoader)
            # 现在先模拟数据信息
            self.file_info.row_count = 1000
            self.file_info.column_count = 5
            self.file_info.columns = ["col1", "col2", "col3", "col4", "col5"]
            self.file_info.time_columns = []
            self.file_info.memory_usage = 50 * 1024  # 50KB
            
            self.progress_updated.emit(100, "文件加载完成")
            self.status_changed.emit(UploadStatus.COMPLETED.value)
            self.file_loaded.emit(self.file_info)
            
        except Exception as e:
            self.error_occurred.emit(str(e))
            self.status_changed.emit(UploadStatus.FAILED.value)
    
    def cancel(self):
        """取消上传"""
        self._cancel_requested = True


class DragDropArea(QFrame, LoggerMixin):
    """拖拽上传区域"""
    
    # 信号
    files_dropped = pyqtSignal(list)  # List[str]
    
    def __init__(self, config: UploadConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self.setAcceptDrops(self.config.enable_drag_drop)
        self._setup_ui()
        self._setup_styles()
    
    def _setup_ui(self):
        """设置UI"""
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(20)
        
        # 上传图标
        if HAS_FLUENT_WIDGETS:
            icon_label = BodyLabel("📁")
        else:
            icon_label = QLabel("📁")
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_label.setObjectName("uploadIcon")
        layout.addWidget(icon_label)
        
        # 主要文本
        if HAS_FLUENT_WIDGETS:
            main_label = SubtitleLabel("拖拽文件到此处")
        else:
            main_label = QLabel("拖拽文件到此处")
        main_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_label.setObjectName("mainLabel")
        layout.addWidget(main_label)
        
        # 支持格式文本
        formats = ", ".join(self.config.supported_extensions)
        if HAS_FLUENT_WIDGETS:
            format_label = CaptionLabel(f"支持格式: {formats}")
        else:
            format_label = QLabel(f"支持格式: {formats}")
        format_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        format_label.setObjectName("formatLabel")
        layout.addWidget(format_label)
        
        # 或者文本
        if HAS_FLUENT_WIDGETS:
            or_label = CaptionLabel("或者")
        else:
            or_label = QLabel("或者")
        or_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(or_label)
        
        # 选择文件按钮
        if HAS_FLUENT_WIDGETS:
            self.select_btn = PrimaryPushButton("选择文件", FluentIcon.FOLDER)
        else:
            self.select_btn = QPushButton("选择文件")
        self.select_btn.setObjectName("selectBtn")
        self.select_btn.clicked.connect(self._select_files)
        layout.addWidget(self.select_btn, 0, Qt.AlignmentFlag.AlignCenter)
    
    def _setup_styles(self):
        """设置样式"""
        self.setObjectName("dragDropArea")
        self.setMinimumHeight(200)
        self.setFrameStyle(QFrame.Shape.Box)
        self.setLineWidth(2)
        
        # 设置样式表
        self.setStyleSheet("""
            #dragDropArea {
                border: 2px dashed #cccccc;
                border-radius: 10px;
                background-color: #fafafa;
            }
            #dragDropArea:hover {
                border-color: #0078d4;
                background-color: #f0f8ff;
            }
            #uploadIcon {
                font-size: 48px;
                color: #666666;
            }
            #mainLabel {
                font-size: 18px;
                font-weight: bold;
                color: #333333;
            }
            #formatLabel {
                color: #666666;
            }
            #selectBtn {
                min-width: 120px;
                min-height: 32px;
            }
        """)
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        """拖拽进入事件"""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet(self.styleSheet() + """
                #dragDropArea {
                    border-color: #0078d4 !important;
                    background-color: #e6f3ff !important;
                }
            """)
    
    def dragLeaveEvent(self, event):
        """拖拽离开事件"""
        self._setup_styles()
    
    def dropEvent(self, event: QDropEvent):
        """拖拽放置事件"""
        files = []
        for url in event.mimeData().urls():
            if url.isLocalFile():
                files.append(url.toLocalFile())
        
        if files:
            self.files_dropped.emit(files)
            self.logger.info(f"拖拽文件: {files}")
        
        self._setup_styles()
        event.acceptProposedAction()
    
    def _select_files(self):
        """选择文件对话框"""
        try:
            file_filter = "支持的文件 ("
            file_filter += " ".join(f"*{ext}" for ext in self.config.supported_extensions)
            file_filter += ")"
            
            files, _ = QFileDialog.getOpenFileNames(
                self,
                "选择数据文件",
                "",
                file_filter
            )
            
            if files:
                self.files_dropped.emit(files)
                self.logger.info(f"选择文件: {files}")
                
        except Exception as e:
            self.logger.error(f"选择文件失败: {str(e)}")


class UploadProgressWidget(QWidget, LoggerMixin):
    """上传进度组件"""
    
    # 信号
    cancel_requested = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_status = UploadStatus.IDLE
        self._setup_ui()
        self.hide()  # 初始隐藏
    
    def _setup_ui(self):
        """设置UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        # 进度信息
        info_layout = QHBoxLayout()
        
        if HAS_FLUENT_WIDGETS:
            self.status_label = BodyLabel("准备中...")
        else:
            self.status_label = QLabel("准备中...")
        self.status_label.setObjectName("statusLabel")
        info_layout.addWidget(self.status_label)
        
        info_layout.addStretch()
        
        # 取消按钮
        if HAS_FLUENT_WIDGETS:
            self.cancel_btn = ToolButton(FluentIcon.CANCEL)
        else:
            self.cancel_btn = QPushButton("✕")
        self.cancel_btn.setObjectName("cancelBtn")
        self.cancel_btn.clicked.connect(self.cancel_requested.emit)
        self.cancel_btn.setToolTip("取消上传")
        info_layout.addWidget(self.cancel_btn)
        
        layout.addLayout(info_layout)
        
        # 进度条
        if HAS_FLUENT_WIDGETS:
            self.progress_bar = ProgressBar()
        else:
            self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("progressBar")
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        layout.addWidget(self.progress_bar)
        
        # 详细信息
        if HAS_FLUENT_WIDGETS:
            self.detail_label = CaptionLabel("")
        else:
            self.detail_label = QLabel("")
        self.detail_label.setObjectName("detailLabel")
        layout.addWidget(self.detail_label)
    
    def update_progress(self, progress: int, message: str):
        """更新进度"""
        self.progress_bar.setValue(progress)
        self.detail_label.setText(message)
        self.logger.debug(f"进度更新: {progress}% - {message}")
    
    def update_status(self, status: UploadStatus):
        """更新状态"""
        self.current_status = status
        
        status_texts = {
            UploadStatus.IDLE: "准备中...",
            UploadStatus.VALIDATING: "验证文件...",
            UploadStatus.UPLOADING: "上传中...",
            UploadStatus.PROCESSING: "处理中...",
            UploadStatus.COMPLETED: "完成",
            UploadStatus.FAILED: "失败",
            UploadStatus.CANCELLED: "已取消"
        }
        
        self.status_label.setText(status_texts.get(status, "未知状态"))
        
        # 根据状态调整UI
        if status in [UploadStatus.COMPLETED, UploadStatus.FAILED, UploadStatus.CANCELLED]:
            self.cancel_btn.hide()
        else:
            self.cancel_btn.show()
        
        self.logger.info(f"状态更新: {status.value}")
    
    def show_progress(self):
        """显示进度组件"""
        self.show()
        self.update_progress(0, "")
    
    def hide_progress(self):
        """隐藏进度组件"""
        self.hide()


class UploadPage(QWidget, LoggerMixin):
    """数据上传页面"""
    
    # 信号
    file_uploaded = pyqtSignal(object)  # FileInfo
    upload_failed = pyqtSignal(str)  # error_message
    
    def __init__(self, config: Optional[UploadConfig] = None, parent=None):
        super().__init__(parent)
        self.config = config or UploadConfig()
        self.validator = FileValidator(self.config)
        self.upload_worker: Optional[UploadWorker] = None
        self.current_file_info: Optional[FileInfo] = None
        self.current_data: Optional[Any] = None  # 存储加载的数据
        
        self._setup_ui()
        self._setup_connections()
        self.logger.info("上传页面初始化完成")
    
    def _setup_ui(self):
        """设置UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # 页面标题
        if HAS_FLUENT_WIDGETS:
            title_label = SubtitleLabel("数据文件上传")
        else:
            title_label = QLabel("数据文件上传")
        title_label.setObjectName("titleLabel")
        layout.addWidget(title_label)
        
        # 上传区域
        if HAS_FLUENT_WIDGETS:
            upload_card = HeaderCardWidget()
            upload_card.setTitle("选择数据文件")
        else:
            upload_card = QFrame()
            upload_card.setFrameStyle(QFrame.Shape.Box)
        
        upload_layout = QVBoxLayout(upload_card)
        
        # 拖拽区域
        self.drag_drop_area = DragDropArea(self.config)
        upload_layout.addWidget(self.drag_drop_area)
        
        # 进度组件
        self.progress_widget = UploadProgressWidget()
        upload_layout.addWidget(self.progress_widget)
        
        layout.addWidget(upload_card)
        
        # 创建水平分割布局
        content_layout = QHBoxLayout()
        
        # 左侧：文件信息区域
        if HAS_FLUENT_WIDGETS:
            self.info_card = HeaderCardWidget()
            self.info_card.setTitle("文件信息")
        else:
            self.info_card = QFrame()
            self.info_card.setFrameStyle(QFrame.Shape.Box)
        
        self.info_layout = QVBoxLayout(self.info_card)
        
        # 无文件时的提示
        if HAS_FLUENT_WIDGETS:
            self.no_file_label = CaptionLabel("请先选择文件")
        else:
            self.no_file_label = QLabel("请先选择文件")
        self.no_file_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info_layout.addWidget(self.no_file_label)
        
        # 右侧：数据预览区域
        preview_config = PreviewConfig()
        preview_config.max_preview_rows = 50  # 上传页面显示较少行数
        self.data_preview = DataPreviewWidget(preview_config)
        self.data_preview.hide()  # 初始隐藏
        
        # 添加到水平布局
        content_layout.addWidget(self.info_card, 1)  # 文件信息占1/3
        content_layout.addWidget(self.data_preview, 2)  # 数据预览占2/3
        
        layout.addLayout(content_layout)
        
        # 弹性空间
        layout.addStretch()
        
        self.setObjectName("uploadPage")
    
    def _setup_connections(self):
        """设置信号连接"""
        self.drag_drop_area.files_dropped.connect(self._handle_files_dropped)
        self.progress_widget.cancel_requested.connect(self._cancel_upload)
    
    def _handle_files_dropped(self, files: List[str]):
        """处理拖拽的文件"""
        if not files:
            return
        
        # 只处理第一个文件
        file_path = files[0]
        
        try:
            # 验证文件
            self.progress_widget.show_progress()
            self.progress_widget.update_status(UploadStatus.VALIDATING)
            self.progress_widget.update_progress(0, "验证文件...")
            
            file_info = self.validator.validate_file(file_path)
            
            if not file_info.is_valid:
                self._show_error(f"文件验证失败: {file_info.error_message}")
                self.progress_widget.hide_progress()
                return
            
            # 开始上传
            self.current_file_info = file_info
            self._start_upload(file_info)
            
        except Exception as e:
            self.logger.error(f"处理文件失败: {str(e)}")
            self._show_error(f"处理文件失败: {str(e)}")
            self.progress_widget.hide_progress()
    
    def _start_upload(self, file_info: FileInfo):
        """开始上传"""
        try:
            # 停止之前的上传
            if self.upload_worker and self.upload_worker.isRunning():
                self.upload_worker.cancel()
                self.upload_worker.wait()
            
            # 创建新的工作线程
            self.upload_worker = UploadWorker(file_info, self.config)
            
            # 连接信号
            self.upload_worker.progress_updated.connect(self.progress_widget.update_progress)
            self.upload_worker.status_changed.connect(
                lambda status: self.progress_widget.update_status(UploadStatus(status))
            )
            self.upload_worker.file_loaded.connect(self._on_file_loaded)
            self.upload_worker.error_occurred.connect(self._on_upload_error)
            
            # 开始上传
            self.upload_worker.start()
            
        except Exception as e:
            self.logger.error(f"开始上传失败: {str(e)}")
            self._show_error(f"开始上传失败: {str(e)}")
    
    def _cancel_upload(self):
        """取消上传"""
        if self.upload_worker and self.upload_worker.isRunning():
            self.upload_worker.cancel()
            self.upload_worker.wait()
        
        self.progress_widget.hide_progress()
        self.logger.info("用户取消上传")
    
    def _on_file_loaded(self, file_info: FileInfo):
        """文件加载完成"""
        self.current_file_info = file_info
        self.progress_widget.hide_progress()
        
        # 模拟数据加载（这里应该集成实际的DataLoader）
        self.current_data = self._create_mock_data(file_info)
        
        # 显示文件信息
        self._show_file_info(file_info)
        
        # 显示数据预览
        if self.current_data is not None:
            try:
                self.data_preview.load_file_data(file_info, self.current_data)
                self.data_preview.show()
            except Exception as e:
                self.logger.error(f"加载数据预览失败: {str(e)}")
                self._show_error(f"数据预览加载失败: {str(e)}")
        
        self.file_uploaded.emit(file_info)
        self.logger.info(f"文件上传完成: {file_info.file_name}")
    
    def _create_mock_data(self, file_info: FileInfo) -> Optional[Any]:
        """创建模拟数据（临时方法，后续将替换为实际数据加载）"""
        try:
            # 根据文件类型创建不同的模拟数据
            if HAS_PANDAS:
                import numpy as np
                
                if file_info.file_type == FileType.CSV:
                    # 模拟CSV数据
                    np.random.seed(42)
                    data = {
                        'ID': range(1, 101),
                        'Name': [f'Item_{i}' for i in range(1, 101)],
                        'Value': np.random.normal(100, 15, 100),
                        'Category': np.random.choice(['A', 'B', 'C'], 100),
                        'DateTime': pd.date_range('2023-01-01', periods=100, freq='D'),
                        'Status': np.random.choice(['Active', 'Inactive', None], 100)
                    }
                    return pd.DataFrame(data)
                
                elif file_info.file_type == FileType.PARQUET:
                    # 模拟Parquet数据
                    np.random.seed(42)
                    data = {
                        'sensor_id': range(1, 201),
                        'timestamp': pd.date_range('2023-01-01', periods=200, freq='H'),
                        'temperature': np.random.normal(25, 5, 200),
                        'humidity': np.random.uniform(30, 80, 200),
                        'pressure': np.random.normal(1013, 10, 200)
                    }
                    return pd.DataFrame(data)
                
                else:
                    # 默认数据
                    np.random.seed(42)
                    data = {
                        'column1': np.random.randn(50),
                        'column2': np.random.randint(1, 100, 50),
                        'column3': [f'text_{i}' for i in range(50)],
                        'tagTime': pd.date_range('2023-01-01', periods=50, freq='T')
                    }
                    return pd.DataFrame(data)
            
            return None
            
        except Exception as e:
            self.logger.error(f"创建模拟数据失败: {str(e)}")
            return None
    
    def _on_upload_error(self, error_message: str):
        """上传错误"""
        self.progress_widget.hide_progress()
        self._show_error(f"上传失败: {error_message}")
        self.upload_failed.emit(error_message)
    
    def _show_file_info(self, file_info: FileInfo):
        """显示文件信息"""
        # 清除现有内容
        self._clear_info_layout()
        
        # 文件基本信息
        info_items = [
            ("文件名", file_info.file_name),
            ("文件大小", self._format_file_size(file_info.file_size)),
            ("文件类型", file_info.file_type.value.upper()),
        ]
        
        if file_info.row_count is not None:
            info_items.append(("行数", f"{file_info.row_count:,}"))
        
        if file_info.column_count is not None:
            info_items.append(("列数", str(file_info.column_count)))
        
        if file_info.memory_usage is not None:
            info_items.append(("内存占用", self._format_file_size(file_info.memory_usage)))
        
        for label, value in info_items:
            item_layout = QHBoxLayout()
            
            if HAS_FLUENT_WIDGETS:
                label_widget = BodyLabel(f"{label}:")
                value_widget = BodyLabel(str(value))
            else:
                label_widget = QLabel(f"{label}:")
                value_widget = QLabel(str(value))
            
            label_widget.setMinimumWidth(80)
            item_layout.addWidget(label_widget)
            item_layout.addWidget(value_widget)
            item_layout.addStretch()
            
            self.info_layout.addLayout(item_layout)
        
        # 列信息
        if file_info.columns:
            self.info_layout.addSpacing(10)
            
            if HAS_FLUENT_WIDGETS:
                columns_label = BodyLabel("列名:")
            else:
                columns_label = QLabel("列名:")
            self.info_layout.addWidget(columns_label)
            
            columns_text = ", ".join(file_info.columns[:10])  # 只显示前10列
            if len(file_info.columns) > 10:
                columns_text += f" ... (共{len(file_info.columns)}列)"
            
            if HAS_FLUENT_WIDGETS:
                columns_value = CaptionLabel(columns_text)
            else:
                columns_value = QLabel(columns_text)
            columns_value.setWordWrap(True)
            self.info_layout.addWidget(columns_value)
        
        # 时间列信息
        if file_info.time_columns:
            self.info_layout.addSpacing(5)
            
            if HAS_FLUENT_WIDGETS:
                time_label = BodyLabel("时间列:")
                time_value = CaptionLabel(", ".join(file_info.time_columns))
            else:
                time_label = QLabel("时间列:")
                time_value = QLabel(", ".join(file_info.time_columns))
            
            self.info_layout.addWidget(time_label)
            self.info_layout.addWidget(time_value)
    
    def _clear_info_layout(self):
        """清除信息布局"""
        while self.info_layout.count():
            item = self.info_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                self._clear_layout(item.layout())
    
    def _clear_layout(self, layout):
        """递归清除布局"""
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                self._clear_layout(item.layout())
    
    def _format_file_size(self, size_bytes: int) -> str:
        """格式化文件大小"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"
    
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
    
    def get_current_file_info(self) -> Optional[FileInfo]:
        """获取当前文件信息"""
        return self.current_file_info
    
    def clear_upload(self):
        """清除上传"""
        if self.upload_worker and self.upload_worker.isRunning():
            self.upload_worker.cancel()
            self.upload_worker.wait()
        
        self.current_file_info = None
        self.current_data = None
        self.progress_widget.hide_progress()
        self._clear_info_layout()
        
        # 隐藏数据预览
        self.data_preview.clear_preview()
        self.data_preview.hide()
        
        # 恢复无文件提示
        self.info_layout.addWidget(self.no_file_label)
        
        self.logger.info("上传已清除")
    
    def apply_responsive_layout(self, layout_mode: str):
        """应用响应式布局"""
        if layout_mode == 'mobile':
            # 移动端布局调整 - 垂直布局
            self.setContentsMargins(10, 10, 10, 10)
            self.drag_drop_area.setMinimumHeight(150)
            # 在移动端可以考虑隐藏数据预览或使用标签页
        elif layout_mode == 'tablet':
            # 平板布局调整
            self.setContentsMargins(15, 15, 15, 15)
            self.drag_drop_area.setMinimumHeight(175)
        else:  # desktop
            # 桌面布局调整
            self.setContentsMargins(20, 20, 20, 20)
            self.drag_drop_area.setMinimumHeight(200)
        
        # 将布局模式传递给数据预览组件
        if hasattr(self.data_preview, 'apply_responsive_layout'):
            self.data_preview.apply_responsive_layout(layout_mode)
        
        self.logger.debug(f"应用响应式布局: {layout_mode}")
    
    def get_current_data(self) -> Optional[Any]:
        """获取当前加载的数据"""
        return self.current_data


def create_upload_page(config: Optional[UploadConfig] = None) -> UploadPage:
    """创建上传页面的工厂函数"""
    try:
        return UploadPage(config)
    except Exception as e:
        print(f"创建上传页面失败: {str(e)}")
        raise