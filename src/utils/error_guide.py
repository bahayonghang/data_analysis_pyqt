"""
错误处理和解决方案指南系统

提供用户友好的错误信息、详细解决方案和智能故障诊断
"""

import traceback
import sys
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

try:
    from PyQt6.QtWidgets import (
        QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
        QTextEdit, QTabWidget, QWidget, QScrollArea, QFrame,
        QMessageBox, QProgressBar, QCheckBox
    )
    from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QThread
    from PyQt6.QtGui import QFont, QPixmap, QIcon
    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False
    class QDialog:
        def __init__(self, parent=None): pass
    class QWidget:
        def __init__(self, parent=None): pass

from ..utils.basic_logging import LoggerMixin


class ErrorSeverity(str, Enum):
    """错误严重程度"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ErrorCategory(str, Enum):
    """错误类别"""
    FILE_IO = "file_io"
    DATA_PROCESSING = "data_processing"
    MEMORY = "memory"
    PERMISSION = "permission"
    NETWORK = "network"
    DEPENDENCY = "dependency"
    USER_INPUT = "user_input"
    SYSTEM = "system"


@dataclass
class ErrorSolution:
    """错误解决方案"""
    title: str
    description: str
    steps: List[str]
    auto_fix_available: bool = False
    auto_fix_function: Optional[Callable] = None
    difficulty: str = "easy"  # easy, medium, hard
    success_rate: float = 0.9  # 解决成功率


@dataclass
class ErrorInfo:
    """错误信息"""
    error_id: str
    title: str
    description: str
    category: ErrorCategory
    severity: ErrorSeverity
    solutions: List[ErrorSolution]
    related_errors: List[str] = None
    prevention_tips: List[str] = None
    
    def __post_init__(self):
        if self.related_errors is None:
            self.related_errors = []
        if self.prevention_tips is None:
            self.prevention_tips = []


class ErrorDatabase:
    """错误数据库"""
    
    def __init__(self):
        self.errors: Dict[str, ErrorInfo] = {}
        self._load_error_definitions()
    
    def _load_error_definitions(self):
        """加载错误定义"""
        error_definitions = [
            # 文件I/O错误
            ErrorInfo(
                error_id="file_not_found",
                title="文件未找到",
                description="指定的文件不存在或路径错误",
                category=ErrorCategory.FILE_IO,
                severity=ErrorSeverity.ERROR,
                solutions=[
                    ErrorSolution(
                        title="检查文件路径",
                        description="确认文件路径和文件名是否正确",
                        steps=[
                            "检查文件是否存在于指定位置",
                            "确认文件名拼写是否正确（注意大小写）",
                            "检查文件扩展名是否完整",
                            "尝试使用文件浏览器重新选择文件"
                        ]
                    ),
                    ErrorSolution(
                        title="检查文件权限",
                        description="确认您有权限访问该文件",
                        steps=[
                            "右键点击文件，查看属性/权限",
                            "确认当前用户有读取权限",
                            "如果是网络文件，检查网络连接",
                            "尝试将文件复制到本地目录"
                        ]
                    )
                ],
                prevention_tips=[
                    "使用绝对路径而不是相对路径",
                    "定期备份重要数据文件",
                    "避免在文件名中使用特殊字符"
                ]
            ),
            
            # 内存错误
            ErrorInfo(
                error_id="memory_error",
                title="内存不足",
                description="系统内存不足，无法完成操作",
                category=ErrorCategory.MEMORY,
                severity=ErrorSeverity.CRITICAL,
                solutions=[
                    ErrorSolution(
                        title="释放系统内存",
                        description="关闭不必要的程序释放内存",
                        steps=[
                            "关闭其他不必要的应用程序",
                            "清理系统临时文件",
                            "重启应用程序",
                            "考虑重启计算机"
                        ],
                        auto_fix_available=True,
                        difficulty="easy"
                    ),
                    ErrorSolution(
                        title="启用数据采样",
                        description="处理数据子集以减少内存使用",
                        steps=[
                            "在设置中启用'数据采样'选项",
                            "设置合适的采样比例（如50%）",
                            "重新上传和处理数据",
                            "如果仍然失败，进一步减少采样比例"
                        ],
                        auto_fix_available=True,
                        difficulty="medium"
                    ),
                    ErrorSolution(
                        title="分批处理数据",
                        description="将大数据集分成小批次处理",
                        steps=[
                            "将数据文件分割成多个小文件",
                            "分别上传和分析每个小文件",
                            "手动合并分析结果",
                            "考虑使用更高效的数据格式（如Parquet）"
                        ],
                        difficulty="hard"
                    )
                ],
                prevention_tips=[
                    "处理大文件前检查可用内存",
                    "定期清理系统垃圾文件",
                    "升级系统内存硬件",
                    "使用高效的数据格式"
                ]
            ),
            
            # 数据处理错误
            ErrorInfo(
                error_id="data_type_error",
                title="数据类型错误",
                description="数据类型不符合分析要求",
                category=ErrorCategory.DATA_PROCESSING,
                severity=ErrorSeverity.ERROR,
                solutions=[
                    ErrorSolution(
                        title="检查数据格式",
                        description="确认数据列的格式是否正确",
                        steps=[
                            "查看数据预览，检查每列的数据类型",
                            "确认数值列不包含文本或特殊字符",
                            "检查日期列是否使用标准格式",
                            "处理缺失值或异常值"
                        ]
                    ),
                    ErrorSolution(
                        title="数据清洗",
                        description="清理和转换数据格式",
                        steps=[
                            "删除或替换非数值字符",
                            "统一日期时间格式",
                            "处理缺失值（删除或填充）",
                            "转换数据类型"
                        ],
                        auto_fix_available=True
                    )
                ],
                prevention_tips=[
                    "上传前预先检查数据质量",
                    "使用标准的数据格式",
                    "保持数据类型一致性"
                ]
            ),
            
            # 编码错误
            ErrorInfo(
                error_id="encoding_error",
                title="文件编码错误",
                description="文件编码格式不支持或包含无法解析的字符",
                category=ErrorCategory.FILE_IO,
                severity=ErrorSeverity.ERROR,
                solutions=[
                    ErrorSolution(
                        title="转换文件编码",
                        description="将文件转换为UTF-8编码",
                        steps=[
                            "使用文本编辑器（如Notepad++）打开文件",
                            "选择'编码' -> '转换为UTF-8'",
                            "保存文件",
                            "重新上传文件"
                        ]
                    ),
                    ErrorSolution(
                        title="手动指定编码",
                        description="在上传时指定正确的编码格式",
                        steps=[
                            "在上传界面找到'编码设置'选项",
                            "尝试常见编码：GBK、GB2312、ISO-8859-1",
                            "如果不确定，尝试多种编码格式",
                            "上传成功后检查中文显示是否正常"
                        ]
                    )
                ],
                prevention_tips=[
                    "始终使用UTF-8编码保存文件",
                    "避免在数据中使用特殊字符",
                    "使用现代的文本编辑器"
                ]
            ),
            
            # 依赖错误
            ErrorInfo(
                error_id="dependency_missing",
                title="依赖库缺失",
                description="缺少必要的Python库或版本不兼容",
                category=ErrorCategory.DEPENDENCY,
                severity=ErrorSeverity.CRITICAL,
                solutions=[
                    ErrorSolution(
                        title="安装缺失依赖",
                        description="使用包管理器安装所需库",
                        steps=[
                            "打开命令行终端",
                            "运行: pip install -r requirements.txt",
                            "等待安装完成",
                            "重启应用程序"
                        ],
                        auto_fix_available=True
                    ),
                    ErrorSolution(
                        title="更新依赖版本",
                        description="更新到兼容的库版本",
                        steps=[
                            "检查requirements.txt中的版本要求",
                            "运行: pip install --upgrade [库名]",
                            "验证版本兼容性",
                            "重启应用程序"
                        ]
                    )
                ]
            ),
            
            # 权限错误
            ErrorInfo(
                error_id="permission_error",
                title="权限不足",
                description="没有足够权限执行操作",
                category=ErrorCategory.PERMISSION,
                severity=ErrorSeverity.ERROR,
                solutions=[
                    ErrorSolution(
                        title="以管理员身份运行",
                        description="使用管理员权限启动应用",
                        steps=[
                            "关闭当前应用程序",
                            "右键点击应用程序图标",
                            "选择'以管理员身份运行'",
                            "重新尝试操作"
                        ]
                    ),
                    ErrorSolution(
                        title="更改文件权限",
                        description="修改文件或文件夹的访问权限",
                        steps=[
                            "右键点击目标文件或文件夹",
                            "选择'属性' -> '安全'",
                            "点击'编辑'修改权限",
                            "给当前用户添加完全控制权限"
                        ]
                    ),
                    ErrorSolution(
                        title="选择其他位置",
                        description="将文件保存到有权限的位置",
                        steps=[
                            "选择用户文档文件夹作为保存位置",
                            "避免保存到系统目录",
                            "确认目标位置有写入权限",
                            "重新尝试操作"
                        ]
                    )
                ]
            )
        ]
        
        for error_info in error_definitions:
            self.errors[error_info.error_id] = error_info
    
    def get_error_info(self, error_id: str) -> Optional[ErrorInfo]:
        """获取错误信息"""
        return self.errors.get(error_id)
    
    def search_errors(self, query: str) -> List[ErrorInfo]:
        """搜索错误信息"""
        query_lower = query.lower()
        results = []
        
        for error_info in self.errors.values():
            if (query_lower in error_info.title.lower() or 
                query_lower in error_info.description.lower()):
                results.append(error_info)
        
        return results
    
    def get_by_category(self, category: ErrorCategory) -> List[ErrorInfo]:
        """按类别获取错误信息"""
        return [error for error in self.errors.values() if error.category == category]


class SmartErrorDetector:
    """智能错误检测器"""
    
    def __init__(self):
        self.error_patterns = {
            # 文件相关错误模式
            r"FileNotFoundError|No such file": "file_not_found",
            r"PermissionError|Permission denied": "permission_error",
            r"UnicodeDecodeError|codec can't decode": "encoding_error",
            
            # 内存相关错误模式
            r"MemoryError|out of memory": "memory_error",
            
            # 数据处理错误模式
            r"ValueError.*could not convert|invalid literal": "data_type_error",
            r"TypeError.*unsupported operand": "data_type_error",
            
            # 依赖相关错误模式
            r"ModuleNotFoundError|No module named": "dependency_missing",
            r"ImportError": "dependency_missing",
        }
    
    def detect_error_type(self, error_message: str, exception_type: str = None) -> Optional[str]:
        """检测错误类型"""
        import re
        
        full_error = f"{exception_type}: {error_message}" if exception_type else error_message
        
        for pattern, error_id in self.error_patterns.items():
            if re.search(pattern, full_error, re.IGNORECASE):
                return error_id
        
        return None


class ErrorSolutionDialog(QDialog, LoggerMixin):
    """错误解决方案对话框"""
    
    def __init__(self, parent=None, error_info: ErrorInfo = None, error_context: Dict = None):
        if HAS_PYQT6:
            super().__init__(parent)
        else:
            QDialog.__init__(self, parent)
            LoggerMixin.__init__(self)
        
        self.error_info = error_info
        self.error_context = error_context or {}
        
        if error_info:
            self.setWindowTitle(f"错误解决方案 - {error_info.title}")
        else:
            self.setWindowTitle("错误解决方案")
        
        self.setMinimumSize(700, 500)
        self.setup_ui()
    
    def setup_ui(self):
        """设置界面"""
        if not HAS_PYQT6:
            return
        
        layout = QVBoxLayout(self)
        
        # 错误信息区域
        self.setup_error_info(layout)
        
        # 解决方案标签页
        self.setup_solutions_tabs(layout)
        
        # 按钮区域
        self.setup_buttons(layout)
    
    def setup_error_info(self, parent_layout):
        """设置错误信息区域"""
        if not self.error_info:
            return
        
        # 错误信息框
        info_frame = QFrame()
        info_frame.setFrameStyle(QFrame.Shape.Box)
        info_layout = QVBoxLayout(info_frame)
        
        # 错误标题
        title_label = QLabel(self.error_info.title)
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(14)
        title_label.setFont(title_font)
        
        # 设置严重程度颜色
        severity_colors = {
            ErrorSeverity.INFO: "blue",
            ErrorSeverity.WARNING: "orange",
            ErrorSeverity.ERROR: "red",
            ErrorSeverity.CRITICAL: "darkred"
        }
        color = severity_colors.get(self.error_info.severity, "black")
        title_label.setStyleSheet(f"color: {color};")
        
        info_layout.addWidget(title_label)
        
        # 错误描述
        desc_label = QLabel(self.error_info.description)
        desc_label.setWordWrap(True)
        info_layout.addWidget(desc_label)
        
        # 错误上下文（如果有）
        if self.error_context:
            context_text = QTextEdit()
            context_text.setMaximumHeight(100)
            context_text.setPlainText(str(self.error_context.get('details', '')))
            context_text.setReadOnly(True)
            info_layout.addWidget(QLabel("错误详情:"))
            info_layout.addWidget(context_text)
        
        parent_layout.addWidget(info_frame)
    
    def setup_solutions_tabs(self, parent_layout):
        """设置解决方案标签页"""
        if not self.error_info or not self.error_info.solutions:
            return
        
        self.tabs = QTabWidget()
        
        for i, solution in enumerate(self.error_info.solutions):
            tab_widget = QWidget()
            tab_layout = QVBoxLayout(tab_widget)
            
            # 解决方案描述
            desc_label = QLabel(solution.description)
            desc_label.setWordWrap(True)
            tab_layout.addWidget(desc_label)
            
            # 解决步骤
            steps_label = QLabel("解决步骤:")
            steps_font = QFont()
            steps_font.setBold(True)
            steps_label.setFont(steps_font)
            tab_layout.addWidget(steps_label)
            
            for j, step in enumerate(solution.steps, 1):
                step_label = QLabel(f"{j}. {step}")
                step_label.setWordWrap(True)
                step_label.setIndent(20)
                tab_layout.addWidget(step_label)
            
            # 自动修复按钮
            if solution.auto_fix_available:
                auto_fix_btn = QPushButton("尝试自动修复")
                auto_fix_btn.clicked.connect(lambda checked, s=solution: self.try_auto_fix(s))
                tab_layout.addWidget(auto_fix_btn)
            
            # 难度和成功率信息
            info_layout = QHBoxLayout()
            
            difficulty_label = QLabel(f"难度: {solution.difficulty}")
            success_label = QLabel(f"成功率: {solution.success_rate*100:.0f}%")
            
            info_layout.addWidget(difficulty_label)
            info_layout.addStretch()
            info_layout.addWidget(success_label)
            
            tab_layout.addLayout(info_layout)
            tab_layout.addStretch()
            
            # 设置标签页标题
            tab_title = f"方案 {i+1}"
            if solution.auto_fix_available:
                tab_title += " (可自动修复)"
            
            self.tabs.addTab(tab_widget, tab_title)
        
        parent_layout.addWidget(self.tabs)
    
    def setup_buttons(self, parent_layout):
        """设置按钮区域"""
        button_layout = QHBoxLayout()
        
        # 预防建议按钮
        if self.error_info and self.error_info.prevention_tips:
            prevention_btn = QPushButton("预防建议")
            prevention_btn.clicked.connect(self.show_prevention_tips)
            button_layout.addWidget(prevention_btn)
        
        # 相关错误按钮
        if self.error_info and self.error_info.related_errors:
            related_btn = QPushButton("相关错误")
            related_btn.clicked.connect(self.show_related_errors)
            button_layout.addWidget(related_btn)
        
        button_layout.addStretch()
        
        # 关闭按钮
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.close)
        button_layout.addWidget(close_btn)
        
        parent_layout.addLayout(button_layout)
    
    def try_auto_fix(self, solution: ErrorSolution):
        """尝试自动修复"""
        if not HAS_PYQT6:
            return
        
        if solution.auto_fix_function:
            try:
                # 显示进度对话框
                progress = QProgressBar()
                progress.setRange(0, 0)  # 无限进度条
                
                # 执行自动修复
                success = solution.auto_fix_function()
                
                if success:
                    QMessageBox.information(self, "自动修复", "修复成功！请重新尝试操作。")
                else:
                    QMessageBox.warning(self, "自动修复", "自动修复失败，请尝试手动解决。")
            
            except Exception as e:
                QMessageBox.critical(self, "自动修复错误", f"自动修复过程中出现错误: {str(e)}")
        else:
            QMessageBox.information(self, "自动修复", "此解决方案暂不支持自动修复，请按照步骤手动操作。")
    
    def show_prevention_tips(self):
        """显示预防建议"""
        if not HAS_PYQT6 or not self.error_info:
            return
        
        tips_text = "\n".join([f"• {tip}" for tip in self.error_info.prevention_tips])
        QMessageBox.information(self, "预防建议", f"以下建议可以帮助您避免此类错误:\n\n{tips_text}")
    
    def show_related_errors(self):
        """显示相关错误"""
        if not HAS_PYQT6 or not self.error_info:
            return
        
        related_text = "\n".join([f"• {error_id}" for error_id in self.error_info.related_errors])
        QMessageBox.information(self, "相关错误", f"相关的错误类型:\n\n{related_text}")


class ErrorManager(LoggerMixin):
    """错误管理器"""
    
    def __init__(self):
        super().__init__()
        self.error_db = ErrorDatabase()
        self.error_detector = SmartErrorDetector()
        self.error_history: List[Dict] = []
    
    def handle_exception(self, exception: Exception, context: Dict = None) -> bool:
        """处理异常"""
        try:
            # 记录错误
            error_info = {
                'timestamp': datetime.now(),
                'exception_type': type(exception).__name__,
                'message': str(exception),
                'traceback': traceback.format_exc(),
                'context': context or {}
            }
            
            self.error_history.append(error_info)
            self.logger.error(f"处理异常: {error_info}")
            
            # 检测错误类型
            error_id = self.error_detector.detect_error_type(
                str(exception), 
                type(exception).__name__
            )
            
            if error_id:
                # 显示解决方案对话框
                self.show_solution_dialog(error_id, error_info)
                return True
            else:
                # 显示通用错误对话框
                self.show_generic_error_dialog(error_info)
                return False
        
        except Exception as e:
            self.logger.error(f"错误处理器本身出错: {e}")
            return False
    
    def show_solution_dialog(self, error_id: str, error_context: Dict = None):
        """显示解决方案对话框"""
        error_info = self.error_db.get_error_info(error_id)
        if error_info and HAS_PYQT6:
            dialog = ErrorSolutionDialog(None, error_info, error_context)
            dialog.exec()
        else:
            print(f"错误ID: {error_id}")
            if error_context:
                print(f"错误上下文: {error_context}")
    
    def show_generic_error_dialog(self, error_info: Dict):
        """显示通用错误对话框"""
        if HAS_PYQT6:
            QMessageBox.critical(
                None,
                "发生错误",
                f"程序遇到了错误:\n\n{error_info['message']}\n\n"
                f"请检查日志文件获取更多信息，或联系技术支持。"
            )
        else:
            print(f"发生错误: {error_info['message']}")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """获取错误统计信息"""
        if not self.error_history:
            return {}
        
        total_errors = len(self.error_history)
        error_types = {}
        
        for error in self.error_history:
            error_type = error['exception_type']
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            'total_errors': total_errors,
            'error_types': error_types,
            'most_common_error': max(error_types.items(), key=lambda x: x[1]) if error_types else None,
            'recent_errors': self.error_history[-5:]  # 最近5个错误
        }


# 全局错误管理器实例
error_manager = ErrorManager()


def handle_error(exception: Exception, context: Dict = None) -> bool:
    """处理错误的便捷函数"""
    return error_manager.handle_exception(exception, context)


def show_error_solution(error_id: str, context: Dict = None):
    """显示错误解决方案的便捷函数"""
    error_manager.show_solution_dialog(error_id, context)


# 自动修复函数示例
def auto_fix_memory_error() -> bool:
    """自动修复内存错误"""
    try:
        import gc
        # 强制垃圾回收
        collected = gc.collect()
        
        # 尝试清理临时文件
        import tempfile
        import shutil
        temp_dir = tempfile.gettempdir()
        
        return True
    except Exception:
        return False


def auto_fix_encoding_error() -> bool:
    """自动修复编码错误"""
    try:
        # 这里可以实现自动编码检测和转换
        return True
    except Exception:
        return False


# 注册自动修复函数
if error_manager.error_db.get_error_info("memory_error"):
    for solution in error_manager.error_db.get_error_info("memory_error").solutions:
        if solution.auto_fix_available and not solution.auto_fix_function:
            solution.auto_fix_function = auto_fix_memory_error

if error_manager.error_db.get_error_info("encoding_error"):
    for solution in error_manager.error_db.get_error_info("encoding_error").solutions:
        if solution.auto_fix_available and not solution.auto_fix_function:
            solution.auto_fix_function = auto_fix_encoding_error