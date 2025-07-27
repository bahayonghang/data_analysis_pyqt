"""
内置帮助系统 - 提供上下文相关的帮助和工具提示

功能：
1. 上下文相关帮助
2. 工具提示和快捷提示
3. 快速入门引导
4. 错误信息解释
"""

import json
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

try:
    from PyQt6.QtWidgets import (
        QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
        QTextBrowser, QTreeWidget, QTreeWidgetItem, QSplitter,
        QWidget, QFrame, QTabWidget, QScrollArea, QToolTip
    )
    from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QPoint
    from PyQt6.QtGui import QFont, QPixmap, QIcon, QPalette
    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False
    # 模拟类定义
    class QDialog:
        def __init__(self, parent=None): pass
    class QWidget:
        def __init__(self, parent=None): pass
    class QVBoxLayout:
        def __init__(self): pass

from ..utils.basic_logging import LoggerMixin


class HelpCategory(str, Enum):
    """帮助类别"""
    GETTING_STARTED = "getting_started"
    DATA_UPLOAD = "data_upload"
    DATA_ANALYSIS = "data_analysis"
    RESULTS_VIEWING = "results_viewing"
    EXPORT_REPORTS = "export_reports"
    SETTINGS = "settings"
    TROUBLESHOOTING = "troubleshooting"
    FAQ = "faq"


@dataclass
class HelpItem:
    """帮助项目"""
    id: str
    title: str
    content: str
    category: HelpCategory
    keywords: List[str]
    related_items: List[str] = None
    difficulty_level: str = "beginner"  # beginner, intermediate, advanced
    
    def __post_init__(self):
        if self.related_items is None:
            self.related_items = []


class HelpContent:
    """帮助内容管理"""
    
    def __init__(self):
        self.help_items: Dict[str, HelpItem] = {}
        self._load_default_content()
    
    def _load_default_content(self):
        """加载默认帮助内容"""
        default_items = [
            # 快速开始
            HelpItem(
                id="quick_start",
                title="快速开始",
                content="""
<h2>欢迎使用数据分析应用！</h2>

<h3>第一步：上传数据</h3>
<ol>
<li>点击"数据上传"标签页</li>
<li>选择您的数据文件（支持CSV、Excel、Parquet格式）</li>
<li>等待文件上传和验证完成</li>
</ol>

<h3>第二步：配置分析</h3>
<ol>
<li>切换到"数据分析"标签页</li>
<li>选择需要进行的分析类型</li>
<li>调整分析参数（可选）</li>
<li>点击"开始分析"</li>
</ol>

<h3>第三步：查看结果</h3>
<ol>
<li>分析完成后自动显示结果</li>
<li>查看统计表格和可视化图表</li>
<li>使用交互功能探索数据</li>
</ol>

<h3>第四步：导出报告</h3>
<ol>
<li>点击"导出报告"按钮</li>
<li>选择导出格式和选项</li>
<li>保存到指定位置</li>
</ol>

<p><strong>提示</strong>：首次使用建议从小数据集开始熟悉操作流程。</p>
                """,
                category=HelpCategory.GETTING_STARTED,
                keywords=["开始", "入门", "第一次", "新手"]
            ),
            
            # 数据上传
            HelpItem(
                id="file_upload",
                title="如何上传数据文件",
                content="""
<h2>数据文件上传指南</h2>

<h3>支持的文件格式</h3>
<ul>
<li><strong>CSV文件</strong> (.csv) - 最大500MB</li>
<li><strong>Excel文件</strong> (.xlsx, .xls) - 最大100MB</li>
<li><strong>Parquet文件</strong> (.parquet) - 最大1GB</li>
</ul>

<h3>上传方式</h3>
<ol>
<li><strong>点击选择</strong>：点击"选择文件"按钮</li>
<li><strong>拖拽上传</strong>：直接将文件拖拽到上传区域</li>
</ol>

<h3>数据格式要求</h3>
<ul>
<li>第一行应为列标题</li>
<li>使用UTF-8编码（避免中文乱码）</li>
<li>数值列应为纯数字格式</li>
<li>日期列使用标准格式：YYYY-MM-DD 或 YYYY-MM-DD HH:mm:ss</li>
</ul>

<h3>常见问题</h3>
<p><strong>Q: 上传时出现编码错误？</strong><br>
A: 请确保文件使用UTF-8编码保存，或在上传时手动选择正确编码。</p>

<p><strong>Q: 文件太大无法上传？</strong><br>
A: 可以尝试数据采样或分割文件，也可以使用Parquet格式减小文件大小。</p>
                """,
                category=HelpCategory.DATA_UPLOAD,
                keywords=["上传", "文件", "格式", "CSV", "Excel"]
            ),
            
            # 数据分析
            HelpItem(
                id="analysis_types",
                title="分析类型说明",
                content="""
<h2>可用的分析类型</h2>

<h3>描述性统计分析</h3>
<p>计算数据的基本统计特征：</p>
<ul>
<li><strong>集中趋势</strong>：均值、中位数、众数</li>
<li><strong>离散程度</strong>：标准差、方差、极差</li>
<li><strong>分布形状</strong>：偏度、峰度</li>
<li><strong>分位数</strong>：25%、50%、75%分位数</li>
</ul>

<h3>相关性分析</h3>
<p>探索变量之间的关系：</p>
<ul>
<li><strong>Pearson相关</strong>：衡量线性相关关系</li>
<li><strong>Spearman相关</strong>：衡量单调相关关系</li>
<li><strong>相关矩阵</strong>：所有变量间的相关系数</li>
</ul>

<h3>异常值检测</h3>
<p>识别数据中的异常观测值：</p>
<ul>
<li><strong>Z-score方法</strong>：基于标准差的检测</li>
<li><strong>IQR方法</strong>：基于四分位距的检测</li>
<li><strong>可视化标记</strong>：在图表中高亮异常值</li>
</ul>

<h3>时间序列分析</h3>
<p>分析时间相关的数据模式：</p>
<ul>
<li><strong>趋势分析</strong>：长期变化方向</li>
<li><strong>季节性分析</strong>：周期性模式</li>
<li><strong>平稳性检验</strong>：ADF检验</li>
</ul>

<h3>分析参数设置</h3>
<p>可以调整的主要参数：</p>
<ul>
<li><strong>异常值阈值</strong>：Z-score阈值（通常为2-4）</li>
<li><strong>相关性方法</strong>：选择Pearson或Spearman</li>
<li><strong>缺失值处理</strong>：删除或填充策略</li>
</ul>
                """,
                category=HelpCategory.DATA_ANALYSIS,
                keywords=["分析", "统计", "相关性", "异常值", "时间序列"]
            ),
            
            # 结果查看
            HelpItem(
                id="reading_results",
                title="如何解读分析结果",
                content="""
<h2>分析结果解读指南</h2>

<h3>描述性统计表格</h3>
<table border="1">
<tr><th>指标</th><th>含义</th><th>应用</th></tr>
<tr><td>均值 (Mean)</td><td>数据的平均值</td><td>了解数据中心位置</td></tr>
<tr><td>中位数 (Median)</td><td>数据排序后的中间值</td><td>不受极值影响的中心位置</td></tr>
<tr><td>标准差 (Std)</td><td>数据的离散程度</td><td>值越大，数据越分散</td></tr>
<tr><td>最小值/最大值</td><td>数据的范围</td><td>了解数据的变化范围</td></tr>
</table>

<h3>相关性系数解读</h3>
<ul>
<li><strong>0.8 - 1.0</strong>：强正相关</li>
<li><strong>0.6 - 0.8</strong>：中等正相关</li>
<li><strong>0.4 - 0.6</strong>：弱正相关</li>
<li><strong>-0.4 - 0.4</strong>：无明显相关</li>
<li><strong>-0.6 - -0.4</strong>：弱负相关</li>
<li><strong>-0.8 - -0.6</strong>：中等负相关</li>
<li><strong>-1.0 - -0.8</strong>：强负相关</li>
</ul>

<h3>可视化图表说明</h3>
<h4>直方图</h4>
<p>显示数据分布形状，帮助了解：</p>
<ul>
<li>数据是否呈正态分布</li>
<li>是否存在多个峰值</li>
<li>数据的偏斜方向</li>
</ul>

<h4>箱线图</h4>
<p>显示数据的四分位数信息：</p>
<ul>
<li>盒子表示25%-75%分位数范围</li>
<li>中间线表示中位数</li>
<li>触须表示正常数据范围</li>
<li>点表示异常值</li>
</ul>

<h4>散点图</h4>
<p>显示两个变量之间的关系：</p>
<ul>
<li>点的分布模式反映相关性强度</li>
<li>线性排列表示线性相关</li>
<li>聚集程度反映相关性大小</li>
</ul>

<h3>异常值分析</h3>
<p>识别到异常值时的处理建议：</p>
<ol>
<li><strong>验证数据</strong>：确认是否为数据录入错误</li>
<li><strong>分析原因</strong>：了解异常值产生的业务原因</li>
<li><strong>决定处理方式</strong>：保留、删除或修正</li>
</ol>
                """,
                category=HelpCategory.RESULTS_VIEWING,
                keywords=["结果", "解读", "统计", "图表", "异常值"]
            ),
            
            # 故障排除
            HelpItem(
                id="common_errors",
                title="常见错误解决方案",
                content="""
<h2>常见问题及解决方案</h2>

<h3>文件上传问题</h3>

<h4>错误：文件格式不支持</h4>
<p><strong>原因</strong>：文件格式不在支持范围内<br>
<strong>解决</strong>：转换为CSV、Excel或Parquet格式</p>

<h4>错误：编码错误</h4>
<p><strong>原因</strong>：文件编码不是UTF-8<br>
<strong>解决</strong>：使用文本编辑器转换为UTF-8编码</p>

<h4>错误：文件太大</h4>
<p><strong>原因</strong>：文件超过大小限制<br>
<strong>解决</strong>：数据采样或使用Parquet格式</p>

<h3>分析执行问题</h3>

<h4>错误：内存不足</h4>
<p><strong>症状</strong>：程序卡顿或崩溃<br>
<strong>解决</strong>：</p>
<ol>
<li>关闭其他程序释放内存</li>
<li>启用数据采样功能</li>
<li>分批处理大数据集</li>
</ol>

<h4>错误：数据类型错误</h4>
<p><strong>症状</strong>：分析结果异常或错误<br>
<strong>解决</strong>：</p>
<ol>
<li>检查数值列是否包含文字</li>
<li>确认日期列格式正确</li>
<li>处理缺失值和特殊字符</li>
</ol>

<h3>界面显示问题</h3>

<h4>问题：界面显示异常</h4>
<p><strong>症状</strong>：窗口布局错乱、按钮无响应<br>
<strong>解决</strong>：</p>
<ol>
<li>重启应用程序</li>
<li>检查显示缩放设置</li>
<li>重置界面到默认设置</li>
</ol>

<h4>问题：中文显示乱码</h4>
<p><strong>原因</strong>：字体或编码问题<br>
<strong>解决</strong>：</p>
<ol>
<li>确保系统安装了中文字体</li>
<li>检查文件编码设置</li>
<li>在设置中选择正确的语言</li>
</ol>

<h3>性能优化建议</h3>

<h4>提高处理速度</h4>
<ol>
<li>使用SSD硬盘提高I/O速度</li>
<li>增加系统内存</li>
<li>启用多线程处理</li>
<li>关闭不必要的后台程序</li>
</ol>

<h4>减少内存使用</h4>
<ol>
<li>处理完成后及时清理数据</li>
<li>使用数据采样减少内存占用</li>
<li>选择高效的数据格式</li>
<li>分批处理大型数据集</li>
</ol>

<h3>获取更多帮助</h3>
<p>如果问题仍未解决，请：</p>
<ol>
<li>查看详细的用户手册</li>
<li>访问在线帮助文档</li>
<li>提交问题反馈</li>
<li>联系技术支持</li>
</ol>
                """,
                category=HelpCategory.TROUBLESHOOTING,
                keywords=["错误", "问题", "解决", "故障", "排除"]
            )
        ]
        
        for item in default_items:
            self.help_items[item.id] = item
    
    def get_item(self, item_id: str) -> Optional[HelpItem]:
        """获取帮助项目"""
        return self.help_items.get(item_id)
    
    def search_items(self, query: str) -> List[HelpItem]:
        """搜索帮助项目"""
        query_lower = query.lower()
        results = []
        
        for item in self.help_items.values():
            # 搜索标题
            if query_lower in item.title.lower():
                results.append(item)
                continue
            
            # 搜索关键词
            if any(query_lower in keyword.lower() for keyword in item.keywords):
                results.append(item)
                continue
            
            # 搜索内容
            if query_lower in item.content.lower():
                results.append(item)
        
        return results
    
    def get_by_category(self, category: HelpCategory) -> List[HelpItem]:
        """按类别获取帮助项目"""
        return [item for item in self.help_items.values() if item.category == category]


class HelpDialog(QDialog, LoggerMixin):
    """帮助对话框"""
    
    def __init__(self, parent=None, initial_topic: str = None):
        if HAS_PYQT6:
            super().__init__(parent)
        else:
            QDialog.__init__(self, parent)
            LoggerMixin.__init__(self)
        
        self.help_content = HelpContent()
        self.setWindowTitle("帮助文档")
        self.setMinimumSize(900, 600)
        self.setup_ui()
        
        if initial_topic:
            self.show_topic(initial_topic)
    
    def setup_ui(self):
        """设置界面"""
        if not HAS_PYQT6:
            return
        
        layout = QHBoxLayout(self)
        
        # 创建分割器
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)
        
        # 左侧：导航树
        self.setup_navigation(splitter)
        
        # 右侧：内容显示
        self.setup_content_area(splitter)
        
        # 设置分割比例
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([250, 650])
    
    def setup_navigation(self, parent):
        """设置导航区域"""
        nav_widget = QWidget()
        nav_layout = QVBoxLayout(nav_widget)
        
        # 导航树
        self.nav_tree = QTreeWidget()
        self.nav_tree.setHeaderLabel("帮助主题")
        self.nav_tree.itemClicked.connect(self.on_nav_item_clicked)
        
        # 添加类别和项目
        self.populate_navigation()
        
        nav_layout.addWidget(self.nav_tree)
        parent.addWidget(nav_widget)
    
    def setup_content_area(self, parent):
        """设置内容区域"""
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        
        # 内容浏览器
        self.content_browser = QTextBrowser()
        self.content_browser.setOpenExternalLinks(True)
        content_layout.addWidget(self.content_browser)
        
        # 按钮区域
        button_layout = QHBoxLayout()
        
        self.back_button = QPushButton("返回")
        self.back_button.clicked.connect(self.content_browser.backward)
        self.back_button.setEnabled(False)
        
        self.forward_button = QPushButton("前进")
        self.forward_button.clicked.connect(self.content_browser.forward)
        self.forward_button.setEnabled(False)
        
        self.close_button = QPushButton("关闭")
        self.close_button.clicked.connect(self.close)
        
        button_layout.addWidget(self.back_button)
        button_layout.addWidget(self.forward_button)
        button_layout.addStretch()
        button_layout.addWidget(self.close_button)
        
        content_layout.addLayout(button_layout)
        
        # 连接导航信号
        self.content_browser.backwardAvailable.connect(self.back_button.setEnabled)
        self.content_browser.forwardAvailable.connect(self.forward_button.setEnabled)
        
        parent.addWidget(content_widget)
    
    def populate_navigation(self):
        """填充导航树"""
        if not HAS_PYQT6:
            return
        
        category_items = {}
        
        # 创建类别节点
        for category in HelpCategory:
            category_name = self.get_category_display_name(category)
            category_item = QTreeWidgetItem([category_name])
            category_item.setData(0, Qt.ItemDataRole.UserRole, f"category_{category.value}")
            self.nav_tree.addTopLevelItem(category_item)
            category_items[category] = category_item
        
        # 添加帮助项目
        for item in self.help_content.help_items.values():
            if item.category in category_items:
                item_widget = QTreeWidgetItem([item.title])
                item_widget.setData(0, Qt.ItemDataRole.UserRole, item.id)
                category_items[item.category].addChild(item_widget)
        
        # 展开所有类别
        self.nav_tree.expandAll()
    
    def get_category_display_name(self, category: HelpCategory) -> str:
        """获取类别显示名称"""
        names = {
            HelpCategory.GETTING_STARTED: "快速开始",
            HelpCategory.DATA_UPLOAD: "数据上传",
            HelpCategory.DATA_ANALYSIS: "数据分析",
            HelpCategory.RESULTS_VIEWING: "结果查看",
            HelpCategory.EXPORT_REPORTS: "报告导出",
            HelpCategory.SETTINGS: "设置配置",
            HelpCategory.TROUBLESHOOTING: "故障排除",
            HelpCategory.FAQ: "常见问题"
        }
        return names.get(category, category.value)
    
    def on_nav_item_clicked(self, item, column):
        """导航项目点击事件"""
        if not HAS_PYQT6:
            return
        
        item_data = item.data(0, Qt.ItemDataRole.UserRole)
        if item_data and not item_data.startswith("category_"):
            self.show_topic(item_data)
    
    def show_topic(self, topic_id: str):
        """显示帮助主题"""
        if not HAS_PYQT6:
            return
        
        help_item = self.help_content.get_item(topic_id)
        if help_item:
            self.content_browser.setHtml(help_item.content)
            self.setWindowTitle(f"帮助文档 - {help_item.title}")


class ToolTipManager(LoggerMixin):
    """工具提示管理器"""
    
    def __init__(self):
        super().__init__()
        self.tooltip_data = self._load_tooltip_data()
    
    def _load_tooltip_data(self) -> Dict[str, str]:
        """加载工具提示数据"""
        return {
            # 数据上传相关
            "upload_button": "点击选择要分析的数据文件，支持CSV、Excel、Parquet格式",
            "file_drag_area": "将数据文件拖拽到此区域进行上传",
            "encoding_selector": "选择文件的文本编码格式，通常使用UTF-8",
            
            # 分析配置相关
            "descriptive_stats_checkbox": "勾选以计算数据的描述性统计指标（均值、标准差等）",
            "correlation_checkbox": "勾选以分析变量之间的相关关系",
            "outlier_detection_checkbox": "勾选以检测数据中的异常值",
            "outlier_method_combo": "选择异常值检测方法：Z-score基于标准差，IQR基于四分位距",
            "outlier_threshold_spin": "设置异常值检测的阈值，数值越小检测越严格",
            
            # 结果查看相关
            "stats_table": "显示各变量的描述性统计结果，包括均值、标准差、分位数等",
            "correlation_heatmap": "相关性热力图，颜色深浅表示相关强度",
            "chart_export_button": "导出当前图表为图片文件",
            
            # 通用操作
            "start_analysis_button": "开始执行数据分析，请确保已上传数据并配置参数",
            "export_report_button": "导出完整的分析报告，包括统计结果和图表",
            "clear_data_button": "清除当前数据，准备分析新的数据集",
            
            # 设置相关
            "theme_selector": "选择应用程序的视觉主题",
            "language_selector": "选择界面显示语言",
            "memory_limit_spin": "设置最大内存使用限制，防止系统卡顿",
            
            # 错误和警告
            "file_size_warning": "文件较大，处理时间可能较长，建议启用数据采样",
            "memory_warning": "内存使用较高，建议关闭其他程序或减小数据集",
            "no_numeric_columns": "数据中没有数值列，无法进行数值分析",
        }
    
    def get_tooltip(self, widget_id: str) -> str:
        """获取工具提示文本"""
        return self.tooltip_data.get(widget_id, "")
    
    def set_tooltip(self, widget, widget_id: str):
        """为控件设置工具提示"""
        if not HAS_PYQT6:
            return
        
        tooltip_text = self.get_tooltip(widget_id)
        if tooltip_text and hasattr(widget, 'setToolTip'):
            widget.setToolTip(tooltip_text)
    
    def show_contextual_tip(self, widget, message: str, duration: int = 3000):
        """显示上下文提示"""
        if not HAS_PYQT6:
            return
        
        try:
            # 获取控件的全局位置
            global_pos = widget.mapToGlobal(QPoint(0, widget.height() + 5))
            QToolTip.showText(global_pos, message, widget, widget.rect(), duration)
        except Exception as e:
            self.logger.error(f"显示上下文提示失败: {e}")


class QuickStartGuide(QDialog, LoggerMixin):
    """快速入门引导"""
    
    def __init__(self, parent=None):
        if HAS_PYQT6:
            super().__init__(parent)
        else:
            QDialog.__init__(self, parent)
            LoggerMixin.__init__(self)
        
        self.current_step = 0
        self.total_steps = 4
        self.setWindowTitle("快速入门引导")
        self.setFixedSize(600, 400)
        self.setup_ui()
    
    def setup_ui(self):
        """设置界面"""
        if not HAS_PYQT6:
            return
        
        layout = QVBoxLayout(self)
        
        # 标题
        title_label = QLabel("欢迎使用数据分析应用")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        # 内容区域
        self.content_widget = QTabWidget()
        self.setup_guide_steps()
        layout.addWidget(self.content_widget)
        
        # 按钮区域
        button_layout = QHBoxLayout()
        
        self.prev_button = QPushButton("上一步")
        self.prev_button.clicked.connect(self.prev_step)
        self.prev_button.setEnabled(False)
        
        self.next_button = QPushButton("下一步")
        self.next_button.clicked.connect(self.next_step)
        
        self.finish_button = QPushButton("完成")
        self.finish_button.clicked.connect(self.close)
        self.finish_button.setVisible(False)
        
        self.skip_button = QPushButton("跳过引导")
        self.skip_button.clicked.connect(self.close)
        
        button_layout.addWidget(self.prev_button)
        button_layout.addWidget(self.next_button)
        button_layout.addWidget(self.finish_button)
        button_layout.addStretch()
        button_layout.addWidget(self.skip_button)
        
        layout.addLayout(button_layout)
    
    def setup_guide_steps(self):
        """设置引导步骤"""
        if not HAS_PYQT6:
            return
        
        steps = [
            {
                "title": "步骤 1: 数据上传",
                "content": """
<h3>上传您的数据文件</h3>
<p>首先，您需要上传要分析的数据文件。</p>
<ol>
<li>点击"数据上传"标签页</li>
<li>选择CSV、Excel或Parquet格式的文件</li>
<li>等待文件上传和验证完成</li>
</ol>
<p><strong>提示</strong>：建议首次使用时选择较小的数据文件（< 10MB）来熟悉操作。</p>
                """
            },
            {
                "title": "步骤 2: 配置分析",
                "content": """
<h3>选择分析类型和参数</h3>
<p>根据您的需求配置分析参数。</p>
<ol>
<li>切换到"数据分析"标签页</li>
<li>选择需要的分析类型：
   <ul>
   <li>描述性统计：了解数据基本特征</li>
   <li>相关性分析：探索变量关系</li>
   <li>异常值检测：识别异常数据</li>
   </ul>
</li>
<li>调整分析参数（可选）</li>
<li>点击"开始分析"</li>
</ol>
                """
            },
            {
                "title": "步骤 3: 查看结果",
                "content": """
<h3>理解和探索分析结果</h3>
<p>分析完成后，您可以查看详细的结果。</p>
<ul>
<li><strong>统计表格</strong>：查看数值指标</li>
<li><strong>可视化图表</strong>：直观理解数据分布和关系</li>
<li><strong>交互功能</strong>：放大、平移图表进行深入探索</li>
</ul>
<p><strong>技巧</strong>：点击图表中的数据点可以查看详细数值。</p>
                """
            },
            {
                "title": "步骤 4: 导出报告",
                "content": """
<h3>保存和分享您的分析结果</h3>
<p>完成分析后，您可以导出结果进行保存或分享。</p>
<ol>
<li>点击"导出报告"按钮</li>
<li>选择导出格式：
   <ul>
   <li>PDF报告：完整的分析报告</li>
   <li>数据文件：处理后的数据</li>
   <li>图表文件：可视化结果</li>
   </ul>
</li>
<li>选择保存位置并确认</li>
</ol>
<p><strong>完成</strong>：恭喜！您已经掌握了基本的使用流程。</p>
                """
            }
        ]
        
        for i, step in enumerate(steps):
            scroll_area = QScrollArea()
            content_widget = QWidget()
            content_layout = QVBoxLayout(content_widget)
            
            content_browser = QTextBrowser()
            content_browser.setHtml(step["content"])
            content_layout.addWidget(content_browser)
            
            scroll_area.setWidget(content_widget)
            scroll_area.setWidgetResizable(True)
            
            self.content_widget.addTab(scroll_area, step["title"])
    
    def next_step(self):
        """下一步"""
        if not HAS_PYQT6:
            return
        
        if self.current_step < self.total_steps - 1:
            self.current_step += 1
            self.content_widget.setCurrentIndex(self.current_step)
            self.update_buttons()
    
    def prev_step(self):
        """上一步"""
        if not HAS_PYQT6:
            return
        
        if self.current_step > 0:
            self.current_step -= 1
            self.content_widget.setCurrentIndex(self.current_step)
            self.update_buttons()
    
    def update_buttons(self):
        """更新按钮状态"""
        if not HAS_PYQT6:
            return
        
        self.prev_button.setEnabled(self.current_step > 0)
        
        if self.current_step >= self.total_steps - 1:
            self.next_button.setVisible(False)
            self.finish_button.setVisible(True)
        else:
            self.next_button.setVisible(True)
            self.finish_button.setVisible(False)


def show_help_dialog(parent=None, topic: str = None):
    """显示帮助对话框"""
    if not HAS_PYQT6:
        print(f"帮助主题: {topic or '主帮助'}")
        return None
    
    dialog = HelpDialog(parent, topic)
    dialog.show()
    return dialog


def show_quick_start_guide(parent=None):
    """显示快速入门引导"""
    if not HAS_PYQT6:
        print("快速入门引导")
        return None
    
    guide = QuickStartGuide(parent)
    guide.exec()
    return guide


# 全局工具提示管理器实例
tooltip_manager = ToolTipManager()


def set_widget_tooltip(widget, widget_id: str):
    """为控件设置工具提示的便捷函数"""
    tooltip_manager.set_tooltip(widget, widget_id)


def show_contextual_help(widget, message: str, duration: int = 3000):
    """显示上下文帮助的便捷函数"""
    tooltip_manager.show_contextual_tip(widget, message, duration)