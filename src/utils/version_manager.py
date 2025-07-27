"""
版本检查和自动更新系统

提供应用程序版本检查、更新通知和自动更新功能
"""

import asyncio
import json
import os
import hashlib
import platform
import tempfile
import zipfile
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import logging

try:
    import aiohttp
    import aiofiles
    from PyQt6.QtWidgets import (
        QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
        QProgressBar, QTextEdit, QCheckBox, QMessageBox, QFrame
    )
    from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
    from PyQt6.QtGui import QFont, QPixmap
    HAS_DEPENDENCIES = True
except ImportError:
    HAS_DEPENDENCIES = False
    # 模拟类定义
    class QDialog:
        def __init__(self, parent=None): pass
    class QThread:
        def __init__(self, parent=None): pass
    pyqtSignal = lambda *args: lambda func: func


class UpdateChannel(str, Enum):
    """更新渠道"""
    STABLE = "stable"
    BETA = "beta"
    DEV = "dev"


class UpdateStatus(str, Enum):
    """更新状态"""
    UP_TO_DATE = "up_to_date"
    UPDATE_AVAILABLE = "update_available"
    CRITICAL_UPDATE = "critical_update"
    ERROR = "error"
    CHECKING = "checking"


@dataclass
class VersionInfo:
    """版本信息"""
    version: str
    build: str
    release_date: datetime
    channel: UpdateChannel
    download_url: str
    size: int
    checksum: str
    changelog: List[str]
    critical: bool = False
    min_compatible_version: str = "1.0.0"


@dataclass
class UpdateConfig:
    """更新配置"""
    server_url: str = "https://api.example.com/updates"
    channel: UpdateChannel = UpdateChannel.STABLE
    auto_check: bool = True
    auto_download: bool = False
    auto_install: bool = False
    check_interval: int = 86400  # 24小时
    download_timeout: int = 300  # 5分钟
    retry_count: int = 3
    backup_enabled: bool = True


class VersionChecker:
    """版本检查器"""
    
    def __init__(self, current_version: str, config: UpdateConfig):
        self.current_version = current_version
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    async def check_for_updates(self) -> Tuple[UpdateStatus, Optional[VersionInfo]]:
        """检查更新"""
        try:
            self.logger.info(f"检查更新中... 当前版本: {self.current_version}")
            
            # 构建请求URL
            url = f"{self.config.server_url}/check"
            params = {
                'current_version': self.current_version,
                'channel': self.config.channel.value,
                'platform': platform.system().lower(),
                'arch': platform.machine().lower()
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=30) as response:
                    if response.status != 200:
                        self.logger.error(f"更新检查失败: HTTP {response.status}")
                        return UpdateStatus.ERROR, None
                    
                    data = await response.json()
                    
                    if not data.get('update_available', False):
                        self.logger.info("已是最新版本")
                        return UpdateStatus.UP_TO_DATE, None
                    
                    # 解析版本信息
                    version_info = VersionInfo(
                        version=data['version'],
                        build=data['build'],
                        release_date=datetime.fromisoformat(data['release_date']),
                        channel=UpdateChannel(data['channel']),
                        download_url=data['download_url'],
                        size=data['size'],
                        checksum=data['checksum'],
                        changelog=data.get('changelog', []),
                        critical=data.get('critical', False),
                        min_compatible_version=data.get('min_compatible_version', '1.0.0')
                    )
                    
                    status = UpdateStatus.CRITICAL_UPDATE if version_info.critical else UpdateStatus.UPDATE_AVAILABLE
                    self.logger.info(f"发现新版本: {version_info.version} (状态: {status.value})")
                    
                    return status, version_info
        
        except asyncio.TimeoutError:
            self.logger.error("更新检查超时")
            return UpdateStatus.ERROR, None
        except Exception as e:
            self.logger.error(f"更新检查失败: {e}")
            return UpdateStatus.ERROR, None
    
    def is_version_compatible(self, version_info: VersionInfo) -> bool:
        """检查版本兼容性"""
        try:
            current_parts = [int(x) for x in self.current_version.split('.')]
            min_parts = [int(x) for x in version_info.min_compatible_version.split('.')]
            
            # 比较版本号
            for i in range(max(len(current_parts), len(min_parts))):
                current = current_parts[i] if i < len(current_parts) else 0
                minimum = min_parts[i] if i < len(min_parts) else 0
                
                if current > minimum:
                    return True
                elif current < minimum:
                    return False
            
            return True
        except Exception as e:
            self.logger.error(f"版本兼容性检查失败: {e}")
            return False


class UpdateDownloader:
    """更新下载器"""
    
    def __init__(self, config: UpdateConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    async def download_update(self, version_info: VersionInfo, 
                            progress_callback=None) -> Optional[Path]:
        """下载更新"""
        try:
            self.logger.info(f"开始下载更新: {version_info.version}")
            
            # 创建临时下载目录
            download_dir = Path(tempfile.gettempdir()) / "data_analysis_updates"
            download_dir.mkdir(exist_ok=True)
            
            filename = f"DataAnalysisApp-{version_info.version}.zip"
            download_path = download_dir / filename
            
            # 下载文件
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    version_info.download_url,
                    timeout=self.config.download_timeout
                ) as response:
                    
                    if response.status != 200:
                        self.logger.error(f"下载失败: HTTP {response.status}")
                        return None
                    
                    total_size = int(response.headers.get('content-length', 0))
                    downloaded = 0
                    
                    async with aiofiles.open(download_path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(8192):
                            await f.write(chunk)
                            downloaded += len(chunk)
                            
                            if progress_callback and total_size > 0:
                                progress = int((downloaded / total_size) * 100)
                                progress_callback(progress)
            
            # 验证文件完整性
            if not self._verify_checksum(download_path, version_info.checksum):
                self.logger.error("文件校验失败")
                download_path.unlink(missing_ok=True)
                return None
            
            self.logger.info(f"更新下载完成: {download_path}")
            return download_path
        
        except Exception as e:
            self.logger.error(f"下载更新失败: {e}")
            return None
    
    def _verify_checksum(self, file_path: Path, expected_checksum: str) -> bool:
        """验证文件校验和"""
        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            
            actual_checksum = sha256_hash.hexdigest()
            return actual_checksum.lower() == expected_checksum.lower()
        
        except Exception as e:
            self.logger.error(f"校验和验证失败: {e}")
            return False


class UpdateInstaller:
    """更新安装器"""
    
    def __init__(self, config: UpdateConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    async def install_update(self, update_file: Path, 
                           progress_callback=None) -> bool:
        """安装更新"""
        try:
            self.logger.info(f"开始安装更新: {update_file}")
            
            # 获取应用程序目录
            app_dir = Path(__file__).parent.parent
            
            # 创建备份
            if self.config.backup_enabled:
                backup_path = await self._create_backup(app_dir)
                if not backup_path:
                    self.logger.error("创建备份失败")
                    return False
            
            # 解压更新文件
            extract_dir = update_file.parent / "extract"
            extract_dir.mkdir(exist_ok=True)
            
            with zipfile.ZipFile(update_file, 'r') as zip_file:
                total_files = len(zip_file.namelist())
                
                for i, member in enumerate(zip_file.namelist()):
                    zip_file.extract(member, extract_dir)
                    
                    if progress_callback:
                        progress = int(((i + 1) / total_files) * 100)
                        progress_callback(progress)
            
            # 应用更新
            await self._apply_update(extract_dir, app_dir)
            
            # 清理临时文件
            self._cleanup_temp_files(extract_dir, update_file)
            
            self.logger.info("更新安装完成")
            return True
        
        except Exception as e:
            self.logger.error(f"安装更新失败: {e}")
            return False
    
    async def _create_backup(self, app_dir: Path) -> Optional[Path]:
        """创建应用程序备份"""
        try:
            backup_dir = app_dir.parent / "backups"
            backup_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = backup_dir / f"backup_{timestamp}.zip"
            
            with zipfile.ZipFile(backup_file, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for file_path in app_dir.rglob('*'):
                    if file_path.is_file():
                        arcname = file_path.relative_to(app_dir)
                        zip_file.write(file_path, arcname)
            
            self.logger.info(f"备份创建完成: {backup_file}")
            return backup_file
        
        except Exception as e:
            self.logger.error(f"创建备份失败: {e}")
            return None
    
    async def _apply_update(self, extract_dir: Path, app_dir: Path):
        """应用更新文件"""
        import shutil
        
        for item in extract_dir.rglob('*'):
            if item.is_file():
                rel_path = item.relative_to(extract_dir)
                target_path = app_dir / rel_path
                
                # 确保目标目录存在
                target_path.parent.mkdir(parents=True, exist_ok=True)
                
                # 复制文件
                shutil.copy2(item, target_path)
    
    def _cleanup_temp_files(self, extract_dir: Path, update_file: Path):
        """清理临时文件"""
        import shutil
        
        try:
            if extract_dir.exists():
                shutil.rmtree(extract_dir)
            
            if update_file.exists():
                update_file.unlink()
        
        except Exception as e:
            self.logger.warning(f"清理临时文件失败: {e}")


class UpdateDialog(QDialog):
    """更新对话框"""
    
    def __init__(self, parent=None, version_info: VersionInfo = None, 
                 status: UpdateStatus = UpdateStatus.UPDATE_AVAILABLE):
        if HAS_DEPENDENCIES:
            super().__init__(parent)
        else:
            return
        
        self.version_info = version_info
        self.status = status
        self.setWindowTitle("软件更新")
        self.setMinimumSize(500, 400)
        self.setup_ui()
    
    def setup_ui(self):
        """设置界面"""
        layout = QVBoxLayout(self)
        
        # 状态信息
        self.setup_status_info(layout)
        
        # 版本信息
        if self.version_info:
            self.setup_version_info(layout)
        
        # 更新日志
        if self.version_info and self.version_info.changelog:
            self.setup_changelog(layout)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # 按钮
        self.setup_buttons(layout)
    
    def setup_status_info(self, parent_layout):
        """设置状态信息"""
        status_frame = QFrame()
        status_layout = QVBoxLayout(status_frame)
        
        # 状态标题
        status_messages = {
            UpdateStatus.UP_TO_DATE: "您的软件已是最新版本",
            UpdateStatus.UPDATE_AVAILABLE: "发现新版本更新",
            UpdateStatus.CRITICAL_UPDATE: "发现重要安全更新",
            UpdateStatus.ERROR: "检查更新时出现错误"
        }
        
        title_label = QLabel(status_messages.get(self.status, "更新状态"))
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(14)
        title_label.setFont(title_font)
        
        # 根据状态设置颜色
        colors = {
            UpdateStatus.UP_TO_DATE: "green",
            UpdateStatus.UPDATE_AVAILABLE: "blue", 
            UpdateStatus.CRITICAL_UPDATE: "red",
            UpdateStatus.ERROR: "red"
        }
        color = colors.get(self.status, "black")
        title_label.setStyleSheet(f"color: {color};")
        
        status_layout.addWidget(title_label)
        parent_layout.addWidget(status_frame)
    
    def setup_version_info(self, parent_layout):
        """设置版本信息"""
        info_frame = QFrame()
        info_frame.setFrameStyle(QFrame.Shape.Box)
        info_layout = QVBoxLayout(info_frame)
        
        # 版本号
        version_label = QLabel(f"新版本: {self.version_info.version}")
        version_font = QFont()
        version_font.setBold(True)
        version_label.setFont(version_font)
        info_layout.addWidget(version_label)
        
        # 构建号
        build_label = QLabel(f"构建: {self.version_info.build}")
        info_layout.addWidget(build_label)
        
        # 发布日期
        date_str = self.version_info.release_date.strftime("%Y年%m月%d日")
        date_label = QLabel(f"发布日期: {date_str}")
        info_layout.addWidget(date_label)
        
        # 文件大小
        size_mb = self.version_info.size / (1024 * 1024)
        size_label = QLabel(f"下载大小: {size_mb:.1f} MB")
        info_layout.addWidget(size_label)
        
        # 更新渠道
        channel_label = QLabel(f"更新渠道: {self.version_info.channel.value}")
        info_layout.addWidget(channel_label)
        
        parent_layout.addWidget(info_frame)
    
    def setup_changelog(self, parent_layout):
        """设置更新日志"""
        changelog_label = QLabel("更新内容:")
        changelog_font = QFont()
        changelog_font.setBold(True)
        changelog_label.setFont(changelog_font)
        parent_layout.addWidget(changelog_label)
        
        changelog_text = QTextEdit()
        changelog_text.setMaximumHeight(150)
        changelog_content = "\n".join([f"• {item}" for item in self.version_info.changelog])
        changelog_text.setPlainText(changelog_content)
        changelog_text.setReadOnly(True)
        parent_layout.addWidget(changelog_text)
    
    def setup_buttons(self, parent_layout):
        """设置按钮"""
        button_layout = QHBoxLayout()
        
        if self.status in [UpdateStatus.UPDATE_AVAILABLE, UpdateStatus.CRITICAL_UPDATE]:
            # 下载并安装按钮
            self.download_button = QPushButton("下载并安装")
            self.download_button.clicked.connect(self.start_download)
            button_layout.addWidget(self.download_button)
            
            # 仅下载按钮
            self.download_only_button = QPushButton("仅下载")
            self.download_only_button.clicked.connect(self.download_only)
            button_layout.addWidget(self.download_only_button)
            
            # 自动更新选项
            self.auto_update_checkbox = QCheckBox("以后自动下载并安装更新")
            parent_layout.addWidget(self.auto_update_checkbox)
        
        # 稍后提醒按钮
        if self.status == UpdateStatus.UPDATE_AVAILABLE:
            self.remind_button = QPushButton("稍后提醒")
            self.remind_button.clicked.connect(self.remind_later)
            button_layout.addWidget(self.remind_button)
        
        # 关闭按钮
        self.close_button = QPushButton("关闭")
        self.close_button.clicked.connect(self.close)
        button_layout.addWidget(self.close_button)
        
        parent_layout.addLayout(button_layout)
    
    def start_download(self):
        """开始下载并安装"""
        # 这里应该启动下载和安装过程
        self.progress_bar.setVisible(True)
        self.download_button.setEnabled(False)
        # 实际实现需要连接到UpdateManager
        pass
    
    def download_only(self):
        """仅下载"""
        # 这里应该仅启动下载过程
        self.progress_bar.setVisible(True)
        self.download_only_button.setEnabled(False)
        # 实际实现需要连接到UpdateManager
        pass
    
    def remind_later(self):
        """稍后提醒"""
        # 设置稍后提醒
        self.close()


class UpdateManager:
    """更新管理器"""
    
    def __init__(self, current_version: str, config: UpdateConfig = None):
        self.current_version = current_version
        self.config = config or UpdateConfig()
        
        self.checker = VersionChecker(current_version, self.config)
        self.downloader = UpdateDownloader(self.config)
        self.installer = UpdateInstaller(self.config)
        
        self.logger = logging.getLogger(__name__)
        
        # 自动检查定时器
        if HAS_DEPENDENCIES:
            self.check_timer = QTimer()
            self.check_timer.timeout.connect(self.check_for_updates_async)
            
            if self.config.auto_check:
                self.start_auto_check()
    
    def start_auto_check(self):
        """启动自动检查"""
        if HAS_DEPENDENCIES:
            # 启动时检查一次
            QTimer.singleShot(5000, self.check_for_updates_async)  # 5秒后检查
            
            # 设置定期检查
            self.check_timer.start(self.config.check_interval * 1000)
    
    def check_for_updates_async(self):
        """异步检查更新（槽函数）"""
        asyncio.create_task(self.check_for_updates())
    
    async def check_for_updates(self, show_dialog: bool = True) -> Tuple[UpdateStatus, Optional[VersionInfo]]:
        """检查更新"""
        try:
            status, version_info = await self.checker.check_for_updates()
            
            if show_dialog and status in [UpdateStatus.UPDATE_AVAILABLE, UpdateStatus.CRITICAL_UPDATE]:
                self.show_update_dialog(status, version_info)
            
            return status, version_info
        
        except Exception as e:
            self.logger.error(f"检查更新失败: {e}")
            return UpdateStatus.ERROR, None
    
    def show_update_dialog(self, status: UpdateStatus, version_info: VersionInfo):
        """显示更新对话框"""
        if HAS_DEPENDENCIES:
            dialog = UpdateDialog(None, version_info, status)
            dialog.exec()
        else:
            print(f"发现更新: {version_info.version}")
    
    async def download_and_install_update(self, version_info: VersionInfo,
                                        progress_callback=None) -> bool:
        """下载并安装更新"""
        try:
            # 下载更新
            update_file = await self.downloader.download_update(
                version_info, 
                progress_callback
            )
            
            if not update_file:
                return False
            
            # 安装更新
            success = await self.installer.install_update(
                update_file,
                progress_callback
            )
            
            return success
        
        except Exception as e:
            self.logger.error(f"下载安装更新失败: {e}")
            return False
    
    def get_update_settings(self) -> Dict[str, Any]:
        """获取更新设置"""
        return {
            'channel': self.config.channel.value,
            'auto_check': self.config.auto_check,
            'auto_download': self.config.auto_download,
            'auto_install': self.config.auto_install,
            'check_interval': self.config.check_interval
        }
    
    def update_settings(self, settings: Dict[str, Any]):
        """更新设置"""
        if 'channel' in settings:
            self.config.channel = UpdateChannel(settings['channel'])
        if 'auto_check' in settings:
            self.config.auto_check = settings['auto_check']
        if 'auto_download' in settings:
            self.config.auto_download = settings['auto_download']
        if 'auto_install' in settings:
            self.config.auto_install = settings['auto_install']
        if 'check_interval' in settings:
            self.config.check_interval = settings['check_interval']
        
        # 重新启动自动检查
        if HAS_DEPENDENCIES and self.config.auto_check:
            self.check_timer.start(self.config.check_interval * 1000)
        elif HAS_DEPENDENCIES:
            self.check_timer.stop()


# 全局更新管理器实例
update_manager = None

def init_update_manager(current_version: str, config: UpdateConfig = None):
    """初始化更新管理器"""
    global update_manager
    update_manager = UpdateManager(current_version, config)
    return update_manager

def check_for_updates_now():
    """立即检查更新"""
    if update_manager:
        return asyncio.create_task(update_manager.check_for_updates())
    return None

def get_current_version() -> str:
    """获取当前版本"""
    # 从配置或环境变量获取版本信息
    return "1.0.0"