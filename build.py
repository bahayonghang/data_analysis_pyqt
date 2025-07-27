#!/usr/bin/env python3
"""
应用程序构建脚本

自动化构建、打包和发布数据分析应用程序
支持多平台构建和多种打包格式
"""

import os
import sys
import shutil
import subprocess
import platform
import argparse
import json
import zipfile
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# 确保可以导入项目模块
sys.path.insert(0, str(Path(__file__).parent))

class BuildConfig:
    """构建配置"""
    
    def __init__(self):
        self.project_dir = Path(__file__).parent
        self.src_dir = self.project_dir / "src"
        self.build_dir = self.project_dir / "build"
        self.dist_dir = self.project_dir / "dist"
        self.assets_dir = self.project_dir / "assets"
        
        # 版本信息
        self.version = "1.0.0"
        self.build_number = datetime.now().strftime("%Y%m%d")
        
        # 平台信息
        self.platform = platform.system().lower()
        self.arch = platform.machine().lower()
        
        # 构建目标
        self.app_name = "DataAnalysisApp"
        self.display_name = "Data Analysis Pro"
        
    def get_platform_suffix(self) -> str:
        """获取平台后缀"""
        platform_map = {
            'windows': 'win',
            'darwin': 'macos',
            'linux': 'linux'
        }
        
        arch_map = {
            'x86_64': 'x64',
            'amd64': 'x64',
            'arm64': 'arm64',
            'aarch64': 'arm64'
        }
        
        platform_name = platform_map.get(self.platform, self.platform)
        arch_name = arch_map.get(self.arch, self.arch)
        
        return f"{platform_name}-{arch_name}"


class BuildTool:
    """构建工具基类"""
    
    def __init__(self, config: BuildConfig):
        self.config = config
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        """设置日志"""
        import logging
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(self.__class__.__name__)
    
    def run_command(self, cmd: List[str], cwd: Optional[Path] = None) -> bool:
        """运行命令"""
        try:
            self.logger.info(f"执行命令: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                cwd=cwd or self.config.project_dir,
                check=True,
                capture_output=True,
                text=True
            )
            
            if result.stdout:
                self.logger.info(f"输出: {result.stdout}")
            
            return True
        
        except subprocess.CalledProcessError as e:
            self.logger.error(f"命令执行失败: {e}")
            if e.stderr:
                self.logger.error(f"错误输出: {e.stderr}")
            return False
    
    def clean_build_dirs(self):
        """清理构建目录"""
        self.logger.info("清理构建目录...")
        
        for dir_path in [self.config.build_dir, self.config.dist_dir]:
            if dir_path.exists():
                shutil.rmtree(dir_path)
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def copy_assets(self, target_dir: Path):
        """复制资源文件"""
        self.logger.info("复制资源文件...")
        
        assets_target = target_dir / "assets"
        if self.config.assets_dir.exists():
            shutil.copytree(self.config.assets_dir, assets_target, dirs_exist_ok=True)
        
        # 复制文档
        docs_files = ["README.md", "LICENSE", "docs/user_manual.md"]
        docs_target = target_dir / "docs"
        docs_target.mkdir(exist_ok=True)
        
        for doc_file in docs_files:
            doc_path = self.config.project_dir / doc_file
            if doc_path.exists():
                if doc_file == "docs/user_manual.md":
                    shutil.copy2(doc_path, docs_target / "user_manual.md")
                else:
                    shutil.copy2(doc_path, target_dir / doc_path.name)
    
    def create_version_info(self, target_dir: Path):
        """创建版本信息文件"""
        version_info = {
            "version": self.config.version,
            "build": self.config.build_number,
            "platform": self.config.platform,
            "arch": self.config.arch,
            "build_date": datetime.now().isoformat(),
            "app_name": self.config.app_name,
            "display_name": self.config.display_name
        }
        
        version_file = target_dir / "version.json"
        with open(version_file, 'w', encoding='utf-8') as f:
            json.dump(version_info, f, indent=2, ensure_ascii=False)


class PyInstallerBuilder(BuildTool):
    """PyInstaller 构建器"""
    
    def build(self, onefile: bool = False) -> bool:
        """使用 PyInstaller 构建"""
        self.logger.info("开始 PyInstaller 构建...")
        
        # 构建命令
        cmd = [
            sys.executable, "-m", "PyInstaller",
            "--name", self.config.app_name,
            "--windowed",  # 不显示控制台
            "--clean",
            "--noconfirm",
            f"--distpath={self.config.dist_dir}",
            f"--workpath={self.config.build_dir / 'pyinstaller'}",
            f"--specpath={self.config.build_dir}",
        ]
        
        if onefile:
            cmd.append("--onefile")
        else:
            cmd.append("--onedir")
        
        # 添加隐藏导入
        hidden_imports = [
            "PyQt6.QtCore", "PyQt6.QtGui", "PyQt6.QtWidgets",
            "qfluentwidgets", "polars", "numpy", "matplotlib.backends.qt_agg",
            "plotly.graph_objects", "reportlab.pdfgen", "sqlite3"
        ]
        
        for module in hidden_imports:
            cmd.extend(["--hidden-import", module])
        
        # 添加数据文件
        data_files = [
            f"{self.config.assets_dir};assets",
            f"{self.config.project_dir / 'docs' / 'user_manual.md'};docs",
        ]
        
        for data_file in data_files:
            if Path(data_file.split(';')[0]).exists():
                cmd.extend(["--add-data", data_file])
        
        # 排除模块
        excludes = ["tkinter", "test", "unittest", "doctest", "pdb"]
        for module in excludes:
            cmd.extend(["--exclude-module", module])
        
        # 主脚本
        cmd.append(str(self.config.project_dir / "main.py"))
        
        # 执行构建
        if not self.run_command(cmd):
            return False
        
        # 后处理
        self._post_process_pyinstaller()
        return True
    
    def _post_process_pyinstaller(self):
        """PyInstaller 后处理"""
        app_dir = self.config.dist_dir / self.config.app_name
        
        if app_dir.exists():
            # 复制额外资源
            self.copy_assets(app_dir)
            self.create_version_info(app_dir)
            
            # 创建启动脚本
            self._create_launcher_script(app_dir)
    
    def _create_launcher_script(self, app_dir: Path):
        """创建启动脚本"""
        if self.config.platform == "windows":
            launcher_content = f"""@echo off
cd /d "%~dp0"
"{self.config.app_name}.exe" %*
"""
            launcher_file = app_dir / "start.bat"
        else:
            launcher_content = f"""#!/bin/bash
cd "$(dirname "$0")"
./{self.config.app_name} "$@"
"""
            launcher_file = app_dir / "start.sh"
        
        with open(launcher_file, 'w', encoding='utf-8') as f:
            f.write(launcher_content)
        
        if self.config.platform != "windows":
            os.chmod(launcher_file, 0o755)


class CxFreezeBuilder(BuildTool):
    """cx_Freeze 构建器"""
    
    def build(self) -> bool:
        """使用 cx_Freeze 构建"""
        self.logger.info("开始 cx_Freeze 构建...")
        
        # 创建 setup 脚本
        setup_script = self._create_setup_script()
        
        # 执行构建
        cmd = [sys.executable, str(setup_script), "build"]
        return self.run_command(cmd)
    
    def _create_setup_script(self) -> Path:
        """创建 cx_Freeze setup 脚本"""
        setup_content = f'''
import sys
from cx_Freeze import setup, Executable

# 构建选项
build_exe_options = {{
    "packages": [
        "PyQt6", "qfluentwidgets", "polars", "numpy", "scipy",
        "matplotlib", "plotly", "reportlab", "sqlite3", "asyncio"
    ],
    "excludes": ["tkinter", "test", "unittest"],
    "include_files": [
        ("assets/", "assets/"),
        ("docs/user_manual.md", "docs/user_manual.md"),
    ],
    "zip_include_packages": ["*"],
    "zip_exclude_packages": []
}}

# 可执行文件配置
base = None
if sys.platform == "win32":
    base = "Win32GUI"

executables = [
    Executable(
        "main.py",
        base=base,
        target_name="{self.config.app_name}",
        icon="assets/icons/app_icon.ico"
    )
]

setup(
    name="{self.config.display_name}",
    version="{self.config.version}",
    description="Professional data analysis software",
    options={{"build_exe": build_exe_options}},
    executables=executables
)
'''
        
        setup_file = self.config.build_dir / "setup_cx_freeze.py"
        with open(setup_file, 'w', encoding='utf-8') as f:
            f.write(setup_content)
        
        return setup_file


class NuitkaBuilder(BuildTool):
    """Nuitka 构建器"""
    
    def build(self) -> bool:
        """使用 Nuitka 构建"""
        self.logger.info("开始 Nuitka 构建...")
        
        cmd = [
            sys.executable, "-m", "nuitka",
            "--standalone",
            "--enable-plugin=pyqt6",
            f"--output-dir={self.config.dist_dir}",
            f"--output-filename={self.config.app_name}",
            "--include-package=src",
            "--include-data-dir=assets=assets",
            "--include-data-file=docs/user_manual.md=docs/user_manual.md",
            "--exclude-module=tkinter",
            "--exclude-module=test",
            "--exclude-module=unittest",
        ]
        
        if self.config.platform == "windows":
            cmd.extend([
                "--windows-disable-console",
                "--windows-icon-from-ico=assets/icons/app_icon.ico"
            ])
        
        cmd.append(str(self.config.project_dir / "main.py"))
        
        return self.run_command(cmd)


class PackageBuilder:
    """包构建器"""
    
    def __init__(self, config: BuildConfig):
        self.config = config
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        """设置日志"""
        import logging
        return logging.getLogger(self.__class__.__name__)
    
    def create_zip_package(self, source_dir: Path) -> Path:
        """创建 ZIP 包"""
        self.logger.info("创建 ZIP 包...")
        
        platform_suffix = self.config.get_platform_suffix()
        zip_name = f"{self.config.app_name}-{self.config.version}-{platform_suffix}.zip"
        zip_path = self.config.dist_dir / zip_name
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for file_path in source_dir.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(source_dir.parent)
                    zip_file.write(file_path, arcname)
        
        self.logger.info(f"ZIP 包创建完成: {zip_path}")
        return zip_path
    
    def create_installer(self, source_dir: Path) -> Optional[Path]:
        """创建安装包"""
        if self.config.platform == "windows":
            return self._create_nsis_installer(source_dir)
        elif self.config.platform == "darwin":
            return self._create_dmg_installer(source_dir)
        elif self.config.platform == "linux":
            return self._create_deb_package(source_dir)
        
        return None
    
    def _create_nsis_installer(self, source_dir: Path) -> Optional[Path]:
        """创建 NSIS 安装包"""
        # 这里需要 NSIS 工具
        self.logger.info("创建 NSIS 安装包 (需要手动配置)")
        return None
    
    def _create_dmg_installer(self, source_dir: Path) -> Optional[Path]:
        """创建 macOS DMG 安装包"""
        # 这里需要 macOS 工具
        self.logger.info("创建 DMG 安装包 (需要 macOS 环境)")
        return None
    
    def _create_deb_package(self, source_dir: Path) -> Optional[Path]:
        """创建 DEB 包"""
        # 这里需要 dpkg 工具
        self.logger.info("创建 DEB 包 (需要 dpkg 工具)")
        return None


class BuildManager:
    """构建管理器"""
    
    def __init__(self):
        self.config = BuildConfig()
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        """设置日志"""
        import logging
        return logging.getLogger(self.__class__.__name__)
    
    def pre_build_checks(self) -> bool:
        """构建前检查"""
        self.logger.info("执行构建前检查...")
        
        # 检查 Python 版本
        if sys.version_info < (3, 11):
            self.logger.error("需要 Python 3.11 或更高版本")
            return False
        
        # 检查主要依赖
        required_modules = ["PyQt6", "polars", "numpy", "matplotlib"]
        for module in required_modules:
            try:
                __import__(module)
            except ImportError:
                self.logger.error(f"缺少必需模块: {module}")
                return False
        
        # 检查主脚本
        main_script = self.config.project_dir / "main.py"
        if not main_script.exists():
            self.logger.error("主脚本 main.py 不存在")
            return False
        
        self.logger.info("构建前检查通过")
        return True
    
    def build(self, builder_type: str = "pyinstaller", **kwargs) -> bool:
        """执行构建"""
        if not self.pre_build_checks():
            return False
        
        # 清理构建目录
        self.logger.info("清理构建目录...")
        for dir_path in [self.config.build_dir, self.config.dist_dir]:
            if dir_path.exists():
                shutil.rmtree(dir_path)
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 选择构建器
        builders = {
            "pyinstaller": PyInstallerBuilder,
            "cx_freeze": CxFreezeBuilder,
            "nuitka": NuitkaBuilder
        }
        
        builder_class = builders.get(builder_type)
        if not builder_class:
            self.logger.error(f"不支持的构建器: {builder_type}")
            return False
        
        # 执行构建
        builder = builder_class(self.config)
        if not builder.build(**kwargs):
            self.logger.error("构建失败")
            return False
        
        # 创建包
        source_dir = self.config.dist_dir / self.config.app_name
        if source_dir.exists():
            packager = PackageBuilder(self.config)
            packager.create_zip_package(source_dir)
        
        self.logger.info("构建完成")
        return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="数据分析应用构建脚本")
    
    parser.add_argument(
        "--builder", "-b",
        choices=["pyinstaller", "cx_freeze", "nuitka"],
        default="pyinstaller",
        help="选择构建工具"
    )
    
    parser.add_argument(
        "--onefile",
        action="store_true",
        help="创建单文件可执行程序 (仅 PyInstaller)"
    )
    
    parser.add_argument(
        "--clean",
        action="store_true",
        help="构建前清理所有临时文件"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="详细输出"
    )
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 创建构建管理器
    build_manager = BuildManager()
    
    # 执行构建
    kwargs = {}
    if args.builder == "pyinstaller":
        kwargs["onefile"] = args.onefile
    
    success = build_manager.build(args.builder, **kwargs)
    
    if success:
        print("\n✅ 构建成功完成!")
        print(f"输出目录: {build_manager.config.dist_dir}")
    else:
        print("\n❌ 构建失败")
        sys.exit(1)


if __name__ == "__main__":
    main()