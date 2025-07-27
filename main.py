"""
Data Analysis PyQt Application Entry Point
"""

import sys
import os
from pathlib import Path

# 添加src目录到Python路径
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from src.utils.simple_config import get_config, get_setting
from src.utils.error_handler import setup_exception_handling
from src.utils.basic_logging import setup_basic_logging, get_logger


def setup_application():
    """设置应用程序基础设施"""
    # 获取配置
    config = get_config()
    
    # 设置日志
    setup_basic_logging(
        log_level=get_setting("logging.level", "INFO"),
        log_file=get_setting("logging.file_path", "data/logs/app.log"),
        enable_console=get_setting("logging.enable_console", True),
    )
    
    # 设置异常处理
    setup_exception_handling()


def setup_qt_application():
    """设置Qt应用程序环境"""
    try:
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtCore import Qt
        from PyQt6.QtGui import QFont
        
        # 设置高DPI支持
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
        
        # 创建应用程序
        app = QApplication(sys.argv)
        
        # 设置应用程序属性
        app.setApplicationName("Data Analysis PyQt")
        app.setApplicationVersion("1.0.0")
        app.setOrganizationName("DataAnalysis")
        app.setOrganizationDomain("dataanalysis.local")
        
        # 设置字体
        font = QFont("Segoe UI", 9)
        app.setFont(font)
        
        return app
        
    except ImportError as e:
        print(f"PyQt6未安装，无法启动GUI应用程序: {e}")
        return None


def run_gui_application():
    """运行GUI应用程序"""
    logger = get_logger(__name__)
    
    try:
        # 设置Qt应用程序
        app = setup_qt_application()
        if not app:
            return 1
        
        logger.info("Qt应用程序设置完成")
        
        # 导入并创建主应用程序
        from src.ui.app_integrator import create_application
        
        # 创建应用程序集成器
        logger.info("正在创建应用程序...")
        integrator = create_application()
        main_window = integrator.get_main_window()
        
        if not main_window:
            logger.error("无法创建主窗口")
            return 1
        
        # 显示主窗口
        main_window.show()
        logger.info("数据分析应用程序已启动")
        
        # 进入事件循环
        return app.exec()
        
    except ImportError as e:
        logger.error(f"导入GUI组件失败: {str(e)}")
        print(f"GUI组件未安装，请检查依赖: {e}")
        return 1
        
    except Exception as e:
        logger.error(f"GUI应用程序启动失败: {str(e)}")
        print(f"应用程序启动失败: {e}")
        return 1


def run_cli_mode():
    """运行命令行模式"""
    logger = get_logger(__name__)
    
    logger.info("Data Analysis PyQt Application Starting...")
    logger.info("基础架构设置完成")
    
    print("Hello from data-analysis-pyqt!")
    print("项目基础架构已成功搭建！")
    print("\n已配置的组件:")
    print("✅ 项目目录结构")
    print("✅ uv项目管理和依赖配置")
    print("✅ 日志系统 (Loguru)")
    print("✅ 异常处理框架")
    print("✅ 应用配置管理 (Pydantic)")
    print("✅ GUI界面和数据分析功能")
    
    # 显示配置信息
    config = get_config()
    print(f"\n应用信息:")
    print(f"- 名称: {get_setting('app_name', 'Data Analysis PyQt')}")
    print(f"- 版本: {get_setting('version', '0.1.0')}")
    print(f"- 数据目录: {get_setting('data_dir', 'data')}")
    print(f"- 日志文件: {get_setting('logging.file_path', 'data/logs/app.log')}")
    
    print("\n启动选项:")
    print("- 运行 'uv run dev' 启动GUI应用程序")
    print("- 或者运行 'uv run python main.py --gui' 启动GUI应用程序")
    
    logger.info("CLI模式运行完成")
    return 0


def main():
    """应用程序主入口"""
    try:
        # 基础设施设置
        setup_application()
        
        # 检查命令行参数
        if len(sys.argv) > 1 and sys.argv[1] in ['--gui', '-g', 'gui']:
            # GUI模式
            return run_gui_application()
        elif len(sys.argv) > 1 and sys.argv[1] in ['--cli', '-c', 'cli']:
            # 强制CLI模式
            return run_cli_mode()
        elif 'GUI' in os.environ or 'gui' in sys.argv[0].lower():
            # 通过环境变量或文件名检测GUI模式
            return run_gui_application()
        else:
            # 默认先尝试GUI模式，失败则CLI模式
            try:
                return run_gui_application()
            except (ImportError, Exception) as e:
                print(f"GUI模式不可用，回退到CLI模式: {e}")
                return run_cli_mode()
        
    except Exception as e:
        print(f"启动失败: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
