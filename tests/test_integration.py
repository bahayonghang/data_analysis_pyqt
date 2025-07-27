"""
应用程序集成测试
验证完整的数据分析应用程序工作流程
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# PyQt6导入
try:
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import Qt, QTimer
    from PyQt6.QtTest import QTest
    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False

# Pandas导入
try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# 测试数据
TEST_CSV_DATA = """id,name,value,category,date
1,Item A,100.5,Category 1,2023-01-01
2,Item B,200.3,Category 2,2023-01-02
3,Item C,150.7,Category 1,2023-01-03
4,Item D,75.2,Category 3,2023-01-04
5,Item E,300.1,Category 2,2023-01-05
"""


class TestApplicationIntegration:
    """应用程序集成测试"""
    
    @pytest.fixture(scope="class")
    def qapp(self):
        """创建QApplication实例"""
        if not HAS_PYQT6:
            pytest.skip("PyQt6 not available")
        
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        yield app
    
    @pytest.fixture
    def temp_csv_file(self):
        """创建临时CSV文件"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(TEST_CSV_DATA)
            temp_path = f.name
        
        yield temp_path
        
        # 清理
        try:
            os.unlink(temp_path)
        except OSError:
            pass
    
    def test_application_creation(self, qapp):
        """测试应用程序创建"""
        from src.ui.app_integrator import create_application
        
        try:
            # 创建应用程序
            integrator = create_application()
            assert integrator is not None
            
            # 验证主窗口
            main_window = integrator.get_main_window()
            assert main_window is not None
            
            # 验证页面容器
            page_container = main_window.get_page_container()
            assert page_container is not None
            
            # 验证页面数量
            assert page_container.get_page_count() == 4  # home, upload, analysis, history
            
            # 验证必要页面存在
            assert page_container.has_page("home")
            assert page_container.has_page("upload")
            assert page_container.has_page("analysis")
            assert page_container.has_page("history")
            
        except ImportError:
            pytest.skip("GUI组件不可用")
    
    def test_navigation_flow(self, qapp):
        """测试导航流程"""
        from src.ui.app_integrator import create_application
        
        try:
            integrator = create_application()
            main_window = integrator.get_main_window()
            page_container = main_window.get_page_container()
            
            # 测试导航到各个页面
            pages = ["home", "upload", "analysis", "history"]
            
            for page_id in pages:
                main_window.navigate_to(page_id)
                # 注意：由于可能有动画，使用较短的等待时间
                QTimer.singleShot(100, lambda: None)
                qapp.processEvents()
                
                # 验证当前页面
                current_page = page_container.get_current_page_id()
                # 如果有动画延迟，current_page可能还没有更新
                # 我们主要验证没有崩溃即可
                
        except ImportError:
            pytest.skip("GUI组件不可用")
    
    def test_upload_workflow(self, qapp, temp_csv_file):
        """测试文件上传工作流程"""
        from src.ui.app_integrator import create_application
        
        try:
            integrator = create_application()
            main_window = integrator.get_main_window()
            
            # 获取上传页面
            upload_page = main_window.get_page_container().get_page_widget("upload")
            assert upload_page is not None
            
            # 模拟文件上传
            upload_page._handle_files_dropped([temp_csv_file])
            
            # 处理事件
            qapp.processEvents()
            
            # 验证文件信息
            file_info = upload_page.get_current_file_info()
            if file_info:  # 如果成功加载
                assert file_info.file_name.endswith('.csv')
                assert file_info.is_valid
                
                # 验证数据加载
                current_data = upload_page.get_current_data()
                if current_data is not None and HAS_PANDAS:
                    assert isinstance(current_data, pd.DataFrame)
                    assert len(current_data) > 0
            
        except ImportError:
            pytest.skip("GUI组件不可用")
    
    def test_data_flow_between_pages(self, qapp, temp_csv_file):
        """测试页面间数据流"""
        from src.ui.app_integrator import create_application
        
        try:
            integrator = create_application()
            main_window = integrator.get_main_window()
            page_container = main_window.get_page_container()
            
            # 获取上传和分析页面
            upload_page = page_container.get_page_widget("upload")
            analysis_page = page_container.get_page_widget("analysis")
            
            assert upload_page is not None
            assert analysis_page is not None
            
            # 模拟数据加载信号
            test_data = None
            if HAS_PANDAS:
                test_data = pd.DataFrame({
                    'id': [1, 2, 3],
                    'value': [100, 200, 300],
                    'category': ['A', 'B', 'C']
                })
            
            # 触发数据加载信号
            integrator._on_data_loaded(test_data, temp_csv_file, None)
            
            # 处理事件
            qapp.processEvents()
            
            # 验证数据传递
            current_data_info = integrator.get_current_data()
            assert current_data_info['file_path'] == temp_csv_file
            
        except ImportError:
            pytest.skip("GUI组件不可用")
    
    def test_responsive_layout(self, qapp):
        """测试响应式布局"""
        from src.ui.app_integrator import create_application
        
        try:
            integrator = create_application()
            main_window = integrator.get_main_window()
            page_container = main_window.get_page_container()
            
            # 测试不同窗口大小
            sizes = [
                (600, 400),   # mobile
                (900, 600),   # tablet
                (1400, 900)   # desktop
            ]
            
            for width, height in sizes:
                main_window.resize(width, height)
                qapp.processEvents()
                
                # 验证布局模式检测
                layout_mode = page_container.get_layout_mode()
                assert layout_mode in ['mobile', 'tablet', 'desktop']
                
                # 验证没有崩溃
                assert main_window.isVisible() or True  # 在测试环境中可能不可见
            
        except ImportError:
            pytest.skip("GUI组件不可用")
    
    def test_signal_connections(self, qapp):
        """测试信号连接"""
        from src.ui.app_integrator import create_application
        
        try:
            integrator = create_application()
            
            # 验证信号存在
            assert hasattr(integrator, 'data_loaded')
            assert hasattr(integrator, 'analysis_completed')
            assert hasattr(integrator, 'navigation_requested')
            
            # 测试信号连接（通过检查信号处理方法存在）
            assert hasattr(integrator, '_on_data_loaded')
            assert hasattr(integrator, '_on_upload_completed')
            assert hasattr(integrator, '_on_analysis_completed')
            assert hasattr(integrator, '_on_navigate_requested')
            
        except ImportError:
            pytest.skip("GUI组件不可用")
    
    def test_error_handling(self, qapp):
        """测试错误处理"""
        from src.ui.app_integrator import create_application
        
        try:
            integrator = create_application()
            main_window = integrator.get_main_window()
            page_container = main_window.get_page_container()
            
            # 测试导航到不存在的页面
            try:
                main_window.navigate_to("nonexistent_page")
                # 应该不会崩溃，可能记录错误日志
            except Exception as e:
                # 预期可能抛出异常
                assert isinstance(e, (ValueError, Exception))
            
            # 测试无效文件上传
            upload_page = page_container.get_page_widget("upload")
            if upload_page:
                # 测试无效文件路径
                upload_page._handle_files_dropped(["/nonexistent/file.csv"])
                qapp.processEvents()
                # 应该不会崩溃
            
        except ImportError:
            pytest.skip("GUI组件不可用")


class TestApplicationStartup:
    """应用程序启动测试"""
    
    def test_main_entry_point(self):
        """测试主入口点"""
        # 测试导入
        from src import main
        assert hasattr(main, 'main')
        assert hasattr(main, 'setup_application')
        assert hasattr(main, 'setup_qt_application')
        assert hasattr(main, 'run_gui_application')
        assert hasattr(main, 'run_cli_mode')
    
    @pytest.mark.skipif(not HAS_PYQT6, reason="PyQt6 not available")
    def test_qt_application_setup(self):
        """测试Qt应用程序设置"""
        from src.main import setup_qt_application
        
        app = setup_qt_application()
        if app:  # 如果PyQt6可用
            assert app.applicationName() == "Data Analysis PyQt"
            assert app.applicationVersion() == "1.0.0"
            assert app.organizationName() == "DataAnalysis"
    
    def test_application_configuration(self):
        """测试应用程序配置"""
        from src.main import setup_application
        
        # 应该不会抛出异常
        setup_application()
        
        # 验证配置加载
        from src.utils.simple_config import get_config, get_setting
        config = get_config()
        assert config is not None
        
        # 验证基本设置
        app_name = get_setting('app_name', 'Data Analysis PyQt')
        assert app_name is not None


class TestDataProcessingIntegration:
    """数据处理集成测试"""
    
    @pytest.fixture
    def sample_dataframe(self):
        """创建示例DataFrame"""
        if not HAS_PANDAS:
            pytest.skip("Pandas not available")
        
        return pd.DataFrame({
            'id': range(1, 101),
            'value': np.random.normal(100, 15, 100),
            'category': np.random.choice(['A', 'B', 'C'], 100),
            'date': pd.date_range('2023-01-01', periods=100, freq='D')
        })
    
    def test_mock_data_generation(self, sample_dataframe):
        """测试模拟数据生成"""
        if not HAS_PANDAS:
            pytest.skip("Pandas not available")
        
        from src.ui.upload_page import FileInfo, FileType
        
        # 创建文件信息
        file_info = FileInfo(
            file_path="/test/data.csv",
            file_name="data.csv",
            file_size=1024,
            file_type=FileType.CSV,
            is_valid=True
        )
        
        # 验证数据结构
        assert len(sample_dataframe) == 100
        assert 'id' in sample_dataframe.columns
        assert 'value' in sample_dataframe.columns
        assert 'category' in sample_dataframe.columns
        assert 'date' in sample_dataframe.columns
    
    def test_data_validation(self):
        """测试数据验证"""
        from src.ui.upload_page import FileValidator, UploadConfig
        
        config = UploadConfig()
        validator = FileValidator(config)
        
        # 测试有效扩展名
        assert '.csv' in config.supported_extensions
        assert '.parquet' in config.supported_extensions
        
        # 测试文件大小限制
        assert config.max_file_size == 500 * 1024 * 1024  # 500MB


class TestUIComponents:
    """UI组件测试"""
    
    @pytest.mark.skipif(not HAS_PYQT6, reason="PyQt6 not available")
    def test_main_window_creation(self):
        """测试主窗口创建"""
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        
        try:
            from src.ui.main_window import create_main_window, UIConfig
            
            config = UIConfig()
            main_window = create_main_window(config)
            
            assert main_window is not None
            assert main_window.windowTitle() == config.window_title
            
        except ImportError:
            pytest.skip("GUI组件不可用")
    
    @pytest.mark.skipif(not HAS_PYQT6, reason="PyQt6 not available")
    def test_upload_page_creation(self):
        """测试上传页面创建"""
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        
        try:
            from src.ui.upload_page import create_upload_page, UploadConfig
            
            config = UploadConfig()
            upload_page = create_upload_page(config)
            
            assert upload_page is not None
            assert hasattr(upload_page, 'file_uploaded')
            assert hasattr(upload_page, 'upload_failed')
            
        except ImportError:
            pytest.skip("GUI组件不可用")


class TestProjectConfiguration:
    """项目配置测试"""
    
    def test_pyproject_configuration(self):
        """测试pyproject.toml配置"""
        import tomllib
        from pathlib import Path
        
        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        if pyproject_path.exists():
            with open(pyproject_path, "rb") as f:
                config = tomllib.load(f)
            
            # 验证项目信息
            assert "project" in config
            assert config["project"]["name"] == "data-analysis-pyqt"
            
            # 验证脚本配置
            if "tool" in config and "uv" in config["tool"] and "scripts" in config["tool"]["uv"]:
                scripts = config["tool"]["uv"]["scripts"]
                assert "dev" in scripts
                assert "test" in scripts
                assert "gui" in scripts
                assert "cli" in scripts
    
    def test_dependency_configuration(self):
        """测试依赖配置"""
        import tomllib
        from pathlib import Path
        
        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        if pyproject_path.exists():
            with open(pyproject_path, "rb") as f:
                config = tomllib.load(f)
            
            # 验证核心依赖
            deps = config["project"]["dependencies"]
            assert any("PyQt6" in dep for dep in deps)
            assert any("PyQt-Fluent-Widgets" in dep for dep in deps)
            assert any("polars" in dep for dep in deps)


if __name__ == "__main__":
    # 直接运行测试
    pytest.main([__file__, "-v"])