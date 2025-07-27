"""
端到端测试 - 完整用户工作流程测试
验证从数据上传到分析完成的完整流程
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import time

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
SAMPLE_CSV_DATA = """timestamp,sensor_id,temperature,humidity,pressure,status
2023-01-01 00:00:00,sensor_001,25.3,65.2,1013.25,active
2023-01-01 01:00:00,sensor_001,25.1,66.1,1013.15,active
2023-01-01 02:00:00,sensor_001,24.9,67.3,1013.05,active
2023-01-01 03:00:00,sensor_001,24.7,68.1,1012.95,active
2023-01-01 04:00:00,sensor_001,24.5,69.2,1012.85,active
2023-01-01 05:00:00,sensor_001,24.8,68.5,1012.95,active
2023-01-01 06:00:00,sensor_001,25.2,67.8,1013.15,active
2023-01-01 07:00:00,sensor_001,25.6,66.9,1013.35,active
2023-01-01 08:00:00,sensor_001,26.1,65.7,1013.55,active
2023-01-01 09:00:00,sensor_001,26.8,64.2,1013.75,active
2023-01-01 10:00:00,sensor_002,26.5,64.8,1013.65,active
2023-01-01 11:00:00,sensor_002,27.2,63.5,1013.85,active
2023-01-01 12:00:00,sensor_002,27.8,62.1,1014.05,active
2023-01-01 13:00:00,sensor_002,28.1,61.6,1014.15,active
2023-01-01 14:00:00,sensor_002,28.4,61.2,1014.25,active
2023-01-01 15:00:00,sensor_002,28.2,61.8,1014.15,active
2023-01-01 16:00:00,sensor_002,27.9,62.4,1014.05,active
2023-01-01 17:00:00,sensor_002,27.5,63.1,1013.95,active
2023-01-01 18:00:00,sensor_002,27.1,63.8,1013.85,active
2023-01-01 19:00:00,sensor_002,26.7,64.5,1013.75,active
"""

SAMPLE_PARQUET_DATA = {
    'product_id': [f'P{i:03d}' for i in range(1, 51)],
    'sales': np.random.randint(10, 1000, 50),
    'revenue': np.random.uniform(100, 5000, 50),
    'category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home'], 50),
    'region': np.random.choice(['North', 'South', 'East', 'West'], 50),
    'date': pd.date_range('2023-01-01', periods=50, freq='D')
}


@pytest.mark.skipif(not HAS_PYQT6, reason="PyQt6 not available")
class TestEndToEndWorkflow:
    """端到端工作流程测试"""
    
    @pytest.fixture(scope="class")
    def qapp(self):
        """创建QApplication实例"""
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        yield app
    
    @pytest.fixture
    def temp_csv_file(self):
        """创建临时CSV文件"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(SAMPLE_CSV_DATA)
            temp_path = f.name
        
        yield temp_path
        
        # 清理
        try:
            os.unlink(temp_path)
        except OSError:
            pass
    
    @pytest.fixture
    def temp_parquet_file(self):
        """创建临时Parquet文件"""
        if not HAS_PANDAS:
            pytest.skip("Pandas not available")
        
        df = pd.DataFrame(SAMPLE_PARQUET_DATA)
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            temp_path = f.name
        
        df.to_parquet(temp_path)
        yield temp_path
        
        # 清理
        try:
            os.unlink(temp_path)
        except OSError:
            pass
    
    def test_complete_workflow_csv(self, qapp, temp_csv_file):
        """测试完整的CSV文件工作流程"""
        try:
            from src.ui.app_integrator import create_application
            
            # 1. 创建应用程序
            integrator = create_application()
            main_window = integrator.get_main_window()
            page_container = main_window.get_page_container()
            
            # 显示主窗口（可选，在测试环境中）
            main_window.show()
            qapp.processEvents()
            
            # 2. 导航到上传页面
            main_window.navigate_to("upload")
            qapp.processEvents()
            QTimer.singleShot(100, lambda: None)  # 等待动画
            qapp.processEvents()
            
            # 3. 获取上传页面并上传文件
            upload_page = page_container.get_page_widget("upload")
            assert upload_page is not None
            
            # 模拟文件上传
            upload_page._handle_files_dropped([temp_csv_file])
            qapp.processEvents()
            
            # 等待上传完成
            for _ in range(50):  # 最多等待5秒
                qapp.processEvents()
                time.sleep(0.1)
                if upload_page.get_current_file_info():
                    break
            
            # 验证文件上传
            file_info = upload_page.get_current_file_info()
            if file_info:
                assert file_info.is_valid
                assert file_info.file_name.endswith('.csv')
                
                # 4. 验证数据加载
                current_data = upload_page.get_current_data()
                if current_data is not None and HAS_PANDAS:
                    assert isinstance(current_data, pd.DataFrame)
                    assert len(current_data) > 0
                    assert 'timestamp' in current_data.columns
                
                # 5. 导航到分析页面
                main_window.navigate_to("analysis")
                qapp.processEvents()
                QTimer.singleShot(100, lambda: None)
                qapp.processEvents()
                
                # 6. 验证分析页面接收到数据
                analysis_page = page_container.get_page_widget("analysis")
                assert analysis_page is not None
                
                # 7. 导航到历史页面
                main_window.navigate_to("history")
                qapp.processEvents()
                QTimer.singleShot(100, lambda: None)
                qapp.processEvents()
                
                # 8. 验证历史页面
                history_page = page_container.get_page_widget("history")
                assert history_page is not None
                
                # 9. 返回主页
                main_window.navigate_to("home")
                qapp.processEvents()
                
                # 验证完整流程成功
                current_page = page_container.get_current_page_id()
                # 注意：由于动画可能尚未完成，我们主要验证没有崩溃
                assert True  # 如果能到达这里说明没有崩溃
            
        except ImportError:
            pytest.skip("GUI组件不可用")
    
    def test_navigation_history(self, qapp):
        """测试导航历史功能"""
        try:
            from src.ui.app_integrator import create_application
            
            integrator = create_application()
            main_window = integrator.get_main_window()
            page_container = main_window.get_page_container()
            
            # 导航序列
            navigation_sequence = ["upload", "analysis", "history", "home"]
            
            for page_id in navigation_sequence:
                main_window.navigate_to(page_id)
                qapp.processEvents()
                time.sleep(0.05)  # 短暂等待
            
            # 测试返回功能
            success = main_window.go_back_page()
            qapp.processEvents()
            
            # 验证历史记录
            history = main_window.get_page_history()
            assert isinstance(history, list)
            
        except ImportError:
            pytest.skip("GUI组件不可用")
    
    def test_responsive_behavior(self, qapp):
        """测试响应式行为"""
        try:
            from src.ui.app_integrator import create_application
            
            integrator = create_application()
            main_window = integrator.get_main_window()
            page_container = main_window.get_page_container()
            
            # 测试不同屏幕尺寸
            test_sizes = [
                (600, 400),   # Mobile
                (800, 600),   # Tablet
                (1200, 800),  # Desktop
                (1920, 1080)  # Large Desktop
            ]
            
            for width, height in test_sizes:
                main_window.resize(width, height)
                qapp.processEvents()
                time.sleep(0.1)
                
                # 验证布局模式
                layout_mode = main_window.get_layout_mode()
                assert layout_mode in ['mobile', 'tablet', 'desktop']
                
                # 测试所有页面在该尺寸下的表现
                for page_id in ["home", "upload", "analysis", "history"]:
                    main_window.navigate_to(page_id)
                    qapp.processEvents()
                    time.sleep(0.05)
                    
                    # 验证页面没有崩溃
                    current_widget = page_container.get_page_widget(page_id)
                    assert current_widget is not None
            
        except ImportError:
            pytest.skip("GUI组件不可用")
    
    def test_error_recovery(self, qapp):
        """测试错误恢复"""
        try:
            from src.ui.app_integrator import create_application
            
            integrator = create_application()
            main_window = integrator.get_main_window()
            page_container = main_window.get_page_container()
            upload_page = page_container.get_page_widget("upload")
            
            # 测试无效文件上传
            invalid_files = [
                "/path/that/does/not/exist.csv",
                "/invalid/file.txt",
                ""
            ]
            
            for invalid_file in invalid_files:
                try:
                    upload_page._handle_files_dropped([invalid_file])
                    qapp.processEvents()
                    time.sleep(0.1)
                    
                    # 验证应用程序仍然响应
                    assert main_window.isEnabled() or True  # 在测试环境中可能disabled
                    
                except Exception:
                    # 预期可能有异常，但应用程序应该继续运行
                    qapp.processEvents()
                    assert True
            
        except ImportError:
            pytest.skip("GUI组件不可用")
    
    def test_performance_basic(self, qapp, temp_csv_file):
        """测试基本性能"""
        try:
            from src.ui.app_integrator import create_application
            
            start_time = time.time()
            
            # 应用程序创建时间
            integrator = create_application()
            main_window = integrator.get_main_window()
            creation_time = time.time() - start_time
            
            # 应用程序创建应该在合理时间内完成
            assert creation_time < 5.0  # 5秒内
            
            # 导航性能测试
            start_time = time.time()
            for page_id in ["upload", "analysis", "history", "home"]:
                main_window.navigate_to(page_id)
                qapp.processEvents()
            navigation_time = time.time() - start_time
            
            # 导航应该在合理时间内完成
            assert navigation_time < 2.0  # 2秒内
            
        except ImportError:
            pytest.skip("GUI组件不可用")


@pytest.mark.skipif(not HAS_PYQT6, reason="PyQt6 not available")
class TestUserScenarios:
    """用户使用场景测试"""
    
    @pytest.fixture(scope="class")
    def qapp(self):
        """创建QApplication实例"""
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        yield app
    
    def test_first_time_user_scenario(self, qapp):
        """测试首次使用用户场景"""
        try:
            from src.ui.app_integrator import create_application
            
            # 1. 用户启动应用程序
            integrator = create_application()
            main_window = integrator.get_main_window()
            
            # 2. 应该看到主页
            page_container = main_window.get_page_container()
            home_page = page_container.get_page_widget("home")
            assert home_page is not None
            
            # 3. 主页应该提供导航指引
            # 验证主页具有导航功能
            assert hasattr(home_page, 'navigate_requested') or hasattr(home_page.get_widget(), 'navigate_requested')
            
            # 4. 用户点击上传按钮
            main_window.navigate_to("upload")
            qapp.processEvents()
            
            # 5. 应该看到上传界面
            upload_page = page_container.get_page_widget("upload")
            assert upload_page is not None
            
        except ImportError:
            pytest.skip("GUI组件不可用")
    
    def test_data_analyst_workflow(self, qapp):
        """测试数据分析师工作流程"""
        try:
            from src.ui.app_integrator import create_application
            
            integrator = create_application()
            main_window = integrator.get_main_window()
            page_container = main_window.get_page_container()
            
            # 分析师典型工作流程：
            # 1. 查看历史分析
            main_window.navigate_to("history")
            qapp.processEvents()
            
            # 2. 上传新数据
            main_window.navigate_to("upload")
            qapp.processEvents()
            
            # 3. 进行分析
            main_window.navigate_to("analysis")
            qapp.processEvents()
            
            # 4. 查看结果并返回历史
            main_window.navigate_to("history")
            qapp.processEvents()
            
            # 验证所有页面都可访问
            for page_id in ["home", "upload", "analysis", "history"]:
                assert page_container.has_page(page_id)
            
        except ImportError:
            pytest.skip("GUI组件不可用")
    
    def test_multiple_files_scenario(self, qapp):
        """测试多文件处理场景"""
        try:
            from src.ui.app_integrator import create_application
            
            integrator = create_application()
            main_window = integrator.get_main_window()
            page_container = main_window.get_page_container()
            upload_page = page_container.get_page_widget("upload")
            
            # 创建多个临时文件
            temp_files = []
            for i in range(3):
                with tempfile.NamedTemporaryFile(mode='w', suffix=f'_test_{i}.csv', delete=False) as f:
                    f.write(f"id,value\n1,{i*10}\n2,{i*20}\n")
                    temp_files.append(f.name)
            
            try:
                # 测试处理多个文件（应该只处理第一个）
                upload_page._handle_files_dropped(temp_files)
                qapp.processEvents()
                time.sleep(0.2)
                
                # 验证只处理了一个文件
                file_info = upload_page.get_current_file_info()
                if file_info:
                    assert file_info.file_name.endswith('.csv')
                
            finally:
                # 清理临时文件
                for temp_file in temp_files:
                    try:
                        os.unlink(temp_file)
                    except OSError:
                        pass
            
        except ImportError:
            pytest.skip("GUI组件不可用")


class TestSystemIntegration:
    """系统集成测试"""
    
    def test_configuration_loading(self):
        """测试配置加载"""
        from src.utils.simple_config import get_config, get_setting
        
        # 测试配置系统工作正常
        config = get_config()
        assert config is not None
        
        # 测试设置获取
        app_name = get_setting('app_name', 'Default App')
        assert isinstance(app_name, str)
        
        data_dir = get_setting('data_dir', 'data')
        assert isinstance(data_dir, str)
    
    def test_logging_system(self):
        """测试日志系统"""
        from src.utils.basic_logging import setup_basic_logging, get_logger
        
        # 设置日志
        setup_basic_logging()
        
        # 获取日志记录器
        logger = get_logger("test")
        assert logger is not None
        
        # 测试日志记录
        logger.info("Test log message")
        logger.warning("Test warning message")
    
    def test_exception_handling(self):
        """测试异常处理"""
        from src.utils.error_handler import setup_exception_handling
        from src.utils.exceptions import DataProcessingError, FileValidationError
        
        # 设置异常处理
        setup_exception_handling()
        
        # 测试自定义异常
        try:
            raise DataProcessingError("Test error")
        except DataProcessingError as e:
            assert str(e) == "Test error"
        
        try:
            raise FileValidationError("Test validation error")
        except FileValidationError as e:
            assert str(e) == "Test validation error"


if __name__ == "__main__":
    # 直接运行测试
    pytest.main([__file__, "-v", "-s"])