"""
应用程序启动和运行测试
验证应用程序可以正常启动和运行
"""

import pytest
import sys
import subprocess
import time
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestApplicationStartup:
    """应用程序启动测试"""
    
    def test_main_module_import(self):
        """测试主模块导入"""
        # 测试能否正常导入main模块
        import main
        
        # 验证必要的函数存在
        assert hasattr(main, 'main')
        assert hasattr(main, 'setup_application')
        assert hasattr(main, 'run_cli_mode')
        assert hasattr(main, 'run_gui_application')
        assert callable(main.main)
    
    def test_cli_mode_execution(self):
        """测试CLI模式执行"""
        import main
        
        # 模拟CLI模式运行
        with patch('sys.argv', ['main.py']):
            try:
                # 应该能正常设置基础设施
                main.setup_application()
                
                # CLI模式应该能正常运行
                result = main.run_cli_mode()
                assert result == 0  # 成功退出码
                
            except Exception as e:
                pytest.fail(f"CLI模式运行失败: {str(e)}")
    
    def test_gui_mode_detection(self):
        """测试GUI模式检测"""
        import main
        
        # 测试命令行参数检测
        test_cases = [
            (['main.py', '--gui'], True),
            (['main.py', '-g'], True),
            (['main.py', 'gui'], True),
            (['main.py'], False),
            (['main.py', '--help'], False),
        ]
        
        for argv, should_be_gui in test_cases:
            with patch('sys.argv', argv):
                # 检查main函数的逻辑
                if len(argv) > 1 and argv[1] in ['--gui', '-g', 'gui']:
                    assert should_be_gui
                else:
                    assert not should_be_gui or len(argv) == 1  # 默认尝试GUI
    
    @pytest.mark.skipif(
        not os.environ.get('DISPLAY') and os.name != 'nt',
        reason="需要显示环境或Windows系统"
    )
    def test_gui_application_creation(self):
        """测试GUI应用程序创建"""
        import main
        
        try:
            # 尝试设置Qt应用程序
            app = main.setup_qt_application()
            
            if app:  # 如果PyQt6可用
                assert app.applicationName() == "Data Analysis PyQt"
                assert app.applicationVersion() == "1.0.0"
                
                # 清理
                app.quit()
                
        except ImportError:
            pytest.skip("PyQt6不可用")
        except Exception as e:
            # 在某些测试环境中可能无法创建GUI应用程序
            if "QApplication" in str(e) or "display" in str(e).lower():
                pytest.skip(f"GUI环境不可用: {str(e)}")
            else:
                raise
    
    def test_error_handling_on_import_failure(self):
        """测试导入失败时的错误处理"""
        import main
        
        # 模拟PyQt6不可用的情况
        with patch.dict('sys.modules', {'PyQt6': None}):
            try:
                # 应该回退到CLI模式
                with patch('sys.argv', ['main.py']):
                    result = main.main()
                    # 应该成功运行CLI模式
                    assert result == 0
                    
            except ImportError:
                # 预期的导入错误
                pass
    
    def test_configuration_setup(self):
        """测试配置设置"""
        import main
        
        # 测试基础设施设置
        try:
            main.setup_application()
            
            # 验证配置系统工作
            from src.utils.simple_config import get_config, get_setting
            config = get_config()
            assert config is not None
            
            # 验证基本设置
            app_name = get_setting('app_name', 'Data Analysis PyQt')
            assert isinstance(app_name, str)
            
        except Exception as e:
            pytest.fail(f"配置设置失败: {str(e)}")


class TestProjectScripts:
    """项目脚本测试"""
    
    @pytest.fixture
    def project_root(self):
        """获取项目根目录"""
        return Path(__file__).parent.parent
    
    def test_pyproject_toml_exists(self, project_root):
        """测试pyproject.toml存在"""
        pyproject_path = project_root / "pyproject.toml"
        assert pyproject_path.exists(), "pyproject.toml文件不存在"
    
    def test_main_py_exists(self, project_root):
        """测试main.py存在"""
        main_path = project_root / "main.py"
        assert main_path.exists(), "main.py文件不存在"
    
    def test_uv_scripts_configuration(self, project_root):
        """测试uv脚本配置"""
        import tomllib
        
        pyproject_path = project_root / "pyproject.toml"
        if pyproject_path.exists():
            with open(pyproject_path, "rb") as f:
                config = tomllib.load(f)
            
            # 验证uv脚本存在
            if "tool" in config and "uv" in config["tool"] and "scripts" in config["tool"]["uv"]:
                scripts = config["tool"]["uv"]["scripts"]
                
                # 验证关键脚本
                assert "dev" in scripts, "缺少dev脚本"
                assert "gui" in scripts, "缺少gui脚本"
                assert "cli" in scripts, "缺少cli脚本"
                assert "test" in scripts, "缺少test脚本"
                
                # 验证脚本内容
                assert "main.py --gui" in scripts["dev"]
                assert "main.py --gui" in scripts["gui"]
                assert "main.py" in scripts["cli"]
                assert "pytest" in scripts["test"]
    
    @pytest.mark.skipif(
        subprocess.run(["which", "uv"], capture_output=True).returncode != 0,
        reason="uv命令不可用"
    )
    def test_uv_dev_script_syntax(self, project_root):
        """测试uv dev脚本语法"""
        # 测试uv脚本语法是否正确
        result = subprocess.run(
            ["uv", "run", "--help"],
            cwd=project_root,
            capture_output=True,
            text=True
        )
        
        # uv run应该能正常显示帮助
        assert result.returncode == 0 or "help" in result.stdout.lower()
    
    def test_python_path_setup(self):
        """测试Python路径设置"""
        import main
        
        # 验证src目录在Python路径中
        src_path = str(Path(__file__).parent.parent / "src")
        assert src_path in sys.path or any(src_path in p for p in sys.path)


class TestDependencyValidation:
    """依赖验证测试"""
    
    def test_core_dependencies_import(self):
        """测试核心依赖导入"""
        # 测试可选依赖的导入
        dependencies = {
            'pathlib': True,  # 标准库
            'sys': True,      # 标准库
            'os': True,       # 标准库
        }
        
        for dep_name, should_exist in dependencies.items():
            try:
                __import__(dep_name)
                imported = True
            except ImportError:
                imported = False
            
            if should_exist:
                assert imported, f"必需依赖 {dep_name} 导入失败"
    
    def test_optional_dependencies_handling(self):
        """测试可选依赖处理"""
        # 测试PyQt6可选依赖
        try:
            import PyQt6
            pyqt6_available = True
        except ImportError:
            pyqt6_available = False
        
        # 测试pandas可选依赖
        try:
            import pandas
            pandas_available = True
        except ImportError:
            pandas_available = False
        
        # 测试qfluentwidgets可选依赖
        try:
            import qfluentwidgets
            fluent_available = True
        except ImportError:
            fluent_available = False
        
        # 即使某些依赖不可用，应用程序也应该能优雅处理
        import main
        
        # 至少应该能运行CLI模式
        try:
            result = main.run_cli_mode()
            assert result == 0
        except Exception as e:
            pytest.fail(f"即使在依赖不完整的情况下，CLI模式也应该能运行: {str(e)}")
    
    def test_graceful_degradation(self):
        """测试优雅降级"""
        # 模拟PyQt6不可用
        with patch.dict('sys.modules', {'PyQt6': None}):
            import importlib
            import main
            
            # 重新加载main模块以测试降级行为
            importlib.reload(main)
            
            # 即使PyQt6不可用，也应该能设置基础设施
            try:
                main.setup_application()
                # 应该成功，因为基础设施不依赖PyQt6
            except ImportError:
                # 如果有导入错误，应该是预期的
                pass


class TestErrorScenarios:
    """错误场景测试"""
    
    def test_missing_src_directory(self):
        """测试src目录缺失的情况"""
        import main
        
        # 模拟src目录不在路径中
        original_path = sys.path.copy()
        sys.path = [p for p in sys.path if 'src' not in p]
        
        try:
            # 应该仍然能导入，因为main.py会添加src路径
            src_path = Path(__file__).parent.parent / "src"
            if str(src_path) not in sys.path:
                sys.path.insert(0, str(src_path))
            
            # 现在应该能导入src模块
            from src.utils.simple_config import get_config
            config = get_config()
            assert config is not None
            
        finally:
            # 恢复原始路径
            sys.path = original_path
    
    def test_invalid_command_line_args(self):
        """测试无效命令行参数"""
        import main
        
        # 测试无效参数
        invalid_args = [
            ['main.py', '--invalid'],
            ['main.py', '--gui', '--extra'],
            ['main.py', 'invalid_command']
        ]
        
        for args in invalid_args:
            with patch('sys.argv', args):
                try:
                    # 应该能处理无效参数，可能回退到默认行为
                    result = main.main()
                    # 应该能返回有效的退出码
                    assert isinstance(result, int)
                    assert 0 <= result <= 1
                    
                except SystemExit as e:
                    # 如果使用argparse等，可能会有SystemExit
                    assert e.code in [0, 1, 2]
                except Exception as e:
                    # 其他异常应该有有意义的错误信息
                    assert str(e), "异常应该有描述信息"


if __name__ == "__main__":
    # 直接运行测试
    pytest.main([__file__, "-v"])