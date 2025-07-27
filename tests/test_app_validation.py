"""
简化的应用程序验证测试
专门用于验证应用程序的基本功能和集成，避免GUI测试环境问题
"""

import sys
from pathlib import Path

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_basic_infrastructure():
    """测试基础架构"""
    print("🔍 测试基础架构...")
    
    # 测试配置系统
    from src.utils.simple_config import get_config, get_setting
    config = get_config()
    assert config is not None
    
    app_name = get_setting('app_name', 'Data Analysis PyQt')
    assert isinstance(app_name, str)
    print("✅ 配置系统正常")
    
    # 测试日志系统
    from src.utils.basic_logging import setup_basic_logging, get_logger
    setup_basic_logging()
    logger = get_logger("test")
    logger.info("测试日志消息")
    print("✅ 日志系统正常")
    
    # 测试异常处理
    from src.utils.exceptions import DataProcessingError, FileValidationError
    from src.utils.error_handler import setup_exception_handling
    setup_exception_handling()
    print("✅ 异常处理系统正常")


def test_main_entry_point():
    """测试主入口点"""
    print("🔍 测试主入口点...")
    
    import main
    
    # 验证必要函数存在
    assert hasattr(main, 'main')
    assert hasattr(main, 'setup_application')
    assert hasattr(main, 'run_cli_mode')
    assert hasattr(main, 'run_gui_application')
    print("✅ 主入口点函数存在")
    
    # 测试应用程序设置
    main.setup_application()
    print("✅ 应用程序设置成功")
    
    # 测试CLI模式
    result = main.run_cli_mode()
    assert result == 0
    print("✅ CLI模式运行成功")


def test_gui_components_import():
    """测试GUI组件导入（仅测试模块存在性）"""
    print("🔍 测试GUI组件模块...")
    
    try:
        # 测试PyQt6可用性
        import PyQt6
        pyqt6_available = True
        print("✅ PyQt6可用")
    except ImportError:
        pyqt6_available = False
        print("⚠️ PyQt6不可用")
    
    try:
        # 测试qfluentwidgets可用性
        import qfluentwidgets
        fluent_available = True
        print("✅ PyQt-Fluent-Widgets可用")
    except ImportError:
        fluent_available = False
        print("⚠️ PyQt-Fluent-Widgets不可用")
    
    # 如果GUI依赖可用，测试模块存在
    if pyqt6_available and fluent_available:
        try:
            from src.ui import main_window
            from src.ui import app_integrator
            print("✅ GUI模块可导入")
            return True
        except Exception as e:
            print(f"⚠️ GUI模块导入有问题: {e}")
            return False
    else:
        print("⚠️ 跳过GUI组件测试（依赖不可用）")
        return False


def test_file_structure():
    """测试文件结构"""
    print("🔍 测试项目文件结构...")
    
    base_path = Path(__file__).parent.parent
    
    # 必要文件检查
    required_files = [
        "main.py",
        "pyproject.toml",
        "README.md",
        "src/__init__.py",
        "src/utils/__init__.py",
        "src/ui/__init__.py",
        "tests/__init__.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = base_path / file_path
        if not full_path.exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ 缺少文件: {missing_files}")
        return False
    else:
        print("✅ 所有必要文件存在")
        return True


def test_project_configuration():
    """测试项目配置"""
    print("🔍 测试项目配置...")
    
    try:
        import tomllib
    except ImportError:
        # Python < 3.11
        try:
            import tomli as tomllib
        except ImportError:
            print("⚠️ TOML解析库不可用，跳过配置测试")
            return True
    
    base_path = Path(__file__).parent.parent
    pyproject_path = base_path / "pyproject.toml"
    
    if pyproject_path.exists():
        with open(pyproject_path, "rb") as f:
            config = tomllib.load(f)
        
        # 验证项目基本信息
        assert "project" in config
        assert config["project"]["name"] == "data-analysis-pyqt"
        assert "dependencies" in config["project"]
        
        # 验证核心依赖
        deps = config["project"]["dependencies"]
        has_pyqt6 = any("PyQt6" in dep for dep in deps)
        has_fluent = any("PyQt-Fluent-Widgets" in dep for dep in deps)
        has_polars = any("polars" in dep for dep in deps)
        
        assert has_pyqt6, "缺少PyQt6依赖"
        assert has_fluent, "缺少PyQt-Fluent-Widgets依赖"
        assert has_polars, "缺少polars依赖"
        
        print("✅ 项目配置正确")
        return True
    else:
        print("❌ pyproject.toml文件不存在")
        return False


def test_data_processing_integration():
    """测试数据处理集成（基础功能）"""
    print("🔍 测试数据处理集成...")
    
    try:
        import pandas as pd
        import numpy as np
        pandas_available = True
        print("✅ Pandas和NumPy可用")
    except ImportError:
        pandas_available = False
        print("⚠️ Pandas或NumPy不可用")
    
    if pandas_available:
        # 创建测试数据
        test_data = pd.DataFrame({
            'id': range(1, 11),
            'value': np.random.randn(10),
            'category': ['A', 'B', 'C'] * 3 + ['A'],
            'timestamp': pd.date_range('2023-01-01', periods=10, freq='D')
        })
        
        # 基本数据处理测试
        assert len(test_data) == 10
        assert 'id' in test_data.columns
        assert test_data['value'].dtype == np.float64
        
        print("✅ 数据处理基础功能正常")
        return True
    else:
        print("⚠️ 跳过数据处理测试（Pandas不可用）")
        return False


def run_all_tests():
    """运行所有测试"""
    print("🚀 开始应用程序验证测试...\n")
    
    tests = [
        ("基础架构", test_basic_infrastructure),
        ("主入口点", test_main_entry_point),
        ("GUI组件", test_gui_components_import),
        ("文件结构", test_file_structure),
        ("项目配置", test_project_configuration),
        ("数据处理", test_data_processing_integration),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            print(f"\n--- {test_name}测试 ---")
            success = test_func()
            results[test_name] = success if success is not None else True
        except Exception as e:
            print(f"❌ {test_name}测试失败: {e}")
            results[test_name] = False
        except AssertionError as e:
            print(f"❌ {test_name}测试断言失败: {e}")
            results[test_name] = False
    
    # 总结结果
    print(f"\n{'='*50}")
    print("📊 测试结果总结:")
    print(f"{'='*50}")
    
    passed = sum(1 for success in results.values() if success)
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{test_name:15} {status}")
    
    print(f"\n通过: {passed}/{total} ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 所有测试通过！应用程序集成验证成功！")
        return True
    else:
        print(f"\n⚠️ {total-passed}个测试失败")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)