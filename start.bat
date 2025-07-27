@echo off
REM Data Analysis App Windows 启动脚本
REM 版本: 1.0.0

setlocal enabledelayedexpansion

REM 应用程序信息
set "APP_NAME=Data Analysis Pro"
set "APP_VERSION=1.0.0"
set "APP_DIR=%~dp0"
set "MAIN_SCRIPT=%APP_DIR%main.py"

REM 颜色定义（Windows 10+）
for /F %%a in ('echo prompt $E ^| cmd') do set "ESC=%%a"
set "RED=%ESC%[31m"
set "GREEN=%ESC%[32m"
set "YELLOW=%ESC%[33m"
set "BLUE=%ESC%[34m"
set "NC=%ESC%[0m"

REM 日志函数
:log_info
echo %BLUE%[INFO]%NC% %~1
goto :eof

:log_success
echo %GREEN%[SUCCESS]%NC% %~1
goto :eof

:log_warning
echo %YELLOW%[WARNING]%NC% %~1
goto :eof

:log_error
echo %RED%[ERROR]%NC% %~1
goto :eof

REM 显示帮助信息
:show_help
echo %APP_NAME% v%APP_VERSION% 启动脚本
echo.
echo 用法: %~nx0 [选项] [文件]
echo.
echo 选项:
echo     /h, /help          显示此帮助信息
echo     /v, /version       显示版本信息
echo     /gui               启动GUI界面（默认）
echo     /cli               启动命令行界面
echo     /debug             启用调试模式
echo     /config FILE       指定配置文件路径
echo     /new               启动时创建新分析
echo     /open              打开文件对话框
echo     /check             检查系统环境
echo     /install-deps      安装依赖包
echo.
echo 示例:
echo     %~nx0                     启动GUI界面
echo     %~nx0 /debug              调试模式启动
echo     %~nx0 data.csv            启动并打开文件
echo     %~nx0 /check              检查系统环境
echo.
echo 更多信息请访问: https://github.com/example/data-analysis-pyqt
goto :eof

REM 显示版本信息
:show_version
echo %APP_NAME%
echo 版本: %APP_VERSION%
echo 构建: %date:~0,4%%date:~5,2%%date:~8,2%
echo Python要求: 3.11+
echo 平台: Windows %PROCESSOR_ARCHITECTURE%
echo.
echo Copyright (c) 2024 Data Analysis Team
echo License: MIT
goto :eof

REM 检查Python环境
:check_python
call :log_info "检查Python环境..."

REM 检查python命令
python --version >nul 2>&1
if !errorlevel! neq 0 (
    py --version >nul 2>&1
    if !errorlevel! neq 0 (
        call :log_error "Python未安装或不在PATH中"
        exit /b 1
    ) else (
        set "PYTHON_CMD=py"
    )
) else (
    set "PYTHON_CMD=python"
)

REM 检查Python版本
for /f "tokens=2" %%a in ('!PYTHON_CMD! --version 2^>^&1') do set "PYTHON_VERSION=%%a"
call :log_success "Python版本: !PYTHON_VERSION! ✓"

REM 检查版本号是否符合要求（简化检查）
!PYTHON_CMD! -c "import sys; exit(0 if sys.version_info >= (3, 11) else 1)"
if !errorlevel! neq 0 (
    call :log_error "Python版本过低 (需要 >= 3.11)"
    exit /b 1
)

exit /b 0

REM 检查依赖包
:check_dependencies
call :log_info "检查依赖包..."

set "missing_deps="
set "deps=PyQt6 polars numpy matplotlib pandas"

for %%d in (%deps%) do (
    !PYTHON_CMD! -c "import %%d" >nul 2>&1
    if !errorlevel! neq 0 (
        set "missing_deps=!missing_deps! %%d"
    )
)

if "!missing_deps!"=="" (
    call :log_success "所有依赖包已安装 ✓"
    exit /b 0
) else (
    call :log_warning "缺失依赖包:!missing_deps!"
    exit /b 1
)

REM 安装依赖包
:install_dependencies
call :log_info "安装依赖包..."

if exist "%APP_DIR%requirements.txt" (
    !PYTHON_CMD! -m pip install -r "%APP_DIR%requirements.txt"
    if !errorlevel! equ 0 (
        call :log_success "依赖包安装完成"
    ) else (
        call :log_error "依赖包安装失败"
        exit /b 1
    )
) else (
    call :log_error "未找到requirements.txt文件"
    exit /b 1
)
exit /b 0

REM 检查系统环境
:check_system
call :log_info "系统环境检查"
echo ====================

REM 操作系统信息
call :log_info "操作系统: %OS%"
call :log_info "架构: %PROCESSOR_ARCHITECTURE%"

REM 内存信息
for /f "skip=1 tokens=2 delims==" %%a in ('wmic computersystem get TotalPhysicalMemory /value') do (
    set /a "memory_gb=%%a/1024/1024/1024"
    call :log_info "系统内存: !memory_gb! GB"
    goto :memory_done
)
:memory_done

REM 磁盘空间
for /f "tokens=3" %%a in ('dir /-c "%APP_DIR%" 2^>nul ^| find "bytes free"') do (
    set /a "disk_gb=%%a/1024/1024/1024"
    call :log_info "可用磁盘空间: !disk_gb! GB"
    goto :disk_done
)
:disk_done

REM Python环境
call :check_python
if !errorlevel! equ 0 (
    call :check_dependencies
)

echo ====================
exit /b 0

REM 启动应用程序
:start_app
REM 检查主脚本是否存在
if not exist "%MAIN_SCRIPT%" (
    call :log_error "主脚本未找到: %MAIN_SCRIPT%"
    pause
    exit /b 1
)

REM 设置Python路径
set "PYTHONPATH=%APP_DIR%;%PYTHONPATH%"

REM 启动应用程序
call :log_info "启动 %APP_NAME%..."

cd /d "%APP_DIR%"
!PYTHON_CMD! "%MAIN_SCRIPT%" %*
exit /b %errorlevel%

REM 主函数
:main
set "app_args="
set "show_help_flag="
set "show_version_flag="
set "check_system_flag="
set "install_deps_flag="

REM 解析命令行参数
:parse_args
if "%~1"=="" goto :args_done

if /i "%~1"=="/h" set "show_help_flag=1" & shift & goto :parse_args
if /i "%~1"=="/help" set "show_help_flag=1" & shift & goto :parse_args
if /i "%~1"=="/v" set "show_version_flag=1" & shift & goto :parse_args
if /i "%~1"=="/version" set "show_version_flag=1" & shift & goto :parse_args
if /i "%~1"=="/check" set "check_system_flag=1" & shift & goto :parse_args
if /i "%~1"=="/install-deps" set "install_deps_flag=1" & shift & goto :parse_args

set "app_args=%app_args% %~1"
shift
goto :parse_args

:args_done

REM 处理特殊标志
if defined show_help_flag (
    call :show_help
    pause
    exit /b 0
)

if defined show_version_flag (
    call :show_version
    pause
    exit /b 0
)

if defined check_system_flag (
    call :check_system
    pause
    exit /b 0
)

if defined install_deps_flag (
    call :install_dependencies
    pause
    exit /b %errorlevel%
)

REM 启动应用程序
call :start_app %app_args%
exit /b %errorlevel%

REM 脚本入口
call :main %*