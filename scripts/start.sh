#!/bin/bash
# Data Analysis App 启动脚本
# 版本: 1.0.0

set -e

# 应用程序信息
APP_NAME="Data Analysis Pro"
APP_VERSION="1.0.0"
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../" && pwd)"
MAIN_SCRIPT="$APP_DIR/main.py"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 显示帮助信息
show_help() {
    cat << EOF
$APP_NAME v$APP_VERSION 启动脚本

用法: $(basename "$0") [选项] [文件]

选项:
    -h, --help          显示此帮助信息
    -v, --version       显示版本信息
    --gui               启动GUI界面（默认）
    --cli               启动命令行界面
    --debug             启用调试模式
    --config FILE       指定配置文件路径
    --new               启动时创建新分析
    --open              打开文件对话框
    --check             检查系统环境
    --install-deps      安装依赖包

示例:
    $(basename "$0")                    # 启动GUI界面
    $(basename "$0") --debug            # 调试模式启动
    $(basename "$0") data.csv           # 启动并打开文件
    $(basename "$0") --check            # 检查系统环境

更多信息请访问: https://github.com/example/data-analysis-pyqt
EOF
}

# 显示版本信息
show_version() {
    cat << EOF
$APP_NAME
版本: $APP_VERSION
构建: $(date '+%Y%m%d')
Python要求: 3.11+
平台: $(uname -s) $(uname -m)

Copyright (c) 2024 Data Analysis Team
License: MIT
EOF
}

# 检查Python环境
check_python() {
    log_info "检查Python环境..."
    
    if ! command -v python3 &> /dev/null; then
        log_error "Python3未安装或不在PATH中"
        return 1
    fi
    
    local python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    local required_version="3.11"
    
    if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 11) else 1)"; then
        log_error "Python版本过低: $python_version (需要 >= $required_version)"
        return 1
    fi
    
    log_success "Python版本: $python_version ✓"
    return 0
}

# 检查依赖包
check_dependencies() {
    log_info "检查依赖包..."
    
    local missing_deps=()
    local deps=("PyQt6" "polars" "numpy" "matplotlib" "pandas")
    
    for dep in "${deps[@]}"; do
        if ! python3 -c "import $dep" &> /dev/null; then
            missing_deps+=("$dep")
        fi
    done
    
    if [ ${#missing_deps[@]} -eq 0 ]; then
        log_success "所有依赖包已安装 ✓"
        return 0
    else
        log_warning "缺失依赖包: ${missing_deps[*]}"
        return 1
    fi
}

# 安装依赖包
install_dependencies() {
    log_info "安装依赖包..."
    
    if [ -f "$APP_DIR/requirements.txt" ]; then
        python3 -m pip install -r "$APP_DIR/requirements.txt"
        log_success "依赖包安装完成"
    else
        log_error "未找到requirements.txt文件"
        return 1
    fi
}

# 检查系统环境
check_system() {
    log_info "系统环境检查"
    echo "===================="
    
    # 操作系统信息
    log_info "操作系统: $(uname -s) $(uname -r)"
    log_info "架构: $(uname -m)"
    
    # 内存信息
    if command -v free &> /dev/null; then
        local memory=$(free -h | awk '/^Mem:/ {print $2}')
        log_info "系统内存: $memory"
    fi
    
    # 磁盘空间
    local disk_space=$(df -h "$APP_DIR" | awk 'NR==2 {print $4}')
    log_info "可用磁盘空间: $disk_space"
    
    # Python环境
    if check_python; then
        check_dependencies
    fi
    
    # GUI环境检查
    if [ -n "$DISPLAY" ] || [ -n "$WAYLAND_DISPLAY" ]; then
        log_success "GUI环境可用 ✓"
    else
        log_warning "未检测到GUI环境"
    fi
    
    echo "===================="
}

# 启动应用程序
start_app() {
    local args=("$@")
    
    # 检查主脚本是否存在
    if [ ! -f "$MAIN_SCRIPT" ]; then
        log_error "主脚本未找到: $MAIN_SCRIPT"
        exit 1
    fi
    
    # 设置Python路径
    export PYTHONPATH="$APP_DIR:$PYTHONPATH"
    
    # 启动应用程序
    log_info "启动 $APP_NAME..."
    
    cd "$APP_DIR"
    exec python3 "$MAIN_SCRIPT" "${args[@]}"
}

# 主函数
main() {
    local app_args=()
    local show_help_flag=false
    local show_version_flag=false
    local check_system_flag=false
    local install_deps_flag=false
    
    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help_flag=true
                shift
                ;;
            -v|--version)
                show_version_flag=true
                shift
                ;;
            --check)
                check_system_flag=true
                shift
                ;;
            --install-deps)
                install_deps_flag=true
                shift
                ;;
            *)
                app_args+=("$1")
                shift
                ;;
        esac
    done
    
    # 处理特殊标志
    if [ "$show_help_flag" = true ]; then
        show_help
        exit 0
    fi
    
    if [ "$show_version_flag" = true ]; then
        show_version
        exit 0
    fi
    
    if [ "$check_system_flag" = true ]; then
        check_system
        exit 0
    fi
    
    if [ "$install_deps_flag" = true ]; then
        install_dependencies
        exit $?
    fi
    
    # 启动应用程序
    start_app "${app_args[@]}"
}

# 脚本入口
main "$@"