# Data Analysis PyQt

基于PyQt6的现代化数据分析桌面应用程序，使用Polars和Fluent Design构建。

## 功能特性

- 📊 **现代数据分析**: 支持CSV和Parquet文件格式
- 🎨 **Fluent Design界面**: 使用PyQt-Fluent-Widgets的美观现代界面
- ⚡ **高性能处理**: 基于Rust的Polars进行快速数据处理
- 📈 **丰富可视化**: 使用matplotlib和plotly创建Nature风格图表
- 🗄️ **历史管理**: 基于SQLite的分析历史记录和缓存
- 🔄 **异步处理**: 后台数据处理，界面不冻结

## 快速开始

### 环境要求

- Python 3.11 或更高版本
- uv 包管理器

### 安装

```bash
# 克隆仓库
git clone https://github.com/example/data-analysis-pyqt.git
cd data-analysis-pyqt

# 使用uv安装依赖
uv install

# 运行应用程序
uv run dev
```

### 开发

```bash
# 安装开发依赖
uv install --extra dev

# 运行测试
uv run test

# 运行测试并生成覆盖率报告
uv run test-cov

# 格式化代码
uv run format

# 类型检查
uv run type-check

# 代码检查
uv run lint
```

## 架构设计

应用程序采用模块化架构：

- `src/ui/` - 用户界面组件和页面
- `src/core/` - 核心业务逻辑和数据处理
- `src/models/` - 数据模型和类型定义
- `src/utils/` - 工具函数和辅助类
- `resources/` - 静态资源（图标、样式、配置）
- `data/` - 本地数据存储和数据库
- `tests/` - 测试套件

## 功能概述

### 数据上传
- 拖拽文件上传
- 支持CSV和Parquet格式
- 自动时间列检测
- 数据预览和验证

### 分析引擎
- 描述性统计
- 关联矩阵和热力图
- 异常值检测
- 时间序列平稳性检验

### 可视化
- 使用plotly的交互式图表
- 使用matplotlib的出版级图表
- Nature期刊风格格式
- 导出多种格式（PNG、SVG、PDF）

### 历史管理
- 自动分析缓存
- 基于SQLite的存储
- 搜索和过滤功能
- 导出分析报告

## 许可证

MIT许可证 - 详见LICENSE文件。

## 贡献

欢迎贡献！请阅读我们的贡献指南并提交拉取请求。
