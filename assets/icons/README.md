# 应用程序图标

这个目录包含应用程序的图标文件。

## 图标文件

- `app_icon.ico` - Windows应用程序图标 (256x256, 128x128, 64x64, 48x48, 32x32, 16x16)
- `app_icon.png` - Linux应用程序图标 (512x512)
- `app_icon.icns` - macOS应用程序图标 (512x512, 256x256, 128x128, 64x64, 32x32, 16x16)
- `csv_icon.ico` - CSV文件关联图标
- `excel_icon.ico` - Excel文件关联图标  
- `parquet_icon.ico` - Parquet文件关联图标

## 图标设计要求

### 主应用图标
- **尺寸**: 512x512像素的基础设计
- **格式**: PNG (透明背景)
- **风格**: 现代化、扁平化设计
- **颜色**: 蓝色主调 (#2563EB)，渐变到浅蓝 (#3B82F6)
- **元素**: 数据图表元素（柱状图、折线图、饼图）

### 设计概念
```
┌─────────────────────┐
│  📊 Data Analysis   │  
│                     │
│    ╭─╮ ╭─╮ ╭─╮     │
│    │█│ │█│ │█│     │  
│    │█│ │█│ │█│     │
│    ╰─╯ ╰─╯ ╰─╯     │
│                     │
│  ∼∼∼∼∼∼∼∼∼∼∼∼∼    │
│                     │
└─────────────────────┘
```

### 文件格式图标
- **CSV**: 表格图标 + CSV标识
- **Excel**: 电子表格图标 + Excel绿色
- **Parquet**: 列式存储图标 + 橙色标识

## 图标生成工具

推荐使用以下工具生成多格式图标：

1. **在线工具**:
   - [Favicon.io](https://favicon.io/) - ICO格式生成
   - [IconGenerator](https://icongenerator.net/) - 多平台图标

2. **桌面工具**:
   - [IcoFX](https://icofx.ro/) - Windows图标编辑器
   - [Icon Composer](https://developer.apple.com/xcode/) - macOS图标工具

3. **命令行工具**:
   ```bash
   # 使用ImageMagick生成多尺寸ICO
   magick app_icon.png -resize 256x256 -resize 128x128 -resize 64x64 -resize 48x48 -resize 32x32 -resize 16x16 app_icon.ico
   
   # 生成macOS ICNS
   iconutil -c icns app_icon.iconset/
   ```

## 使用说明

1. 将设计好的图标文件放置在对应位置
2. 确保文件名与pyproject.toml中的配置一致
3. 验证图标在不同尺寸下的显示效果
4. 测试在不同操作系统上的兼容性

## 注意事项

- 图标文件不应过大（建议每个文件<1MB）
- 确保在高DPI显示器上清晰显示
- 遵循各平台的图标设计规范
- 支持透明背景以适应不同主题