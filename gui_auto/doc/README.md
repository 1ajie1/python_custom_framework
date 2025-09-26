# GUI自动化框架 API文档

## 概述

GUI自动化框架是一个功能强大、易于使用的Python库，用于自动化GUI操作。该框架采用模块化设计，支持跨平台操作，提供丰富的图像识别、鼠标键盘操作和窗口管理功能。

## 版本信息

- **版本**: 2.0.0
- **作者**: GUI Auto Framework Team
- **邮箱**: support@gui-auto-framework.com

## 目录

- [快速开始](quick_start.md) - 快速入门指南
- [主类API](main_class.md) - 主类GuiAutoToolV2的完整API
- [核心模块](core_modules.md) - 核心配置和工具模块
- [算法模块](algorithms.md) - 图像匹配算法
- [操作模块](operations.md) - GUI操作模块
- [平台模块](platform.md) - 跨平台适配
- [工具模块](utils.md) - 实用工具函数
- [示例代码](examples.md) - 完整的使用示例
- [故障排除](troubleshooting.md) - 常见问题和解决方案

## 主要特性

- 🎯 **多种图像匹配算法**: 模板匹配、特征匹配、混合匹配、金字塔匹配
- 🖱️ **完整的GUI操作**: 鼠标点击、键盘输入、窗口管理
- 🌐 **跨平台支持**: Windows、Linux平台适配
- ⚡ **高性能**: 优化的图像处理和坐标转换
- 🔧 **易于扩展**: 模块化设计，支持自定义算法和操作
- 📊 **详细日志**: 完整的操作日志和错误追踪

## 安装

```bash
# 使用uv安装
uv add gui-auto-framework

# 或使用pip安装
pip install gui-auto-framework
```

## 快速示例

```python
from gui_auto import GuiAutoTool, create_tool

# 创建工具实例
tool = create_tool()

# 截图
screenshot = tool.capture_screen()

# 查找并点击图像
result = tool.find_and_click("button.png")

# 输入文本
tool.type_text("Hello World")

# 点击坐标
tool.click(100, 200)
```

## 架构图

```
GUI自动化框架
├── 主类 (GuiAutoToolV2)
├── 核心模块
│   ├── 配置管理 (GuiConfig)
│   ├── 日志系统 (Logger)
│   └── 异常处理 (Exceptions)
├── 算法模块
│   ├── 模板匹配 (TemplateMatching)
│   ├── 特征匹配 (FeatureMatching)
│   ├── 混合匹配 (HybridMatching)
│   └── 金字塔匹配 (PyramidMatching)
├── 操作模块
│   ├── 图像操作 (ImageOperations)
│   ├── 点击操作 (ClickOperations)
│   ├── 键盘操作 (KeyboardOperations)
│   └── 窗口操作 (WindowOperations)
├── 平台模块
│   ├── Windows平台 (WindowsPlatform)
│   └── Linux平台 (LinuxPlatform)
└── 工具模块
    ├── 图像工具 (ImageUtils)
    ├── 坐标工具 (CoordinateUtils)
    ├── 缩放工具 (ScaleUtils)
    └── 格式工具 (FormatUtils)
```

## 许可证

本项目采用MIT许可证。详情请参阅LICENSE文件。
