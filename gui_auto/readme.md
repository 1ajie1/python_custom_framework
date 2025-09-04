# GUI自动化工具 v1.1 使用手册

## 简介

GUI自动化工具是一个基于图像识别的Python自动化框架，支持跨分辨率和DPI缩放的智能图像匹配，提供丰富的GUI操作功能。该工具特别适用于需要处理不同屏幕分辨率和DPI设置的自动化场景。

## 主要特性

- 🎯 **智能图像匹配**：支持多种匹配算法（模板匹配、特征匹配、混合匹配）
- 🔍 **多尺度匹配**：自动处理不同分辨率和DPI缩放
- 🖼️ **图像增强**：多种增强级别，提高匹配精度
- 🎮 **完整GUI操作**：点击、双击、拖拽、键盘输入等
- 🔄 **重试机制**：内置重试装饰器，提高操作稳定性
- 📊 **详细日志**：完整的操作日志和调试信息
- ⚡ **高性能**：优化的算法实现，快速响应

## 安装

### 系统要求

- Python 3.11+
- Windows 10/11（当前版本主要针对Windows优化）

### 依赖安装

```bash
# 使用pip安装依赖
pip install opencv-python>=4.8.0 pyautogui>=0.9.54 Pillow>=10.0.0 numpy>=1.24.0 typing-extensions>=4.7.0 rich>=14.1.0

# 或使用uv安装
uv sync
```

## 快速开始

### 基本使用

```python
from auto_gui_tool import GuiAutoTool

# 创建工具实例
tool = GuiAutoTool()

# 获取屏幕截图
screenshot = tool.get_screen_screenshot()

# 查找图像
location = tool.find_image_in_target(
    template="button.png",  # 模板图像路径
    target_image=screenshot  # 目标图像（可以是路径或numpy数组）
)

if location:
    print(f"找到图像，位置: {location}")
    # 点击图像
    tool.click_image("button.png", screenshot)
else:
    print("未找到图像")
```

## 详细配置

### 初始化参数

```python
tool = GuiAutoTool(
    confidence=0.8,                    # 匹配置信度阈值 (0.0-1.0)
    timeout=10.0,                      # 操作超时时间（秒）
    default_method="TM_CCOEFF_NORMED", # 默认匹配方法
    auto_scale=True,                   # 是否自动处理DPI和分辨率缩放
    default_max_retries=3,             # 默认最大重试次数
    default_retry_delay=0.5,           # 默认重试延迟时间（秒）
    default_use_multi_scale=False,     # 默认是否启用多尺度匹配
    default_enhancement_level="light", # 默认图像增强级别
    template_scale_info={              # 默认模板图像缩放信息
        'dpi_scale': 1.0,
        'resolution': (1920, 1080)
    },
    base_scale=1.0,                    # 自定义基准DPI缩放
    base_resolution=(1920, 1080)       # 自定义基准分辨率
)
```

### 匹配方法

支持以下OpenCV模板匹配方法：

- `TM_CCOEFF_NORMED`：归一化相关系数匹配（推荐）
- `TM_CCORR_NORMED`：归一化相关匹配
- `TM_SQDIFF_NORMED`：归一化平方差匹配

### 图像增强级别

- `light`：轻度增强，适用于高质量图像
- `standard`：标准增强，平衡效果和性能
- `aggressive`：激进增强，适用于低质量或模糊图像
- `adaptive`：自适应增强，根据图像特性自动选择策略

## 核心功能

### 1. 图像查找

```python
# 基本查找
location = tool.find_image_in_target(
    template="template.png",
    target_image="screenshot.png"
)

# 高级查找（多方法尝试）
location = tool.find_image_in_target(
    template="template.png",
    target_image="screenshot.png",
    try_multiple_methods=True,  # 尝试多种方法
    use_multi_scale=True,       # 启用多尺度匹配
    enhancement_level="adaptive" # 自适应增强
)

# 在指定区域查找
location = tool.find_image_in_target(
    template="template.png",
    target_image="screenshot.png",
    region=(100, 100, 500, 300)  # (x, y, width, height)
)
```

### 2. 图像比较

```python
# 详细比较结果
result = tool.compare_images(
    template_path="template.png",
    target_path="screenshot.png",
    method="TM_CCOEFF_NORMED",
    enhance_images=True,
    save_result=True,              # 保存标注结果
    result_path="result.png",      # 结果图片路径
    return_system_coordinates=True, # 返回系统坐标
    use_multi_scale=True,
    enhancement_level="adaptive"
)

print(f"置信度: {result['confidence']}")
print(f"位置: {result['location']}")
print(f"中心点: {result['center']}")
print(f"是否找到: {result['found']}")
```

### 3. 鼠标操作

```python
# 点击图像
success = tool.click_image(
    template="button.png",
    target_image=screenshot,
    button="left",              # 鼠标按键
    offset=(10, 5),            # 点击偏移量
    enhancement_level="standard"
)

# 双击图像
success = tool.double_click_image(
    template="icon.png",
    target_image=screenshot,
    offset=(0, 0)
)

# 拖拽操作
success = tool.drag_from_to_image(
    from_template="source.png",
    to_template="target.png",
    target_image=screenshot,
    duration=1.0
)

# 移动鼠标到图像位置
success = tool.move_to_image(
    template="target.png",
    target_image=screenshot,
    offset=(0, 0),
    duration=0.5
)
```

### 4. 键盘操作

```python
# 输入文本
tool.type_text("Hello World", interval=0.05)

# 按键
tool.press_key("enter")

# 组合键
tool.press_keys(["ctrl", "c"])
```

## 高级功能

### 1. 缩放信息配置

```python
# 为特定模板配置缩放信息
template_scale_info = {
    'dpi_scale': 1.25,           # 模板图像的DPI缩放
    'resolution': (2560, 1440)   # 模板图像的分辨率
}

location = tool.find_image_in_target(
    template="template.png",
    target_image=screenshot,
    template_scale_info=template_scale_info
)
```

### 2. 特征匹配

```python
# 强制使用特征匹配（对旋转和缩放更鲁棒）
result = tool.compare_images(
    template_path="template.png",
    target_path="screenshot.png",
    use_feature_matching=True
)
```

### 3. 混合匹配

```python
# 使用混合匹配（结合模板匹配和特征匹配）
result = tool.compare_images(
    template_path="template.png",
    target_path="screenshot.png",
    use_multi_scale=True,
    enhancement_level="adaptive"  # 自适应模式会使用混合匹配
)
```

### 4. 重试装饰器

```python
from auto_gui_tool import retry

@retry(max_attempts=5, delay=1.0, backoff=2.0)
def my_operation():
    # 您的操作代码
    pass
```

## 实际应用示例

### 示例1：自动化登录

```python
from auto_gui_tool import GuiAutoTool
import time

tool = GuiAutoTool(confidence=0.8)

# 获取屏幕截图
screenshot = tool.get_screen_screenshot()

# 点击用户名输入框
if tool.click_image("username_field.png", screenshot):
    tool.type_text("my_username")
    
# 点击密码输入框
if tool.click_image("password_field.png", screenshot):
    tool.type_text("my_password")
    
# 点击登录按钮
if tool.click_image("login_button.png", screenshot):
    print("登录成功")
```

### 示例2：文件操作

```python
# 拖拽文件
screenshot = tool.get_screen_screenshot()
success = tool.drag_from_to_image(
    from_template="file.png",
    to_template="folder.png",
    target_image=screenshot,
    duration=1.5
)

if success:
    print("文件拖拽成功")
```

### 示例3：批量处理

```python
# 批量点击多个按钮
buttons = ["button1.png", "button2.png", "button3.png"]
screenshot = tool.get_screen_screenshot()

for button in buttons:
    if tool.click_image(button, screenshot):
        print(f"点击 {button} 成功")
        time.sleep(0.5)  # 等待界面响应
    else:
        print(f"未找到 {button}")
```

## 最佳实践

### 1. 模板图像准备

- 使用清晰的模板图像，避免模糊或压缩
- 模板图像应该包含足够的特征点
- 避免使用纯色或过于简单的图像作为模板

### 2. 缩放配置

- 为不同分辨率的模板图像配置正确的缩放信息
- 使用`auto_scale=True`自动处理系统缩放
- 在混合分辨率环境中测试您的脚本

### 3. 性能优化

- 使用`region`参数限制搜索区域
- 根据图像质量选择合适的增强级别
- 对于简单场景，使用`light`增强级别

### 4. 错误处理

```python
try:
    location = tool.find_image_in_target("template.png", screenshot)
    if location:
        tool.click_image("template.png", screenshot)
    else:
        print("未找到目标图像")
except Exception as e:
    print(f"操作失败: {e}")
```

### 5. 调试技巧

```python
# 保存匹配结果用于调试
result = tool.compare_images(
    template_path="template.png",
    target_path="screenshot.png",
    save_result=True,
    result_path="debug_result.png"
)

# 查看详细匹配信息
print(f"匹配方法: {result['method_used']}")
print(f"增强级别: {result['enhancement_level']}")
print(f"多尺度匹配: {result['multi_scale_enabled']}")
```

## 故障排除

### 常见问题

1. **找不到图像**
   - 检查模板图像是否清晰
   - 尝试不同的匹配方法
   - 调整置信度阈值
   - 使用多尺度匹配

2. **点击位置不准确**
   - 检查DPI缩放设置
   - 使用偏移量调整点击位置
   - 验证坐标转换是否正确

3. **性能问题**
   - 限制搜索区域
   - 使用更轻量的增强级别
   - 减少重试次数

### 调试模式

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 现在会显示详细的调试信息
tool = GuiAutoTool()
```

## 版本信息

- **当前版本**: 1.1
- **Python要求**: 3.11+
- **主要依赖**: OpenCV, PyAutoGUI, Pillow, NumPy

## 许可证

请查看项目根目录的LICENSE文件了解详细许可信息。

## 贡献

欢迎提交Issue和Pull Request来改进这个工具。

## 更新日志

### v1.1
- 新增混合匹配算法
- 改进多尺度匹配性能
- 增强图像预处理功能
- 优化坐标转换系统
- 添加详细的使用手册

---

如有问题或建议，请通过Issue联系我们。
