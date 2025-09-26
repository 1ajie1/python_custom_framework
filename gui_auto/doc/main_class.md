# 主类API - GuiAutoToolV2

## 概述

`GuiAutoToolV2` 是GUI自动化框架的主类，提供统一的API接口来执行各种GUI自动化操作。

## 类定义

```python
class GuiAutoToolV2:
    """重构后的GUI自动化工具主类"""
```

## 构造函数

### `__init__(config: GuiConfig = None)`

初始化GUI自动化工具实例。

**参数:**
- `config` (GuiConfig, 可选): 配置对象。如果为None，将使用默认配置。

**示例:**
```python
from gui_auto import GuiAutoToolV2, GuiConfig

# 使用默认配置
tool = GuiAutoToolV2()

# 使用自定义配置
config = GuiConfig(
    match_confidence=0.9,
    match_timeout=15.0,
    auto_scale=True
)
tool = GuiAutoToolV2(config)
```

## 图像操作API

### `find_image(template, target=None, algorithm="template", **kwargs)`

在目标图像中查找模板图像。

**参数:**
- `template` (str | np.ndarray): 模板图像路径或numpy数组
- `target` (str | np.ndarray, 可选): 目标图像路径或numpy数组。如果为None，将截取当前屏幕
- `algorithm` (str): 匹配算法名称，可选值：'template', 'feature', 'hybrid', 'pyramid'
- `**kwargs`: 其他算法特定参数

**返回:**
- `MatchResult`: 匹配结果对象，包含匹配位置、置信度等信息

**示例:**
```python
# 在屏幕中查找按钮
result = tool.find_image("button.png")

# 在指定图像中查找
result = tool.find_image("button.png", "screenshot.png", algorithm="feature")

# 使用自定义参数
result = tool.find_image("button.png", confidence=0.9, scale_range=(0.8, 1.2))
```

### `find_and_click(template, target=None, algorithm="template", **kwargs)`

查找图像并点击。

**参数:**
- `template` (str | np.ndarray): 模板图像路径或numpy数组
- `target` (str | np.ndarray, 可选): 目标图像路径或numpy数组
- `algorithm` (str): 匹配算法名称
- `**kwargs`: 其他参数

**返回:**
- `bool`: 是否成功找到并点击

**示例:**
```python
# 查找并点击按钮
success = tool.find_and_click("button.png")

# 使用特征匹配算法
success = tool.find_and_click("button.png", algorithm="feature")
```

### `capture_screen(region=None, target_format="bgr", **kwargs)`

截取屏幕图像。

**参数:**
- `region` (tuple, 可选): 截取区域 (x, y, width, height)
- `target_format` (str): 目标格式，'bgr' 或 'rgb'
- `**kwargs`: 其他参数

**返回:**
- `np.ndarray`: 截取的图像数组

**示例:**
```python
# 截取整个屏幕
screenshot = tool.capture_screen()

# 截取指定区域
region_screenshot = tool.capture_screen(region=(100, 100, 800, 600))

# 截取为RGB格式
rgb_screenshot = tool.capture_screen(target_format="rgb")
```

### `compare_images(image1, image2, method="ssim", **kwargs)`

比较两个图像的相似度。

**参数:**
- `image1` (str | np.ndarray): 第一个图像
- `image2` (str | np.ndarray): 第二个图像
- `method` (str): 比较方法，'ssim', 'mse', 'histogram'
- `**kwargs`: 其他参数

**返回:**
- `float`: 相似度分数 (0-1)

**示例:**
```python
# 比较两个图像
similarity = tool.compare_images("image1.png", "image2.png")

# 使用MSE方法比较
mse_score = tool.compare_images("image1.png", "image2.png", method="mse")
```

## 鼠标操作API

### `click(x, y, button="left", clicks=1, interval=0.0, **kwargs)`

在指定坐标点击鼠标。

**参数:**
- `x` (int | float): X坐标
- `y` (int | float): Y坐标
- `button` (str): 鼠标按钮，'left', 'right', 'middle'
- `clicks` (int): 点击次数
- `interval` (float): 点击间隔（秒）
- `**kwargs`: 其他参数

**返回:**
- `bool`: 是否成功点击

**示例:**
```python
# 左键单击
tool.click(100, 200)

# 右键双击
tool.click(100, 200, button="right", clicks=2)

# 带间隔的多次点击
tool.click(100, 200, clicks=3, interval=0.1)
```

### `double_click(x, y, button="left", **kwargs)`

在指定坐标双击鼠标。

**参数:**
- `x` (int | float): X坐标
- `y` (int | float): Y坐标
- `button` (str): 鼠标按钮
- `**kwargs`: 其他参数

**返回:**
- `bool`: 是否成功双击

**示例:**
```python
# 左键双击
tool.double_click(100, 200)

# 右键双击
tool.double_click(100, 200, button="right")
```

### `right_click(x, y, **kwargs)`

在指定坐标右键点击。

**参数:**
- `x` (int | float): X坐标
- `y` (int | float): Y坐标
- `**kwargs`: 其他参数

**返回:**
- `bool`: 是否成功右键点击

**示例:**
```python
tool.right_click(100, 200)
```

### `drag(x1, y1, x2, y2, duration=0.5, **kwargs)`

从一点拖拽到另一点。

**参数:**
- `x1` (int | float): 起始X坐标
- `y1` (int | float): 起始Y坐标
- `x2` (int | float): 结束X坐标
- `y2` (int | float): 结束Y坐标
- `duration` (float): 拖拽持续时间（秒）
- `**kwargs`: 其他参数

**返回:**
- `bool`: 是否成功拖拽

**示例:**
```python
# 从(100, 100)拖拽到(200, 200)
tool.drag(100, 100, 200, 200)

# 慢速拖拽
tool.drag(100, 100, 200, 200, duration=2.0)
```

### `scroll(x, y, clicks=3, **kwargs)`

在指定位置滚动鼠标滚轮。

**参数:**
- `x` (int | float): X坐标
- `y` (int | float): Y坐标
- `clicks` (int): 滚动次数，正数向上，负数向下
- `**kwargs`: 其他参数

**返回:**
- `bool`: 是否成功滚动

**示例:**
```python
# 向上滚动3次
tool.scroll(100, 200, clicks=3)

# 向下滚动5次
tool.scroll(100, 200, clicks=-5)
```

## 键盘操作API

### `type_text(text, interval=0.0, **kwargs)`

输入文本。

**参数:**
- `text` (str): 要输入的文本
- `interval` (float): 字符间间隔（秒）
- `**kwargs`: 其他参数

**返回:**
- `bool`: 是否成功输入

**示例:**
```python
# 输入文本
tool.type_text("Hello World")

# 慢速输入
tool.type_text("Hello World", interval=0.1)
```

### `press_key(key, **kwargs)`

按下单个按键。

**参数:**
- `key` (str): 按键名称
- `**kwargs`: 其他参数

**返回:**
- `bool`: 是否成功按下

**示例:**
```python
# 按下回车键
tool.press_key("enter")

# 按下空格键
tool.press_key("space")
```

### `hotkey(*keys, **kwargs)`

按下组合键。

**参数:**
- `*keys` (str): 按键序列
- `**kwargs`: 其他参数

**返回:**
- `bool`: 是否成功按下组合键

**示例:**
```python
# Ctrl+C
tool.hotkey("ctrl", "c")

# Alt+Tab
tool.hotkey("alt", "tab")

# Ctrl+Shift+Z
tool.hotkey("ctrl", "shift", "z")
```

### `type_with_delay(text, delay=0.1, **kwargs)`

带延迟的文本输入。

**参数:**
- `text` (str): 要输入的文本
- `delay` (float): 延迟时间（秒）
- `**kwargs`: 其他参数

**返回:**
- `bool`: 是否成功输入

**示例:**
```python
# 带延迟输入
tool.type_with_delay("Hello World", delay=0.2)
```

### `key_combination(keys, **kwargs)`

按下按键组合。

**参数:**
- `keys` (list): 按键列表
- `**kwargs`: 其他参数

**返回:**
- `bool`: 是否成功按下

**示例:**
```python
# 按键组合
tool.key_combination(["ctrl", "alt", "delete"])
```

### `clear_text(**kwargs)`

清空当前文本（Ctrl+A + Delete）。

**参数:**
- `**kwargs`: 其他参数

**返回:**
- `bool`: 是否成功清空

**示例:**
```python
tool.clear_text()
```

## 窗口操作API

### `find_window(title, **kwargs)`

查找窗口。

**参数:**
- `title` (str): 窗口标题
- `**kwargs`: 其他参数

**返回:**
- `str | None`: 窗口ID，如果未找到返回None

**示例:**
```python
# 查找记事本窗口
window_id = tool.find_window("记事本")

# 查找Chrome窗口
window_id = tool.find_window("Google Chrome")
```

### `activate_window(window_id, **kwargs)`

激活窗口。

**参数:**
- `window_id` (str): 窗口ID
- `**kwargs`: 其他参数

**返回:**
- `bool`: 是否成功激活

**示例:**
```python
window_id = tool.find_window("记事本")
if window_id:
    tool.activate_window(window_id)
```

### `minimize_window(window_id, **kwargs)`

最小化窗口。

**参数:**
- `window_id` (str): 窗口ID
- `**kwargs`: 其他参数

**返回:**
- `bool`: 是否成功最小化

**示例:**
```python
window_id = tool.find_window("记事本")
if window_id:
    tool.minimize_window(window_id)
```

### `maximize_window(window_id, **kwargs)`

最大化窗口。

**参数:**
- `window_id` (str): 窗口ID
- `**kwargs`: 其他参数

**返回:**
- `bool`: 是否成功最大化

**示例:**
```python
window_id = tool.find_window("记事本")
if window_id:
    tool.maximize_window(window_id)
```

### `close_window(window_id, **kwargs)`

关闭窗口。

**参数:**
- `window_id` (str): 窗口ID
- `**kwargs`: 其他参数

**返回:**
- `bool`: 是否成功关闭

**示例:**
```python
window_id = tool.find_window("记事本")
if window_id:
    tool.close_window(window_id)
```

### `move_window(window_id, x, y, **kwargs)`

移动窗口。

**参数:**
- `window_id` (str): 窗口ID
- `x` (int): 新X坐标
- `y` (int): 新Y坐标
- `**kwargs`: 其他参数

**返回:**
- `bool`: 是否成功移动

**示例:**
```python
window_id = tool.find_window("记事本")
if window_id:
    tool.move_window(window_id, 100, 100)
```

### `resize_window(window_id, width, height, **kwargs)`

调整窗口大小。

**参数:**
- `window_id` (str): 窗口ID
- `width` (int): 新宽度
- `height` (int): 新高度
- `**kwargs`: 其他参数

**返回:**
- `bool`: 是否成功调整

**示例:**
```python
window_id = tool.find_window("记事本")
if window_id:
    tool.resize_window(window_id, 800, 600)
```

### `get_window_info(window_id, **kwargs)`

获取窗口信息。

**参数:**
- `window_id` (str): 窗口ID
- `**kwargs`: 其他参数

**返回:**
- `dict`: 窗口信息字典

**示例:**
```python
window_id = tool.find_window("记事本")
if window_id:
    info = tool.get_window_info(window_id)
    print(f"窗口位置: {info.get('position')}")
    print(f"窗口大小: {info.get('size')}")
```

## 系统信息API

### `get_screen_size()`

获取屏幕尺寸。

**返回:**
- `tuple[int, int]`: 屏幕宽度和高度

**示例:**
```python
width, height = tool.get_screen_size()
print(f"屏幕尺寸: {width}x{height}")
```

### `get_dpi_scale()`

获取DPI缩放比例。

**返回:**
- `float`: DPI缩放比例

**示例:**
```python
scale = tool.get_dpi_scale()
print(f"DPI缩放: {scale}")
```

### `get_platform_info()`

获取平台信息。

**返回:**
- `dict`: 平台信息字典

**示例:**
```python
info = tool.get_platform_info()
print(f"平台: {info.get('name')}")
print(f"版本: {info.get('version')}")
```

## 配置管理API

### `get_config()`

获取当前配置。

**返回:**
- `GuiConfig`: 配置对象

**示例:**
```python
config = tool.get_config()
print(f"匹配置信度: {config.match_confidence}")
print(f"超时时间: {config.match_timeout}")
```

### `update_config(**kwargs)`

更新配置。

**参数:**
- `**kwargs`: 要更新的配置项

**返回:**
- `bool`: 是否成功更新

**示例:**
```python
# 更新匹配置信度
tool.update_config(match_confidence=0.9)

# 更新多个配置项
tool.update_config(
    match_confidence=0.9,
    match_timeout=15.0,
    auto_scale=True
)
```

## 便捷属性

### `screen_size`

获取屏幕尺寸（属性形式）。

**返回:**
- `tuple[int, int]`: 屏幕宽度和高度

**示例:**
```python
width, height = tool.screen_size
print(f"屏幕尺寸: {width}x{height}")
```

### `dpi_scale`

获取DPI缩放比例（属性形式）。

**返回:**
- `float`: DPI缩放比例

**示例:**
```python
scale = tool.dpi_scale
print(f"DPI缩放: {scale}")
```

## 版本信息

### `get_version()`

获取框架版本。

**返回:**
- `str`: 版本号

**示例:**
```python
version = tool.get_version()
print(f"框架版本: {version}")
```

## 错误处理

所有方法都可能抛出以下异常：

- `OperationError`: 操作执行错误
- `AlgorithmError`: 算法相关错误
- `PlatformError`: 平台相关错误
- `ConfigError`: 配置相关错误

**示例:**
```python
try:
    result = tool.find_image("button.png")
    if result.success:
        print(f"找到图像，位置: {result.position}")
    else:
        print("未找到图像")
except OperationError as e:
    print(f"操作失败: {e}")
except Exception as e:
    print(f"未知错误: {e}")
```
