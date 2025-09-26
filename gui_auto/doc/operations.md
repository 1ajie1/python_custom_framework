# 操作模块API

## 概述

操作模块提供各种GUI操作的统一接口，包括图像操作、鼠标操作、键盘操作和窗口操作。

## 操作基类 (Operation)

### 类定义

```python
class Operation(ABC):
    """操作基类"""
```

### 抽象方法

#### `execute(operation: str, *args, **kwargs) -> OperationResult`

执行操作。

**参数:**
- `operation` (str): 操作名称
- `*args`: 位置参数
- `**kwargs`: 关键字参数

**返回:**
- `OperationResult`: 操作结果

## 操作结果 (OperationResult)

### 类定义

```python
@dataclass
class OperationResult:
    """操作结果类"""
```

### 属性

| 属性 | 类型 | 描述 |
|------|------|------|
| `success` | bool | 操作是否成功 |
| `data` | Any | 操作返回的数据 |
| `message` | str | 操作消息 |
| `error` | str | 错误信息 |
| `execution_time` | float | 执行时间（秒） |
| `details` | dict | 详细信息 |

### 方法

#### `is_success() -> bool`

检查操作是否成功。

**返回:**
- `bool`: 是否成功

**示例:**
```python
result = tool.click(100, 200)
if result.is_success():
    print("点击成功")
```

## 图像操作 (ImageOperations)

### 类定义

```python
class ImageOperations(Operation):
    """图像操作类"""
```

### 支持的操作

| 操作 | 描述 | 参数 |
|------|------|------|
| `find_image` | 查找图像 | template, target, algorithm, **kwargs |
| `find_and_click` | 查找并点击 | template, target, algorithm, **kwargs |
| `capture_screen` | 截取屏幕 | region, target_format, **kwargs |
| `compare_images` | 比较图像 | image1, image2, method, **kwargs |
| `extract_text` | 提取文本 | image, **kwargs |

### 方法

#### `execute(operation: str, *args, **kwargs) -> OperationResult`

执行图像操作。

**参数:**
- `operation` (str): 操作名称
- `*args`: 位置参数
- `**kwargs`: 关键字参数

**返回:**
- `OperationResult`: 操作结果

**示例:**
```python
from gui_auto.operations import ImageOperations

image_ops = ImageOperations()

# 查找图像
result = image_ops.execute("find_image", "button.png", algorithm="template")

# 截取屏幕
result = image_ops.execute("capture_screen", region=(0, 0, 800, 600))

# 比较图像
result = image_ops.execute("compare_images", "img1.png", "img2.png", method="ssim")
```

#### `find_image(template, target=None, algorithm="template", **kwargs) -> OperationResult`

查找图像。

**参数:**
- `template` (str | np.ndarray): 模板图像
- `target` (str | np.ndarray, 可选): 目标图像
- `algorithm` (str): 匹配算法
- `**kwargs`: 其他参数

**返回:**
- `OperationResult`: 操作结果

**示例:**
```python
result = image_ops.find_image("button.png", algorithm="feature")
if result.success:
    print(f"找到图像: {result.data}")
```

#### `find_and_click(template, target=None, algorithm="template", **kwargs) -> OperationResult`

查找并点击图像。

**参数:**
- `template` (str | np.ndarray): 模板图像
- `target` (str | np.ndarray, 可选): 目标图像
- `algorithm` (str): 匹配算法
- `**kwargs`: 其他参数

**返回:**
- `OperationResult`: 操作结果

**示例:**
```python
result = image_ops.find_and_click("button.png")
if result.success:
    print("成功点击图像")
```

#### `capture_screen(region=None, target_format="bgr", **kwargs) -> OperationResult`

截取屏幕。

**参数:**
- `region` (tuple, 可选): 截取区域 (x, y, width, height)
- `target_format` (str): 目标格式
- `**kwargs`: 其他参数

**返回:**
- `OperationResult`: 操作结果

**示例:**
```python
# 截取整个屏幕
result = image_ops.capture_screen()

# 截取指定区域
result = image_ops.capture_screen(region=(100, 100, 800, 600))
```

#### `compare_images(image1, image2, method="ssim", **kwargs) -> OperationResult`

比较图像相似度。

**参数:**
- `image1` (str | np.ndarray): 第一个图像
- `image2` (str | np.ndarray): 第二个图像
- `method` (str): 比较方法
- `**kwargs`: 其他参数

**返回:**
- `OperationResult`: 操作结果

**示例:**
```python
result = image_ops.compare_images("img1.png", "img2.png", method="ssim")
if result.success:
    similarity = result.data
    print(f"相似度: {similarity}")
```

## 点击操作 (ClickOperations)

### 类定义

```python
class ClickOperations(Operation):
    """点击操作类"""
```

### 支持的操作

| 操作 | 描述 | 参数 |
|------|------|------|
| `click` | 点击 | x, y, button, clicks, interval, **kwargs |
| `double_click` | 双击 | x, y, button, **kwargs |
| `right_click` | 右键点击 | x, y, **kwargs |
| `drag` | 拖拽 | x1, y1, x2, y2, duration, **kwargs |
| `scroll` | 滚动 | x, y, clicks, **kwargs |

### 方法

#### `click(x, y, button="left", clicks=1, interval=0.0, **kwargs) -> OperationResult`

点击指定坐标。

**参数:**
- `x` (int | float): X坐标
- `y` (int | float): Y坐标
- `button` (str): 鼠标按钮
- `clicks` (int): 点击次数
- `interval` (float): 点击间隔
- `**kwargs`: 其他参数

**返回:**
- `OperationResult`: 操作结果

**示例:**
```python
from gui_auto.operations import ClickOperations

click_ops = ClickOperations()

# 左键单击
result = click_ops.click(100, 200)

# 右键双击
result = click_ops.click(100, 200, button="right", clicks=2)
```

#### `double_click(x, y, button="left", **kwargs) -> OperationResult`

双击指定坐标。

**参数:**
- `x` (int | float): X坐标
- `y` (int | float): Y坐标
- `button` (str): 鼠标按钮
- `**kwargs`: 其他参数

**返回:**
- `OperationResult`: 操作结果

**示例:**
```python
result = click_ops.double_click(100, 200)
```

#### `right_click(x, y, **kwargs) -> OperationResult`

右键点击指定坐标。

**参数:**
- `x` (int | float): X坐标
- `y` (int | float): Y坐标
- `**kwargs`: 其他参数

**返回:**
- `OperationResult`: 操作结果

**示例:**
```python
result = click_ops.right_click(100, 200)
```

#### `drag(x1, y1, x2, y2, duration=0.5, **kwargs) -> OperationResult`

从一点拖拽到另一点。

**参数:**
- `x1` (int | float): 起始X坐标
- `y1` (int | float): 起始Y坐标
- `x2` (int | float): 结束X坐标
- `y2` (int | float): 结束Y坐标
- `duration` (float): 拖拽持续时间
- `**kwargs`: 其他参数

**返回:**
- `OperationResult`: 操作结果

**示例:**
```python
result = click_ops.drag(100, 100, 200, 200, duration=1.0)
```

#### `scroll(x, y, clicks=3, **kwargs) -> OperationResult`

在指定位置滚动鼠标滚轮。

**参数:**
- `x` (int | float): X坐标
- `y` (int | float): Y坐标
- `clicks` (int): 滚动次数
- `**kwargs`: 其他参数

**返回:**
- `OperationResult`: 操作结果

**示例:**
```python
# 向上滚动
result = click_ops.scroll(100, 200, clicks=3)

# 向下滚动
result = click_ops.scroll(100, 200, clicks=-3)
```

## 键盘操作 (KeyboardOperations)

### 类定义

```python
class KeyboardOperations(Operation):
    """键盘操作类"""
```

### 支持的操作

| 操作 | 描述 | 参数 |
|------|------|------|
| `type_text` | 输入文本 | text, interval, **kwargs |
| `press_key` | 按下按键 | key, **kwargs |
| `hotkey` | 组合键 | *keys, **kwargs |
| `type_with_delay` | 带延迟输入 | text, delay, **kwargs |
| `key_combination` | 按键组合 | keys, **kwargs |
| `clear_text` | 清空文本 | **kwargs |

### 方法

#### `type_text(text, interval=0.0, **kwargs) -> OperationResult`

输入文本。

**参数:**
- `text` (str): 要输入的文本
- `interval` (float): 字符间间隔
- `**kwargs`: 其他参数

**返回:**
- `OperationResult`: 操作结果

**示例:**
```python
from gui_auto.operations import KeyboardOperations

keyboard_ops = KeyboardOperations()

# 输入文本
result = keyboard_ops.type_text("Hello World")

# 慢速输入
result = keyboard_ops.type_text("Hello World", interval=0.1)
```

#### `press_key(key, **kwargs) -> OperationResult`

按下单个按键。

**参数:**
- `key` (str): 按键名称
- `**kwargs`: 其他参数

**返回:**
- `OperationResult`: 操作结果

**示例:**
```python
# 按下回车键
result = keyboard_ops.press_key("enter")

# 按下空格键
result = keyboard_ops.press_key("space")
```

#### `hotkey(*keys, **kwargs) -> OperationResult`

按下组合键。

**参数:**
- `*keys` (str): 按键序列
- `**kwargs`: 其他参数

**返回:**
- `OperationResult`: 操作结果

**示例:**
```python
# Ctrl+C
result = keyboard_ops.hotkey("ctrl", "c")

# Alt+Tab
result = keyboard_ops.hotkey("alt", "tab")
```

#### `type_with_delay(text, delay=0.1, **kwargs) -> OperationResult`

带延迟的文本输入。

**参数:**
- `text` (str): 要输入的文本
- `delay` (float): 延迟时间
- `**kwargs`: 其他参数

**返回:**
- `OperationResult`: 操作结果

**示例:**
```python
result = keyboard_ops.type_with_delay("Hello World", delay=0.2)
```

#### `key_combination(keys, **kwargs) -> OperationResult`

按下按键组合。

**参数:**
- `keys` (list): 按键列表
- `**kwargs`: 其他参数

**返回:**
- `OperationResult`: 操作结果

**示例:**
```python
result = keyboard_ops.key_combination(["ctrl", "alt", "delete"])
```

#### `clear_text(**kwargs) -> OperationResult`

清空当前文本。

**参数:**
- `**kwargs`: 其他参数

**返回:**
- `OperationResult`: 操作结果

**示例:**
```python
result = keyboard_ops.clear_text()
```

## 窗口操作 (WindowOperations)

### 类定义

```python
class WindowOperations(Operation):
    """窗口操作类"""
```

### 支持的操作

| 操作 | 描述 | 参数 |
|------|------|------|
| `find_window` | 查找窗口 | title, **kwargs |
| `activate_window` | 激活窗口 | window_id, **kwargs |
| `minimize_window` | 最小化窗口 | window_id, **kwargs |
| `maximize_window` | 最大化窗口 | window_id, **kwargs |
| `close_window` | 关闭窗口 | window_id, **kwargs |
| `move_window` | 移动窗口 | window_id, x, y, **kwargs |
| `resize_window` | 调整窗口大小 | window_id, width, height, **kwargs |
| `get_window_info` | 获取窗口信息 | window_id, **kwargs |

### 方法

#### `find_window(title, **kwargs) -> OperationResult`

查找窗口。

**参数:**
- `title` (str): 窗口标题
- `**kwargs`: 其他参数

**返回:**
- `OperationResult`: 操作结果

**示例:**
```python
from gui_auto.operations import WindowOperations

window_ops = WindowOperations()

# 查找记事本窗口
result = window_ops.find_window("记事本")
if result.success:
    window_id = result.data
    print(f"找到窗口: {window_id}")
```

#### `activate_window(window_id, **kwargs) -> OperationResult`

激活窗口。

**参数:**
- `window_id` (str): 窗口ID
- `**kwargs`: 其他参数

**返回:**
- `OperationResult`: 操作结果

**示例:**
```python
result = window_ops.activate_window(window_id)
if result.success:
    print("窗口已激活")
```

#### `minimize_window(window_id, **kwargs) -> OperationResult`

最小化窗口。

**参数:**
- `window_id` (str): 窗口ID
- `**kwargs`: 其他参数

**返回:**
- `OperationResult`: 操作结果

**示例:**
```python
result = window_ops.minimize_window(window_id)
```

#### `maximize_window(window_id, **kwargs) -> OperationResult`

最大化窗口。

**参数:**
- `window_id` (str): 窗口ID
- `**kwargs`: 其他参数

**返回:**
- `OperationResult`: 操作结果

**示例:**
```python
result = window_ops.maximize_window(window_id)
```

#### `close_window(window_id, **kwargs) -> OperationResult`

关闭窗口。

**参数:**
- `window_id` (str): 窗口ID
- `**kwargs`: 其他参数

**返回:**
- `OperationResult`: 操作结果

**示例:**
```python
result = window_ops.close_window(window_id)
```

#### `move_window(window_id, x, y, **kwargs) -> OperationResult`

移动窗口。

**参数:**
- `window_id` (str): 窗口ID
- `x` (int): 新X坐标
- `y` (int): 新Y坐标
- `**kwargs`: 其他参数

**返回:**
- `OperationResult`: 操作结果

**示例:**
```python
result = window_ops.move_window(window_id, 100, 100)
```

#### `resize_window(window_id, width, height, **kwargs) -> OperationResult`

调整窗口大小。

**参数:**
- `window_id` (str): 窗口ID
- `width` (int): 新宽度
- `height` (int): 新高度
- `**kwargs`: 其他参数

**返回:**
- `OperationResult`: 操作结果

**示例:**
```python
result = window_ops.resize_window(window_id, 800, 600)
```

#### `get_window_info(window_id, **kwargs) -> OperationResult`

获取窗口信息。

**参数:**
- `window_id` (str): 窗口ID
- `**kwargs`: 其他参数

**返回:**
- `OperationResult`: 操作结果

**示例:**
```python
result = window_ops.get_window_info(window_id)
if result.success:
    info = result.data
    print(f"窗口信息: {info}")
```

## 操作工厂 (OperationFactory)

### 类定义

```python
class OperationFactory:
    """操作工厂类"""
```

### 类方法

#### `create_operation(name: str, config: Any = None) -> Operation`

创建操作实例。

**参数:**
- `name` (str): 操作名称
- `config` (Any, 可选): 配置对象

**返回:**
- `Operation`: 操作实例

**示例:**
```python
from gui_auto.operations import OperationFactory

# 创建图像操作
image_ops = OperationFactory.create_operation("image")

# 创建点击操作
click_ops = OperationFactory.create_operation("click")

# 创建键盘操作
keyboard_ops = OperationFactory.create_operation("keyboard")

# 创建窗口操作
window_ops = OperationFactory.create_operation("window")
```

#### `get_available_operations() -> list`

获取可用的操作列表。

**返回:**
- `list[str]`: 操作名称列表

**示例:**
```python
operations = OperationFactory.get_available_operations()
print(f"可用操作: {operations}")
```

## 操作管理器 (OperationManager)

### 类定义

```python
class OperationManager:
    """操作管理器类"""
```

### 方法

#### `__init__(config: Any = None)`

初始化操作管理器。

**参数:**
- `config` (Any, 可选): 配置对象

**示例:**
```python
from gui_auto.operations import OperationManager

manager = OperationManager()
```

#### `execute_operation(operation_type: str, operation: str, *args, **kwargs) -> OperationResult`

执行操作。

**参数:**
- `operation_type` (str): 操作类型
- `operation` (str): 操作名称
- `*args`: 位置参数
- `**kwargs`: 关键字参数

**返回:**
- `OperationResult`: 操作结果

**示例:**
```python
# 执行图像操作
result = manager.execute_operation("image", "find_image", "button.png")

# 执行点击操作
result = manager.execute_operation("click", "click", 100, 200)

# 执行键盘操作
result = manager.execute_operation("keyboard", "type_text", "Hello World")
```

## 使用示例

### 基本使用

```python
from gui_auto.operations import OperationFactory

# 创建操作实例
image_ops = OperationFactory.create_operation("image")
click_ops = OperationFactory.create_operation("click")
keyboard_ops = OperationFactory.create_operation("keyboard")
window_ops = OperationFactory.create_operation("window")

# 执行操作
result = image_ops.find_image("button.png")
if result.success:
    position = result.data.position
    click_ops.click(position[0], position[1])
```

### 使用操作管理器

```python
from gui_auto.operations import OperationManager

manager = OperationManager()

# 查找并点击图像
result = manager.execute_operation("image", "find_and_click", "button.png")
if result.success:
    print("成功点击图像")

# 输入文本
result = manager.execute_operation("keyboard", "type_text", "Hello World")
if result.success:
    print("文本输入成功")

# 窗口操作
result = manager.execute_operation("window", "find_window", "记事本")
if result.success:
    window_id = result.data
    manager.execute_operation("window", "activate_window", window_id)
```

### 错误处理

```python
from gui_auto.operations import OperationFactory, OperationError

try:
    image_ops = OperationFactory.create_operation("image")
    result = image_ops.find_image("button.png")
    
    if result.success:
        print(f"操作成功: {result.message}")
    else:
        print(f"操作失败: {result.error}")
        
except OperationError as e:
    print(f"操作异常: {e}")
except Exception as e:
    print(f"未知错误: {e}")
```
