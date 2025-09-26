# 平台模块API

## 概述

平台模块提供跨平台的GUI自动化功能，支持Windows和Linux平台，通过统一的接口抽象不同操作系统的差异。

## 平台基类 (PlatformBase)

### 类定义

```python
class PlatformBase(ABC):
    """平台基类"""
```

### 抽象方法

#### `get_screen_size() -> Tuple[int, int]`

获取屏幕尺寸。

**返回:**
- `Tuple[int, int]`: 屏幕宽度和高度

#### `capture_screen(region: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray`

截取屏幕图像。

**参数:**
- `region` (Optional[Tuple[int, int, int, int]]): 截取区域 (x, y, width, height)

**返回:**
- `np.ndarray`: 截取的图像数组

#### `click(x: int, y: int, button: str = "left") -> bool`

点击指定坐标。

**参数:**
- `x` (int): X坐标
- `y` (int): Y坐标
- `button` (str): 鼠标按钮

**返回:**
- `bool`: 是否成功点击

#### `type_text(text: str) -> bool`

输入文本。

**参数:**
- `text` (str): 要输入的文本

**返回:**
- `bool`: 是否成功输入

#### `find_window(title: str) -> Optional[str]`

查找窗口。

**参数:**
- `title` (str): 窗口标题

**返回:**
- `Optional[str]`: 窗口ID，如果未找到返回None

## 平台信息 (PlatformInfo)

### 类定义

```python
@dataclass
class PlatformInfo:
    """平台信息类"""
```

### 属性

| 属性 | 类型 | 描述 |
|------|------|------|
| `name` | str | 平台名称 |
| `version` | str | 平台版本 |
| `architecture` | str | 系统架构 |
| `python_version` | str | Python版本 |
| `features` | List[str] | 支持的功能列表 |

## Windows平台 (WindowsPlatform)

### 类定义

```python
class WindowsPlatform(PlatformBase):
    """Windows平台实现"""
```

### 方法

#### `get_screen_size() -> Tuple[int, int]`

获取Windows屏幕尺寸。

**返回:**
- `Tuple[int, int]`: 屏幕宽度和高度

**示例:**
```python
from gui_auto.platform import WindowsPlatform

platform = WindowsPlatform()
width, height = platform.get_screen_size()
print(f"屏幕尺寸: {width}x{height}")
```

#### `capture_screen(region: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray`

截取Windows屏幕图像。

**参数:**
- `region` (Optional[Tuple[int, int, int, int]]): 截取区域

**返回:**
- `np.ndarray`: 截取的图像数组

**示例:**
```python
# 截取整个屏幕
screenshot = platform.capture_screen()

# 截取指定区域
region_screenshot = platform.capture_screen(region=(100, 100, 800, 600))
```

#### `click(x: int, y: int, button: str = "left") -> bool`

在Windows上点击指定坐标。

**参数:**
- `x` (int): X坐标
- `y` (int): Y坐标
- `button` (str): 鼠标按钮

**返回:**
- `bool`: 是否成功点击

**示例:**
```python
# 左键点击
success = platform.click(100, 200)

# 右键点击
success = platform.click(100, 200, button="right")
```

#### `type_text(text: str) -> bool`

在Windows上输入文本。

**参数:**
- `text` (str): 要输入的文本

**返回:**
- `bool`: 是否成功输入

**示例:**
```python
success = platform.type_text("Hello World")
```

#### `press_key(key: str) -> bool`

按下单个按键。

**参数:**
- `key` (str): 按键名称

**返回:**
- `bool`: 是否成功按下

**示例:**
```python
# 按下回车键
success = platform.press_key("enter")

# 按下空格键
success = platform.press_key("space")
```

#### `hotkey(*keys: str) -> bool`

按下组合键。

**参数:**
- `*keys` (str): 按键序列

**返回:**
- `bool`: 是否成功按下

**示例:**
```python
# Ctrl+C
success = platform.hotkey("ctrl", "c")

# Alt+Tab
success = platform.hotkey("alt", "tab")
```

#### `find_window(title: str) -> Optional[str]`

查找Windows窗口。

**参数:**
- `title` (str): 窗口标题

**返回:**
- `Optional[str]`: 窗口句柄，如果未找到返回None

**示例:**
```python
window_handle = platform.find_window("记事本")
if window_handle:
    print(f"找到窗口: {window_handle}")
```

#### `activate_window(window_id: str) -> bool`

激活Windows窗口。

**参数:**
- `window_id` (str): 窗口句柄

**返回:**
- `bool`: 是否成功激活

**示例:**
```python
window_handle = platform.find_window("记事本")
if window_handle:
    success = platform.activate_window(window_handle)
```

#### `minimize_window(window_id: str) -> bool`

最小化Windows窗口。

**参数:**
- `window_id` (str): 窗口句柄

**返回:**
- `bool`: 是否成功最小化

**示例:**
```python
window_handle = platform.find_window("记事本")
if window_handle:
    success = platform.minimize_window(window_handle)
```

#### `maximize_window(window_id: str) -> bool`

最大化Windows窗口。

**参数:**
- `window_id` (str): 窗口句柄

**返回:**
- `bool`: 是否成功最大化

**示例:**
```python
window_handle = platform.find_window("记事本")
if window_handle:
    success = platform.maximize_window(window_handle)
```

#### `close_window(window_id: str) -> bool`

关闭Windows窗口。

**参数:**
- `window_id` (str): 窗口句柄

**返回:**
- `bool`: 是否成功关闭

**示例:**
```python
window_handle = platform.find_window("记事本")
if window_handle:
    success = platform.close_window(window_handle)
```

#### `move_window(window_id: str, x: int, y: int) -> bool`

移动Windows窗口。

**参数:**
- `window_id` (str): 窗口句柄
- `x` (int): 新X坐标
- `y` (int): 新Y坐标

**返回:**
- `bool`: 是否成功移动

**示例:**
```python
window_handle = platform.find_window("记事本")
if window_handle:
    success = platform.move_window(window_handle, 100, 100)
```

#### `resize_window(window_id: str, width: int, height: int) -> bool`

调整Windows窗口大小。

**参数:**
- `window_id` (str): 窗口句柄
- `width` (int): 新宽度
- `height` (int): 新高度

**返回:**
- `bool`: 是否成功调整

**示例:**
```python
window_handle = platform.find_window("记事本")
if window_handle:
    success = platform.resize_window(window_handle, 800, 600)
```

#### `get_window_info(window_id: str) -> Dict[str, Any]`

获取Windows窗口信息。

**参数:**
- `window_id` (str): 窗口句柄

**返回:**
- `Dict[str, Any]`: 窗口信息字典

**示例:**
```python
window_handle = platform.find_window("记事本")
if window_handle:
    info = platform.get_window_info(window_handle)
    print(f"窗口信息: {info}")
```

#### `get_dpi_scale() -> float`

获取Windows DPI缩放比例。

**返回:**
- `float`: DPI缩放比例

**示例:**
```python
scale = platform.get_dpi_scale()
print(f"DPI缩放: {scale}")
```

#### `is_feature_supported(feature: str) -> bool`

检查是否支持特定功能。

**参数:**
- `feature` (str): 功能名称

**返回:**
- `bool`: 是否支持该功能

**示例:**
```python
# 检查是否支持窗口操作
if platform.is_feature_supported("window_operations"):
    print("支持窗口操作")
```

## Linux平台 (LinuxPlatform)

### 类定义

```python
class LinuxPlatform(PlatformBase):
    """Linux平台实现"""
```

### 方法

#### `get_screen_size() -> Tuple[int, int]`

获取Linux屏幕尺寸。

**返回:**
- `Tuple[int, int]`: 屏幕宽度和高度

**示例:**
```python
from gui_auto.platform import LinuxPlatform

platform = LinuxPlatform()
width, height = platform.get_screen_size()
print(f"屏幕尺寸: {width}x{height}")
```

#### `capture_screen(region: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray`

截取Linux屏幕图像。

**参数:**
- `region` (Optional[Tuple[int, int, int, int]]): 截取区域

**返回:**
- `np.ndarray`: 截取的图像数组

**示例:**
```python
# 截取整个屏幕
screenshot = platform.capture_screen()

# 截取指定区域
region_screenshot = platform.capture_screen(region=(100, 100, 800, 600))
```

#### `click(x: int, y: int, button: str = "left") -> bool`

在Linux上点击指定坐标。

**参数:**
- `x` (int): X坐标
- `y` (int): Y坐标
- `button` (str): 鼠标按钮

**返回:**
- `bool`: 是否成功点击

**示例:**
```python
# 左键点击
success = platform.click(100, 200)

# 右键点击
success = platform.click(100, 200, button="right")
```

#### `type_text(text: str) -> bool`

在Linux上输入文本。

**参数:**
- `text` (str): 要输入的文本

**返回:**
- `bool`: 是否成功输入

**示例:**
```python
success = platform.type_text("Hello World")
```

#### `press_key(key: str) -> bool`

按下单个按键。

**参数:**
- `key` (str): 按键名称

**返回:**
- `bool`: 是否成功按下

**示例:**
```python
# 按下回车键
success = platform.press_key("enter")

# 按下空格键
success = platform.press_key("space")
```

#### `hotkey(*keys: str) -> bool`

按下组合键。

**参数:**
- `*keys` (str): 按键序列

**返回:**
- `bool`: 是否成功按下

**示例:**
```python
# Ctrl+C
success = platform.hotkey("ctrl", "c")

# Alt+Tab
success = platform.hotkey("alt", "tab")
```

#### `find_window(title: str) -> Optional[str]`

查找Linux窗口。

**参数:**
- `title` (str): 窗口标题

**返回:**
- `Optional[str]`: 窗口ID，如果未找到返回None

**示例:**
```python
window_id = platform.find_window("gedit")
if window_id:
    print(f"找到窗口: {window_id}")
```

#### `activate_window(window_id: str) -> bool`

激活Linux窗口。

**参数:**
- `window_id` (str): 窗口ID

**返回:**
- `bool`: 是否成功激活

**示例:**
```python
window_id = platform.find_window("gedit")
if window_id:
    success = platform.activate_window(window_id)
```

#### `minimize_window(window_id: str) -> bool`

最小化Linux窗口。

**参数:**
- `window_id` (str): 窗口ID

**返回:**
- `bool`: 是否成功最小化

**示例:**
```python
window_id = platform.find_window("gedit")
if window_id:
    success = platform.minimize_window(window_id)
```

#### `maximize_window(window_id: str) -> bool`

最大化Linux窗口。

**参数:**
- `window_id` (str): 窗口ID

**返回:**
- `bool`: 是否成功最大化

**示例:**
```python
window_id = platform.find_window("gedit")
if window_id:
    success = platform.maximize_window(window_id)
```

#### `close_window(window_id: str) -> bool`

关闭Linux窗口。

**参数:**
- `window_id` (str): 窗口ID

**返回:**
- `bool`: 是否成功关闭

**示例:**
```python
window_id = platform.find_window("gedit")
if window_id:
    success = platform.close_window(window_id)
```

#### `move_window(window_id: str, x: int, y: int) -> bool`

移动Linux窗口。

**参数:**
- `window_id` (str): 窗口ID
- `x` (int): 新X坐标
- `y` (int): 新Y坐标

**返回:**
- `bool`: 是否成功移动

**示例:**
```python
window_id = platform.find_window("gedit")
if window_id:
    success = platform.move_window(window_id, 100, 100)
```

#### `resize_window(window_id: str, width: int, height: int) -> bool`

调整Linux窗口大小。

**参数:**
- `window_id` (str): 窗口ID
- `width` (int): 新宽度
- `height` (int): 新高度

**返回:**
- `bool`: 是否成功调整

**示例:**
```python
window_id = platform.find_window("gedit")
if window_id:
    success = platform.resize_window(window_id, 800, 600)
```

#### `get_window_info(window_id: str) -> Dict[str, Any]`

获取Linux窗口信息。

**参数:**
- `window_id` (str): 窗口ID

**返回:**
- `Dict[str, Any]`: 窗口信息字典

**示例:**
```python
window_id = platform.find_window("gedit")
if window_id:
    info = platform.get_window_info(window_id)
    print(f"窗口信息: {info}")
```

#### `get_dpi_scale() -> float`

获取Linux DPI缩放比例。

**返回:**
- `float`: DPI缩放比例

**示例:**
```python
scale = platform.get_dpi_scale()
print(f"DPI缩放: {scale}")
```

#### `is_feature_supported(feature: str) -> bool`

检查是否支持特定功能。

**参数:**
- `feature` (str): 功能名称

**返回:**
- `bool`: 是否支持该功能

**示例:**
```python
# 检查是否支持窗口操作
if platform.is_feature_supported("window_operations"):
    print("支持窗口操作")
```

## 平台工厂 (PlatformFactory)

### 类定义

```python
class PlatformFactory:
    """平台工厂类"""
```

### 类方法

#### `create_platform(platform_name: str = None) -> PlatformBase`

创建平台实例。

**参数:**
- `platform_name` (str, 可选): 平台名称。如果为None，将自动检测当前平台

**返回:**
- `PlatformBase`: 平台实例

**示例:**
```python
from gui_auto.platform import PlatformFactory

# 自动检测平台
platform = PlatformFactory.create_platform()

# 指定平台
windows_platform = PlatformFactory.create_platform("windows")
linux_platform = PlatformFactory.create_platform("linux")
```

#### `detect_platform() -> str`

检测当前平台。

**返回:**
- `str`: 平台名称

**示例:**
```python
platform_name = PlatformFactory.detect_platform()
print(f"当前平台: {platform_name}")
```

#### `get_available_platforms() -> List[str]`

获取可用的平台列表。

**返回:**
- `List[str]`: 平台名称列表

**示例:**
```python
platforms = PlatformFactory.get_available_platforms()
print(f"可用平台: {platforms}")
```

#### `register_platform(name: str, platform_class: Type[PlatformBase])`

注册自定义平台。

**参数:**
- `name` (str): 平台名称
- `platform_class` (Type[PlatformBase]): 平台类

**示例:**
```python
from gui_auto.platform import PlatformFactory, PlatformBase

class CustomPlatform(PlatformBase):
    def get_screen_size(self):
        return (1920, 1080)
    
    def capture_screen(self, region=None):
        # 自定义实现
        pass
    
    # 实现其他抽象方法...

# 注册自定义平台
PlatformFactory.register_platform("custom", CustomPlatform)
```

## 平台管理器 (PlatformManager)

### 类定义

```python
class PlatformManager:
    """平台管理器类"""
```

### 方法

#### `__init__(platform: PlatformBase = None)`

初始化平台管理器。

**参数:**
- `platform` (PlatformBase, 可选): 平台实例。如果为None，将自动创建

**示例:**
```python
from gui_auto.platform import PlatformManager

# 使用默认平台
manager = PlatformManager()

# 使用指定平台
from gui_auto.platform import WindowsPlatform
windows_platform = WindowsPlatform()
manager = PlatformManager(windows_platform)
```

#### `get_platform() -> PlatformBase`

获取当前平台实例。

**返回:**
- `PlatformBase`: 平台实例

**示例:**
```python
platform = manager.get_platform()
print(f"当前平台: {platform.__class__.__name__}")
```

#### `set_platform(platform: PlatformBase)`

设置平台实例。

**参数:**
- `platform` (PlatformBase): 平台实例

**示例:**
```python
from gui_auto.platform import LinuxPlatform

linux_platform = LinuxPlatform()
manager.set_platform(linux_platform)
```

#### `get_platform_info() -> PlatformInfo`

获取平台信息。

**返回:**
- `PlatformInfo`: 平台信息

**示例:**
```python
info = manager.get_platform_info()
print(f"平台名称: {info.name}")
print(f"平台版本: {info.version}")
print(f"支持的功能: {info.features}")
```

#### `execute_operation(operation: str, *args, **kwargs) -> Any`

执行平台操作。

**参数:**
- `operation` (str): 操作名称
- `*args`: 位置参数
- `**kwargs`: 关键字参数

**返回:**
- `Any`: 操作结果

**示例:**
```python
# 获取屏幕尺寸
width, height = manager.execute_operation("get_screen_size")

# 截取屏幕
screenshot = manager.execute_operation("capture_screen")

# 点击坐标
success = manager.execute_operation("click", 100, 200)

# 输入文本
success = manager.execute_operation("type_text", "Hello World")
```

## 使用示例

### 基本使用

```python
from gui_auto.platform import PlatformFactory

# 自动检测并创建平台
platform = PlatformFactory.create_platform()

# 获取屏幕尺寸
width, height = platform.get_screen_size()
print(f"屏幕尺寸: {width}x{height}")

# 截取屏幕
screenshot = platform.capture_screen()

# 点击坐标
success = platform.click(100, 200)
if success:
    print("点击成功")

# 输入文本
success = platform.type_text("Hello World")
if success:
    print("文本输入成功")
```

### 使用平台管理器

```python
from gui_auto.platform import PlatformManager

# 创建平台管理器
manager = PlatformManager()

# 获取平台信息
info = manager.get_platform_info()
print(f"平台: {info.name} {info.version}")

# 执行操作
width, height = manager.execute_operation("get_screen_size")
screenshot = manager.execute_operation("capture_screen")
success = manager.execute_operation("click", 100, 200)
```

### 跨平台兼容性

```python
from gui_auto.platform import PlatformFactory
import platform as os_platform

# 检测当前操作系统
current_os = os_platform.system().lower()

if current_os == "windows":
    platform = PlatformFactory.create_platform("windows")
elif current_os == "linux":
    platform = PlatformFactory.create_platform("linux")
else:
    # 使用默认平台
    platform = PlatformFactory.create_platform()

# 执行平台无关的操作
width, height = platform.get_screen_size()
screenshot = platform.capture_screen()
success = platform.click(100, 200)
```

### 错误处理

```python
from gui_auto.platform import PlatformFactory, PlatformError

try:
    platform = PlatformFactory.create_platform()
    success = platform.click(100, 200)
    if not success:
        print("点击操作失败")
except PlatformError as e:
    print(f"平台错误: {e}")
except Exception as e:
    print(f"未知错误: {e}")
```
