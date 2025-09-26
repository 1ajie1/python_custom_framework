# 工具模块API

## 概述

工具模块提供各种实用工具函数，包括图像处理、坐标转换、缩放管理和格式转换等功能。

## 图像工具 (ImageUtils)

### 类定义

```python
class ImageUtils:
    """图像处理工具类"""
```

### 静态方法

#### `load_image(image_path: str, target_format: str = "bgr") -> np.ndarray`

加载图像文件。

**参数:**
- `image_path` (str): 图像文件路径
- `target_format` (str): 目标格式，'bgr' 或 'rgb'

**返回:**
- `np.ndarray`: 图像数组

**示例:**
```python
from gui_auto.utils import ImageUtils

# 加载BGR格式图像
image = ImageUtils.load_image("button.png", "bgr")

# 加载RGB格式图像
image = ImageUtils.load_image("button.png", "rgb")
```

#### `save_image(image: np.ndarray, output_path: str, quality: int = 95) -> bool`

保存图像文件。

**参数:**
- `image` (np.ndarray): 图像数组
- `output_path` (str): 输出文件路径
- `quality` (int): 图像质量 (1-100)

**返回:**
- `bool`: 是否成功保存

**示例:**
```python
# 保存图像
success = ImageUtils.save_image(image, "output.png", quality=95)
if success:
    print("图像保存成功")
```

#### `convert_format(image: np.ndarray, target_format: str) -> np.ndarray`

转换图像格式。

**参数:**
- `image` (np.ndarray): 输入图像
- `target_format` (str): 目标格式，'bgr', 'rgb', 'gray'

**返回:**
- `np.ndarray`: 转换后的图像

**示例:**
```python
# 转换为RGB格式
rgb_image = ImageUtils.convert_format(image, "rgb")

# 转换为灰度图
gray_image = ImageUtils.convert_format(image, "gray")
```

#### `resize_image(image: np.ndarray, size: Tuple[int, int], method: str = "bilinear") -> np.ndarray`

调整图像大小。

**参数:**
- `image` (np.ndarray): 输入图像
- `size` (Tuple[int, int]): 目标尺寸 (width, height)
- `method` (str): 插值方法

**返回:**
- `np.ndarray`: 调整后的图像

**示例:**
```python
# 调整图像大小
resized = ImageUtils.resize_image(image, (800, 600))

# 使用双三次插值
resized = ImageUtils.resize_image(image, (800, 600), "bicubic")
```

#### `crop_image(image: np.ndarray, region: Tuple[int, int, int, int]) -> np.ndarray`

裁剪图像。

**参数:**
- `image` (np.ndarray): 输入图像
- `region` (Tuple[int, int, int, int]): 裁剪区域 (x, y, width, height)

**返回:**
- `np.ndarray`: 裁剪后的图像

**示例:**
```python
# 裁剪图像
cropped = ImageUtils.crop_image(image, (100, 100, 200, 200))
```

#### `get_image_info(image: np.ndarray) -> Dict[str, Any]`

获取图像信息。

**参数:**
- `image` (np.ndarray): 输入图像

**返回:**
- `Dict[str, Any]`: 图像信息字典

**示例:**
```python
info = ImageUtils.get_image_info(image)
print(f"图像尺寸: {info['size']}")
print(f"图像格式: {info['format']}")
print(f"数据类型: {info['dtype']}")
```

#### `validate_image(image: np.ndarray) -> bool`

验证图像是否有效。

**参数:**
- `image` (np.ndarray): 输入图像

**返回:**
- `bool`: 图像是否有效

**示例:**
```python
if ImageUtils.validate_image(image):
    print("图像有效")
else:
    print("图像无效")
```

## 坐标工具 (CoordinateUtils)

### 类定义

```python
class CoordinateUtils:
    """坐标转换工具类"""
```

### 静态方法

#### `convert_coordinates(x: int, y: int, from_system: str, to_system: str, scale_factor: float = 1.0) -> Tuple[int, int]`

转换坐标系统。

**参数:**
- `x` (int): X坐标
- `y` (int): Y坐标
- `from_system` (str): 源坐标系统
- `to_system` (str): 目标坐标系统
- `scale_factor` (float): 缩放因子

**返回:**
- `Tuple[int, int]`: 转换后的坐标

**示例:**
```python
from gui_auto.utils import CoordinateUtils

# 从基础坐标转换为系统坐标
sys_x, sys_y = CoordinateUtils.convert_coordinates(
    100, 200, "base", "system", scale_factor=1.25
)

# 从系统坐标转换为基础坐标
base_x, base_y = CoordinateUtils.convert_coordinates(
    100, 200, "system", "base", scale_factor=1.25
)
```

#### `normalize_coordinates(x: int, y: int, width: int, height: int) -> Tuple[float, float]`

标准化坐标。

**参数:**
- `x` (int): X坐标
- `y` (int): Y坐标
- `width` (int): 图像宽度
- `height` (int): 图像高度

**返回:**
- `Tuple[float, float]`: 标准化坐标 (0-1)

**示例:**
```python
# 标准化坐标
norm_x, norm_y = CoordinateUtils.normalize_coordinates(100, 200, 800, 600)
print(f"标准化坐标: ({norm_x:.3f}, {norm_y:.3f})")
```

#### `denormalize_coordinates(norm_x: float, norm_y: float, width: int, height: int) -> Tuple[int, int]`

反标准化坐标。

**参数:**
- `norm_x` (float): 标准化X坐标
- `norm_y` (float): 标准化Y坐标
- `width` (int): 图像宽度
- `height` (int): 图像高度

**返回:**
- `Tuple[int, int]`: 实际坐标

**示例:**
```python
# 反标准化坐标
x, y = CoordinateUtils.denormalize_coordinates(0.125, 0.333, 800, 600)
print(f"实际坐标: ({x}, {y})")
```

#### `is_point_in_region(x: int, y: int, region: Tuple[int, int, int, int]) -> bool`

检查点是否在区域内。

**参数:**
- `x` (int): X坐标
- `y` (int): Y坐标
- `region` (Tuple[int, int, int, int]): 区域 (x, y, width, height)

**返回:**
- `bool`: 点是否在区域内

**示例:**
```python
# 检查点是否在区域内
in_region = CoordinateUtils.is_point_in_region(100, 200, (50, 150, 200, 200))
if in_region:
    print("点在区域内")
```

#### `get_region_center(region: Tuple[int, int, int, int]) -> Tuple[int, int]`

获取区域中心点。

**参数:**
- `region` (Tuple[int, int, int, int]): 区域 (x, y, width, height)

**返回:**
- `Tuple[int, int]`: 中心点坐标

**示例:**
```python
# 获取区域中心点
center_x, center_y = CoordinateUtils.get_region_center((100, 100, 200, 200))
print(f"中心点: ({center_x}, {center_y})")
```

## 缩放工具 (ScaleUtils)

### 类定义

```python
class ScaleUtils:
    """缩放管理工具类"""
```

### 静态方法

#### `calculate_scale_factor(original_size: Tuple[int, int], target_size: Tuple[int, int]) -> float`

计算缩放因子。

**参数:**
- `original_size` (Tuple[int, int]): 原始尺寸 (width, height)
- `target_size` (Tuple[int, int]): 目标尺寸 (width, height)

**返回:**
- `float`: 缩放因子

**示例:**
```python
from gui_auto.utils import ScaleUtils

# 计算缩放因子
scale = ScaleUtils.calculate_scale_factor((800, 600), (1920, 1080))
print(f"缩放因子: {scale}")
```

#### `apply_scaling(coordinates: Tuple[int, int], scale_factor: float) -> Tuple[int, int]`

应用缩放。

**参数:**
- `coordinates` (Tuple[int, int]): 原始坐标
- `scale_factor` (float): 缩放因子

**返回:**
- `Tuple[int, int]`: 缩放后的坐标

**示例:**
```python
# 应用缩放
scaled_x, scaled_y = ScaleUtils.apply_scaling((100, 200), 1.5)
print(f"缩放后坐标: ({scaled_x}, {scaled_y})")
```

#### `create_scale_record(original_size: Tuple[int, int], scaled_size: Tuple[int, int], scale_factor: float) -> Dict[str, Any]`

创建缩放记录。

**参数:**
- `original_size` (Tuple[int, int]): 原始尺寸
- `scaled_size` (Tuple[int, int]): 缩放后尺寸
- `scale_factor` (float): 缩放因子

**返回:**
- `Dict[str, Any]`: 缩放记录

**示例:**
```python
# 创建缩放记录
record = ScaleUtils.create_scale_record((800, 600), (1920, 1080), 1.5)
print(f"缩放记录: {record}")
```

#### `convert_coordinates_with_scale_record(coordinates: Tuple[int, int], scale_record: Dict[str, Any], direction: str = "forward") -> Tuple[int, int]`

使用缩放记录转换坐标。

**参数:**
- `coordinates` (Tuple[int, int]): 原始坐标
- `scale_record` (Dict[str, Any]): 缩放记录
- `direction` (str): 转换方向，'forward' 或 'backward'

**返回:**
- `Tuple[int, int]`: 转换后的坐标

**示例:**
```python
# 正向转换
converted_x, converted_y = ScaleUtils.convert_coordinates_with_scale_record(
    (100, 200), scale_record, "forward"
)

# 反向转换
original_x, original_y = ScaleUtils.convert_coordinates_with_scale_record(
    (150, 300), scale_record, "backward"
)
```

#### `create_image_pyramid(image: np.ndarray, levels: int = 3) -> List[np.ndarray]`

创建图像金字塔。

**参数:**
- `image` (np.ndarray): 输入图像
- `levels` (int): 金字塔层数

**返回:**
- `List[np.ndarray]`: 金字塔图像列表

**示例:**
```python
# 创建图像金字塔
pyramid = ScaleUtils.create_image_pyramid(image, levels=4)
print(f"金字塔层数: {len(pyramid)}")
```

## 格式工具 (FormatUtils)

### 类定义

```python
class FormatUtils:
    """格式转换工具类"""
```

### 静态方法

#### `detect_image_format(image: np.ndarray) -> str`

检测图像格式。

**参数:**
- `image` (np.ndarray): 输入图像

**返回:**
- `str`: 图像格式 ('bgr', 'rgb', 'gray')

**示例:**
```python
from gui_auto.utils import FormatUtils

# 检测图像格式
format_type = FormatUtils.detect_image_format(image)
print(f"图像格式: {format_type}")
```

#### `convert_bgr_to_rgb(image: np.ndarray) -> np.ndarray`

BGR转RGB。

**参数:**
- `image` (np.ndarray): BGR图像

**返回:**
- `np.ndarray`: RGB图像

**示例:**
```python
# BGR转RGB
rgb_image = FormatUtils.convert_bgr_to_rgb(bgr_image)
```

#### `convert_rgb_to_bgr(image: np.ndarray) -> np.ndarray`

RGB转BGR。

**参数:**
- `image` (np.ndarray): RGB图像

**返回:**
- `np.ndarray`: BGR图像

**示例:**
```python
# RGB转BGR
bgr_image = FormatUtils.convert_rgb_to_bgr(rgb_image)
```

#### `convert_to_grayscale(image: np.ndarray) -> np.ndarray`

转换为灰度图。

**参数:**
- `image` (np.ndarray): 输入图像

**返回:**
- `np.ndarray`: 灰度图像

**示例:**
```python
# 转换为灰度图
gray_image = FormatUtils.convert_to_grayscale(image)
```

#### `normalize_image(image: np.ndarray, min_val: float = 0.0, max_val: float = 1.0) -> np.ndarray`

标准化图像。

**参数:**
- `image` (np.ndarray): 输入图像
- `min_val` (float): 最小值
- `max_val` (float): 最大值

**返回:**
- `np.ndarray`: 标准化后的图像

**示例:**
```python
# 标准化图像到0-1范围
normalized = FormatUtils.normalize_image(image, 0.0, 1.0)
```

#### `denormalize_image(image: np.ndarray, min_val: float = 0.0, max_val: float = 255.0) -> np.ndarray`

反标准化图像。

**参数:**
- `image` (np.ndarray): 标准化图像
- `min_val` (float): 最小值
- `max_val` (float): 最大值

**返回:**
- `np.ndarray`: 反标准化后的图像

**示例:**
```python
# 反标准化图像到0-255范围
denormalized = FormatUtils.denormalize_image(image, 0.0, 255.0)
```

## 重试工具 (RetryUtils)

### 类定义

```python
class RetryUtils:
    """重试工具类"""
```

### 静态方法

#### `retry_with_backoff(func: Callable, max_retries: int = 3, base_delay: float = 1.0, backoff_factor: float = 2.0, max_delay: float = 60.0, **kwargs) -> Any`

带退避的重试装饰器。

**参数:**
- `func` (Callable): 要重试的函数
- `max_retries` (int): 最大重试次数
- `base_delay` (float): 基础延迟时间
- `backoff_factor` (float): 退避因子
- `max_delay` (float): 最大延迟时间
- `**kwargs`: 其他参数

**返回:**
- `Any`: 函数执行结果

**示例:**
```python
from gui_auto.utils import RetryUtils

@RetryUtils.retry_with_backoff(max_retries=5, base_delay=1.0)
def unreliable_function():
    # 可能失败的操作
    pass

# 使用装饰器
result = unreliable_function()
```

#### `retry_on_exception(func: Callable, exceptions: Tuple[Type[Exception], ...] = (Exception,), max_retries: int = 3, delay: float = 1.0, **kwargs) -> Any`

异常重试装饰器。

**参数:**
- `func` (Callable): 要重试的函数
- `exceptions` (Tuple[Type[Exception], ...]): 要捕获的异常类型
- `max_retries` (int): 最大重试次数
- `delay` (float): 重试延迟时间
- `**kwargs`: 其他参数

**返回:**
- `Any`: 函数执行结果

**示例:**
```python
@RetryUtils.retry_on_exception(exceptions=(ConnectionError, TimeoutError), max_retries=3)
def network_operation():
    # 网络操作
    pass
```

#### `calculate_delay(attempt: int, base_delay: float = 1.0, backoff_factor: float = 2.0, max_delay: float = 60.0) -> float`

计算重试延迟时间。

**参数:**
- `attempt` (int): 重试次数
- `base_delay` (float): 基础延迟时间
- `backoff_factor` (float): 退避因子
- `max_delay` (float): 最大延迟时间

**返回:**
- `float`: 延迟时间

**示例:**
```python
# 计算第3次重试的延迟时间
delay = RetryUtils.calculate_delay(3, base_delay=1.0, backoff_factor=2.0)
print(f"延迟时间: {delay}秒")
```

## 使用示例

### 图像处理示例

```python
from gui_auto.utils import ImageUtils, FormatUtils
import cv2

# 加载图像
image = ImageUtils.load_image("button.png", "bgr")

# 获取图像信息
info = ImageUtils.get_image_info(image)
print(f"图像尺寸: {info['size']}")
print(f"图像格式: {info['format']}")

# 转换格式
rgb_image = FormatUtils.convert_bgr_to_rgb(image)
gray_image = FormatUtils.convert_to_grayscale(image)

# 调整大小
resized = ImageUtils.resize_image(image, (400, 300))

# 裁剪图像
cropped = ImageUtils.crop_image(image, (100, 100, 200, 200))

# 保存图像
ImageUtils.save_image(resized, "resized.png", quality=95)
```

### 坐标转换示例

```python
from gui_auto.utils import CoordinateUtils, ScaleUtils

# 坐标转换
base_x, base_y = 100, 200
scale_factor = 1.5

# 转换为系统坐标
sys_x, sys_y = CoordinateUtils.convert_coordinates(
    base_x, base_y, "base", "system", scale_factor
)
print(f"系统坐标: ({sys_x}, {sys_y})")

# 标准化坐标
norm_x, norm_y = CoordinateUtils.normalize_coordinates(
    sys_x, sys_y, 1920, 1080
)
print(f"标准化坐标: ({norm_x:.3f}, {norm_y:.3f})")

# 创建缩放记录
scale_record = ScaleUtils.create_scale_record(
    (800, 600), (1920, 1080), scale_factor
)

# 使用缩放记录转换坐标
converted_x, converted_y = ScaleUtils.convert_coordinates_with_scale_record(
    (100, 200), scale_record, "forward"
)
print(f"转换后坐标: ({converted_x}, {converted_y})")
```

### 重试机制示例

```python
from gui_auto.utils import RetryUtils
import time
import random

# 使用重试装饰器
@RetryUtils.retry_with_backoff(max_retries=5, base_delay=1.0, backoff_factor=2.0)
def unreliable_operation():
    """模拟不可靠的操作"""
    if random.random() < 0.7:  # 70% 失败率
        raise Exception("操作失败")
    return "操作成功"

# 执行操作
try:
    result = unreliable_operation()
    print(f"结果: {result}")
except Exception as e:
    print(f"最终失败: {e}")

# 异常重试
@RetryUtils.retry_on_exception(
    exceptions=(ConnectionError, TimeoutError),
    max_retries=3,
    delay=2.0
)
def network_operation():
    """模拟网络操作"""
    if random.random() < 0.5:
        raise ConnectionError("网络连接失败")
    return "网络操作成功"

# 执行网络操作
try:
    result = network_operation()
    print(f"网络操作结果: {result}")
except Exception as e:
    print(f"网络操作最终失败: {e}")
```

### 综合使用示例

```python
from gui_auto.utils import ImageUtils, CoordinateUtils, ScaleUtils, FormatUtils
from gui_auto import create_tool

# 创建工具
tool = create_tool()

# 截取屏幕
screenshot = tool.capture_screen()

# 查找图像
result = tool.find_image("button.png")
if result.success:
    # 获取匹配位置
    position = result.position
    size = result.size
    
    # 转换坐标
    sys_x, sys_y = CoordinateUtils.convert_coordinates(
        position[0], position[1], "base", "system", result.scale
    )
    
    # 点击坐标
    tool.click(sys_x, sys_y)
    
    # 获取图像信息
    info = ImageUtils.get_image_info(screenshot)
    print(f"屏幕截图信息: {info}")
    
    # 创建缩放记录
    scale_record = ScaleUtils.create_scale_record(
        (800, 600), info['size'], result.scale
    )
    
    # 保存缩放记录
    import json
    with open("scale_record.json", "w") as f:
        json.dump(scale_record, f)
```
