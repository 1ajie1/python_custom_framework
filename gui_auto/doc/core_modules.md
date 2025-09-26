# 核心模块API

## 概述

核心模块提供框架的基础功能，包括配置管理、日志系统和异常处理。

## 配置管理 (GuiConfig)

### 类定义

```python
@dataclass
class GuiConfig:
    """GUI自动化工具配置类"""
```

### 配置项

| 配置项 | 类型 | 默认值 | 描述 |
|--------|------|--------|------|
| `match_confidence` | float | 0.8 | 图像匹配置信度阈值 |
| `match_timeout` | float | 10.0 | 匹配超时时间（秒） |
| `default_method` | str | "TM_CCOEFF_NORMED" | 默认匹配方法 |
| `auto_scale` | bool | True | 是否启用自动缩放 |
| `default_max_retries` | int | 3 | 默认最大重试次数 |
| `match_config` | MatchConfig | MatchConfig() | 匹配配置 |
| `retry_config` | RetryConfig | RetryConfig() | 重试配置 |
| `scale_config` | ScaleConfig | ScaleConfig() | 缩放配置 |
| `image_config` | ImageConfig | ImageConfig() | 图像配置 |

### 方法

#### `__init__(**kwargs)`

初始化配置。

**参数:**
- `**kwargs`: 配置项键值对

**示例:**
```python
from gui_auto import GuiConfig

# 使用默认配置
config = GuiConfig()

# 自定义配置
config = GuiConfig(
    match_confidence=0.9,
    match_timeout=15.0,
    auto_scale=True
)
```

#### `validate() -> bool`

验证配置的有效性。

**返回:**
- `bool`: 配置是否有效

**示例:**
```python
if config.validate():
    print("配置有效")
else:
    print("配置无效")
```

#### `to_dict() -> dict`

将配置转换为字典。

**返回:**
- `dict`: 配置字典

**示例:**
```python
config_dict = config.to_dict()
print(config_dict)
```

#### `from_dict(data: dict) -> GuiConfig`

从字典创建配置。

**参数:**
- `data` (dict): 配置字典

**返回:**
- `GuiConfig`: 配置对象

**示例:**
```python
config_dict = {
    "match_confidence": 0.9,
    "match_timeout": 15.0
}
config = GuiConfig.from_dict(config_dict)
```

#### `save_to_file(file_path: str) -> bool`

保存配置到文件。

**参数:**
- `file_path` (str): 文件路径

**返回:**
- `bool`: 是否成功保存

**示例:**
```python
config.save_to_file("config.json")
```

#### `load_from_file(file_path: str) -> GuiConfig`

从文件加载配置。

**参数:**
- `file_path` (str): 文件路径

**返回:**
- `GuiConfig`: 配置对象

**示例:**
```python
config = GuiConfig.load_from_file("config.json")
```

## 匹配配置 (MatchConfig)

### 类定义

```python
@dataclass
class MatchConfig:
    """图像匹配配置类"""
```

### 配置项

| 配置项 | 类型 | 默认值 | 描述 |
|--------|------|--------|------|
| `confidence` | float | 0.8 | 匹配置信度阈值 |
| `timeout` | float | 10.0 | 匹配超时时间 |
| `method` | str | "TM_CCOEFF_NORMED" | 匹配方法 |
| `scale_range` | tuple | (0.8, 1.2) | 缩放范围 |
| `max_features` | int | 500 | 最大特征点数 |
| `match_threshold` | float | 0.7 | 匹配阈值 |

### 方法

#### `get_opencv_method() -> int`

获取OpenCV匹配方法。

**返回:**
- `int`: OpenCV方法常量

**示例:**
```python
method = match_config.get_opencv_method()
```

## 重试配置 (RetryConfig)

### 类定义

```python
@dataclass
class RetryConfig:
    """重试配置类"""
```

### 配置项

| 配置项 | 类型 | 默认值 | 描述 |
|--------|------|--------|------|
| `max_retries` | int | 3 | 最大重试次数 |
| `retry_delay` | float | 1.0 | 重试延迟（秒） |
| `backoff_factor` | float | 2.0 | 退避因子 |
| `max_delay` | float | 60.0 | 最大延迟（秒） |

### 方法

#### `calculate_delay(attempt: int) -> float`

计算重试延迟。

**参数:**
- `attempt` (int): 重试次数

**返回:**
- `float`: 延迟时间（秒）

**示例:**
```python
delay = retry_config.calculate_delay(2)
```

## 缩放配置 (ScaleConfig)

### 类定义

```python
@dataclass
class ScaleConfig:
    """缩放配置类"""
```

### 配置项

| 配置项 | 类型 | 默认值 | 描述 |
|--------|------|--------|------|
| `enabled` | bool | True | 是否启用缩放 |
| `scale_range` | tuple | (0.8, 1.2) | 缩放范围 |
| `scale_step` | float | 0.1 | 缩放步长 |
| `max_scale` | float | 2.0 | 最大缩放比例 |
| `min_scale` | float | 0.5 | 最小缩放比例 |

## 图像配置 (ImageConfig)

### 类定义

```python
@dataclass
class ImageConfig:
    """图像配置类"""
```

### 配置项

| 配置项 | 类型 | 默认值 | 描述 |
|--------|------|--------|------|
| `default_format` | str | "bgr" | 默认图像格式 |
| `quality` | int | 95 | 图像质量 |
| `compression` | int | 6 | 压缩级别 |
| `max_size` | tuple | (1920, 1080) | 最大图像尺寸 |

## 日志系统

### 获取日志器

#### `get_logger(name: str = None) -> logging.Logger`

获取日志器实例。

**参数:**
- `name` (str, 可选): 日志器名称

**返回:**
- `logging.Logger`: 日志器实例

**示例:**
```python
from gui_auto import get_logger

logger = get_logger("my_module")
logger.info("这是一条信息日志")
logger.error("这是一条错误日志")
```

### 日志级别

框架支持以下日志级别：

- `DEBUG`: 调试信息
- `INFO`: 一般信息
- `WARNING`: 警告信息
- `ERROR`: 错误信息
- `CRITICAL`: 严重错误

### 日志格式

默认日志格式：
```
%(asctime)s - %(name)s - %(levelname)s - %(message)s
```

### 配置日志

```python
import logging
from gui_auto import get_logger

# 设置日志级别
logging.basicConfig(level=logging.INFO)

# 获取日志器
logger = get_logger("my_app")

# 使用日志器
logger.info("应用程序启动")
logger.debug("调试信息")
logger.warning("警告信息")
logger.error("错误信息")
```

## 异常处理

### 异常类层次结构

```
Exception
├── GuiAutoError (基础异常)
│   ├── OperationError (操作异常)
│   ├── AlgorithmError (算法异常)
│   ├── PlatformError (平台异常)
│   ├── ConfigError (配置异常)
│   └── ValidationError (验证异常)
```

### 基础异常 (GuiAutoError)

```python
class GuiAutoError(Exception):
    """GUI自动化框架基础异常类"""
```

**属性:**
- `message` (str): 错误消息
- `error_code` (str): 错误代码
- `details` (dict): 错误详情

**示例:**
```python
from gui_auto.core.exceptions import GuiAutoError

try:
    # 执行操作
    pass
except GuiAutoError as e:
    print(f"错误: {e.message}")
    print(f"错误代码: {e.error_code}")
    print(f"错误详情: {e.details}")
```

### 操作异常 (OperationError)

```python
class OperationError(GuiAutoError):
    """操作执行异常"""
```

**示例:**
```python
from gui_auto.core.exceptions import OperationError

try:
    tool.click(100, 200)
except OperationError as e:
    print(f"点击操作失败: {e.message}")
```

### 算法异常 (AlgorithmError)

```python
class AlgorithmError(GuiAutoError):
    """算法相关异常"""
```

**示例:**
```python
from gui_auto.core.exceptions import AlgorithmError

try:
    result = tool.find_image("template.png")
except AlgorithmError as e:
    print(f"图像匹配失败: {e.message}")
```

### 平台异常 (PlatformError)

```python
class PlatformError(GuiAutoError):
    """平台相关异常"""
```

**示例:**
```python
from gui_auto.core.exceptions import PlatformError

try:
    tool.capture_screen()
except PlatformError as e:
    print(f"屏幕截图失败: {e.message}")
```

### 配置异常 (ConfigError)

```python
class ConfigError(GuiAutoError):
    """配置相关异常"""
```

**示例:**
```python
from gui_auto.core.exceptions import ConfigError

try:
    config = GuiConfig(confidence=2.0)  # 无效的置信度
except ConfigError as e:
    print(f"配置错误: {e.message}")
```

### 验证异常 (ValidationError)

```python
class ValidationError(GuiAutoError):
    """验证异常"""
```

**示例:**
```python
from gui_auto.core.exceptions import ValidationError

try:
    config.validate()
except ValidationError as e:
    print(f"验证失败: {e.message}")
```

## 便捷函数

### `create_tool(config: GuiConfig = None) -> GuiAutoToolV2`

创建GUI自动化工具实例。

**参数:**
- `config` (GuiConfig, 可选): 配置对象

**返回:**
- `GuiAutoToolV2`: 工具实例

**示例:**
```python
from gui_auto import create_tool, GuiConfig

# 使用默认配置
tool = create_tool()

# 使用自定义配置
config = GuiConfig(match_confidence=0.9)
tool = create_tool(config)
```

### `get_version() -> str`

获取框架版本。

**返回:**
- `str`: 版本号

**示例:**
```python
from gui_auto import get_version

version = get_version()
print(f"框架版本: {version}")
```

## 使用示例

### 完整配置示例

```python
from gui_auto import GuiConfig, create_tool, get_logger

# 创建日志器
logger = get_logger("my_app")

# 创建配置
config = GuiConfig(
    match_confidence=0.9,
    match_timeout=15.0,
    auto_scale=True,
    default_max_retries=5
)

# 验证配置
if not config.validate():
    logger.error("配置验证失败")
    exit(1)

# 保存配置
config.save_to_file("my_config.json")

# 创建工具
tool = create_tool(config)

# 使用工具
try:
    result = tool.find_image("button.png")
    if result.success:
        logger.info(f"找到图像，位置: {result.position}")
    else:
        logger.warning("未找到图像")
except Exception as e:
    logger.error(f"操作失败: {e}")
```

### 错误处理示例

```python
from gui_auto import create_tool
from gui_auto.core.exceptions import (
    OperationError, AlgorithmError, PlatformError, ConfigError
)

tool = create_tool()

try:
    result = tool.find_and_click("button.png")
except OperationError as e:
    print(f"操作失败: {e.message}")
except AlgorithmError as e:
    print(f"算法错误: {e.message}")
except PlatformError as e:
    print(f"平台错误: {e.message}")
except ConfigError as e:
    print(f"配置错误: {e.message}")
except Exception as e:
    print(f"未知错误: {e}")
```
