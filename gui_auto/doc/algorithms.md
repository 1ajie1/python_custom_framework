# 算法模块API

## 概述

算法模块提供多种图像匹配算法，包括模板匹配、特征匹配、混合匹配和金字塔匹配。

## 匹配结果 (MatchResult)

### 类定义

```python
@dataclass
class MatchResult:
    """图像匹配结果类"""
```

### 属性

| 属性 | 类型 | 描述 |
|------|------|------|
| `found` | bool | 是否匹配成功 |
| `center` | tuple | 匹配中心位置 (x, y) |
| `confidence` | float | 匹配置信度 (0-1) |
| `bbox` | tuple | 匹配边界框 (x, y, width, height) |
| `scale_factor` | float | 匹配时的缩放比例 |
| `algorithm_name` | str | 使用的算法名称 |
| `metadata` | dict | 详细信息和元数据 |

### 方法

#### `is_valid() -> bool`

检查匹配结果是否有效。

**返回:**
- `bool`: 结果是否有效

**示例:**
```python
result = tool.find_image("button.png")
if result.found:
    print(f"匹配成功，位置: {result.center}")
```

#### `get_center() -> tuple`

获取匹配区域的中心点。

**返回:**
- `tuple[int, int]`: 中心点坐标 (x, y)

**示例:**
```python
center = result.center
print(f"中心点: {center}")
```

#### `get_bounds() -> tuple`

获取匹配区域的边界。

**返回:**
- `tuple`: 边界 (x, y, width, height)

**示例:**
```python
bounds = result.bbox
print(f"边界: {bounds}")
```

## 算法基类 (MatchingAlgorithm)

### 类定义

```python
class MatchingAlgorithm(ABC):
    """图像匹配算法基类"""
```

### 抽象方法

#### `match(template, target, **kwargs) -> MatchResult`

执行图像匹配。

**参数:**
- `template` (np.ndarray): 模板图像
- `target` (np.ndarray): 目标图像
- `**kwargs`: 算法特定参数

**返回:**
- `MatchResult`: 匹配结果

#### `get_name() -> str`

获取算法名称。

**返回:**
- `str`: 算法名称

#### `get_supported_methods() -> list`

获取支持的方法列表。

**返回:**
- `list[str]`: 支持的方法列表

## 模板匹配 (TemplateMatching)

### 类定义

```python
class TemplateMatching(MatchingAlgorithm):
    """模板匹配算法"""
```

### 支持的方法

| 方法 | 描述 | 适用场景 |
|------|------|----------|
| `TM_CCOEFF_NORMED` | 归一化相关系数 | 一般模板匹配 |
| `TM_CCORR_NORMED` | 归一化相关 | 快速匹配 |
| `TM_SQDIFF_NORMED` | 归一化平方差 | 精确匹配 |

### 方法

#### `match(template, target, method="TM_CCOEFF_NORMED", confidence=0.8, **kwargs) -> MatchResult`

执行模板匹配。

**参数:**
- `template` (np.ndarray): 模板图像
- `target` (np.ndarray): 目标图像
- `method` (str): 匹配方法
- `confidence` (float): 置信度阈值
- `**kwargs`: 其他参数

**返回:**
- `MatchResult`: 匹配结果

**示例:**
```python
from gui_auto.algorithms import TemplateMatching

algorithm = TemplateMatching()
result = algorithm.match(template, target, method="TM_CCOEFF_NORMED", confidence=0.9)
```

#### `match_multi_scale(template, target, scale_range=(0.8, 1.2), **kwargs) -> MatchResult`

多尺度模板匹配。

**参数:**
- `template` (np.ndarray): 模板图像
- `target` (np.ndarray): 目标图像
- `scale_range` (tuple): 缩放范围
- `**kwargs`: 其他参数

**返回:**
- `MatchResult`: 匹配结果

**示例:**
```python
result = algorithm.match_multi_scale(template, target, scale_range=(0.5, 2.0))
```

## 特征匹配 (FeatureMatching)

### 类定义

```python
class FeatureMatching(MatchingAlgorithm):
    """特征匹配算法"""
```

### 支持的方法

| 方法 | 描述 | 特点 |
|------|------|------|
| `SIFT` | SIFT特征匹配 | 尺度不变，旋转不变 |
| `SURF` | SURF特征匹配 | 快速SIFT |
| `ORB` | ORB特征匹配 | 快速，二进制特征 |
| `AKAZE` | AKAZE特征匹配 | 快速，多尺度 |

### 方法

#### `match(template, target, method="SIFT", max_features=500, **kwargs) -> MatchResult`

执行特征匹配。

**参数:**
- `template` (np.ndarray): 模板图像
- `target` (np.ndarray): 目标图像
- `method` (str): 特征检测方法
- `max_features` (int): 最大特征点数
- `**kwargs`: 其他参数

**返回:**
- `MatchResult`: 匹配结果

**示例:**
```python
from gui_auto.algorithms import FeatureMatching

algorithm = FeatureMatching()
result = algorithm.match(template, target, method="SIFT", max_features=1000)
```

#### `match_with_ratio_test(template, target, ratio=0.75, **kwargs) -> MatchResult`

使用比率测试的特征匹配。

**参数:**
- `template` (np.ndarray): 模板图像
- `target` (np.ndarray): 目标图像
- `ratio` (float): 比率阈值
- `**kwargs`: 其他参数

**返回:**
- `MatchResult`: 匹配结果

**示例:**
```python
result = algorithm.match_with_ratio_test(template, target, ratio=0.8)
```

## 混合匹配 (HybridMatching)

### 类定义

```python
class HybridMatching(MatchingAlgorithm):
    """混合匹配算法"""
```

### 方法

#### `match(template, target, primary_algorithm="template", fallback_algorithm="feature", **kwargs) -> MatchResult`

执行混合匹配。

**参数:**
- `template` (np.ndarray): 模板图像
- `target` (np.ndarray): 目标图像
- `primary_algorithm` (str): 主要算法
- `fallback_algorithm` (str): 备用算法
- `**kwargs`: 其他参数

**返回:**
- `MatchResult`: 匹配结果

**示例:**
```python
from gui_auto.algorithms import HybridMatching

algorithm = HybridMatching()
result = algorithm.match(template, target, primary_algorithm="template", fallback_algorithm="feature")
```

## 金字塔匹配 (PyramidMatching)

### 类定义

```python
class PyramidMatching(MatchingAlgorithm):
    """金字塔匹配算法"""
```

### 方法

#### `match(template, target, levels=3, **kwargs) -> MatchResult`

执行金字塔匹配。

**参数:**
- `template` (np.ndarray): 模板图像
- `target` (np.ndarray): 目标图像
- `levels` (int): 金字塔层数
- `**kwargs`: 其他参数

**返回:**
- `MatchResult`: 匹配结果

**示例:**
```python
from gui_auto.algorithms import PyramidMatching

algorithm = PyramidMatching()
result = algorithm.match(template, target, levels=4)
```

## 算法工厂 (AlgorithmFactory)

### 类定义

```python
class AlgorithmFactory:
    """算法工厂类"""
```

### 类方法

#### `create_algorithm(name: str, config: Any = None) -> MatchingAlgorithm`

创建算法实例。

**参数:**
- `name` (str): 算法名称
- `config` (Any, 可选): 配置对象

**返回:**
- `MatchingAlgorithm`: 算法实例

**示例:**
```python
from gui_auto.algorithms import AlgorithmFactory

# 创建模板匹配算法
template_algorithm = AlgorithmFactory.create_algorithm("template")

# 创建特征匹配算法
feature_algorithm = AlgorithmFactory.create_algorithm("feature")

# 创建混合匹配算法
hybrid_algorithm = AlgorithmFactory.create_algorithm("hybrid")
```

#### `get_available_algorithms() -> list`

获取可用的算法列表。

**返回:**
- `list[str]`: 算法名称列表

**示例:**
```python
algorithms = AlgorithmFactory.get_available_algorithms()
print(f"可用算法: {algorithms}")
```

#### `register_algorithm(name: str, algorithm_class: Type[MatchingAlgorithm])`

注册自定义算法。

**参数:**
- `name` (str): 算法名称
- `algorithm_class` (Type[MatchingAlgorithm]): 算法类

**示例:**
```python
from gui_auto.algorithms import AlgorithmFactory, MatchingAlgorithm

class CustomAlgorithm(MatchingAlgorithm):
    def match(self, template, target, **kwargs):
        # 自定义匹配逻辑
        pass
    
    def get_name(self):
        return "custom"
    
    def get_supported_methods(self):
        return ["custom_method"]

# 注册自定义算法
AlgorithmFactory.register_algorithm("custom", CustomAlgorithm)
```

## 参数传递功能

### 概述

框架支持为不同的匹配算法传递特定参数，所有参数都可以通过 `find_image()` 方法的 `**kwargs` 参数传递。

### 模板匹配参数

```python
result = tool.find_image(
    "button.png",
    algorithm="template",
    method="TM_CCOEFF_NORMED",  # 匹配方法
    confidence=0.8              # 置信度阈值
)
```

**支持的参数：**
- `method`: 匹配方法 (`TM_CCOEFF_NORMED`, `TM_CCORR_NORMED`, `TM_SQDIFF_NORMED`, `TM_CCOEFF`, `TM_CCORR`, `TM_SQDIFF`)
- `confidence`: 置信度阈值 (0.0-1.0)

### 特征匹配参数

```python
result = tool.find_image(
    "button.png",
    algorithm="feature",
    min_matches=10,             # 最小匹配数
    confidence=0.7,             # 置信度阈值
    nfeatures=1000,             # 特征点数量
    scaleFactor=1.2,            # 缩放因子
    nlevels=8,                  # 金字塔层数
    edgeThreshold=15,           # 边缘阈值
    patchSize=31,               # patch大小
    fastThreshold=20            # FAST阈值
)
```

**支持的参数：**
- `min_matches`: 最小匹配数 (默认: 5)
- `confidence`: 置信度阈值 (0.0-1.0)
- `nfeatures`: ORB特征点数量 (默认: 500)
- `scaleFactor`: 金字塔缩放因子 (默认: 1.1)
- `nlevels`: 金字塔层数 (默认: 4)
- `edgeThreshold`: 边缘阈值 (默认: 5)
- `patchSize`: patch大小 (默认: 15)
- `fastThreshold`: FAST阈值 (默认: 10)

### 金字塔匹配参数

```python
result = tool.find_image(
    "button.png",
    algorithm="pyramid",
    pyramid_levels=6,           # 金字塔层数
    pyramid_scale_factor=0.7,   # 缩放因子
    confidence=0.8,             # 置信度阈值
    method="TM_CCOEFF_NORMED"   # 底层匹配方法
)
```

**支持的参数：**
- `pyramid_levels`: 金字塔层数 (默认: 4)
- `pyramid_scale_factor`: 缩放因子 (默认: 0.5)
- `confidence`: 置信度阈值 (默认: 0.8)
- `method`: 底层匹配方法 (默认: 'TM_CCOEFF_NORMED')

### 混合匹配参数

```python
result = tool.find_image(
    "button.png",
    algorithm="hybrid",
    template_weight=0.7,        # 模板匹配权重
    feature_weight=0.3,         # 特征匹配权重
    confidence=0.8,             # 置信度阈值
    method="TM_CCOEFF_NORMED",  # 模板匹配方法
    min_matches=5               # 特征匹配最小匹配数
)
```

**支持的参数：**
- `template_weight`: 模板匹配权重 (默认: 0.6)
- `feature_weight`: 特征匹配权重 (默认: 0.4)
- `confidence`: 置信度阈值 (默认: 0.8)
- `method`: 模板匹配方法 (默认: 'TM_CCOEFF_NORMED')
- `min_matches`: 特征匹配最小匹配数 (默认: 5)

### 参数配置示例

```python
# 高精度匹配
result = tool.find_image(
    "button.png",
    algorithm="template",
    method="TM_CCOEFF_NORMED",
    confidence=0.95
)

# 快速匹配
result = tool.find_image(
    "button.png",
    algorithm="pyramid",
    pyramid_levels=3,
    pyramid_scale_factor=0.8,
    confidence=0.7
)

# 鲁棒性匹配
result = tool.find_image(
    "button.png",
    algorithm="hybrid",
    template_weight=0.4,
    feature_weight=0.6,
    confidence=0.8,
    min_matches=10
)

# 多尺度匹配
result = tool.find_image(
    "button.png",
    algorithm="pyramid",
    pyramid_levels=8,
    pyramid_scale_factor=0.6,
    confidence=0.8,
    method="TM_CCORR_NORMED"
)
```

## 使用示例

### 基本使用

```python
from gui_auto import create_tool
from gui_auto.algorithms import AlgorithmFactory

# 创建工具
tool = create_tool()

# 使用默认算法查找图像
result = tool.find_image("button.png")
if result.found:
    print(f"找到图像，位置: {result.center}")
    print(f"置信度: {result.confidence}")

# 使用特定算法
result = tool.find_image("button.png", algorithm="feature")
if result.found:
    print(f"特征匹配成功，位置: {result.center}")
```

### 高级使用

```python
from gui_auto.algorithms import AlgorithmFactory, MatchResult
import cv2
import numpy as np

# 加载图像
template = cv2.imread("button.png")
target = cv2.imread("screenshot.png")

# 创建算法
algorithm = AlgorithmFactory.create_algorithm("template")

# 执行匹配
result = algorithm.match(
    template, 
    target, 
    method="TM_CCOEFF_NORMED",
    confidence=0.9
)

# 检查结果
if result.found:
    print(f"匹配成功:")
    print(f"  位置: {result.center}")
    print(f"  置信度: {result.confidence}")
    print(f"  边界框: {result.bbox}")
    print(f"  缩放: {result.scale_factor}")
    print(f"  算法: {result.algorithm_name}")
    
    # 获取中心点
    center = result.center
    print(f"  中心点: {center}")
    
    # 获取边界
    bounds = result.bbox
    print(f"  边界: {bounds}")
else:
    print("匹配失败")
```

### 多算法比较

```python
from gui_auto.algorithms import AlgorithmFactory

template = cv2.imread("button.png")
target = cv2.imread("screenshot.png")

algorithms = ["template", "feature", "hybrid", "pyramid"]
results = {}

for algo_name in algorithms:
    algorithm = AlgorithmFactory.create_algorithm(algo_name)
    result = algorithm.match(template, target)
    results[algo_name] = result

# 比较结果
for algo_name, result in results.items():
    if result.found:
        print(f"{algo_name}: 成功, 置信度={result.confidence:.3f}")
    else:
        print(f"{algo_name}: 失败")
```

### 自定义算法

```python
from gui_auto.algorithms import MatchingAlgorithm, AlgorithmFactory, MatchResult
import cv2
import numpy as np
import time

class CustomTemplateMatching(MatchingAlgorithm):
    """自定义模板匹配算法"""
    
    def __init__(self, config=None):
        self.config = config or {}
    
    def match(self, template, target, **kwargs):
        start_time = time.time()
        
        # 获取参数
        method = kwargs.get('method', 'TM_CCOEFF_NORMED')
        confidence = kwargs.get('confidence', 0.8)
        
        # 转换为灰度图
        if len(template.shape) == 3:
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        else:
            template_gray = template
            
        if len(target.shape) == 3:
            target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
        else:
            target_gray = target
        
        # 执行匹配
        result = cv2.matchTemplate(target_gray, template_gray, getattr(cv2, method))
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        processing_time = time.time() - start_time
        
        # 检查置信度
        if max_val >= confidence:
            h, w = template_gray.shape[:2]
            return MatchResult(
                success=True,
                position=max_loc,
                confidence=float(max_val),
                size=(w, h),
                scale=1.0,
                method=method,
                algorithm="custom_template",
                processing_time=processing_time,
                details={"min_val": min_val, "max_val": max_val}
            )
        else:
            return MatchResult(
                success=False,
                position=(0, 0),
                confidence=float(max_val),
                size=(0, 0),
                scale=1.0,
                method=method,
                algorithm="custom_template",
                processing_time=processing_time,
                details={"min_val": min_val, "max_val": max_val}
            )
    
    def get_name(self):
        return "custom_template"
    
    def get_supported_methods(self):
        return ["TM_CCOEFF_NORMED", "TM_CCORR_NORMED", "TM_SQDIFF_NORMED"]

# 注册自定义算法
AlgorithmFactory.register_algorithm("custom_template", CustomTemplateMatching)

# 使用自定义算法
algorithm = AlgorithmFactory.create_algorithm("custom_template")
result = algorithm.match(template, target, confidence=0.9)
```
