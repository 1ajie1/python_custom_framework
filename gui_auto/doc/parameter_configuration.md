# 参数配置指南

## 概述

本文档详细介绍了GUI自动化框架中各种匹配算法的参数配置方法，帮助用户根据具体需求优化匹配效果。

## 参数传递方式

### 基本语法

```python
result = tool.find_image(
    template_path,
    algorithm="algorithm_name",
    parameter1=value1,
    parameter2=value2,
    ...
)
```

### 配置对象方式

```python
# 创建配置对象
config = type('Config', (), {
    'confidence': 0.8,
    'pyramid_levels': 6,
    'pyramid_scale_factor': 0.7
})()

# 使用配置
result = tool.find_image(template_path, algorithm="pyramid", **config.__dict__)
```

## 模板匹配参数

### 基本参数

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `method` | str | `TM_CCOEFF_NORMED` | 匹配方法 |
| `confidence` | float | 0.8 | 置信度阈值 |

### 匹配方法详解

| 方法 | 描述 | 适用场景 | 值范围 |
|------|------|----------|--------|
| `TM_CCOEFF_NORMED` | 归一化相关系数 | 一般模板匹配 | [0, 1] |
| `TM_CCORR_NORMED` | 归一化相关 | 快速匹配 | [0, 1] |
| `TM_SQDIFF_NORMED` | 归一化平方差 | 精确匹配 | [0, 1] |
| `TM_CCOEFF` | 相关系数 | 特殊场景 | [-1, 1] |
| `TM_CCORR` | 相关 | 特殊场景 | [0, ∞) |
| `TM_SQDIFF` | 平方差 | 特殊场景 | [0, ∞) |

### 使用示例

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
    algorithm="template",
    method="TM_CCORR_NORMED",
    confidence=0.7
)

# 精确匹配
result = tool.find_image(
    "button.png",
    algorithm="template",
    method="TM_SQDIFF_NORMED",
    confidence=0.9
)
```

## 特征匹配参数

### 基本参数

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `min_matches` | int | 5 | 最小匹配数 |
| `confidence` | float | 0.8 | 置信度阈值 |

### ORB检测器参数

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `nfeatures` | int | 500 | 特征点数量 |
| `scaleFactor` | float | 1.1 | 金字塔缩放因子 |
| `nlevels` | int | 4 | 金字塔层数 |
| `edgeThreshold` | int | 5 | 边缘阈值 |
| `patchSize` | int | 15 | patch大小 |
| `fastThreshold` | int | 10 | FAST阈值 |

### 参数调优建议

#### 小图像优化
```python
result = tool.find_image(
    "small_button.png",
    algorithm="feature",
    min_matches=3,
    confidence=0.6,
    nfeatures=200,
    scaleFactor=1.05,
    nlevels=2,
    edgeThreshold=3,
    patchSize=10,
    fastThreshold=5
)
```

#### 大图像优化
```python
result = tool.find_image(
    "large_button.png",
    algorithm="feature",
    min_matches=15,
    confidence=0.8,
    nfeatures=2000,
    scaleFactor=1.2,
    nlevels=8,
    edgeThreshold=20,
    patchSize=31,
    fastThreshold=20
)
```

## 金字塔匹配参数

### 基本参数

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `pyramid_levels` | int | 4 | 金字塔层数 |
| `pyramid_scale_factor` | float | 0.5 | 缩放因子 |
| `confidence` | float | 0.8 | 置信度阈值 |
| `method` | str | `TM_CCOEFF_NORMED` | 底层匹配方法 |

### 层数选择指南

| 图像尺寸 | 推荐层数 | 说明 |
|----------|----------|------|
| < 50x50 | 1-2 | 小图像，层数过多会失效 |
| 50x50 - 200x200 | 3-4 | 中等图像，平衡精度和速度 |
| 200x200 - 500x500 | 4-6 | 大图像，可以增加层数 |
| > 500x500 | 6-8 | 超大图像，多尺度匹配 |

### 缩放因子选择

| 缩放因子 | 适用场景 | 特点 |
|----------|----------|------|
| 0.5 | 快速匹配 | 层数少，速度快 |
| 0.7 | 平衡匹配 | 精度和速度平衡 |
| 0.8 | 精确匹配 | 层数多，精度高 |

### 使用示例

```python
# 快速多尺度匹配
result = tool.find_image(
    "button.png",
    algorithm="pyramid",
    pyramid_levels=3,
    pyramid_scale_factor=0.8,
    confidence=0.7
)

# 精确多尺度匹配
result = tool.find_image(
    "button.png",
    algorithm="pyramid",
    pyramid_levels=6,
    pyramid_scale_factor=0.6,
    confidence=0.8,
    method="TM_CCORR_NORMED"
)
```

## 混合匹配参数

### 基本参数

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `template_weight` | float | 0.6 | 模板匹配权重 |
| `feature_weight` | float | 0.4 | 特征匹配权重 |
| `confidence` | float | 0.8 | 置信度阈值 |
| `method` | str | `TM_CCOEFF_NORMED` | 模板匹配方法 |
| `min_matches` | int | 5 | 特征匹配最小匹配数 |

### 权重配置策略

#### 偏向模板匹配
```python
result = tool.find_image(
    "button.png",
    algorithm="hybrid",
    template_weight=0.8,
    feature_weight=0.2,
    confidence=0.8
)
```

#### 平衡策略
```python
result = tool.find_image(
    "button.png",
    algorithm="hybrid",
    template_weight=0.5,
    feature_weight=0.5,
    confidence=0.8
)
```

#### 偏向特征匹配
```python
result = tool.find_image(
    "button.png",
    algorithm="hybrid",
    template_weight=0.3,
    feature_weight=0.7,
    confidence=0.8,
    min_matches=10
)
```

## 场景化配置

### 高精度场景

```python
# 适用于需要极高匹配精度的场景
result = tool.find_image(
    "critical_button.png",
    algorithm="template",
    method="TM_CCOEFF_NORMED",
    confidence=0.95
)
```

### 快速匹配场景

```python
# 适用于需要快速响应的场景
result = tool.find_image(
    "button.png",
    algorithm="pyramid",
    pyramid_levels=3,
    pyramid_scale_factor=0.8,
    confidence=0.7
)
```

### 鲁棒性场景

```python
# 适用于需要处理旋转、缩放等变换的场景
result = tool.find_image(
    "button.png",
    algorithm="hybrid",
    template_weight=0.4,
    feature_weight=0.6,
    confidence=0.8,
    min_matches=10
)
```

### 多尺度场景

```python
# 适用于需要多尺度匹配的场景
result = tool.find_image(
    "button.png",
    algorithm="pyramid",
    pyramid_levels=8,
    pyramid_scale_factor=0.6,
    confidence=0.8,
    method="TM_CCORR_NORMED"
)
```

## 性能优化建议

### 1. 根据图像特征选择算法

- **小图像 (< 50x50)**: 使用模板匹配
- **中等图像 (50x50 - 200x200)**: 使用模板匹配或混合匹配
- **大图像 (> 200x200)**: 可以使用所有算法
- **需要旋转/缩放鲁棒性**: 使用特征匹配或混合匹配
- **需要多尺度匹配**: 使用金字塔匹配

### 2. 参数调优策略

1. **从默认参数开始**
2. **根据失败情况调整置信度**
3. **根据图像特征调整算法特定参数**
4. **测试不同算法组合**

### 3. 调试技巧

```python
# 启用调试日志
import logging
from gui_auto.core.logger import setup_logging
setup_logging(level=logging.DEBUG)

# 测试不同参数组合
test_configs = [
    {"algorithm": "template", "confidence": 0.8},
    {"algorithm": "feature", "min_matches": 10, "confidence": 0.7},
    {"algorithm": "pyramid", "pyramid_levels": 4, "confidence": 0.8},
    {"algorithm": "hybrid", "template_weight": 0.6, "feature_weight": 0.4}
]

for config in test_configs:
    result = tool.find_image("button.png", **config)
    print(f"配置: {config}, 结果: {result.found}, 置信度: {result.confidence:.3f}")
```

## 常见问题

### Q: 参数设置后没有生效怎么办？

A: 确保参数名称正确，并且算法支持该参数。可以查看日志确认参数是否被正确传递。

### Q: 如何选择合适的置信度？

A: 建议从0.8开始，如果匹配失败则降低到0.7或0.6，如果误匹配则提高到0.9或0.95。

### Q: 金字塔匹配为什么只匹配了几层？

A: 当模板图像太小时，金字塔会自动停止创建更多层。这是正常现象，不会影响匹配效果。

### Q: 混合匹配的权重如何选择？

A: 如果模板匹配效果好，增加`template_weight`；如果需要处理变换，增加`feature_weight`。权重总和应该接近1.0。
