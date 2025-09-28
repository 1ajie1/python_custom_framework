# 快速开始

## 安装

### 使用uv安装（推荐）

```bash
# 在项目目录中
uv add gui-auto-framework
```

### 使用pip安装

```bash
pip install gui-auto-framework
```

### 从源码安装

```bash
git clone https://github.com/your-repo/gui-auto-framework.git
cd gui-auto-framework
pip install -e .
```

## 基本使用

### 1. 导入框架

```python
from gui_auto import GuiAutoTool, create_tool, get_version

# 检查版本
print(f"框架版本: {get_version()}")
```

### 2. 创建工具实例

```python
# 使用默认配置
tool = create_tool()

# 或者直接创建
tool = GuiAutoTool()
```

### 3. 基本操作

```python
# 截取屏幕
screenshot = tool.capture_screen()
print(f"屏幕尺寸: {screenshot.shape}")

# 查找并点击图像
success = tool.find_and_click("button.png")
if success:
    print("成功点击按钮")

# 输入文本
tool.type_text("Hello World")

# 点击坐标
tool.click(100, 200)
```

## 快速示例

### 自动化记事本操作

```python
from gui_auto import create_tool
import time

def automate_notepad():
    tool = create_tool()
    
    # 查找记事本窗口
    window_id = tool.find_window("记事本")
    if window_id:
        # 激活窗口
        tool.activate_window(window_id)
        time.sleep(0.5)
        
        # 输入文本
        tool.type_text("Hello, World!")
        
        # 保存文件
        tool.hotkey("ctrl", "s")
        time.sleep(1.0)
        tool.type_text("test.txt")
        tool.press_key("enter")
        
        print("自动化操作完成")
    else:
        print("未找到记事本窗口")

# 运行示例
automate_notepad()
```

### 图像识别自动化

```python
from gui_auto import create_tool

def image_automation():
    tool = create_tool()
    
    # 查找多个按钮
    buttons = ["button1.png", "button2.png", "button3.png"]
    
    for button in buttons:
        result = tool.find_image(button)
        if result.found:
            print(f"找到按钮 {button}，位置: {result.center}")
            tool.click(result.center[0], result.center[1])
        else:
            print(f"未找到按钮 {button}")

# 运行示例
image_automation()
```

## 配置选项

### 基本配置

```python
from gui_auto import GuiConfig, create_tool

# 创建自定义配置
config = GuiConfig(
    match_confidence=0.9,      # 匹配置信度
    match_timeout=15.0,        # 匹配超时时间
    auto_scale=True,           # 自动缩放
    default_max_retries=5      # 最大重试次数
)

# 使用配置创建工具
tool = create_tool(config)
```

### 高级配置

```python
from gui_auto import GuiConfig, MatchConfig, RetryConfig

# 创建详细配置
config = GuiConfig(
    match_confidence=0.9,
    match_timeout=15.0,
    auto_scale=True,
    match_config=MatchConfig(
        confidence=0.9,
        timeout=15.0,
        method="TM_CCOEFF_NORMED",
        scale_range=(0.8, 1.2)
    ),
    retry_config=RetryConfig(
        max_retries=5,
        retry_delay=1.0,
        backoff_factor=2.0
    )
)

tool = create_tool(config)
```

## 常用操作

### 鼠标操作

```python
# 基本点击
tool.click(100, 200)                    # 左键点击
tool.right_click(100, 200)             # 右键点击
tool.double_click(100, 200)            # 双击

# 拖拽
tool.drag(100, 100, 200, 200)          # 拖拽

# 滚轮
tool.scroll(100, 200, clicks=3)        # 向上滚动
tool.scroll(100, 200, clicks=-3)       # 向下滚动
```

### 键盘操作

```python
# 文本输入
tool.type_text("Hello World")          # 输入文本
tool.type_with_delay("Hello", 0.1)     # 带延迟输入

# 按键
tool.press_key("enter")                # 按下回车
tool.press_key("space")                # 按下空格

# 组合键
tool.hotkey("ctrl", "c")               # Ctrl+C
tool.hotkey("ctrl", "v")               # Ctrl+V
tool.hotkey("alt", "tab")              # Alt+Tab
```

### 窗口操作

```python
# 窗口查找和管理
window_id = tool.find_window("记事本")
if window_id:
    tool.activate_window(window_id)     # 激活窗口
    tool.minimize_window(window_id)     # 最小化
    tool.maximize_window(window_id)     # 最大化
    tool.close_window(window_id)        # 关闭窗口
```

### 图像操作

```python
# 图像查找
result = tool.find_image("button.png")
if result.found:
    print(f"找到图像，位置: {result.center}")
    print(f"置信度: {result.confidence}")

# 图像比较
similarity = tool.compare_images("img1.png", "img2.png")
print(f"相似度: {similarity}")

# 屏幕截图
screenshot = tool.capture_screen()
screenshot = tool.capture_screen(region=(100, 100, 800, 600))  # 指定区域
```

## 错误处理

### 基本错误处理

```python
from gui_auto import create_tool
from gui_auto.core.exceptions import OperationError, AlgorithmError

def safe_operation():
    tool = create_tool()
    
    try:
        result = tool.find_and_click("button.png")
        if result:
            print("操作成功")
        else:
            print("操作失败")
    except OperationError as e:
        print(f"操作错误: {e}")
    except AlgorithmError as e:
        print(f"算法错误: {e}")
    except Exception as e:
        print(f"未知错误: {e}")
```

### 重试机制

```python
import time

def retry_operation(max_retries=3):
    tool = create_tool()
    
    for attempt in range(max_retries):
        try:
            result = tool.find_and_click("button.png")
            if result:
                print("操作成功")
                return True
        except Exception as e:
            print(f"尝试 {attempt + 1} 失败: {e}")
            if attempt < max_retries - 1:
                time.sleep(1.0)  # 等待1秒后重试
    
    print("所有重试都失败了")
    return False
```

## 性能优化

### 预加载图像

```python
# 预加载模板图像
template = tool.load_image("button.png")

# 使用预加载的图像
result = tool.find_image(template)
```

### 批量操作

```python
# 批量执行操作
operations = [
    ("click", 100, 200),
    ("type_text", "Hello"),
    ("click", 200, 300),
    ("type_text", "World")
]

for operation, *args in operations:
    if operation == "click":
        tool.click(*args)
    elif operation == "type_text":
        tool.type_text(*args)
```

## 调试技巧

### 启用日志

```python
import logging
from gui_auto import get_logger

# 设置日志级别
logging.basicConfig(level=logging.INFO)

# 获取日志器
logger = get_logger("my_app")

# 使用日志
logger.info("开始操作")
logger.debug("调试信息")
logger.error("错误信息")
```

### 保存调试图像

```python
# 保存屏幕截图用于调试
screenshot = tool.capture_screen()
tool.save_image(screenshot, "debug_screenshot.png")

# 保存匹配结果
result = tool.find_image("button.png")
if result.found:
    # 在匹配位置画框
    import cv2
    x, y, w, h = result.bbox
    cv2.rectangle(screenshot, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imwrite("debug_match.png", screenshot)
```

## 常见问题

### Q: 图像匹配失败怎么办？

A: 尝试以下方法：
1. 调整置信度阈值
2. 使用不同的匹配算法
3. 检查图像质量和格式
4. 启用自动缩放

```python
# 调整置信度
result = tool.find_image("button.png", confidence=0.7)

# 使用不同算法
result = tool.find_image("button.png", algorithm="feature")

# 启用自动缩放
config = GuiConfig(auto_scale=True)
tool = create_tool(config)
```

### Q: 坐标不准确怎么办？

A: 检查DPI缩放设置：

```python
# 获取DPI缩放
scale = tool.get_dpi_scale()
print(f"DPI缩放: {scale}")

# 手动调整坐标
adjusted_x = int(x * scale)
adjusted_y = int(y * scale)
tool.click(adjusted_x, adjusted_y)
```

### Q: 如何提高匹配速度？

A: 使用以下优化方法：

```python
# 1. 使用更快的算法
result = tool.find_image("button.png", algorithm="template")

# 2. 限制搜索区域
result = tool.find_image("button.png", region=(100, 100, 400, 300))

# 3. 预加载图像
template = tool.load_image("button.png")
result = tool.find_image(template)
```

## 下一步

- 查看[主类API文档](main_class.md)了解完整的API
- 查看[使用示例](examples.md)了解更多用法
- 查看[参数配置指南](parameter_configuration.md)了解详细的参数设置
- 查看[配置管理](core_modules.md)了解高级配置
- 查看[算法模块](algorithms.md)了解图像匹配算法
- 查看[操作模块](operations.md)了解详细操作
- 查看[平台模块](platform.md)了解跨平台支持
- 查看[工具模块](utils.md)了解实用工具
