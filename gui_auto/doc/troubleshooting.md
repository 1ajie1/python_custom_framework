# 故障排除

## 常见问题

### 1. 模块导入错误

#### 问题：`ModuleNotFoundError: No module named 'cv2'`

**原因：** 缺少OpenCV依赖

**解决方案：**
```bash
# 安装OpenCV
pip install opencv-python

# 或使用uv
uv add opencv-python
```

#### 问题：`ModuleNotFoundError: No module named 'pyautogui'`

**原因：** 缺少PyAutoGUI依赖

**解决方案：**
```bash
# 安装PyAutoGUI
pip install pyautogui

# 或使用uv
uv add pyautogui
```

#### 问题：`ModuleNotFoundError: No module named 'yaml'`

**原因：** 缺少PyYAML依赖

**解决方案：**
```bash
# 安装PyYAML
pip install pyyaml

# 或使用uv
uv add pyyaml
```

### 2. 图像匹配问题

#### 问题：图像匹配失败

**可能原因：**
- 图像质量差
- 置信度阈值过高
- 图像格式不匹配
- 缩放问题

**解决方案：**

1. **调整置信度阈值**
```python
# 降低置信度
result = tool.find_image("button.png", confidence=0.7)

# 或使用配置
config = GuiConfig(match_confidence=0.7)
tool = create_tool(config)
```

2. **尝试不同算法**
```python
# 使用特征匹配
result = tool.find_image("button.png", algorithm="feature")

# 使用混合匹配
result = tool.find_image("button.png", algorithm="hybrid")
```

3. **检查图像格式**
```python
from gui_auto.utils import ImageUtils

# 检查图像信息
image = ImageUtils.load_image("button.png")
info = ImageUtils.get_image_info(image)
print(f"图像信息: {info}")
```

4. **启用自动缩放**
```python
config = GuiConfig(auto_scale=True)
tool = create_tool(config)
```

#### 问题：匹配速度慢

**解决方案：**

1. **使用更快的算法**
```python
# 模板匹配通常最快
result = tool.find_image("button.png", algorithm="template")
```

2. **限制搜索区域**
```python
# 在指定区域搜索
result = tool.find_image("button.png", region=(100, 100, 400, 300))
```

3. **预加载图像**
```python
# 预加载模板
template = tool.load_image("button.png")
result = tool.find_image(template)
```

### 3. 坐标问题

#### 问题：点击位置不准确

**可能原因：**
- DPI缩放问题
- 坐标系统不匹配
- 屏幕分辨率变化

**解决方案：**

1. **检查DPI缩放**
```python
# 获取DPI缩放
scale = tool.get_dpi_scale()
print(f"DPI缩放: {scale}")

# 手动调整坐标
adjusted_x = int(x * scale)
adjusted_y = int(y * scale)
tool.click(adjusted_x, adjusted_y)
```

2. **使用相对坐标**
```python
# 获取屏幕尺寸
width, height = tool.get_screen_size()

# 使用相对坐标
relative_x = int(x * width / 1920)  # 假设基准分辨率1920x1080
relative_y = int(y * height / 1080)
tool.click(relative_x, relative_y)
```

3. **使用图像匹配代替坐标点击**
```python
# 查找图像并点击
result = tool.find_and_click("button.png")
```

#### 问题：坐标超出屏幕范围

**解决方案：**
```python
# 检查坐标是否在屏幕范围内
width, height = tool.get_screen_size()

if 0 <= x < width and 0 <= y < height:
    tool.click(x, y)
else:
    print(f"坐标 ({x}, {y}) 超出屏幕范围 ({width}, {height})")
```

### 4. 键盘操作问题

#### 问题：文本输入失败

**可能原因：**
- 焦点不在输入框
- 输入速度过快
- 特殊字符问题

**解决方案：**

1. **确保焦点在输入框**
```python
# 先点击输入框
tool.click(input_x, input_y)
time.sleep(0.5)  # 等待焦点切换
tool.type_text("Hello World")
```

2. **调整输入速度**
```python
# 慢速输入
tool.type_text("Hello World", interval=0.1)

# 或使用带延迟的输入
tool.type_with_delay("Hello World", delay=0.2)
```

3. **处理特殊字符**
```python
# 使用按键输入特殊字符
tool.press_key("enter")
tool.press_key("tab")
tool.press_key("space")
```

#### 问题：组合键不工作

**解决方案：**

1. **检查按键名称**
```python
# 正确的组合键
tool.hotkey("ctrl", "c")  # Ctrl+C
tool.hotkey("alt", "tab")  # Alt+Tab
tool.hotkey("ctrl", "shift", "z")  # Ctrl+Shift+Z
```

2. **添加延迟**
```python
# 在组合键之间添加延迟
tool.hotkey("ctrl", "c")
time.sleep(0.1)
tool.hotkey("ctrl", "v")
```

### 5. 窗口操作问题

#### 问题：找不到窗口

**可能原因：**
- 窗口标题不准确
- 窗口未完全加载
- 权限问题

**解决方案：**

1. **使用部分标题匹配**
```python
# 使用部分标题
window_id = tool.find_window("记事本")
# 或
window_id = tool.find_window("Notepad")
```

2. **等待窗口加载**
```python
import time

# 等待窗口出现
for i in range(10):
    window_id = tool.find_window("记事本")
    if window_id:
        break
    time.sleep(1.0)
```

3. **检查窗口信息**
```python
window_id = tool.find_window("记事本")
if window_id:
    info = tool.get_window_info(window_id)
    print(f"窗口信息: {info}")
```

#### 问题：窗口操作失败

**解决方案：**

1. **确保窗口存在**
```python
window_id = tool.find_window("记事本")
if not window_id:
    print("窗口不存在")
    return

# 执行窗口操作
tool.activate_window(window_id)
```

2. **添加延迟**
```python
# 激活窗口后等待
tool.activate_window(window_id)
time.sleep(0.5)
tool.minimize_window(window_id)
```

### 6. 平台兼容性问题

#### 问题：Windows特定功能不工作

**解决方案：**

1. **检查win32模块**
```python
try:
    import win32gui
    import win32con
    print("win32模块可用")
except ImportError:
    print("win32模块不可用，某些功能可能受限")
```

2. **安装win32模块**
```bash
pip install pywin32
```

#### 问题：Linux平台问题

**解决方案：**

1. **检查X11环境**
```bash
# 检查DISPLAY环境变量
echo $DISPLAY

# 设置DISPLAY
export DISPLAY=:0
```

2. **安装必要的依赖**
```bash
# Ubuntu/Debian
sudo apt-get install python3-tk python3-dev

# CentOS/RHEL
sudo yum install tkinter python3-devel
```

### 7. 性能问题

#### 问题：操作速度慢

**解决方案：**

1. **优化图像匹配**
```python
# 使用更快的算法
result = tool.find_image("button.png", algorithm="template")

# 限制搜索区域
result = tool.find_image("button.png", region=(100, 100, 400, 300))
```

2. **减少不必要的操作**
```python
# 避免重复截图
screenshot = tool.capture_screen()
# 多次使用同一个截图
```

3. **使用批量操作**
```python
# 批量执行操作
operations = [
    ("click", 100, 200),
    ("type_text", "Hello"),
    ("click", 200, 300)
]

for operation, *args in operations:
    if operation == "click":
        tool.click(*args)
    elif operation == "type_text":
        tool.type_text(*args)
```

#### 问题：内存使用过高

**解决方案：**

1. **及时释放图像**
```python
import gc

# 使用完图像后释放
screenshot = tool.capture_screen()
# 处理图像...
del screenshot
gc.collect()
```

2. **使用较小的图像**
```python
# 裁剪图像
cropped = tool.capture_screen(region=(100, 100, 400, 300))
```

### 8. 错误处理

#### 问题：异常处理不当

**解决方案：**

1. **使用具体的异常类型**
```python
from gui_auto.core.exceptions import OperationError, AlgorithmError, PlatformError

try:
    result = tool.find_and_click("button.png")
except OperationError as e:
    print(f"操作错误: {e}")
except AlgorithmError as e:
    print(f"算法错误: {e}")
except PlatformError as e:
    print(f"平台错误: {e}")
except Exception as e:
    print(f"未知错误: {e}")
```

2. **添加重试机制**
```python
import time

def retry_operation(operation, max_retries=3):
    for attempt in range(max_retries):
        try:
            return operation()
        except Exception as e:
            print(f"尝试 {attempt + 1} 失败: {e}")
            if attempt < max_retries - 1:
                time.sleep(1.0)
    return None

# 使用重试
result = retry_operation(lambda: tool.find_and_click("button.png"))
```

## 调试技巧

### 1. 启用详细日志

```python
import logging
from gui_auto import get_logger

# 设置详细日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = get_logger("debug")
logger.debug("调试信息")
```

### 2. 保存调试图像

```python
# 保存屏幕截图
screenshot = tool.capture_screen()
tool.save_image(screenshot, "debug_screenshot.png")

# 保存匹配结果
result = tool.find_image("button.png")
if result.success:
    # 在图像上标记匹配位置
    import cv2
    cv2.rectangle(screenshot, result.position, 
                  (result.position[0] + result.size[0], 
                   result.position[1] + result.size[1]), 
                  (0, 255, 0), 2)
    cv2.imwrite("debug_match.png", screenshot)
```

### 3. 性能监控

```python
import time

def monitor_performance():
    start_time = time.time()
    
    # 执行操作
    result = tool.find_image("button.png")
    
    end_time = time.time()
    print(f"操作耗时: {end_time - start_time:.3f}秒")
    
    if result.success:
        print(f"匹配耗时: {result.processing_time:.3f}秒")
```

### 4. 配置验证

```python
from gui_auto import GuiConfig

# 验证配置
config = GuiConfig()
if config.validate():
    print("配置有效")
else:
    print("配置无效")
    print(config.get_validation_errors())
```

## 获取帮助

### 1. 查看文档

- [快速开始](quick_start.md)
- [主类API](main_class.md)
- [使用示例](examples.md)
- [配置管理](core_modules.md)

### 2. 检查日志

```python
# 启用日志
import logging
logging.basicConfig(level=logging.INFO)

# 查看详细错误信息
try:
    result = tool.find_and_click("button.png")
except Exception as e:
    logging.error(f"详细错误: {e}", exc_info=True)
```

### 3. 测试基本功能

```python
def test_basic_functionality():
    """测试基本功能"""
    tool = create_tool()
    
    # 测试屏幕截图
    try:
        screenshot = tool.capture_screen()
        print(f"✓ 屏幕截图成功: {screenshot.shape}")
    except Exception as e:
        print(f"✗ 屏幕截图失败: {e}")
        return False
    
    # 测试鼠标点击
    try:
        tool.click(100, 200)
        print("✓ 鼠标点击成功")
    except Exception as e:
        print(f"✗ 鼠标点击失败: {e}")
        return False
    
    # 测试键盘输入
    try:
        tool.type_text("Test")
        print("✓ 键盘输入成功")
    except Exception as e:
        print(f"✗ 键盘输入失败: {e}")
        return False
    
    print("所有基本功能测试通过")
    return True

# 运行测试
test_basic_functionality()
```

### 4. 联系支持

如果问题仍然存在，请：

1. 查看错误日志
2. 提供复现步骤
3. 包含系统信息
4. 提供相关代码

```python
# 获取系统信息
import platform
import sys

print(f"操作系统: {platform.system()}")
print(f"Python版本: {sys.version}")
print(f"框架版本: {get_version()}")
```
