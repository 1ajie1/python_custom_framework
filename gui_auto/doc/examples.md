# 使用示例

## 概述

本文档提供了GUI自动化框架的完整使用示例，涵盖各种常见的使用场景。

## 快速开始

### 基本安装和导入

```python
# 安装框架
# pip install gui-auto-framework

# 导入主类
from gui_auto import GuiAutoTool, create_tool, get_version

# 检查版本
print(f"框架版本: {get_version()}")

# 创建工具实例
tool = create_tool()
```

### 简单示例

```python
from gui_auto import create_tool

# 创建工具
tool = create_tool()

# 截取屏幕
screenshot = tool.capture_screen()
print(f"屏幕截图尺寸: {screenshot.shape}")

# 查找并点击按钮
success = tool.find_and_click("button.png")
if success:
    print("成功点击按钮")
else:
    print("未找到按钮")

# 输入文本
tool.type_text("Hello World")

# 点击坐标
tool.click(100, 200)
```

## 图像识别示例

### 基本图像查找

```python
from gui_auto import create_tool

tool = create_tool()

# 查找图像
result = tool.find_image("button.png")
if result.success:
    print(f"找到图像，位置: {result.position}")
    print(f"置信度: {result.confidence}")
    print(f"大小: {result.size}")
else:
    print("未找到图像")
```

### 使用不同算法

```python
from gui_auto import create_tool

tool = create_tool()

# 模板匹配
result1 = tool.find_image("button.png", algorithm="template")
print(f"模板匹配: {result1.success}")

# 特征匹配
result2 = tool.find_image("button.png", algorithm="feature")
print(f"特征匹配: {result2.success}")

# 混合匹配
result3 = tool.find_image("button.png", algorithm="hybrid")
print(f"混合匹配: {result3.success}")

# 金字塔匹配
result4 = tool.find_image("button.png", algorithm="pyramid")
print(f"金字塔匹配: {result4.success}")
```

### 多尺度匹配

```python
from gui_auto import create_tool

tool = create_tool()

# 在指定图像中查找
result = tool.find_image(
    "button.png", 
    "screenshot.png",
    algorithm="template",
    scale_range=(0.8, 1.2)
)

if result.success:
    print(f"找到图像，缩放比例: {result.scale}")
```

### 图像比较

```python
from gui_auto import create_tool

tool = create_tool()

# 比较两个图像
similarity = tool.compare_images("image1.png", "image2.png", method="ssim")
print(f"图像相似度: {similarity}")

# 使用MSE方法比较
mse_score = tool.compare_images("image1.png", "image2.png", method="mse")
print(f"MSE分数: {mse_score}")
```

## 鼠标操作示例

### 基本点击操作

```python
from gui_auto import create_tool

tool = create_tool()

# 左键单击
tool.click(100, 200)

# 右键单击
tool.right_click(100, 200)

# 双击
tool.double_click(100, 200)

# 多次点击
tool.click(100, 200, clicks=3, interval=0.1)
```

### 拖拽操作

```python
from gui_auto import create_tool

tool = create_tool()

# 从(100, 100)拖拽到(200, 200)
tool.drag(100, 100, 200, 200)

# 慢速拖拽
tool.drag(100, 100, 200, 200, duration=2.0)
```

### 滚轮操作

```python
from gui_auto import create_tool

tool = create_tool()

# 向上滚动
tool.scroll(100, 200, clicks=3)

# 向下滚动
tool.scroll(100, 200, clicks=-3)
```

## 键盘操作示例

### 文本输入

```python
from gui_auto import create_tool

tool = create_tool()

# 基本文本输入
tool.type_text("Hello World")

# 慢速输入
tool.type_text("Hello World", interval=0.1)

# 带延迟输入
tool.type_with_delay("Hello World", delay=0.2)
```

### 按键操作

```python
from gui_auto import create_tool

tool = create_tool()

# 按下单个按键
tool.press_key("enter")
tool.press_key("space")
tool.press_key("tab")

# 组合键
tool.hotkey("ctrl", "c")  # Ctrl+C
tool.hotkey("ctrl", "v")  # Ctrl+V
tool.hotkey("alt", "tab")  # Alt+Tab

# 复杂组合键
tool.hotkey("ctrl", "shift", "z")  # Ctrl+Shift+Z
```

### 文本编辑

```python
from gui_auto import create_tool

tool = create_tool()

# 清空文本
tool.clear_text()

# 选择全部文本并删除
tool.hotkey("ctrl", "a")
tool.press_key("delete")
```

## 窗口操作示例

### 窗口查找和激活

```python
from gui_auto import create_tool

tool = create_tool()

# 查找窗口
window_id = tool.find_window("记事本")
if window_id:
    print(f"找到窗口: {window_id}")
    
    # 激活窗口
    tool.activate_window(window_id)
    
    # 获取窗口信息
    info = tool.get_window_info(window_id)
    print(f"窗口信息: {info}")
else:
    print("未找到窗口")
```

### 窗口管理

```python
from gui_auto import create_tool

tool = create_tool()

window_id = tool.find_window("记事本")
if window_id:
    # 最小化窗口
    tool.minimize_window(window_id)
    
    # 等待一秒
    import time
    time.sleep(1)
    
    # 最大化窗口
    tool.maximize_window(window_id)
    
    # 移动窗口
    tool.move_window(window_id, 100, 100)
    
    # 调整窗口大小
    tool.resize_window(window_id, 800, 600)
```

## 配置管理示例

### 基本配置

```python
from gui_auto import GuiConfig, create_tool

# 创建自定义配置
config = GuiConfig(
    match_confidence=0.9,
    match_timeout=15.0,
    auto_scale=True,
    default_max_retries=5
)

# 使用配置创建工具
tool = create_tool(config)

# 获取当前配置
current_config = tool.get_config()
print(f"当前配置: {current_config}")
```

### 配置保存和加载

```python
from gui_auto import GuiConfig

# 创建配置
config = GuiConfig(
    match_confidence=0.9,
    match_timeout=15.0
)

# 保存配置
config.save_to_file("my_config.json")

# 从文件加载配置
loaded_config = GuiConfig.load_from_file("my_config.json")

# 验证配置
if loaded_config.validate():
    print("配置有效")
else:
    print("配置无效")
```

### 动态配置更新

```python
from gui_auto import create_tool

tool = create_tool()

# 更新配置
tool.update_config(
    match_confidence=0.95,
    match_timeout=20.0
)

# 获取更新后的配置
config = tool.get_config()
print(f"更新后的配置: {config}")
```

## 高级使用示例

### 自动化脚本示例

```python
from gui_auto import create_tool
import time

def automate_notepad():
    """自动化记事本操作"""
    tool = create_tool()
    
    # 查找并激活记事本
    window_id = tool.find_window("记事本")
    if not window_id:
        print("未找到记事本窗口")
        return False
    
    tool.activate_window(window_id)
    time.sleep(0.5)
    
    # 输入文本
    tool.type_text("Hello, World!")
    time.sleep(0.5)
    
    # 选择全部文本
    tool.hotkey("ctrl", "a")
    time.sleep(0.5)
    
    # 复制文本
    tool.hotkey("ctrl", "c")
    time.sleep(0.5)
    
    # 换行
    tool.press_key("enter")
    time.sleep(0.5)
    
    # 粘贴文本
    tool.hotkey("ctrl", "v")
    time.sleep(0.5)
    
    # 保存文件
    tool.hotkey("ctrl", "s")
    time.sleep(1.0)
    
    # 输入文件名
    tool.type_text("test.txt")
    time.sleep(0.5)
    
    # 确认保存
    tool.press_key("enter")
    
    print("自动化操作完成")
    return True

# 运行自动化脚本
if __name__ == "__main__":
    automate_notepad()
```

### 图像识别自动化

```python
from gui_auto import create_tool
import time

def automate_image_recognition():
    """图像识别自动化"""
    tool = create_tool()
    
    # 截取屏幕
    screenshot = tool.capture_screen()
    print(f"屏幕截图尺寸: {screenshot.shape}")
    
    # 查找多个按钮
    buttons = ["button1.png", "button2.png", "button3.png"]
    
    for button in buttons:
        result = tool.find_image(button)
        if result.success:
            print(f"找到按钮 {button}，位置: {result.position}")
            # 点击按钮
            tool.click(result.position[0], result.position[1])
            time.sleep(1.0)
        else:
            print(f"未找到按钮 {button}")
    
    return True

# 运行图像识别自动化
if __name__ == "__main__":
    automate_image_recognition()
```

### 错误处理和重试

```python
from gui_auto import create_tool
from gui_auto.core.exceptions import OperationError, AlgorithmError
import time

def robust_automation():
    """健壮的自动化操作"""
    tool = create_tool()
    
    max_retries = 3
    retry_delay = 1.0
    
    for attempt in range(max_retries):
        try:
            # 查找并点击按钮
            result = tool.find_and_click("button.png")
            if result:
                print("操作成功")
                return True
            else:
                print(f"尝试 {attempt + 1} 失败，未找到按钮")
                
        except OperationError as e:
            print(f"操作错误: {e}")
        except AlgorithmError as e:
            print(f"算法错误: {e}")
        except Exception as e:
            print(f"未知错误: {e}")
        
        if attempt < max_retries - 1:
            print(f"等待 {retry_delay} 秒后重试...")
            time.sleep(retry_delay)
            retry_delay *= 2  # 指数退避
    
    print("所有重试都失败了")
    return False

# 运行健壮的自动化
if __name__ == "__main__":
    robust_automation()
```

### 性能监控

```python
from gui_auto import create_tool
import time

def performance_monitoring():
    """性能监控示例"""
    tool = create_tool()
    
    # 监控屏幕截图性能
    start_time = time.time()
    screenshot = tool.capture_screen()
    capture_time = time.time() - start_time
    print(f"屏幕截图耗时: {capture_time:.3f}秒")
    
    # 监控图像查找性能
    start_time = time.time()
    result = tool.find_image("button.png")
    find_time = time.time() - start_time
    print(f"图像查找耗时: {find_time:.3f}秒")
    
    if result.success:
        print(f"找到图像，置信度: {result.confidence}")
        print(f"处理时间: {result.processing_time:.3f}秒")
    
    # 监控点击性能
    start_time = time.time()
    tool.click(100, 200)
    click_time = time.time() - start_time
    print(f"点击操作耗时: {click_time:.3f}秒")
    
    return True

# 运行性能监控
if __name__ == "__main__":
    performance_monitoring()
```

## 集成测试示例

### 基本功能测试

```python
from gui_auto import create_tool
import time

def test_basic_functionality():
    """测试基本功能"""
    tool = create_tool()
    
    print("开始基本功能测试...")
    
    # 测试屏幕截图
    try:
        screenshot = tool.capture_screen()
        print(f"✓ 屏幕截图成功，尺寸: {screenshot.shape}")
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
    
    # 测试窗口查找
    try:
        window_id = tool.find_window("记事本")
        if window_id:
            print(f"✓ 窗口查找成功: {window_id}")
        else:
            print("✓ 窗口查找成功（未找到记事本）")
    except Exception as e:
        print(f"✗ 窗口查找失败: {e}")
        return False
    
    print("所有基本功能测试通过")
    return True

# 运行基本功能测试
if __name__ == "__main__":
    test_basic_functionality()
```

### 压力测试

```python
from gui_auto import create_tool
import time
import random

def stress_test():
    """压力测试"""
    tool = create_tool()
    
    print("开始压力测试...")
    
    # 测试次数
    test_count = 100
    
    # 记录成功和失败次数
    success_count = 0
    failure_count = 0
    
    start_time = time.time()
    
    for i in range(test_count):
        try:
            # 随机点击
            x = random.randint(0, 800)
            y = random.randint(0, 600)
            tool.click(x, y)
            success_count += 1
            
            if i % 10 == 0:
                print(f"已完成 {i}/{test_count} 次测试")
                
        except Exception as e:
            failure_count += 1
            print(f"第 {i+1} 次测试失败: {e}")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"压力测试完成:")
    print(f"总测试次数: {test_count}")
    print(f"成功次数: {success_count}")
    print(f"失败次数: {failure_count}")
    print(f"成功率: {success_count/test_count*100:.1f}%")
    print(f"总耗时: {total_time:.3f}秒")
    print(f"平均耗时: {total_time/test_count:.3f}秒/次")
    
    return success_count == test_count

# 运行压力测试
if __name__ == "__main__":
    stress_test()
```

## 最佳实践

### 1. 错误处理

```python
from gui_auto import create_tool
from gui_auto.core.exceptions import OperationError, AlgorithmError, PlatformError

def safe_automation():
    """安全的自动化操作"""
    tool = create_tool()
    
    try:
        # 执行操作
        result = tool.find_and_click("button.png")
        if result:
            print("操作成功")
        else:
            print("操作失败")
            
    except OperationError as e:
        print(f"操作错误: {e}")
    except AlgorithmError as e:
        print(f"算法错误: {e}")
    except PlatformError as e:
        print(f"平台错误: {e}")
    except Exception as e:
        print(f"未知错误: {e}")
```

### 2. 配置管理

```python
from gui_auto import GuiConfig, create_tool

def configure_tool():
    """配置工具"""
    # 创建配置
    config = GuiConfig(
        match_confidence=0.9,
        match_timeout=15.0,
        auto_scale=True,
        default_max_retries=5
    )
    
    # 验证配置
    if not config.validate():
        print("配置无效")
        return None
    
    # 创建工具
    tool = create_tool(config)
    
    # 保存配置
    config.save_to_file("config.json")
    
    return tool
```

### 3. 性能优化

```python
from gui_auto import create_tool
import time

def optimized_automation():
    """优化的自动化操作"""
    tool = create_tool()
    
    # 预加载图像
    template = tool.load_image("button.png")
    
    # 批量操作
    operations = [
        ("click", 100, 200),
        ("type_text", "Hello"),
        ("click", 200, 300),
        ("type_text", "World")
    ]
    
    for operation, *args in operations:
        start_time = time.time()
        
        if operation == "click":
            tool.click(*args)
        elif operation == "type_text":
            tool.type_text(*args)
        
        execution_time = time.time() - start_time
        print(f"{operation} 耗时: {execution_time:.3f}秒")
```

### 4. 日志记录

```python
from gui_auto import create_tool, get_logger
import logging

def logged_automation():
    """带日志的自动化操作"""
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    logger = get_logger("automation")
    
    tool = create_tool()
    
    logger.info("开始自动化操作")
    
    try:
        # 执行操作
        result = tool.find_and_click("button.png")
        if result:
            logger.info("操作成功")
        else:
            logger.warning("操作失败")
            
    except Exception as e:
        logger.error(f"操作异常: {e}")
    
    logger.info("自动化操作完成")
```

这些示例涵盖了GUI自动化框架的主要使用场景，可以根据具体需求进行修改和扩展。
