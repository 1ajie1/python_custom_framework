#!/usr/bin/env python3
"""
测试智能缩放功能
演示在已知模板和目标图片缩放信息时，如何避免多尺度匹配
"""

import sys
import os
import numpy as np
import cv2
import time
from pathlib import Path

# 添加当前目录到系统路径
sys.path.insert(0, str(Path(__file__).parent))

from auto_gui_tool import GuiAutoTool

def create_test_images():
    """创建测试用的图片"""
    # 创建一个简单的模板图像 (蓝色矩形)
    template = np.zeros((50, 50, 3), dtype=np.uint8)
    template[:, :] = [255, 0, 0]  # 蓝色
    
    # 创建一个目标图像 (白色背景，中间有蓝色矩形)
    target = np.ones((200, 200, 3), dtype=np.uint8) * 255  # 白色背景
    target[75:125, 75:125] = [255, 0, 0]  # 在中间放置蓝色矩形
    
    return template, target

def test_without_scale_info():
    """测试不提供缩放信息的情况（使用默认缩放）"""
    print("=" * 60)
    print("测试不提供缩放信息的情况")
    print("=" * 60)
    
    tool = GuiAutoTool(confidence=0.8)
    template, target = create_test_images()
    
    start_time = time.time()
    result = tool.compare_images(
        template_path=template,
        target_path=target,
        method="TM_CCOEFF_NORMED",
        enhance_images=False,
        # 不提供缩放信息，将使用默认缩放1.0
    )
    end_time = time.time()
    
    print(f"默认缩放匹配结果:")
    print(f"  置信度: {result['confidence']:.3f}")
    print(f"  找到: {result['found']}")
    print(f"  位置: {result['location']}")
    print(f"  缩放: {result.get('scale', 1.0):.3f}")
    print(f"  耗时: {(end_time - start_time)*1000:.1f}ms")
    
    return result, end_time - start_time

def test_smart_scale_matching():
    """测试智能缩放匹配"""
    print("\n" + "=" * 60)
    print("测试智能缩放匹配")
    print("=" * 60)
    
    tool = GuiAutoTool(confidence=0.8)
    template, target = create_test_images()
    
    # 模拟已知的缩放信息
    template_scale_info = {
        'dpi_scale': 1.0,  # 100% DPI缩放
        'resolution': (1920, 1080)  # 1080p分辨率
    }
    
    target_scale_info = {
        'dpi_scale': 1.0,  # 100% DPI缩放  
        'resolution': (1920, 1080)  # 1080p分辨率
    }
    
    start_time = time.time()
    result = tool.compare_images(
        template_path=template,
        target_path=target,
        method="TM_CCOEFF_NORMED",
        enhance_images=False,
        template_scale_info=template_scale_info,
        target_scale_info=target_scale_info,
    )
    end_time = time.time()
    
    print(f"智能缩放匹配结果:")
    print(f"  置信度: {result['confidence']:.3f}")
    print(f"  找到: {result['found']}")
    print(f"  位置: {result['location']}")
    print(f"  缩放: {result.get('scale', 1.0):.3f}")
    print(f"  耗时: {(end_time - start_time)*1000:.1f}ms")
    
    return result, end_time - start_time

def test_different_scale_scenarios():
    """测试不同缩放场景"""
    print("\n" + "=" * 60)
    print("测试不同缩放场景")
    print("=" * 60)
    
    tool = GuiAutoTool(confidence=0.8)
    template, target = create_test_images()
    
    scenarios = [
        {
            "name": "同样缩放和分辨率",
            "template": {'dpi_scale': 1.0, 'resolution': (1920, 1080)},
            "target": {'dpi_scale': 1.0, 'resolution': (1920, 1080)},
            "expected_scale": 1.0
        },
        {
            "name": "目标DPI缩放125%",
            "template": {'dpi_scale': 1.0, 'resolution': (1920, 1080)},
            "target": {'dpi_scale': 1.25, 'resolution': (1920, 1080)},
            "expected_scale": 1.25
        },
        {
            "name": "目标分辨率4K",
            "template": {'dpi_scale': 1.0, 'resolution': (1920, 1080)},
            "target": {'dpi_scale': 1.0, 'resolution': (3840, 2160)},
            "expected_scale": 2.0
        },
        {
            "name": "组合：125% DPI + 4K",
            "template": {'dpi_scale': 1.0, 'resolution': (1920, 1080)},
            "target": {'dpi_scale': 1.25, 'resolution': (3840, 2160)},
            "expected_scale": 2.5
        }
    ]
    
    for scenario in scenarios:
        print(f"\n场景: {scenario['name']}")
        
        start_time = time.time()
        result = tool.compare_images(
            template_path=template,
            target_path=target,
            method="TM_CCOEFF_NORMED",
            template_scale_info=scenario["template"],
            target_scale_info=scenario["target"],
        )
        end_time = time.time()
        
        actual_scale = result.get('scale', 1.0)
        expected_scale = scenario['expected_scale']
        
        print(f"  预期缩放: {expected_scale:.2f}")
        print(f"  实际缩放: {actual_scale:.2f}")
        print(f"  缩放误差: {abs(actual_scale - expected_scale):.3f}")
        print(f"  置信度: {result['confidence']:.3f}")
        print(f"  耗时: {(end_time - start_time)*1000:.1f}ms")

def test_convenience_method():
    """测试便捷方法"""
    print("\n" + "=" * 60)
    print("测试便捷方法 - 自动获取当前系统缩放信息")
    print("=" * 60)
    
    tool = GuiAutoTool(confidence=0.8)
    template, target = create_test_images()
    
    # 获取当前系统缩放信息
    current_scale_info = tool.get_current_system_scale_info()
    print(f"当前系统缩放信息: {current_scale_info}")
    
    # 模拟模板来自1080p 100%缩放的环境
    template_scale_info = {
        'dpi_scale': 1.0,
        'resolution': (1920, 1080)
    }
    
    start_time = time.time()
    result = tool.compare_images(
        template_path=template,
        target_path=target,
        template_scale_info=template_scale_info,
        target_scale_info=current_scale_info,  # 使用当前系统信息
    )
    end_time = time.time()
    
    print(f"\n使用便捷方法的结果:")
    print(f"  置信度: {result['confidence']:.3f}")
    print(f"  找到: {result['found']}")
    print(f"  计算缩放: {result.get('scale', 1.0):.3f}")
    print(f"  耗时: {(end_time - start_time)*1000:.1f}ms")

def main():
    """主测试函数"""
    print("开始测试智能缩放功能")
    print("=" * 60)
    
    try:
        # 测试不提供缩放信息的情况
        default_result, default_time = test_without_scale_info()
        
        # 测试智能缩放匹配
        smart_result, smart_time = test_smart_scale_matching()
        
        # 测试不同缩放场景
        test_different_scale_scenarios()
        
        # 测试便捷方法
        test_convenience_method()
        
        # 性能对比总结
        print("\n" + "=" * 60)
        print("性能对比总结")
        print("=" * 60)
        print(f"默认缩放匹配耗时: {default_time*1000:.1f}ms")
        print(f"智能缩放匹配耗时: {smart_time*1000:.1f}ms")
        print(f"性能对比: 智能缩放相对于默认缩放的时间比例 {(smart_time/default_time*100):.1f}%")
        
        # 结论
        print("\n" + "=" * 60)
        print("结论")
        print("=" * 60)
        print("1. 智能缩放功能的优势：")
        print("   - 精确计算所需的缩放比例，避免盲目尝试")
        print("   - 单次匹配，性能优异")
        print("   - 在已知环境信息时提供更准确的匹配结果")
        print("2. 推荐使用方式：")
        print("   - 手动指定template_scale_info（模板图片来源环境）")
        print("   - 使用get_current_system_scale_info()获取当前系统信息作为target_scale_info")
        print("   - 或者让target_scale_info为None，系统会自动获取当前环境信息")
        print("3. 简化的API：")
        print("   - 移除了复杂的多尺度匹配参数")
        print("   - 保留了智能缩放的核心功能")
        print("   - 如果未提供缩放信息，使用默认缩放1.0")
        
        return True
        
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
