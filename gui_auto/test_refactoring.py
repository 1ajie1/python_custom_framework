#!/usr/bin/env python3
"""
测试重构后的GUI自动化工具
验证图片对比和图片寻找功能使用统一核心函数后是否正常工作
"""

import sys
import os
import numpy as np
import cv2
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

def test_unified_core_function():
    """测试统一的核心函数"""
    print("=" * 50)
    print("测试统一的核心匹配函数")
    print("=" * 50)
    
    tool = GuiAutoTool(confidence=0.8)
    template, target = create_test_images()
    
    # 测试统一核心函数
    result = tool._unified_image_matching_core(
        template=template,
        target_image=target,
        method="TM_CCOEFF_NORMED",
        multi_scale=True,
        enhance_images=False
    )
    
    print(f"统一核心函数结果:")
    print(f"  置信度: {result['confidence']:.3f}")
    print(f"  找到: {result['found']}")
    print(f"  位置: {result['location']}")
    print(f"  中心: {result['center']}")
    print(f"  方法: {result['method']}")
    print(f"  缩放: {result['scale']}")
    
    return result['found']

def test_compare_images():
    """测试图片对比功能"""
    print("\n" + "=" * 50)
    print("测试图片对比功能")
    print("=" * 50)
    
    tool = GuiAutoTool(confidence=0.8)
    template, target = create_test_images()
    
    # 测试compare_images函数
    result = tool.compare_images(
        template_path=template,
        target_path=target,
        method="TM_CCOEFF_NORMED",
        multi_scale=True,
        enhance_images=False
    )
    
    print(f"图片对比结果:")
    print(f"  置信度: {result['confidence']:.3f}")
    print(f"  找到: {result['found']}")
    print(f"  基准位置: {result['base_location']}")
    print(f"  基准中心: {result['base_center']}")
    print(f"  方法: {result['method']}")
    print(f"  缩放: {result['scale']}")
    
    return result['found']

def test_find_image():
    """测试图片寻找功能"""
    print("\n" + "=" * 50)
    print("测试图片寻找功能")
    print("=" * 50)
    
    tool = GuiAutoTool(confidence=0.8)
    template, target = create_test_images()
    
    # 测试find_image_in_target函数
    result = tool.find_image_in_target(
        template=template,
        target_image=target,
        method="TM_CCOEFF_NORMED",
        multi_scale=True,
        enhance_images=False
    )
    
    print(f"图片寻找结果:")
    if result:
        print(f"  找到图片，位置: {result}")
        # 计算中心点
        center = tool.get_image_center(result)
        print(f"  中心点: {center}")
        return True
    else:
        print(f"  未找到图片")
        return False

def test_multiple_methods():
    """测试多方法匹配"""
    print("\n" + "=" * 50)
    print("测试多方法匹配功能")
    print("=" * 50)
    
    tool = GuiAutoTool(confidence=0.8)
    template, target = create_test_images()
    
    # 测试compare_multiple_methods函数
    result = tool.compare_multiple_methods(
        template_path=template,
        target_path=target,
        methods=["TM_CCOEFF_NORMED", "TM_CCORR_NORMED"],
        save_results=False
    )
    
    print(f"多方法对比结果:")
    print(f"  最佳方法: {result['summary']['best_method']}")
    print(f"  最佳置信度: {result['summary']['best_confidence']:.3f}")
    print(f"  尝试方法数: {result['summary']['methods_tried']}")
    
    if result['best_result']:
        print(f"  最佳结果找到: {result['best_result']['found']}")
        return result['best_result']['found']
    
    return False

def main():
    """主测试函数"""
    print("开始测试重构后的GUI自动化工具")
    
    try:
        # 测试统一核心函数
        test1_passed = test_unified_core_function()
        
        # 测试图片对比功能
        test2_passed = test_compare_images()
        
        # 测试图片寻找功能
        test3_passed = test_find_image()
        
        # 测试多方法匹配
        test4_passed = test_multiple_methods()
        
        # 总结测试结果
        print("\n" + "=" * 50)
        print("测试结果总结")
        print("=" * 50)
        print(f"统一核心函数测试: {'通过' if test1_passed else '失败'}")
        print(f"图片对比功能测试: {'通过' if test2_passed else '失败'}")
        print(f"图片寻找功能测试: {'通过' if test3_passed else '失败'}")
        print(f"多方法匹配测试: {'通过' if test4_passed else '失败'}")
        
        all_passed = all([test1_passed, test2_passed, test3_passed, test4_passed])
        print(f"\n整体测试结果: {'全部通过' if all_passed else '部分失败'}")
        
        if all_passed:
            print("✅ 重构成功！所有功能均正常工作，已成功使用统一的核心函数。")
        else:
            print("❌ 重构可能存在问题，请检查失败的测试项。")
            
        return all_passed
        
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
