from auto_gui_tool import GuiAutoTool



def main():
    """示例用法"""
    # 创建GUI自动化工具实例
    gui_tool = GuiAutoTool(
        confidence=0.8,  # 降低阈值以便演示
        timeout=10.0,
        default_method="TM_CCOEFF_NORMED",
        use_advanced_matching=True,
        default_max_retries=3,  # 默认重试3次
        default_retry_delay=0.5,  # 默认重试延迟0.5秒
    )

    print("GUI自动化工具已初始化")
    print("=" * 50)
    print("主要功能:")
    print("- 两张图片对比匹配")
    print("- 多种匹配算法选择")
    print("- 多尺度匹配")
    print("- 图像预处理增强")
    print("- 匹配结果可视化")
    print("- 屏幕自动化操作")
    print("- 智能重试机制")
    print()

    # 演示图像比较功能
    template_path = "templates/example_file3.png"
    target_path = "templates/main_125.png"  # 这里可以是不同的图片

    print("=== 图像比较功能演示 ===")

    # 检查文件是否存在
    from pathlib import Path

    if Path(template_path).exists() and Path(target_path).exists():
        print(f"模板图片: {template_path}")
        print(f"目标图片: {target_path}")

        # 单个方法比较
        print("\n1. 单个方法比较:")
        try:
            result = gui_tool.compare_images(
                template_path=template_path,
                target_path=target_path,
                method="TM_CCOEFF_NORMED",
                multi_scale=True,
                enhance_images=True,
                save_result=True,
                result_path="single_method_result.png",
                max_retries=5,  # 演示自定义重试次数
                retry_delay=0.3,  # 演示自定义重试延迟
            )

            print(f"   置信度: {result['confidence']:.3f}")
            print(f"   坐标类型: {result['coordinate_type']}")
            print(
                f"   返回坐标: {result['location']} ({'基准' if result['coordinate_type'] == 'base' else '系统'})"
            )
            print(
                f"   返回中心点: {result['center']} ({'基准' if result['coordinate_type'] == 'base' else '系统'})"
            )
            print(f"   基准坐标: {result['base_location']}")
            print(f"   基准中心点: {result['base_center']}")
            if result["system_location"]:
                print(f"   系统坐标: {result['system_location']}")
                print(f"   系统中心点: {result['system_center']}")
            print(f"   是否找到: {result['found']}")
            print(f"   使用方法: {result['method']}")
            print(f"   模板缩放: {result['scale']:.1f}")
            print(f"   系统缩放因子: {result['system_scale_factor']:.2f}")
            print(
                f"   当前分辨率: {result['system_info']['resolution'][0]}x{result['system_info']['resolution'][1]}"
            )
            print(f"   DPI缩放: {result['system_info']['dpi_scale']:.2f}")
            if result["result_image_path"]:
                print(f"   结果图片: {result['result_image_path']}")

            # 演示返回系统坐标的情况
            print("\n   演示返回系统坐标:")
            result_sys = gui_tool.compare_images(
                template_path=template_path,
                target_path=target_path,
                method="TM_CCOEFF_NORMED",
                return_system_coordinates=True,
                max_retries=1,
            )
            print(f"   系统坐标模式 - 返回坐标: {result_sys['location']}")
            print(f"   系统坐标模式 - 返回中心点: {result_sys['center']}")
        except Exception as e:
            print(f"   错误: {e}")

        # 多方法比较
        print("\n2. 多方法比较:")
        try:
            results = gui_tool.compare_multiple_methods(
                template_path=template_path,
                target_path=target_path,
                methods=["TM_CCOEFF_NORMED", "TM_CCORR_NORMED", "TM_SQDIFF_NORMED"],
                save_results=True,
                max_retries=3,  # 演示重试功能
                retry_delay=0.5,
            )

            import rich

            rich.print(results)
            # print(results)
            # print("   各方法结果:")
            # for method, result in results.items():
            #     if isinstance(result, dict) and "confidence" in result:
            #         print(f"     {method}: 置信度 {result['confidence']:.3f}")

            #     print(method)
            #     print("==========================================")
            #     print(results)

        except Exception as e:
            print(f"   错误: {e}")

    else:
        print("演示文件不存在，请将图片放入 templates/ 目录")
        print("建议创建以下文件:")
        print("- templates/template.png (模板图片)")
        print("- templates/target.png (目标图片)")

        # 创建演示代码
        print("\n演示代码:")


if __name__ == "__main__":
    main()