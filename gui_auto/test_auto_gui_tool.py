from auto_gui_tool import GuiAutoTool


def main():
    """示例用法"""
    # 创建GUI自动化工具实例
    tool = GuiAutoTool(
        confidence=0.8,
        auto_scale=True,
        # 模板图像的缩放信息（例如：在1920x1080, 100%DPI下制作的模板）
        template_scale_info={
            "dpi_scale": 1.25,
            "resolution": (1920, 1080)
        },
        # 目标图像的缩放信息（例如：当前系统是2560x1440, 125%DPI）
        target_scale_info={
            "dpi_scale": 1.00,
            "resolution": (1920, 1080)
        },
        # 默认使用自适应图像增强
        default_enhancement_level="adaptive"
    )
    template_path = "image/terminal.png"
    target_path = "image/screenshot_125.png"

    # t1 = gui_tool.find_image_in_target(
    #     template=template_path,
    #     target_image=target_path,
    #     enhancement_level="standard",
    # )

    t1 = tool.compare_images(
        template_path=template_path,
        target_path=target_path,
        save_result=True,
        result_path="image/compare_images_result.png",
        return_system_coordinates=True,
    )

    

    import rich
    rich.print("t1", t1)
    print("-"*100)
    t2 = tool.compare_images(
        template_path=template_path,
        target_path=target_path,
        save_result=True,
        result_path="image/compare_images_result_2.png",
        return_system_coordinates=True,
        enhancement_level="aggressive"
    )

    rich.print("t2", t2)
    print("-"*100)
    
    t3 = tool.compare_images(
        template_path=template_path,
        target_path=target_path,
        save_result=True,
        result_path="image/compare_images_result_3.png",
        return_system_coordinates=True,
        enhancement_level="standard"
    )

    rich.print("t3", t3)
    print("-"*100)
    t4 = tool.compare_images(
        template_path=template_path,
        target_path=target_path,
        save_result=True,
        result_path="image/compare_images_result_4.png",
        return_system_coordinates=True,
        enhancement_level="light"
    )
    rich.print("t4", t4)
    print("-"*100)

if __name__ == "__main__":
    main()
