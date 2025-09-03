from auto_gui_tool import GuiAutoTool


def main():
    """示例用法"""
    # 创建GUI自动化工具实例
    gui_tool = GuiAutoTool(
        confidence=0.8,  # 降低阈值以便演示
        timeout=10.0,
        default_method="TM_CCOEFF_NORMED",
        default_max_retries=3,  # 默认重试3次
        default_retry_delay=0.5,  # 默认重试延迟0.5秒
        base_scale=1.25
    )
    template_path = "image/terminal.png"
    target_path = "image/screenshot_125.png"

    # t1 = gui_tool.find_image_in_target(
    #     template=template_path,
    #     target_image=target_path,
    #     enhancement_level="standard",
    # )

    t1 = gui_tool.compare_images(
        template_path=template_path,
        target_path=target_path,
        save_result=True,
        result_path="image/compare_images_result.png",
        return_system_coordinates=True,
    )



    import rich
    rich.print(t1)

if __name__ == "__main__":
    main()
