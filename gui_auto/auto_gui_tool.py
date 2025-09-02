"""
GUI自动化工具类 - 基于图像识别的自动化操作
支持重试机制和常用GUI操作
"""

import time
import functools
import logging
import ctypes
import ctypes.wintypes
from typing import Optional, Tuple, Union, Callable, Any
from pathlib import Path

import cv2
import numpy as np
import pyautogui
from PIL import ImageGrab


# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 禁用pyautogui的安全功能以提高性能
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.1


class SystemInfo:
    """系统信息获取类"""

    # 基准配置
    BASE_SCALE = 1.0  # 100%缩放
    BASE_RESOLUTION = (1920, 1080)  # 基准分辨率

    @staticmethod
    def get_dpi_scale() -> float:
        """获取系统DPI缩放比例"""
        try:
            # Windows API获取DPI
            user32 = ctypes.windll.user32
            user32.SetProcessDPIAware()

            # 获取系统DPI
            dc = user32.GetDC(0)
            dpi_x = ctypes.windll.gdi32.GetDeviceCaps(dc, 88)  # LOGPIXELSX
            dpi_y = ctypes.windll.gdi32.GetDeviceCaps(dc, 90)  # LOGPIXELSY
            user32.ReleaseDC(0, dc)

            # 标准DPI是96，计算缩放比例
            scale_x = dpi_x / 96.0
            scale_y = dpi_y / 96.0

            # 取平均值作为缩放比例
            scale = (scale_x + scale_y) / 2.0

            return scale

        except Exception as e:
            logger.warning(f"无法获取DPI缩放比例，使用默认值1.0: {e}")
            return 1.0

    @staticmethod
    def get_screen_resolution() -> Tuple[int, int]:
        """获取屏幕分辨率"""
        try:
            # 使用pyautogui获取屏幕尺寸
            width, height = pyautogui.size()
            return (width, height)
        except Exception as e:
            logger.warning(f"无法获取屏幕分辨率，使用默认值: {e}")
            return SystemInfo.BASE_RESOLUTION

    @staticmethod
    def get_scale_factors() -> dict:
        """获取相对于基准配置的缩放因子"""
        current_scale = SystemInfo.get_dpi_scale()
        current_resolution = SystemInfo.get_screen_resolution()

        # 计算相对于基准的缩放因子
        dpi_factor = current_scale / SystemInfo.BASE_SCALE

        # 计算分辨率缩放因子
        res_factor_x = current_resolution[0] / SystemInfo.BASE_RESOLUTION[0]
        res_factor_y = current_resolution[1] / SystemInfo.BASE_RESOLUTION[1]
        res_factor = (res_factor_x + res_factor_y) / 2.0

        # 综合缩放因子
        combined_factor = dpi_factor * res_factor

        return {
            "dpi_scale": current_scale,
            "resolution": current_resolution,
            "dpi_factor": dpi_factor,
            "resolution_factor": res_factor,
            "combined_factor": combined_factor,
            "base_scale": SystemInfo.BASE_SCALE,
            "base_resolution": SystemInfo.BASE_RESOLUTION,
        }


def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 1.5):
    """
    重试装饰器

    Args:
        max_attempts: 最大重试次数
        delay: 初始延迟时间（秒）
        backoff: 延迟时间倍数增长因子
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            current_delay = delay
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    result = func(*args, **kwargs)
                    if attempt > 0:
                        logger.info(
                            f"函数 {func.__name__} 在第 {attempt + 1} 次尝试成功"
                        )
                    return result
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"函数 {func.__name__} 第 {attempt + 1} 次尝试失败: {e}, {current_delay:.1f}秒后重试"
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            f"函数 {func.__name__} 所有 {max_attempts} 次尝试均失败"
                        )

            raise last_exception

        return wrapper

    return decorator


class GuiAutoTool:
    """GUI自动化工具类"""

    def __init__(
        self,
        confidence: float = 0.8,
        timeout: float = 10.0,
        default_method: str = "TM_CCOEFF_NORMED",
        use_advanced_matching: bool = True,
        auto_scale: bool = True,
        default_max_retries: int = 3,
        default_retry_delay: float = 0.5,
    ):
        """
        初始化GUI自动化工具

        Args:
            confidence: 图像匹配置信度阈值 (0.0-1.0)
            timeout: 操作超时时间（秒）
            default_method: 默认匹配方法
            use_advanced_matching: 是否默认使用高级匹配
            auto_scale: 是否自动处理DPI和分辨率缩放
            default_max_retries: 默认最大重试次数
            default_retry_delay: 默认重试延迟时间（秒）
        """
        self.confidence = confidence
        self.timeout = timeout
        self.default_method = default_method
        self.use_advanced_matching = use_advanced_matching
        self.auto_scale = auto_scale
        self.default_max_retries = default_max_retries
        self.default_retry_delay = default_retry_delay

        # 获取系统缩放信息
        self.scale_info = SystemInfo.get_scale_factors()
        logger.info(
            f"系统信息: DPI缩放={self.scale_info['dpi_scale']:.2f}, "
            f"分辨率={self.scale_info['resolution']}, "
            f"综合缩放因子={self.scale_info['combined_factor']:.2f}"
        )

    def get_version(self) -> str:
        """
        获取工具版本信息

        Returns:
            工具版本信息
        """
        return 1.0

    def get_screen_screenshot(self) -> np.ndarray:
        """
        获取当前屏幕截图的工具函数

        Returns:
            屏幕截图的numpy数组（BGR格式）
        """
        screenshot = ImageGrab.grab()
        screenshot_np = np.array(screenshot)
        screenshot_cv = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)
        return screenshot_cv

    def adjust_coordinates_from_scale(
        self, coordinates: Tuple[int, int, int, int]
    ) -> Tuple[int, int, int, int]:
        """
        将系统坐标转换为基准坐标

        Args:
            coordinates: 系统坐标 (x, y, w, h)

        Returns:
            基准坐标 (x, y, w, h)
        """
        if not self.auto_scale:
            return coordinates

        x, y, w, h = coordinates
        factor = self.scale_info["combined_factor"]

        # 反向缩放到基准坐标
        base_x = int(x / factor)
        base_y = int(y / factor)
        base_w = int(w / factor)
        base_h = int(h / factor)

        return (base_x, base_y, base_w, base_h)

    def adjust_coordinates_to_scale(
        self, coordinates: Tuple[int, int, int, int]
    ) -> Tuple[int, int, int, int]:
        """
        将基准坐标转换为系统坐标

        Args:
            coordinates: 基准坐标 (x, y, w, h)

        Returns:
            系统坐标 (x, y, w, h)
        """
        if not self.auto_scale:
            return coordinates

        x, y, w, h = coordinates
        factor = self.scale_info["combined_factor"]

        # 缩放到系统坐标
        sys_x = int(x * factor)
        sys_y = int(y * factor)
        sys_w = int(w * factor)
        sys_h = int(h * factor)

        return (sys_x, sys_y, sys_w, sys_h)

    def adjust_target_image_to_base(self, target_image: np.ndarray) -> np.ndarray:
        """
        将目标图像调整到基准尺寸

        Args:
            target_image: 原始目标图像

        Returns:
            调整后的目标图像
        """
        if not self.auto_scale:
            return target_image

        factor = self.scale_info["combined_factor"]

        # 如果缩放因子接近1.0，不需要调整
        if abs(factor - 1.0) < 0.05:
            return target_image

        # 反向缩放到基准尺寸
        new_height = int(target_image.shape[0] / factor)
        new_width = int(target_image.shape[1] / factor)

        if new_height > 0 and new_width > 0:
            adjusted_image = cv2.resize(
                target_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC
            )
            logger.debug(
                f"目标图像尺寸调整: {target_image.shape[:2]} -> {adjusted_image.shape[:2]} (缩放因子: {1/factor:.2f})"
            )
            return adjusted_image

        return target_image

    def adjust_template_size(self, template: np.ndarray) -> np.ndarray:
        """
        根据系统缩放调整模板图像尺寸

        Args:
            template: 原始模板图像

        Returns:
            调整后的模板图像
        """
        if not self.auto_scale:
            return template

        factor = self.scale_info["combined_factor"]

        # 如果缩放因子接近1.0，不需要调整
        if abs(factor - 1.0) < 0.05:
            return template

        # 调整模板尺寸
        new_height = int(template.shape[0] * factor)
        new_width = int(template.shape[1] * factor)

        if new_height > 0 and new_width > 0:
            adjusted_template = cv2.resize(
                template, (new_width, new_height), interpolation=cv2.INTER_CUBIC
            )
            logger.debug(
                f"模板尺寸调整: {template.shape[:2]} -> {adjusted_template.shape[:2]} (缩放因子: {factor:.2f})"
            )
            return adjusted_template

        return template

    def load_template(self, template_path: Union[str, Path]) -> np.ndarray:
        """
        加载模板图像

        Args:
            template_path: 模板图像路径

        Returns:
            模板图像的numpy数组
        """
        template_path = Path(template_path)
        if not template_path.exists():
            raise FileNotFoundError(f"模板图像文件不存在: {template_path}")

        template = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
        if template is None:
            raise ValueError(f"无法加载图像文件: {template_path}")

        return template

    def preprocess_image(self, image: np.ndarray, enhance: bool = True) -> np.ndarray:
        """
        图像预处理以提高匹配精度

        Args:
            image: 输入图像
            enhance: 是否进行图像增强

        Returns:
            处理后的图像
        """
        if not enhance:
            return image

        # 转换为Lab颜色空间进行处理
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # 应用CLAHE (对比度限制自适应直方图均衡) 到L通道
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        # 合并通道并转换回BGR
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        # 轻微的高斯模糊以减少噪声
        enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)

        return enhanced

    def compare_images(
        self,
        template_path: Union[str, Path],
        target_path: Union[str, Path],
        method: str = "TM_CCOEFF_NORMED",
        multi_scale: bool = True,
        enhance_images: bool = True,
        save_result: bool = False,
        result_path: Optional[str] = None,
        max_retries: Optional[int] = None,
        retry_delay: Optional[float] = None,
        return_system_coordinates: bool = False,
    ) -> dict:
        """
        比较两张图片，返回匹配结果

        Args:
            template_path: 模板图片路径
            target_path: 目标图片路径
            method: 匹配方法
            multi_scale: 是否使用多尺度匹配
            enhance_images: 是否进行图像增强
            save_result: 是否保存标注结果图片
            result_path: 结果图片保存路径
            max_retries: 最大重试次数（None时使用默认值）
            retry_delay: 重试延迟时间（None时使用默认值）
            return_system_coordinates: 是否返回系统坐标（默认返回基准坐标）

        Returns:
            包含匹配结果的字典: {
                'confidence': float,  # 置信度
                'location': tuple,    # 匹配位置 (x, y, w, h) - 基准坐标或系统坐标
                'center': tuple,      # 匹配中心点 (x, y) - 基准坐标或系统坐标
                'found': bool,        # 是否找到匹配
                'method': str,        # 使用的匹配方法
                'scale': float,       # 最佳匹配尺度
                'result_image_path': str  # 结果图片路径（如果保存）
            }
        """
        # 设置重试参数
        if max_retries is None:
            max_retries = self.default_max_retries
        if retry_delay is None:
            retry_delay = self.default_retry_delay

        # 定义核心比较逻辑
        def _compare_core():
            return self._compare_images_core(
                template_path,
                target_path,
                method,
                multi_scale,
                enhance_images,
                save_result,
                result_path,
                return_system_coordinates,
            )

        # 执行重试逻辑
        current_delay = retry_delay
        last_exception = None

        for attempt in range(max_retries):
            try:
                result = _compare_core()
                if attempt > 0:
                    logger.info(f"compare_images 在第 {attempt + 1} 次尝试成功")
                return result
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    logger.warning(
                        f"compare_images 第 {attempt + 1} 次尝试失败: {e}, {current_delay:.1f}秒后重试"
                    )
                    time.sleep(current_delay)
                    current_delay *= 1.5  # 延迟递增
                else:
                    logger.error(f"compare_images 所有 {max_retries} 次尝试均失败")

        raise last_exception

    def _compare_images_core(
        self,
        template_path: Union[str, Path],
        target_path: Union[str, Path],
        method: str,
        multi_scale: bool,
        enhance_images: bool,
        save_result: bool,
        result_path: Optional[str],
        return_system_coordinates: bool,
    ) -> dict:
        """
        compare_images 的核心实现逻辑
        """
        # 加载图像
        template = self.load_template(template_path)
        target = self.load_template(target_path)

        if template is None or target is None:
            raise ValueError("无法加载图片文件")

        # 将目标图像调整到基准尺寸（在处理前转换）
        if self.auto_scale:
            target = self.adjust_target_image_to_base(target)
            # 模板图像不需要额外调整，因为已经是基准尺寸

        # 图像预处理
        if enhance_images:
            template_processed = self.preprocess_image(template, enhance=True)
            target_processed = self.preprocess_image(target, enhance=True)
        else:
            template_processed = template
            target_processed = target

        # 方法映射
        methods = {
            "TM_CCOEFF_NORMED": cv2.TM_CCOEFF_NORMED,
            "TM_CCORR_NORMED": cv2.TM_CCORR_NORMED,
            "TM_SQDIFF_NORMED": cv2.TM_SQDIFF_NORMED,
        }

        cv_method = methods.get(method, cv2.TM_CCOEFF_NORMED)

        # 多尺度匹配
        best_confidence = 0
        best_location = None
        best_scale = 1.0
        best_match_loc = None

        if multi_scale:
            scales = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
        else:
            scales = [1.0]

        logger.info(
            f"开始图像匹配，方法: {method}, 多尺度: {multi_scale}, 增强: {enhance_images}"
        )

        for scale in scales:
            # 缩放模板图像
            if scale != 1.0:
                scaled_template = cv2.resize(
                    template_processed,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_CUBIC,
                )
            else:
                scaled_template = template_processed

            # 确保模板不大于目标图像
            if (
                scaled_template.shape[0] > target_processed.shape[0]
                or scaled_template.shape[1] > target_processed.shape[1]
            ):
                continue

            # 执行模板匹配
            result = cv2.matchTemplate(target_processed, scaled_template, cv_method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            # 根据方法选择合适的值和位置
            if cv_method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                confidence = 1 - min_val
                match_loc = min_loc
            else:
                confidence = max_val
                match_loc = max_loc

            logger.debug(f"尺度 {scale:.1f}, 置信度: {confidence:.3f}")

            # 记录最佳匹配
            if confidence > best_confidence:
                best_confidence = confidence
                h, w = scaled_template.shape[:2]
                best_location = (match_loc[0], match_loc[1], w, h)
                best_scale = scale
                best_match_loc = match_loc

        # 现在best_location已经是基准坐标
        base_location = best_location
        base_center = None
        system_location = None
        system_center = None

        if base_location:
            # 计算基准中心点
            base_center = (
                base_location[0] + base_location[2] // 2,
                base_location[1] + base_location[3] // 2,
            )
            
            # 如果需要系统坐标，进行转换
            if return_system_coordinates:
                system_location = self.adjust_coordinates_to_scale(base_location)
                system_center = (
                    system_location[0] + system_location[2] // 2,
                    system_location[1] + system_location[3] // 2,
                )

        # 根据参数决定返回的坐标类型
        output_location = system_location if return_system_coordinates else base_location
        output_center = system_center if return_system_coordinates else base_center

        # 准备返回结果
        result_dict = {
            "confidence": best_confidence,
            "location": output_location,  # 基准坐标或系统坐标（根据参数决定）
            "center": output_center,  # 基准中心点或系统中心点（根据参数决定）
            "base_location": base_location,  # 基准坐标（总是提供）
            "base_center": base_center,  # 基准中心点（总是提供）
            "system_location": system_location,  # 系统坐标（如果需要）
            "system_center": system_center,  # 系统中心点（如果需要）
            "found": best_confidence >= self.confidence,
            "method": method,
            "scale": best_scale,
            "system_scale_factor": self.scale_info["combined_factor"],
            "system_info": self.scale_info,
            "result_image_path": None,
            "coordinate_type": "system" if return_system_coordinates else "base",
        }

        # 保存标注结果图片
        if save_result and base_location:
            if result_path is None:
                result_path = f"match_result_{int(time.time())}.png"

            # 重新加载原始目标图像用于绘制（不进行基准转换）
            original_target = self.load_template(target_path)
            result_image = original_target.copy()

            # 使用系统坐标进行绘制
            draw_location = system_location if system_location else self.adjust_coordinates_to_scale(base_location)
            draw_center = system_center if system_center else (
                draw_location[0] + draw_location[2] // 2,
                draw_location[1] + draw_location[3] // 2,
            )

            # 绘制匹配框（使用系统坐标）
            x, y, w, h = draw_location
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 绘制中心点
            if draw_center:
                cv2.circle(result_image, draw_center, 5, (0, 0, 255), -1)

            # 添加文本信息
            info_texts = [
                f"Confidence: {best_confidence:.3f}",
                f"Method: {method}",
                f"Template Scale: {best_scale:.1f}",
                f"System Scale: {self.scale_info['combined_factor']:.2f}",
                f"Base Location: {base_location}",
                f"System Location: {draw_location}",
                f"Resolution: {self.scale_info['resolution'][0]}x{self.scale_info['resolution'][1]}",
            ]

            # 绘制文本信息
            for i, text in enumerate(info_texts):
                y_offset = y - 10 - (i * 20)
                if y_offset > 0:  # 确保文本在图像范围内
                    cv2.putText(
                        result_image,
                        text,
                        (x, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )

            # 保存结果图像
            cv2.imwrite(result_path, result_image)
            result_dict["result_image_path"] = result_path
            logger.info(f"结果图像已保存到: {result_path}")

        logger.info(
            f"图像匹配完成 - 置信度: {best_confidence:.3f}, 找到: {result_dict['found']}"
        )

        return result_dict

    def compare_multiple_methods(
        self,
        template_path: Union[str, Path],
        target_path: Union[str, Path],
        methods: list = None,
        save_results: bool = False,
        max_retries: Optional[int] = None,
        retry_delay: Optional[float] = None,
    ) -> dict:
        """
        使用多种方法比较图片，返回最佳匹配结果

        Args:
            template_path: 模板图片路径
            target_path: 目标图片路径
            methods: 要尝试的匹配方法列表
            save_results: 是否保存所有结果图片
            max_retries: 最大重试次数（None时使用默认值）
            retry_delay: 重试延迟时间（None时使用默认值）

        Returns:
            包含所有方法结果的字典
        """
        # 设置重试参数
        if max_retries is None:
            max_retries = self.default_max_retries
        if retry_delay is None:
            retry_delay = self.default_retry_delay

        # 定义核心比较逻辑
        def _compare_multiple_core():
            return self._compare_multiple_methods_core(
                template_path, target_path, methods, save_results
            )

        # 执行重试逻辑
        current_delay = retry_delay
        last_exception = None

        for attempt in range(max_retries):
            try:
                result = _compare_multiple_core()
                if attempt > 0:
                    logger.info(
                        f"compare_multiple_methods 在第 {attempt + 1} 次尝试成功"
                    )
                return result
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    logger.warning(
                        f"compare_multiple_methods 第 {attempt + 1} 次尝试失败: {e}, {current_delay:.1f}秒后重试"
                    )
                    time.sleep(current_delay)
                    current_delay *= 1.5  # 延迟递增
                else:
                    logger.error(
                        f"compare_multiple_methods 所有 {max_retries} 次尝试均失败"
                    )

        raise last_exception

    def _compare_multiple_methods_core(
        self,
        template_path: Union[str, Path],
        target_path: Union[str, Path],
        methods: list,
        save_results: bool,
    ) -> dict:
        """
        compare_multiple_methods 的核心实现逻辑
        """
        if methods is None:
            methods = ["TM_CCOEFF_NORMED", "TM_CCORR_NORMED", "TM_SQDIFF_NORMED"]

        results = {}
        best_result = None
        best_confidence = 0

        logger.info(f"开始多方法比较，尝试方法: {methods}")

        for method in methods:
            try:
                result_path = (
                    f"result_{method}_{int(time.time())}.png" if save_results else None
                )
                result = self.compare_images(
                    template_path,
                    target_path,
                    method=method,
                    multi_scale=True,
                    enhance_images=True,
                    save_result=save_results,
                    result_path=result_path,
                )
                results[method] = result

                if result["confidence"] > best_confidence:
                    best_confidence = result["confidence"]
                    best_result = result.copy()
                    best_result["best_method"] = method

                logger.info(f"方法 {method}: 置信度 {result['confidence']:.3f}")

            except Exception as e:
                logger.error(f"方法 {method} 执行失败: {e}")
                results[method] = {"error": str(e)}

        results["best_result"] = best_result
        results["summary"] = {
            "best_method": best_result["best_method"] if best_result else None,
            "best_confidence": best_confidence,
            "methods_tried": len(
                [
                    r
                    for r in results.values()
                    if isinstance(r, dict) and "confidence" in r
                ]
            ),
        }

        return results

    def find_image_in_target(
        self,
        template: Union[str, Path, np.ndarray],
        target_image: Union[str, Path, np.ndarray],
        region: Optional[Tuple[int, int, int, int]] = None,
        method: str = "TM_CCOEFF_NORMED",
        multi_scale: bool = True,
        enhance_images: bool = True,
        max_retries: Optional[int] = None,
        retry_delay: Optional[float] = None,
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        在目标图像中查找模板图像（优化版本）

        Args:
            template: 模板图像路径或numpy数组
            target_image: 目标图像路径或numpy数组
            region: 搜索区域 (left, top, width, height)
            method: 匹配方法 ("TM_CCOEFF_NORMED", "TM_CCORR_NORMED", "TM_SQDIFF_NORMED")
            multi_scale: 是否使用多尺度匹配
            enhance_images: 是否进行图像增强处理
            max_retries: 最大重试次数（None时使用默认值）
            retry_delay: 重试延迟时间（None时使用默认值）

        Returns:
            找到的图像位置 (left, top, width, height) 或 None
        """
        # 设置重试参数
        if max_retries is None:
            max_retries = self.default_max_retries
        if retry_delay is None:
            retry_delay = self.default_retry_delay

        # 定义核心查找逻辑
        def _find_core():
            return self._find_image_in_target_core(
                template, target_image, region, method, multi_scale, enhance_images
            )

        # 执行重试逻辑
        current_delay = retry_delay
        last_exception = None

        for attempt in range(max_retries):
            try:
                result = _find_core()
                if attempt > 0 and result is not None:
                    logger.info(f"find_image_in_target 在第 {attempt + 1} 次尝试成功")
                return result
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    logger.warning(
                        f"find_image_in_target 第 {attempt + 1} 次尝试失败: {e}, {current_delay:.1f}秒后重试"
                    )
                    time.sleep(current_delay)
                    current_delay *= 1.5  # 延迟递增
                else:
                    logger.error(
                        f"find_image_in_target 所有 {max_retries} 次尝试均失败"
                    )

        # 如果所有重试都失败了，可以选择抛出异常或返回None
        # 这里选择返回None，保持原有行为
        return None

    def _find_image_in_target_core(
        self,
        template: Union[str, Path, np.ndarray],
        target_image: Union[str, Path, np.ndarray],
        region: Optional[Tuple[int, int, int, int]],
        method: str,
        multi_scale: bool,
        enhance_images: bool,
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        find_image_in_target 的核心实现逻辑
        """
        # 加载目标图像
        if isinstance(target_image, (str, Path)):
            target_img = self.load_template(target_image)
        else:
            target_img = target_image

        # 将目标图像调整到基准尺寸（在处理前转换）
        if self.auto_scale:
            target_img = self.adjust_target_image_to_base(target_img)

        # 如果指定了搜索区域，裁剪目标图像（region应该是基准坐标）
        if region:
            x, y, w, h = region
            target_img = target_img[y : y + h, x : x + w]

        # 加载模板图像
        if isinstance(template, (str, Path)):
            template_img = self.load_template(template)
        else:
            template_img = template

        # 图像预处理增强
        if enhance_images:
            target_img = self.preprocess_image(target_img, enhance=True)
            template_img = self.preprocess_image(template_img, enhance=True)

        # 方法映射
        methods = {
            "TM_CCOEFF_NORMED": cv2.TM_CCOEFF_NORMED,
            "TM_CCORR_NORMED": cv2.TM_CCORR_NORMED,
            "TM_SQDIFF_NORMED": cv2.TM_SQDIFF_NORMED,
        }

        cv_method = methods.get(method, cv2.TM_CCOEFF_NORMED)

        # 多尺度匹配以提高准确性
        best_confidence = 0
        best_location = None
        best_scale = 1.0

        if multi_scale:
            scales = [0.8, 0.9, 1.0, 1.1, 1.2]  # 尝试不同缩放比例
        else:
            scales = [1.0]

        for scale in scales:
            # 缩放模板图像
            if scale != 1.0:
                scaled_template = cv2.resize(
                    template_img,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_CUBIC,
                )
            else:
                scaled_template = template_img

            # 确保模板不大于目标图像
            if (
                scaled_template.shape[0] > target_img.shape[0]
                or scaled_template.shape[1] > target_img.shape[1]
            ):
                continue

            # 执行模板匹配
            result = cv2.matchTemplate(target_img, scaled_template, cv_method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            # 根据方法选择合适的值和位置
            if cv_method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                confidence = 1 - min_val
                match_loc = min_loc
            else:
                confidence = max_val
                match_loc = max_loc

            logger.debug(f"尺度 {scale:.1f}, 置信度: {confidence:.3f}")

            # 记录最佳匹配
            if confidence > best_confidence:
                best_confidence = confidence
                h, w = scaled_template.shape[:2]
                left, top = match_loc

            # 如果指定了搜索区域，需要调整坐标
            if region:
                left += region[0]
                top += region[1]

                best_location = (left, top, w, h)
                best_scale = scale

        logger.debug(
            f"最佳匹配 - 置信度: {best_confidence:.3f}, 尺度: {best_scale:.1f}"
        )

        if best_confidence >= self.confidence:
            logger.info(
                f"找到图像，置信度: {best_confidence:.3f}, 位置: {best_location}, 尺度: {best_scale:.1f}"
            )
            return best_location

        logger.debug(f"未找到图像，最高置信度: {best_confidence:.3f}")
        return None

    def find_image_advanced_in_target(
        self,
        template: Union[str, Path, np.ndarray],
        target_image: Union[str, Path, np.ndarray],
        region: Optional[Tuple[int, int, int, int]] = None,
        try_multiple_methods: bool = True,
        max_retries: Optional[int] = None,
        retry_delay: Optional[float] = None,
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        高级图像匹配，在目标图像中尝试多种方法和参数组合

        Args:
            template: 模板图像路径或numpy数组
            target_image: 目标图像路径或numpy数组
            region: 搜索区域
            try_multiple_methods: 是否尝试多种匹配方法
            max_retries: 最大重试次数（None时使用默认值）
            retry_delay: 重试延迟时间（None时使用默认值）

        Returns:
            找到的图像位置 (left, top, width, height) 或 None
        """
        # 设置重试参数
        if max_retries is None:
            max_retries = self.default_max_retries
        if retry_delay is None:
            retry_delay = self.default_retry_delay

        # 定义核心查找逻辑
        def _find_advanced_core():
            return self._find_image_advanced_in_target_core(
                template, target_image, region, try_multiple_methods
            )

        # 执行重试逻辑
        current_delay = retry_delay
        last_exception = None

        for attempt in range(max_retries):
            try:
                result = _find_advanced_core()
                if attempt > 0 and result is not None:
                    logger.info(
                        f"find_image_advanced_in_target 在第 {attempt + 1} 次尝试成功"
                    )
                return result
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    logger.warning(
                        f"find_image_advanced_in_target 第 {attempt + 1} 次尝试失败: {e}, {current_delay:.1f}秒后重试"
                    )
                    time.sleep(current_delay)
                    current_delay *= 1.5  # 延迟递增
                else:
                    logger.error(
                        f"find_image_advanced_in_target 所有 {max_retries} 次尝试均失败"
                    )

        # 如果所有重试都失败了，返回None保持原有行为
        return None

    def _find_image_advanced_in_target_core(
        self,
        template: Union[str, Path, np.ndarray],
        target_image: Union[str, Path, np.ndarray],
        region: Optional[Tuple[int, int, int, int]],
        try_multiple_methods: bool,
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        find_image_advanced_in_target 的核心实现逻辑
        """
        methods_to_try = (
            ["TM_CCOEFF_NORMED", "TM_CCORR_NORMED"]
            if try_multiple_methods
            else ["TM_CCOEFF_NORMED"]
        )
        enhancement_options = [True, False]

        best_result = None
        best_confidence = 0

        for method in methods_to_try:
            for enhance in enhancement_options:
                try:
                    result = self.find_image_in_target(
                        template,
                        target_image,
                        region=region,
                        method=method,
                        multi_scale=True,
                        enhance_images=enhance,
                    )

                    if result:
                        # 重新计算置信度以比较不同方法的结果
                        if isinstance(target_image, (str, Path)):
                            target_img = self.load_template(target_image)
                        else:
                            target_img = target_image

                        if isinstance(template, (str, Path)):
                            template_img = self.load_template(template)
                        else:
                            template_img = template

                        if enhance:
                            target_img = self.preprocess_image(target_img, enhance=True)
                            template_img = self.preprocess_image(
                                template_img, enhance=True
                            )

                        # 使用结果位置计算置信度
                        left, top, w, h = result
                        if region:
                            left -= region[0]
                            top -= region[1]

                        roi = target_img[top : top + h, left : left + w]
                        if roi.shape[:2] == template_img.shape[:2]:
                            # 计算结构相似性
                            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                            template_gray = cv2.cvtColor(
                                template_img, cv2.COLOR_BGR2GRAY
                            )

                            # 使用归一化互相关计算相似度
                            match_result = cv2.matchTemplate(
                                roi_gray.reshape(1, -1),
                                template_gray.reshape(1, -1),
                                cv2.TM_CCOEFF_NORMED,
                            )
                            confidence = float(match_result[0, 0])

                            if confidence > best_confidence:
                                best_confidence = confidence
                                best_result = result
                                logger.debug(
                                    f"找到更好匹配 - 方法: {method}, 增强: {enhance}, 置信度: {confidence:.3f}"
                                )

                except Exception as e:
                    logger.debug(f"方法 {method} (增强={enhance}) 失败: {e}")
                    continue

        if best_result:
            logger.info(f"高级匹配成功，最终置信度: {best_confidence:.3f}")

        return best_result

    def find_template_in_target(
        self,
        template: Union[str, Path, np.ndarray],
        target_image: Union[str, Path, np.ndarray],
        region: Optional[Tuple[int, int, int, int]] = None,
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        在目标图像中查找模板（简化版本）

        Args:
            template: 模板图像路径或numpy数组
            target_image: 目标图像路径或numpy数组
            region: 搜索区域

        Returns:
            图像位置 (left, top, width, height) 或 None
        """
        # 根据设置选择匹配方法
        if self.use_advanced_matching:
            return self.find_image_advanced_in_target(template, target_image, region)
        else:
            return self.find_image_in_target(
                template, target_image, region, method=self.default_method
            )

    def get_image_center(self, location: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """
        获取图像中心点坐标

        Args:
            location: 图像位置 (left, top, width, height)

        Returns:
            中心点坐标 (x, y)
        """
        left, top, width, height = location
        center_x = left + width // 2
        center_y = top + height // 2
        return (center_x, center_y)

    def wait_for_image(
        self,
        template: Union[str, Path, np.ndarray],
        target_image: Union[str, Path, np.ndarray],
        region: Optional[Tuple[int, int, int, int]] = None,
        max_retries: Optional[int] = None,
        retry_delay: Optional[float] = None,
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        在目标图像中等待/查找模板图像

        Args:
            template: 模板图像路径或numpy数组
            target_image: 目标图像路径或numpy数组
            region: 搜索区域
            max_retries: 最大重试次数（None时使用默认值）
            retry_delay: 重试延迟时间（None时使用默认值）

        Returns:
            找到的图像位置 (left, top, width, height) 或 None
        """
        if max_retries is None:
            max_retries = self.default_max_retries
        if retry_delay is None:
            retry_delay = self.default_retry_delay

        # 执行重试逻辑
        current_delay = retry_delay
        last_exception = None

        for attempt in range(max_retries):
            try:
                # 使用现有的图像查找方法
                location = self.find_image_in_target(
                    template,
                    target_image,
                    region=region,
                    max_retries=1,  # 内部不再重试，由外层控制
                    retry_delay=0,
                )
                if location:
                    if attempt > 0:
                        logger.info(f"wait_for_image 在第 {attempt + 1} 次尝试成功")
                    return location
                else:
                    # 如果没找到，不抛异常，继续重试
                    if attempt < max_retries - 1:
                        logger.debug(
                            f"wait_for_image 第 {attempt + 1} 次尝试未找到图像，{current_delay:.1f}秒后重试"
                        )
                        time.sleep(current_delay)
                        current_delay *= 1.5  # 延迟递增
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    logger.warning(
                        f"wait_for_image 第 {attempt + 1} 次尝试失败: {e}, {current_delay:.1f}秒后重试"
                    )
                    time.sleep(current_delay)
                    current_delay *= 1.5  # 延迟递增
                else:
                    logger.error(f"wait_for_image 所有 {max_retries} 次尝试均失败")

        # 如果所有重试都失败了，返回None
        logger.debug(f"wait_for_image 在 {max_retries} 次尝试后仍未找到图像")
        return None

    @retry(max_attempts=3, delay=0.3)
    def click_image(
        self,
        template: Union[str, Path, np.ndarray],
        target_image: Union[str, Path, np.ndarray],
        button: str = "left",
        offset: Tuple[int, int] = (0, 0),
        region: Optional[Tuple[int, int, int, int]] = None,
    ) -> bool:
        """
        点击图像

        Args:
            template: 模板图像路径或numpy数组
            target_image: 目标图像路径或numpy数组
            button: 鼠标按键 ('left', 'right', 'middle')
            offset: 点击偏移量 (x_offset, y_offset) - 基准坐标偏移
            region: 搜索区域 - 基准坐标

        Returns:
            是否成功点击
        """
        try:
            # 获取基准坐标位置
            base_location = self.wait_for_image(template, target_image, region=region)
            if not base_location:
                logger.error("点击失败：未找到目标图像")
                return False
                
            # 计算基准中心点
            base_center_x, base_center_y = self.get_image_center(base_location)

            # 应用基准偏移量
            base_click_x = base_center_x + offset[0]
            base_click_y = base_center_y + offset[1]

            # 转换为系统坐标进行实际点击
            system_location = self.adjust_coordinates_to_scale((base_click_x, base_click_y, 1, 1))
            click_x, click_y = system_location[0], system_location[1]

            pyautogui.click(click_x, click_y, button=button)
            logger.info(f"点击图像成功，基准位置: ({base_click_x}, {base_click_y}), 系统位置: ({click_x}, {click_y})")
            return True
        except Exception as e:
            logger.error(f"点击失败：{e}")
            return False

    @retry(max_attempts=3, delay=0.3)
    def double_click_image(
        self,
        template: Union[str, Path, np.ndarray],
        target_image: Union[str, Path, np.ndarray],
        offset: Tuple[int, int] = (0, 0),
        region: Optional[Tuple[int, int, int, int]] = None,
    ) -> bool:
        """
        双击图像

        Args:
            template: 模板图像路径或numpy数组
            target_image: 目标图像路径或numpy数组
            offset: 点击偏移量 - 基准坐标偏移
            region: 搜索区域 - 基准坐标

        Returns:
            是否成功双击
        """
        try:
            # 获取基准坐标位置
            base_location = self.wait_for_image(template, target_image, region=region)
            if not base_location:
                logger.error("双击失败：未找到目标图像")
                return False
                
            # 计算基准中心点
            base_center_x, base_center_y = self.get_image_center(base_location)

            # 应用基准偏移量
            base_click_x = base_center_x + offset[0]
            base_click_y = base_center_y + offset[1]

            # 转换为系统坐标进行实际双击
            system_location = self.adjust_coordinates_to_scale((base_click_x, base_click_y, 1, 1))
            click_x, click_y = system_location[0], system_location[1]

            pyautogui.doubleClick(click_x, click_y)
            logger.info(f"双击图像成功，基准位置: ({base_click_x}, {base_click_y}), 系统位置: ({click_x}, {click_y})")
            return True
        except Exception as e:
            logger.error(f"双击失败：{e}")
            return False

    @retry(max_attempts=3, delay=0.3)
    def drag_from_to_image(
        self,
        from_template: Union[str, Path, np.ndarray],
        to_template: Union[str, Path, np.ndarray],
        target_image: Union[str, Path, np.ndarray],
        duration: float = 1.0,
    ) -> bool:
        """
        从一个图像拖拽到另一个图像

        Args:
            from_template: 起始图像模板
            to_template: 目标图像模板
            target_image: 目标图像路径或numpy数组
            duration: 拖拽持续时间

        Returns:
            是否成功拖拽
        """
        try:
            # 找到起始位置（基准坐标）
            from_base_location = self.wait_for_image(from_template, target_image)
            if not from_base_location:
                logger.error("拖拽失败：未找到起始图像")
                return False
                
            from_base_x, from_base_y = self.get_image_center(from_base_location)

            # 找到目标位置（基准坐标）
            to_base_location = self.wait_for_image(to_template, target_image)
            if not to_base_location:
                logger.error("拖拽失败：未找到目标图像")
                return False
                
            to_base_x, to_base_y = self.get_image_center(to_base_location)

            # 转换为系统坐标进行实际拖拽
            from_system = self.adjust_coordinates_to_scale((from_base_x, from_base_y, 1, 1))
            to_system = self.adjust_coordinates_to_scale((to_base_x, to_base_y, 1, 1))
            
            from_x, from_y = from_system[0], from_system[1]
            to_x, to_y = to_system[0], to_system[1]

            # 执行拖拽
            pyautogui.drag(
                to_x - from_x, to_y - from_y, duration=duration, button="left"
            )
            logger.info(f"拖拽成功：基准坐标从 ({from_base_x}, {from_base_y}) 到 ({to_base_x}, {to_base_y})")
            logger.info(f"         系统坐标从 ({from_x}, {from_y}) 到 ({to_x}, {to_y})")
            return True
        except Exception as e:
            logger.error(f"拖拽失败：{e}")
            return False

    def type_text(self, text: str, interval: float = 0.05) -> None:
        """
        输入文本

        Args:
            text: 要输入的文本
            interval: 字符间隔时间
        """
        pyautogui.write(text, interval=interval)
        logger.info(f"输入文本成功: {text}")

    def press_key(self, key: str) -> None:
        """
        按下键盘按键

        Args:
            key: 按键名称 (如 'enter', 'ctrl', 'alt' 等)
        """
        pyautogui.press(key)
        logger.info(f"按键成功: {key}")

    def press_keys(self, keys: list) -> None:
        """
        按下组合键

        Args:
            keys: 按键列表 (如 ['ctrl', 'c'])
        """
        pyautogui.hotkey(*keys)
        logger.info(f"组合键成功: {'+'.join(keys)}")

    def move_to_image(
        self,
        template: Union[str, Path, np.ndarray],
        target_image: Union[str, Path, np.ndarray],
        offset: Tuple[int, int] = (0, 0),
        duration: float = 0.5,
    ) -> bool:
        """
        移动鼠标到图像位置

        Args:
            template: 模板图像
            target_image: 目标图像路径或numpy数组
            offset: 偏移量 - 基准坐标偏移
            duration: 移动持续时间

        Returns:
            是否成功移动
        """
        try:
            # 获取基准坐标位置
            base_location = self.wait_for_image(template, target_image)
            if not base_location:
                logger.error("鼠标移动失败：未找到目标图像")
                return False
                
            # 计算基准中心点
            base_center_x, base_center_y = self.get_image_center(base_location)

            # 应用基准偏移量
            base_move_x = base_center_x + offset[0]
            base_move_y = base_center_y + offset[1]

            # 转换为系统坐标进行实际移动
            system_location = self.adjust_coordinates_to_scale((base_move_x, base_move_y, 1, 1))
            move_x, move_y = system_location[0], system_location[1]

            pyautogui.moveTo(move_x, move_y, duration=duration)
            logger.info(f"鼠标移动成功，基准位置: ({base_move_x}, {base_move_y}), 系统位置: ({move_x}, {move_y})")
            return True
        except Exception as e:
            logger.error(f"鼠标移动失败：{e}")
            return False


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
            print(f"   返回坐标: {result['location']} ({'基准' if result['coordinate_type'] == 'base' else '系统'})")
            print(f"   返回中心点: {result['center']} ({'基准' if result['coordinate_type'] == 'base' else '系统'})")
            print(f"   基准坐标: {result['base_location']}")
            print(f"   基准中心点: {result['base_center']}")
            if result['system_location']:
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
                max_retries=1
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
