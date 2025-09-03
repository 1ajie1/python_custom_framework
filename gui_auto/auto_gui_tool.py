"""
GUI自动化工具类 - 基于图像识别的自动化操作
支持重试机制和常用GUI操作
"""

import time
import functools
import logging
import ctypes
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
    def get_scale_factors(
        base_scale: Optional[float] = None,
        base_resolution: Optional[Tuple[int, int]] = None,
    ) -> dict:
        """获取相对于基准配置的缩放因子"""
        current_scale = SystemInfo.get_dpi_scale()
        current_resolution = SystemInfo.get_screen_resolution()

        # 使用自定义基准值或默认值
        effective_base_scale = (
            base_scale if base_scale is not None else SystemInfo.BASE_SCALE
        )
        effective_base_resolution = (
            base_resolution
            if base_resolution is not None
            else SystemInfo.BASE_RESOLUTION
        )

        # 计算相对于基准的缩放因子
        dpi_factor = current_scale / effective_base_scale

        # 计算分辨率缩放因子
        res_factor_x = current_resolution[0] / effective_base_resolution[0]
        res_factor_y = current_resolution[1] / effective_base_resolution[1]
        res_factor = (res_factor_x + res_factor_y) / 2.0

        # 综合缩放因子
        combined_factor = dpi_factor * res_factor

        return {
            "dpi_scale": current_scale,
            "resolution": current_resolution,
            "dpi_factor": dpi_factor,
            "resolution_factor": res_factor,
            "combined_factor": combined_factor,
            "base_scale": effective_base_scale,
            "base_resolution": effective_base_resolution,
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
        auto_scale: bool = True,
        default_max_retries: int = 3,
        default_retry_delay: float = 0.5,
        base_scale: Optional[float] = None,
        base_resolution: Optional[Tuple[int, int]] = None,
    ):
        """
        初始化GUI自动化工具

        Args:
            confidence: 图像匹配置信度阈值 (0.0-1.0)
            timeout: 操作超时时间（秒）
            default_method: 默认匹配方法
            auto_scale: 是否自动处理DPI和分辨率缩放
            default_max_retries: 默认最大重试次数
            default_retry_delay: 默认重试延迟时间（秒）
            base_scale: 自定义基准DPI缩放，不指定则使用默认值1.0
            base_resolution: 自定义基准分辨率，不指定则使用默认值(1920, 1080)
        """
        self.confidence = confidence
        self.timeout = timeout
        self.default_method = default_method
        self.auto_scale = auto_scale
        self.default_max_retries = default_max_retries
        self.default_retry_delay = default_retry_delay

        # 保存自定义基准值
        self.base_scale = base_scale
        self.base_resolution = base_resolution

        # 获取系统缩放信息（使用自定义基准值）
        self.scale_info = SystemInfo.get_scale_factors(
            base_scale=self.base_scale, base_resolution=self.base_resolution
        )
        logger.info(
            f"系统信息: DPI缩放={self.scale_info['dpi_scale']:.2f}, "
            f"分辨率={self.scale_info['resolution']}, "
            f"基准缩放={self.scale_info['base_scale']:.2f}, "
            f"基准分辨率={self.scale_info['base_resolution']}, "
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

    def load_image(self, image_input: Union[str, Path, np.ndarray]) -> np.ndarray:
        """
        加载图像，支持路径或numpy数组输入

        Args:
            image_input: 图像路径或numpy数组

        Returns:
            图像的numpy数组
        """
        if isinstance(image_input, np.ndarray):
            # 验证是否为有效的图像数组
            if len(image_input.shape) not in [2, 3]:
                raise ValueError(f"无效的图像数组形状: {image_input.shape}")
            return image_input
        else:
            # 使用现有的 load_template 方法加载文件
            return self.load_template(image_input)

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

    def _calculate_optimal_scale(
        self, 
        template_scale_info: dict, 
        target_scale_info: dict
    ) -> float:
        """
        根据模板和目标图像的缩放信息计算最优缩放比例
        
        Args:
            template_scale_info: 模板图像缩放信息 {'dpi_scale': float, 'resolution': (w, h)}
            target_scale_info: 目标图像缩放信息 {'dpi_scale': float, 'resolution': (w, h)}
            
        Returns:
            计算出的最优缩放比例
        """
        try:
            # 获取DPI缩放比例
            template_dpi = template_scale_info.get('dpi_scale', 1.0)
            target_dpi = target_scale_info.get('dpi_scale', 1.0)
            
            # 获取分辨率信息
            template_res = template_scale_info.get('resolution', (1920, 1080))
            target_res = target_scale_info.get('resolution', (1920, 1080))
            
            # 计算DPI缩放比例
            dpi_scale_ratio = target_dpi / template_dpi
            
            # 计算分辨率缩放比例（取平均值）
            res_scale_x = target_res[0] / template_res[0]
            res_scale_y = target_res[1] / template_res[1]
            res_scale_ratio = (res_scale_x + res_scale_y) / 2.0
            
            # 综合缩放比例
            optimal_scale = dpi_scale_ratio * res_scale_ratio
            
            # 限制缩放比例在合理范围内
            optimal_scale = max(0.3, min(3.0, optimal_scale))
            
            logger.debug(
                f"缩放计算: 模板DPI={template_dpi:.2f}, 目标DPI={target_dpi:.2f}, "
                f"DPI比例={dpi_scale_ratio:.3f}, 分辨率比例={res_scale_ratio:.3f}, "
                f"最终缩放={optimal_scale:.3f}"
            )
            
            return optimal_scale
            
        except Exception as e:
            logger.warning(f"计算最优缩放比例失败，使用默认值1.0: {e}")
            return 1.0

    def get_current_system_scale_info(self) -> dict:
        """
        获取当前系统的缩放信息，用作target_scale_info的默认值
        
        Returns:
            当前系统缩放信息字典 {'dpi_scale': float, 'resolution': (w, h)}
        """
        return {
            'dpi_scale': self.scale_info['dpi_scale'],
            'resolution': self.scale_info['resolution']
        }

    def compare_images(
        self,
        template_path: Union[str, Path, np.ndarray],
        target_path: Union[str, Path, np.ndarray],
        method: str = "TM_CCOEFF_NORMED",
        enhance_images: bool = True,
        save_result: bool = False,
        result_path: Optional[str] = None,
        return_system_coordinates: bool = False,
        template_scale_info: Optional[dict] = None,
        target_scale_info: Optional[dict] = None,
    ) -> dict:
        """
        比较两张图片，返回匹配结果

        Args:
            template_path: 模板图片路径或numpy数组
            target_path: 目标图片路径或numpy数组
            method: 匹配方法
            enhance_images: 是否进行图像增强
            save_result: 是否保存标注结果图片
            result_path: 结果图片保存路径
            return_system_coordinates: 是否返回系统坐标（默认返回基准坐标）
            template_scale_info: 模板图像缩放信息 {'dpi_scale': float, 'resolution': (w, h)}
            target_scale_info: 目标图像缩放信息 {'dpi_scale': float, 'resolution': (w, h)}

        Returns:
            包含匹配结果的字典: {
                'confidence': float,  # 置信度
                'location': tuple,    # 匹配位置 (x, y, w, h) - 基准坐标或系统坐标
                'center': tuple,      # 匹配中心点 (x, y) - 基准坐标或系统坐标
                'found': bool,        # 是否找到匹配
                'method': str,        # 使用的匹配方法
                'scale': float,       # 匹配尺度
                'result_image_path': str  # 结果图片路径（如果保存）
            }
        """
        # 定义核心比较逻辑
        return self._compare_images_core(
            template_path,
            target_path,
            method,
            enhance_images,
            save_result,
            result_path,
            return_system_coordinates,
            template_scale_info,
            target_scale_info,
        )

    def _unified_image_matching_core(
        self,
        template: Union[str, Path, np.ndarray],
        target_image: Union[str, Path, np.ndarray],
        method: str = "TM_CCOEFF_NORMED",
        enhance_images: bool = True,
        region: Optional[Tuple[int, int, int, int]] = None,
        template_scale_info: Optional[dict] = None,
        target_scale_info: Optional[dict] = None,
    ) -> dict:
        """
        统一的图像匹配核心函数
        
        Args:
            template: 模板图像路径或numpy数组
            target_image: 目标图像路径或numpy数组
            method: 匹配方法
            enhance_images: 是否进行图像增强
            region: 搜索区域 (x, y, w, h) - 基准坐标
            template_scale_info: 模板图像的缩放信息 {'dpi_scale': float, 'resolution': (w, h)}
            target_scale_info: 目标图像的缩放信息 {'dpi_scale': float, 'resolution': (w, h)}
            
        Returns:
            包含匹配结果的字典: {
                'confidence': float,    # 置信度
                'location': tuple,      # 匹配位置 (x, y, w, h) - 基准坐标
                'center': tuple,        # 匹配中心点 (x, y) - 基准坐标
                'found': bool,          # 是否找到匹配
                'method': str,          # 使用的匹配方法
                'scale': float,         # 匹配尺度
                'template_shape': tuple,# 模板图像形状
                'target_shape': tuple,  # 目标图像形状
            }
        """
        try:
            # 加载图像
            template_img = self.load_image(template)
            target_img = self.load_image(target_image)

            if template_img is None or target_img is None:
                raise ValueError("无法加载图片文件")

            # 将目标图像调整到基准尺寸（在处理前转换）
            if self.auto_scale:
                target_img = self.adjust_target_image_to_base(target_img)

            # 如果指定了搜索区域，裁剪目标图像（region应该是基准坐标）
            region_offset = (0, 0)
            if region:
                x, y, w, h = region
                target_img = target_img[y : y + h, x : x + w]
                region_offset = (x, y)

            # 图像预处理
            if enhance_images:
                template_processed = self.preprocess_image(template_img, enhance=True)
                target_processed = self.preprocess_image(target_img, enhance=True)
            else:
                template_processed = template_img
                target_processed = target_img

            # 方法映射
            methods = {
                "TM_CCOEFF_NORMED": cv2.TM_CCOEFF_NORMED,
                "TM_CCORR_NORMED": cv2.TM_CCORR_NORMED,
                "TM_SQDIFF_NORMED": cv2.TM_SQDIFF_NORMED,
            }

            cv_method = methods.get(method, cv2.TM_CCOEFF_NORMED)

            # 智能缩放逻辑：计算精确的缩放比例
            if template_scale_info and target_scale_info:
                scale = self._calculate_optimal_scale(template_scale_info, target_scale_info)
                logger.info(f"智能缩放：计算出的精确缩放比例为 {scale:.3f}")
            else:
                scale = 1.0
                logger.info("未提供缩放信息，使用默认缩放比例 1.0")

            # 单次精确匹配
            best_confidence = 0
            best_location = None
            best_scale = scale
            best_match_loc = None

            logger.debug(f"开始图像匹配，方法: {method}, 缩放: {scale:.3f}, 增强: {enhance_images}")
            
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
                scaled_template.shape[0] <= target_processed.shape[0]
                and scaled_template.shape[1] <= target_processed.shape[1]
            ):
                # 执行模板匹配
                result = cv2.matchTemplate(target_processed, scaled_template, cv_method)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

                # 根据方法选择合适的值和位置
                if cv_method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                    best_confidence = 1 - min_val
                    match_loc = min_loc
                else:
                    best_confidence = max_val
                    match_loc = max_loc

                logger.debug(f"缩放 {scale:.3f}, 置信度: {best_confidence:.3f}")

                # 记录匹配结果
                h, w = scaled_template.shape[:2]
                # 计算在原始目标图像中的位置（加上region偏移）
                location_x = match_loc[0] + region_offset[0]
                location_y = match_loc[1] + region_offset[1]
                best_location = (location_x, location_y, w, h)
                best_match_loc = match_loc
            else:
                logger.warning(f"缩放后的模板图像 {scaled_template.shape[:2]} 大于目标图像 {target_processed.shape[:2]}")

            # 计算中心点
            best_center = None
            if best_location:
                best_center = (
                    best_location[0] + best_location[2] // 2,
                    best_location[1] + best_location[3] // 2,
                )

            # 准备返回结果
            result_dict = {
                "confidence": best_confidence,
                "location": best_location,  # 基准坐标
                "center": best_center,      # 基准中心点
                "found": best_confidence >= self.confidence,
                "method": method,
                "scale": best_scale,
                "template_shape": template_img.shape,
                "target_shape": target_img.shape,
            }

            logger.debug(
                f"图像匹配完成 - 置信度: {best_confidence:.3f}, 找到: {result_dict['found']}"
            )

            return result_dict

        except Exception as e:
            logger.error(f"图像匹配失败: {e}")
            return {
                "confidence": 0.0,
                "location": None,
                "center": None,
                "found": False,
                "method": method,
                "scale": 1.0,
                "template_shape": None,
                "target_shape": None,
                "error": str(e),
            }

    def _compare_images_core(
        self,
        template_path: Union[str, Path, np.ndarray],
        target_path: Union[str, Path, np.ndarray],
        method: str,
        enhance_images: bool,
        save_result: bool,
        result_path: Optional[str],
        return_system_coordinates: bool,
        template_scale_info: Optional[dict] = None,
        target_scale_info: Optional[dict] = None,
    ) -> dict:
        """
        compare_images 的核心实现逻辑 - 使用统一的核心匹配函数
        """
        # 使用统一的核心匹配函数进行图像匹配
        match_result = self._unified_image_matching_core(
            template=template_path,
            target_image=target_path,
            method=method,
            enhance_images=enhance_images,
            region=None,  # compare_images 不使用区域限制
            template_scale_info=template_scale_info,
            target_scale_info=target_scale_info,
        )

        # 检查匹配是否成功
        if "error" in match_result:
            raise ValueError(f"图像匹配失败: {match_result['error']}")

        # 获取基准坐标
        base_location = match_result["location"]
        base_center = match_result["center"]
        best_confidence = match_result["confidence"]
        best_scale = match_result["scale"]

        # 计算系统坐标
        system_location = None
        system_center = None
        if base_location:
            # 如果需要系统坐标，进行转换
            if return_system_coordinates:
                system_location = self.adjust_coordinates_to_scale(base_location)
                system_center = (
                    system_location[0] + system_location[2] // 2,
                    system_location[1] + system_location[3] // 2,
                )

        # 根据参数决定返回的坐标类型
        output_location = (
            system_location if return_system_coordinates else base_location
        )
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
            "found": match_result["found"],
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

            # 获取原始目标图像用于绘制（不进行基准转换）
            if isinstance(target_path, np.ndarray):
                # 如果输入是numpy数组，直接使用
                original_target = target_path.copy()
            else:
                # 如果输入是路径，重新加载原始图像
                original_target = self.load_template(target_path)
            result_image = original_target.copy()

            # 使用系统坐标进行绘制
            draw_location = (
                system_location
                if system_location
                else self.adjust_coordinates_to_scale(base_location)
            )
            draw_center = (
                system_center
                if system_center
                else (
                    draw_location[0] + draw_location[2] // 2,
                    draw_location[1] + draw_location[3] // 2,
                )
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
        template_path: Union[str, Path, np.ndarray],
        target_path: Union[str, Path, np.ndarray],
        methods: list = None,
        save_results: bool = False,
        max_retries: Optional[int] = None,
        retry_delay: Optional[float] = None,
    ) -> dict:
        """
        使用多种方法比较图片，返回最佳匹配结果

        Args:
            template_path: 模板图片路径或numpy数组
            target_path: 目标图片路径或numpy数组
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
        template_path: Union[str, Path, np.ndarray],
        target_path: Union[str, Path, np.ndarray],
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
        enhance_images: bool = True,
        try_multiple_methods: bool = False,
        max_retries: Optional[int] = None,
        retry_delay: Optional[float] = None,
        template_scale_info: Optional[dict] = None,
        target_scale_info: Optional[dict] = None,
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        在目标图像中查找模板图像

        Args:
            template: 模板图像路径或numpy数组
            target_image: 目标图像路径或numpy数组
            region: 搜索区域 (left, top, width, height)
            method: 匹配方法 ("TM_CCOEFF_NORMED", "TM_CCORR_NORMED", "TM_SQDIFF_NORMED")
            enhance_images: 是否进行图像增强处理
            try_multiple_methods: 是否尝试多种方法和参数组合（高级匹配）
            max_retries: 最大重试次数（None时使用默认值）
            retry_delay: 重试延迟时间（None时使用默认值）
            template_scale_info: 模板图像缩放信息 {'dpi_scale': float, 'resolution': (w, h)}
            target_scale_info: 目标图像缩放信息 {'dpi_scale': float, 'resolution': (w, h)}

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
            if try_multiple_methods:
                return self._find_image_multiple_methods_core(
                    template, target_image, region, template_scale_info, target_scale_info
                )
            else:
                return self._find_image_in_target_core(
                    template, target_image, region, method, enhance_images, template_scale_info, target_scale_info
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
        return last_exception

    def _find_image_in_target_core(
        self,
        template: Union[str, Path, np.ndarray],
        target_image: Union[str, Path, np.ndarray],
        region: Optional[Tuple[int, int, int, int]],
        method: str,
        enhance_images: bool,
        template_scale_info: Optional[dict] = None,
        target_scale_info: Optional[dict] = None,
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        find_image_in_target 的核心实现逻辑 - 使用统一的核心匹配函数
        """
        try:
            # 使用统一的核心匹配函数进行图像匹配
            match_result = self._unified_image_matching_core(
                template=template,
                target_image=target_image,
                method=method,
                enhance_images=enhance_images,
                region=region,
                template_scale_info=template_scale_info,
                target_scale_info=target_scale_info,
            )

            # 检查匹配是否成功
            if "error" in match_result:
                logger.error(f"图像匹配失败: {match_result['error']}")
                return None

            best_confidence = match_result["confidence"]
            best_location = match_result["location"]
            best_scale = match_result["scale"]

            logger.debug(
                f"最佳匹配 - 置信度: {best_confidence:.3f}, 尺度: {best_scale:.1f}"
            )

            if match_result["found"]:
                logger.info(
                    f"找到图像，置信度: {best_confidence:.3f}, 位置: {best_location}, 尺度: {best_scale:.1f}"
                )
                return best_location

            print(f"未找到图像，最高置信度: {best_confidence:.3f}")
            logger.debug(f"未找到图像，最高置信度: {best_confidence:.3f}")
            return None

        except Exception as e:
            logger.error(f"查找图像失败: {e}")
            return None

    def _find_image_multiple_methods_core(
        self,
        template: Union[str, Path, np.ndarray],
        target_image: Union[str, Path, np.ndarray],
        region: Optional[Tuple[int, int, int, int]],
        template_scale_info: Optional[dict] = None,
        target_scale_info: Optional[dict] = None,
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        多方法图像匹配的核心实现逻辑 - 使用统一的核心匹配函数
        """
        methods_to_try = ["TM_CCOEFF_NORMED", "TM_CCORR_NORMED"]
        enhancement_options = [True, False]

        best_result = None
        best_confidence = 0
        best_method = None
        best_enhance = None

        logger.debug("开始多方法匹配尝试")

        for method in methods_to_try:
            for enhance in enhancement_options:
                try:
                    # 使用统一的核心匹配函数
                    match_result = self._unified_image_matching_core(
                        template=template,
                        target_image=target_image,
                        method=method,
                        enhance_images=enhance,
                        region=region,
                        template_scale_info=template_scale_info,
                        target_scale_info=target_scale_info,
                    )

                    if match_result["found"]:
                        result = match_result["location"]
                        confidence = match_result["confidence"]
                        
                        # 使用实际置信度进行比较，同时考虑方法优先级
                        # 优先级：CCOEFF_NORMED > CCORR_NORMED，增强 > 不增强
                        priority_bonus = 0
                        if method == "TM_CCOEFF_NORMED":
                            priority_bonus += 0.1
                        if enhance:
                            priority_bonus += 0.05
                            
                        adjusted_confidence = confidence + priority_bonus

                        if adjusted_confidence > best_confidence or best_result is None:
                            best_confidence = adjusted_confidence
                            best_result = result
                            best_method = method
                            best_enhance = enhance
                            logger.debug(
                                f"找到匹配 - 方法: {method}, 增强: {enhance}, 置信度: {confidence:.3f}, 位置: {result}"
                            )

                        # 如果用最优配置找到了结果，直接返回
                        if method == "TM_CCOEFF_NORMED" and enhance:
                            logger.info(
                                f"使用最优配置找到匹配: {method}, 增强: {enhance}, 置信度: {confidence:.3f}"
                            )
                            return result

                except Exception as e:
                    logger.debug(f"方法 {method} (增强={enhance}) 失败: {e}")
                    continue

        if best_result:
            logger.info(
                f"多方法匹配成功 - 最佳方法: {best_method}, 增强: {best_enhance}, 置信度: {best_confidence:.3f}"
            )

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
        return self.find_image_in_target(
            template,
            target_image,
            region=region,
            method=self.default_method,
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
            base_location = self.find_template_in_target(template, target_image, region=region)
            logger.info(base_location)
            if not base_location:
                logger.error("点击失败：未找到目标图像")
                return False

            # 计算基准中心点
            base_center_x, base_center_y = self.get_image_center(base_location)

            # 应用基准偏移量
            base_click_x = base_center_x + offset[0]
            base_click_y = base_center_y + offset[1]

            # 转换为系统坐标进行实际点击
            system_location = self.adjust_coordinates_to_scale(
                (base_click_x, base_click_y, 1, 1)
            )
            click_x, click_y = system_location[0], system_location[1]

            pyautogui.click(click_x, click_y, button=button)
            logger.info(
                f"点击图像成功，基准位置: ({base_click_x}, {base_click_y}), 系统位置: ({click_x}, {click_y})"
            )
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
            base_location = self.find_template_in_target(template, target_image, region=region)
            if not base_location:
                logger.error("双击失败：未找到目标图像")
                return False

            # 计算基准中心点
            base_center_x, base_center_y = self.get_image_center(base_location)

            # 应用基准偏移量
            base_click_x = base_center_x + offset[0]
            base_click_y = base_center_y + offset[1]

            # 转换为系统坐标进行实际双击
            system_location = self.adjust_coordinates_to_scale(
                (base_click_x, base_click_y, 1, 1)
            )
            click_x, click_y = system_location[0], system_location[1]

            pyautogui.doubleClick(click_x, click_y)
            logger.info(
                f"双击图像成功，基准位置: ({base_click_x}, {base_click_y}), 系统位置: ({click_x}, {click_y})"
            )
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
            from_base_location = self.find_template_in_target(from_template, target_image)
            if not from_base_location:
                logger.error("拖拽失败：未找到起始图像")
                return False

            from_base_x, from_base_y = self.get_image_center(from_base_location)

            # 找到目标位置（基准坐标）
            to_base_location = self.find_template_in_target(to_template, target_image)
            if not to_base_location:
                logger.error("拖拽失败：未找到目标图像")
                return False

            to_base_x, to_base_y = self.get_image_center(to_base_location)

            # 转换为系统坐标进行实际拖拽
            from_system = self.adjust_coordinates_to_scale(
                (from_base_x, from_base_y, 1, 1)
            )
            to_system = self.adjust_coordinates_to_scale((to_base_x, to_base_y, 1, 1))

            from_x, from_y = from_system[0], from_system[1]
            to_x, to_y = to_system[0], to_system[1]

            # 执行拖拽
            pyautogui.drag(
                to_x - from_x, to_y - from_y, duration=duration, button="left"
            )
            logger.info(
                f"拖拽成功：基准坐标从 ({from_base_x}, {from_base_y}) 到 ({to_base_x}, {to_base_y})"
            )
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
            base_location = self.find_template_in_target(template, target_image)
            if not base_location:
                logger.error("鼠标移动失败：未找到目标图像")
                return False

            # 计算基准中心点
            base_center_x, base_center_y = self.get_image_center(base_location)

            # 应用基准偏移量
            base_move_x = base_center_x + offset[0]
            base_move_y = base_center_y + offset[1]

            # 转换为系统坐标进行实际移动
            system_location = self.adjust_coordinates_to_scale(
                (base_move_x, base_move_y, 1, 1)
            )
            move_x, move_y = system_location[0], system_location[1]

            pyautogui.moveTo(move_x, move_y, duration=duration)
            logger.info(
                f"鼠标移动成功，基准位置: ({base_move_x}, {base_move_y}), 系统位置: ({move_x}, {move_y})"
            )
            return True
        except Exception as e:
            logger.error(f"鼠标移动失败：{e}")
            return False


