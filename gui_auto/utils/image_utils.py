"""
图像工具模块
提供图像加载、保存、格式转换、质量分析、增强等基础功能
"""

import cv2
import numpy as np
from PIL import Image, ImageGrab
from pathlib import Path
from typing import Union, Tuple, Optional, Dict, Any
import logging

from ..core.exceptions import ImageProcessingError
from .format_utils import FormatUtils

logger = logging.getLogger(__name__)


class ImageUtils:
    """图像处理工具类"""
    
    @staticmethod
    def load_image(
        image_source: Union[str, Path, np.ndarray],
        target_format: str = "bgr",
        validate: bool = True
    ) -> np.ndarray:
        """
        加载图像
        
        Args:
            image_source: 图像源（文件路径、PIL图像或numpy数组）
            target_format: 目标格式 ("bgr" 或 "rgb")
            validate: 是否验证图像
            
        Returns:
            np.ndarray: 加载的图像数组
            
        Raises:
            ImageProcessingError: 加载失败时抛出异常
        """
        try:
            if isinstance(image_source, (str, Path)):
                # 从文件加载
                image_path = Path(image_source)
                if not image_path.exists():
                    raise ImageProcessingError(f"Image file not found: {image_path}")
                
                # 使用OpenCV加载图像
                image = cv2.imread(str(image_path))
                if image is None:
                    raise ImageProcessingError(f"Failed to load image: {image_path}")
                
                # 确保是BGR格式
                if target_format.lower() == "rgb":
                    image = FormatUtils.convert_bgr_to_rgb(image)
                    
            elif isinstance(image_source, np.ndarray):
                # 已经是numpy数组
                image = image_source.copy()
                
                # 转换格式
                if target_format.lower() == "bgr" and FormatUtils.detect_image_format(image) == "rgb":
                    image = FormatUtils.convert_rgb_to_bgr(image)
                elif target_format.lower() == "rgb" and FormatUtils.detect_image_format(image) == "bgr":
                    image = FormatUtils.convert_bgr_to_rgb(image)
                    
            elif hasattr(image_source, 'mode'):  # PIL Image
                # 从PIL图像转换
                image = np.array(image_source)
                
                # 转换格式
                if target_format.lower() == "bgr" and FormatUtils.detect_image_format(image) == "rgb":
                    image = FormatUtils.convert_rgb_to_bgr(image)
                elif target_format.lower() == "rgb" and FormatUtils.detect_image_format(image) == "bgr":
                    image = FormatUtils.convert_bgr_to_rgb(image)
            else:
                raise ImageProcessingError(f"Unsupported image source type: {type(image_source)}")
            
            # 验证图像
            if validate:
                ImageUtils.validate_image(image)
            
            logger.debug(f"Image loaded successfully, shape: {image.shape}, format: {target_format}")
            return image
            
        except Exception as e:
            raise ImageProcessingError(f"Failed to load image: {e}")
    
    @staticmethod
    def save_image(
        image: np.ndarray,
        output_path: Union[str, Path],
        quality: int = 95
    ) -> None:
        """
        保存图像
        
        Args:
            image: 要保存的图像数组
            output_path: 输出路径
            quality: 保存质量 (1-100)
            
        Raises:
            ImageProcessingError: 保存失败时抛出异常
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 确保图像格式正确
            if FormatUtils.detect_image_format(image) == "rgb":
                image = FormatUtils.convert_rgb_to_bgr(image)
            
            # 保存图像
            success = cv2.imwrite(str(output_path), image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if not success:
                raise ImageProcessingError(f"Failed to save image to {output_path}")
            
            logger.debug(f"Image saved successfully to {output_path}")
            
        except Exception as e:
            raise ImageProcessingError(f"Failed to save image: {e}")
    
    @staticmethod
    def convert_image_format(
        image: np.ndarray,
        target_format: str
    ) -> np.ndarray:
        """
        转换图像格式
        
        Args:
            image: 输入图像
            target_format: 目标格式 ("bgr" 或 "rgb")
            
        Returns:
            np.ndarray: 转换后的图像
            
        Raises:
            ImageProcessingError: 转换失败时抛出异常
        """
        try:
            current_format = FormatUtils.detect_image_format(image)
            
            if current_format == target_format.lower():
                return image.copy()
            
            if target_format.lower() == "bgr":
                return FormatUtils.convert_rgb_to_bgr(image)
            elif target_format.lower() == "rgb":
                return FormatUtils.convert_bgr_to_rgb(image)
            else:
                raise ImageProcessingError(f"Unsupported target format: {target_format}")
                
        except Exception as e:
            raise ImageProcessingError(f"Failed to convert image format: {e}")
    
    @staticmethod
    def validate_image(image: np.ndarray) -> bool:
        """
        验证图像
        
        Args:
            image: 要验证的图像
            
        Returns:
            bool: 图像是否有效
            
        Raises:
            ImageProcessingError: 图像无效时抛出异常
        """
        try:
            if not isinstance(image, np.ndarray):
                raise ImageProcessingError("Image must be a numpy array")
            
            if image.ndim not in [2, 3]:
                raise ImageProcessingError(f"Image must be 2D or 3D array, got {image.ndim}D")
            
            if image.ndim == 3 and image.shape[2] not in [1, 3, 4]:
                raise ImageProcessingError(f"Image must have 1, 3, or 4 channels, got {image.shape[2]}")
            
            if image.size == 0:
                raise ImageProcessingError("Image is empty")
            
            if not np.isfinite(image).all():
                raise ImageProcessingError("Image contains invalid values (NaN or Inf)")
            
            logger.debug(f"Image validation passed, shape: {image.shape}")
            return True
            
        except Exception as e:
            raise ImageProcessingError(f"Image validation failed: {e}")
    
    @staticmethod
    def analyze_image_quality(image: np.ndarray) -> Dict[str, Any]:
        """
        分析图像质量
        
        Args:
            image: 要分析的图像
            
        Returns:
            Dict[str, Any]: 质量分析结果
        """
        try:
            # 确保图像格式正确
            if FormatUtils.detect_image_format(image) == "rgb":
                image = FormatUtils.convert_rgb_to_bgr(image)
            
            # 转换为灰度图
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # 计算图像质量指标
            quality_info = {
                "shape": image.shape,
                "dtype": str(image.dtype),
                "channels": image.shape[2] if len(image.shape) == 3 else 1,
                "mean_brightness": float(np.mean(gray)),
                "std_brightness": float(np.std(gray)),
                "min_value": int(np.min(image)),
                "max_value": int(np.max(image)),
                "dynamic_range": int(np.max(image) - np.min(image))
            }
            
            # 计算清晰度（拉普拉斯方差）
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            quality_info["sharpness"] = float(laplacian_var)
            
            # 计算对比度
            contrast = gray.std()
            quality_info["contrast"] = float(contrast)
            
            # 质量评级
            if laplacian_var > 1000 and contrast > 50:
                quality_info["quality_rating"] = "high"
            elif laplacian_var > 500 and contrast > 25:
                quality_info["quality_rating"] = "medium"
            else:
                quality_info["quality_rating"] = "low"
            
            logger.debug(f"Image quality analysis completed: {quality_info['quality_rating']}")
            return quality_info
            
        except Exception as e:
            raise ImageProcessingError(f"Failed to analyze image quality: {e}")
    
    @staticmethod
    def enhance_image(
        image: np.ndarray,
        level: str = "light"
    ) -> np.ndarray:
        """
        增强图像
        
        Args:
            image: 输入图像
            level: 增强级别 ("light", "medium", "heavy")
            
        Returns:
            np.ndarray: 增强后的图像
        """
        try:
            # 确保图像格式正确
            if FormatUtils.detect_image_format(image) == "rgb":
                image = FormatUtils.convert_rgb_to_bgr(image)
            
            enhanced = image.copy()
            
            if level == "none":
                return enhanced
            
            # 转换为LAB色彩空间进行增强
            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # 应用CLAHE（对比度限制自适应直方图均衡化）
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # 根据级别调整参数
            if level == "light":
                alpha = 1.1  # 对比度
                beta = 10    # 亮度
            elif level == "medium":
                alpha = 1.2
                beta = 15
            elif level == "heavy":
                alpha = 1.3
                beta = 20
            else:
                alpha = 1.0
                beta = 0
            
            # 应用对比度和亮度调整
            l = cv2.convertScaleAbs(l, alpha=alpha, beta=beta)
            
            # 合并通道
            enhanced_lab = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            
            # 确保值在有效范围内
            enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
            
            logger.debug(f"Image enhanced with level: {level}")
            return enhanced
            
        except Exception as e:
            raise ImageProcessingError(f"Failed to enhance image: {e}")
    
    @staticmethod
    def resize_image(
        image: np.ndarray,
        target_size: Union[Tuple[int, int], float],
        interpolation: int = cv2.INTER_LINEAR
    ) -> np.ndarray:
        """
        调整图像大小
        
        Args:
            image: 输入图像
            target_size: 目标大小（(width, height) 或 缩放比例）
            interpolation: 插值方法
            
        Returns:
            np.ndarray: 调整后的图像
        """
        try:
            if isinstance(target_size, (int, float)):
                # 按比例缩放
                scale = target_size
                height, width = image.shape[:2]
                new_width = int(width * scale)
                new_height = int(height * scale)
            else:
                # 指定尺寸
                new_width, new_height = target_size
            
            resized = cv2.resize(image, (new_width, new_height), interpolation=interpolation)
            
            logger.debug(f"Image resized from {image.shape[:2]} to {(new_height, new_width)}")
            return resized
            
        except Exception as e:
            raise ImageProcessingError(f"Failed to resize image: {e}")
    
    @staticmethod
    def crop_image(
        image: np.ndarray,
        x: int, y: int, width: int, height: int
    ) -> np.ndarray:
        """
        裁剪图像
        
        Args:
            image: 输入图像
            x, y: 裁剪起始坐标
            width, height: 裁剪尺寸
            
        Returns:
            np.ndarray: 裁剪后的图像
        """
        try:
            # 确保坐标在有效范围内
            h, w = image.shape[:2]
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))
            width = min(width, w - x)
            height = min(height, h - y)
            
            cropped = image[y:y+height, x:x+width]
            
            logger.debug(f"Image cropped to {(height, width)} at ({x}, {y})")
            return cropped
            
        except Exception as e:
            raise ImageProcessingError(f"Failed to crop image: {e}")
    
    @staticmethod
    def capture_screen(
        region: Optional[Tuple[int, int, int, int]] = None,
        target_format: str = "bgr"
    ) -> np.ndarray:
        """
        捕获屏幕截图
        
        Args:
            region: 捕获区域 (x, y, width, height)，None表示全屏
            target_format: 目标格式 ("bgr" 或 "rgb")
            
        Returns:
            np.ndarray: 屏幕截图
        """
        try:
            if region is None:
                # 全屏截图
                screenshot = ImageGrab.grab()
            else:
                # 区域截图
                x, y, width, height = region
                screenshot = ImageGrab.grab(bbox=(x, y, x + width, y + height))
            
            # 转换为numpy数组
            image = np.array(screenshot)
            
            # 转换格式
            if target_format.lower() == "bgr":
                image = FormatUtils.convert_rgb_to_bgr(image)
            
            logger.debug(f"Screen captured, shape: {image.shape}")
            return image
            
        except Exception as e:
            raise ImageProcessingError(f"Failed to capture screen: {e}")


# 便捷函数
def load_image(image_source: Union[str, Path, np.ndarray], **kwargs) -> np.ndarray:
    """便捷的图像加载函数"""
    return ImageUtils.load_image(image_source, **kwargs)


def save_image(image: np.ndarray, output_path: Union[str, Path], **kwargs) -> None:
    """便捷的图像保存函数"""
    ImageUtils.save_image(image, output_path, **kwargs)


def convert_image_format(image: np.ndarray, target_format: str) -> np.ndarray:
    """便捷的图像格式转换函数"""
    return ImageUtils.convert_image_format(image, target_format)


def validate_image(image: np.ndarray) -> bool:
    """便捷的图像验证函数"""
    return ImageUtils.validate_image(image)


def analyze_image_quality(image: np.ndarray) -> Dict[str, Any]:
    """便捷的图像质量分析函数"""
    return ImageUtils.analyze_image_quality(image)


def enhance_image(image: np.ndarray, level: str = "light") -> np.ndarray:
    """便捷的图像增强函数"""
    return ImageUtils.enhance_image(image, level)


def resize_image(image: np.ndarray, target_size: Union[Tuple[int, int], float], **kwargs) -> np.ndarray:
    """便捷的图像大小调整函数"""
    return ImageUtils.resize_image(image, target_size, **kwargs)


def crop_image(image: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray:
    """便捷的图像裁剪函数"""
    return ImageUtils.crop_image(image, x, y, width, height)
