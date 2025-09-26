"""
图像格式工具模块
提供图像格式检测、转换、验证等功能
"""

import cv2
import numpy as np
from typing import Union, Tuple, Dict, Any
import logging

from ..core.exceptions import ImageProcessingError

logger = logging.getLogger(__name__)


class FormatUtils:
    """图像格式处理工具类"""
    
    @staticmethod
    def detect_image_format(image: np.ndarray) -> str:
        """
        检测图像格式
        
        Args:
            image: 输入图像数组
            
        Returns:
            str: 图像格式 ("bgr", "rgb", "gray", "unknown")
        """
        try:
            if len(image.shape) == 2:
                return "gray"
            
            if len(image.shape) != 3 or image.shape[2] not in [3, 4]:
                return "unknown"
            
            # 通过颜色通道特征判断格式
            # BGR和RGB在颜色分布上有细微差异
            # 这里使用简单的启发式方法
            
            # 检查蓝色和红色通道的分布
            blue_channel = image[:, :, 0]
            green_channel = image[:, :, 1]
            red_channel = image[:, :, 2]
            
            # 计算各通道的均值和方差
            blue_mean = np.mean(blue_channel)
            green_mean = np.mean(green_channel)
            red_mean = np.mean(red_channel)
            
            # 在自然图像中，绿色通道通常有最高的均值
            # 如果蓝色通道均值最高，可能是BGR格式
            # 如果红色通道均值最高，可能是RGB格式
            if blue_mean > red_mean and blue_mean > green_mean:
                return "bgr"
            elif red_mean > blue_mean and red_mean > green_mean:
                return "rgb"
            else:
                # 如果无法确定，默认返回BGR（OpenCV默认格式）
                return "bgr"
                
        except Exception as e:
            logger.warning(f"Failed to detect image format: {e}")
            return "unknown"
    
    @staticmethod
    def convert_rgb_to_bgr(image: np.ndarray) -> np.ndarray:
        """
        将RGB格式转换为BGR格式
        
        Args:
            image: RGB格式的图像
            
        Returns:
            np.ndarray: BGR格式的图像
        """
        try:
            if len(image.shape) == 2:
                # 灰度图像不需要转换
                return image.copy()
            
            if len(image.shape) != 3 or image.shape[2] not in [3, 4]:
                raise ImageProcessingError(f"Invalid image shape for RGB to BGR conversion: {image.shape}")
            
            # 交换红色和蓝色通道
            bgr_image = image.copy()
            bgr_image[:, :, 0] = image[:, :, 2]  # B = R
            bgr_image[:, :, 2] = image[:, :, 0]  # R = B
            # G通道保持不变
            
            logger.debug("Image converted from RGB to BGR")
            return bgr_image
            
        except Exception as e:
            raise ImageProcessingError(f"Failed to convert RGB to BGR: {e}")
    
    @staticmethod
    def convert_bgr_to_rgb(image: np.ndarray) -> np.ndarray:
        """
        将BGR格式转换为RGB格式
        
        Args:
            image: BGR格式的图像
            
        Returns:
            np.ndarray: RGB格式的图像
        """
        try:
            if len(image.shape) == 2:
                # 灰度图像不需要转换
                return image.copy()
            
            if len(image.shape) != 3 or image.shape[2] not in [3, 4]:
                raise ImageProcessingError(f"Invalid image shape for BGR to RGB conversion: {image.shape}")
            
            # 使用OpenCV的转换函数
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            logger.debug("Image converted from BGR to RGB")
            return rgb_image
            
        except Exception as e:
            raise ImageProcessingError(f"Failed to convert BGR to RGB: {e}")
    
    @staticmethod
    def ensure_bgr_format(image: np.ndarray) -> np.ndarray:
        """
        确保图像为BGR格式
        
        Args:
            image: 输入图像
            
        Returns:
            np.ndarray: BGR格式的图像
        """
        try:
            current_format = FormatUtils.detect_image_format(image)
            
            if current_format == "bgr":
                return image.copy()
            elif current_format == "rgb":
                return FormatUtils.convert_rgb_to_bgr(image)
            elif current_format == "gray":
                # 灰度图像转换为3通道BGR
                return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                # 未知格式，尝试直接使用
                logger.warning(f"Unknown image format, using as-is: {current_format}")
                return image.copy()
                
        except Exception as e:
            raise ImageProcessingError(f"Failed to ensure BGR format: {e}")
    
    @staticmethod
    def ensure_rgb_format(image: np.ndarray) -> np.ndarray:
        """
        确保图像为RGB格式
        
        Args:
            image: 输入图像
            
        Returns:
            np.ndarray: RGB格式的图像
        """
        try:
            current_format = FormatUtils.detect_image_format(image)
            
            if current_format == "rgb":
                return image.copy()
            elif current_format == "bgr":
                return FormatUtils.convert_bgr_to_rgb(image)
            elif current_format == "gray":
                # 灰度图像转换为3通道RGB
                return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                # 未知格式，尝试直接使用
                logger.warning(f"Unknown image format, using as-is: {current_format}")
                return image.copy()
                
        except Exception as e:
            raise ImageProcessingError(f"Failed to ensure RGB format: {e}")
    
    @staticmethod
    def get_image_info(image: np.ndarray) -> Dict[str, Any]:
        """
        获取图像信息
        
        Args:
            image: 输入图像
            
        Returns:
            Dict[str, Any]: 图像信息
        """
        try:
            info = {
                "shape": image.shape,
                "dtype": str(image.dtype),
                "ndim": image.ndim,
                "size": image.size,
                "format": FormatUtils.detect_image_format(image)
            }
            
            if len(image.shape) == 3:
                info["channels"] = image.shape[2]
                info["height"] = image.shape[0]
                info["width"] = image.shape[1]
            else:
                info["channels"] = 1
                info["height"] = image.shape[0]
                info["width"] = image.shape[1]
            
            # 计算内存使用
            info["memory_bytes"] = image.nbytes
            info["memory_mb"] = round(image.nbytes / (1024 * 1024), 2)
            
            # 计算像素值范围
            info["min_value"] = int(np.min(image))
            info["max_value"] = int(np.max(image))
            info["mean_value"] = float(np.mean(image))
            info["std_value"] = float(np.std(image))
            
            logger.debug(f"Image info: {info}")
            return info
            
        except Exception as e:
            raise ImageProcessingError(f"Failed to get image info: {e}")
    
    @staticmethod
    def validate_format(image: np.ndarray, expected_format: str) -> bool:
        """
        验证图像格式是否符合预期
        
        Args:
            image: 输入图像
            expected_format: 期望的格式
            
        Returns:
            bool: 格式是否符合预期
        """
        try:
            actual_format = FormatUtils.detect_image_format(image)
            is_valid = actual_format == expected_format.lower()
            
            if not is_valid:
                logger.warning(f"Format mismatch: expected {expected_format}, got {actual_format}")
            
            return is_valid
            
        except Exception as e:
            logger.error(f"Failed to validate format: {e}")
            return False
    
    @staticmethod
    def normalize_image(image: np.ndarray, target_dtype: np.dtype = np.uint8) -> np.ndarray:
        """
        标准化图像数据类型和值范围
        
        Args:
            image: 输入图像
            target_dtype: 目标数据类型
            
        Returns:
            np.ndarray: 标准化后的图像
        """
        try:
            # 确保值在0-1范围内
            if image.dtype == np.uint8:
                normalized = image.astype(np.float32) / 255.0
            elif image.dtype == np.uint16:
                normalized = image.astype(np.float32) / 65535.0
            else:
                normalized = image.astype(np.float32)
            
            # 确保值在0-1范围内
            normalized = np.clip(normalized, 0.0, 1.0)
            
            # 转换为目标类型
            if target_dtype == np.uint8:
                result = (normalized * 255).astype(np.uint8)
            elif target_dtype == np.uint16:
                result = (normalized * 65535).astype(np.uint16)
            else:
                result = normalized.astype(target_dtype)
            
            logger.debug(f"Image normalized to {target_dtype}")
            return result
            
        except Exception as e:
            raise ImageProcessingError(f"Failed to normalize image: {e}")


# 便捷函数
def detect_image_format(image: np.ndarray) -> str:
    """便捷的图像格式检测函数"""
    return FormatUtils.detect_image_format(image)


def convert_rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    """便捷的RGB到BGR转换函数"""
    return FormatUtils.convert_rgb_to_bgr(image)


def convert_bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """便捷的BGR到RGB转换函数"""
    return FormatUtils.convert_bgr_to_rgb(image)


def ensure_bgr_format(image: np.ndarray) -> np.ndarray:
    """便捷的BGR格式确保函数"""
    return FormatUtils.ensure_bgr_format(image)


def ensure_rgb_format(image: np.ndarray) -> np.ndarray:
    """便捷的RGB格式确保函数"""
    return FormatUtils.ensure_rgb_format(image)


def get_image_info(image: np.ndarray) -> Dict[str, Any]:
    """便捷的图像信息获取函数"""
    return FormatUtils.get_image_info(image)
