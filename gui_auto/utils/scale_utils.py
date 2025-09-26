"""
缩放工具模块
提供图像缩放、坐标缩放、DPI处理等基础功能
"""

import cv2
import numpy as np
import ctypes
from typing import Union, Tuple, Dict, Any, Optional, List
import logging

from ..core.exceptions import ImageProcessingError

logger = logging.getLogger(__name__)


class ScaleUtils:
    """缩放处理工具类"""
    
    @staticmethod
    def calculate_scale_factors(
        current_resolution: Tuple[int, int],
        current_dpi: float,
        base_resolution: Tuple[int, int] = (1920, 1080),
        base_dpi: float = 96.0
    ) -> Dict[str, float]:
        """
        计算缩放因子
        
        Args:
            current_resolution: 当前分辨率 (width, height)
            current_dpi: 当前DPI
            base_resolution: 基准分辨率 (width, height)
            base_dpi: 基准DPI
            
        Returns:
            Dict[str, float]: 缩放因子字典
        """
        try:
            # DPI缩放因子
            dpi_factor = current_dpi / base_dpi
            
            # 分辨率缩放因子
            res_factor_x = current_resolution[0] / base_resolution[0]
            res_factor_y = current_resolution[1] / base_resolution[1]
            res_factor = (res_factor_x + res_factor_y) / 2.0
            
            # 综合缩放因子
            total_factor = dpi_factor * res_factor
            
            scale_factors = {
                "dpi_factor": dpi_factor,
                "res_factor_x": res_factor_x,
                "res_factor_y": res_factor_y,
                "res_factor": res_factor,
                "total_factor": total_factor,
                "current_resolution": current_resolution,
                "current_dpi": current_dpi,
                "base_resolution": base_resolution,
                "base_dpi": base_dpi
            }
            
            logger.debug(f"Scale factors calculated: {scale_factors}")
            return scale_factors
            
        except Exception as e:
            raise ImageProcessingError(f"Failed to calculate scale factors: {e}")
    
    @staticmethod
    def apply_scale(
        image: np.ndarray,
        scale_factor: Union[float, Tuple[float, float]],
        interpolation: int = cv2.INTER_LINEAR
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        应用缩放到图像
        
        Args:
            image: 输入图像
            scale_factor: 缩放因子（单个值或(x, y)元组）
            interpolation: 插值方法
            
        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: 缩放后的图像和缩放信息
        """
        try:
            original_shape = image.shape
            
            if isinstance(scale_factor, (int, float)):
                scale_x = scale_y = scale_factor
            else:
                scale_x, scale_y = scale_factor
            
            # 计算新尺寸
            new_width = int(round(original_shape[1] * scale_x))
            new_height = int(round(original_shape[0] * scale_y))
            
            # 应用缩放
            scaled_image = cv2.resize(
                image, 
                (new_width, new_height), 
                interpolation=interpolation
            )
            
            scale_info = {
                "original_shape": original_shape,
                "scaled_shape": scaled_image.shape,
                "scale_x": scale_x,
                "scale_y": scale_y,
                "interpolation": interpolation
            }
            
            logger.debug(f"Image scaled from {original_shape} to {scaled_image.shape}")
            return scaled_image, scale_info
            
        except Exception as e:
            raise ImageProcessingError(f"Failed to apply scale: {e}")
    
    @staticmethod
    def get_system_scale() -> float:
        """
        获取系统DPI缩放比例
        
        Returns:
            float: 系统缩放比例
        """
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
            
            logger.debug(f"System scale detected: {scale}")
            return scale
            
        except Exception as e:
            logger.warning(f"Failed to get system scale, using default 1.0: {e}")
            return 1.0
    
    @staticmethod
    def get_resolution_scale(
        current_resolution: Tuple[int, int],
        base_resolution: Tuple[int, int] = (1920, 1080)
    ) -> Tuple[float, float]:
        """
        获取分辨率缩放比例
        
        Args:
            current_resolution: 当前分辨率 (width, height)
            base_resolution: 基准分辨率 (width, height)
            
        Returns:
            Tuple[float, float]: (x_scale, y_scale)
        """
        try:
            scale_x = current_resolution[0] / base_resolution[0]
            scale_y = current_resolution[1] / base_resolution[1]
            
            logger.debug(f"Resolution scale: x={scale_x:.3f}, y={scale_y:.3f}")
            return (scale_x, scale_y)
            
        except Exception as e:
            raise ImageProcessingError(f"Failed to get resolution scale: {e}")
    
    @staticmethod
    def normalize_coordinates(
        coordinates: Union[Tuple[int, int], List[Tuple[int, int]]],
        scale_factors: Dict[str, float],
        from_system: str = "system",
        to_system: str = "base"
    ) -> Union[Tuple[int, int], List[Tuple[int, int]]]:
        """
        标准化坐标到基准系统
        
        Args:
            coordinates: 要标准化的坐标
            scale_factors: 缩放因子字典
            from_system: 源坐标系统
            to_system: 目标坐标系统
            
        Returns:
            标准化后的坐标
        """
        try:
            # 处理单个坐标
            if isinstance(coordinates, (tuple, list)) and len(coordinates) == 2 and isinstance(coordinates[0], (int, float)):
                x, y = coordinates
                
                if from_system == "system" and to_system == "base":
                    # 从系统坐标转换到基准坐标
                    norm_x = x / scale_factors.get("dpi_factor", 1.0)
                    norm_y = y / scale_factors.get("dpi_factor", 1.0)
                elif from_system == "base" and to_system == "system":
                    # 从基准坐标转换到系统坐标
                    norm_x = x * scale_factors.get("dpi_factor", 1.0)
                    norm_y = y * scale_factors.get("dpi_factor", 1.0)
                else:
                    # 其他转换
                    norm_x, norm_y = x, y
                
                return (int(round(norm_x)), int(round(norm_y)))
            
            # 处理坐标列表
            if isinstance(coordinates, list):
                return [
                    ScaleUtils.normalize_coordinates(coord, scale_factors, from_system, to_system)
                    for coord in coordinates
                ]
            
            raise ImageProcessingError(f"Invalid coordinates format: {type(coordinates)}")
            
        except Exception as e:
            raise ImageProcessingError(f"Failed to normalize coordinates: {e}")
    
    @staticmethod
    def scale_image_to_template(
        target_image: np.ndarray,
        template_image: np.ndarray,
        scale_factors: Dict[str, float]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        将目标图像缩放到模板图像的尺寸
        
        Args:
            target_image: 目标图像
            template_image: 模板图像
            scale_factors: 缩放因子
            
        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: 缩放后的图像和缩放信息
        """
        try:
            template_height, template_width = template_image.shape[:2]
            target_height, target_width = target_image.shape[:2]
            
            # 计算缩放比例
            scale_x = template_width / target_width
            scale_y = template_height / target_height
            
            # 应用缩放
            scaled_target, scale_info = ScaleUtils.apply_scale(
                target_image, (scale_x, scale_y)
            )
            
            # 更新缩放信息
            scale_info.update({
                "template_size": (template_width, template_height),
                "target_size": (target_width, target_height),
                "scale_to_template": True
            })
            
            logger.debug(f"Target image scaled to template size: {scaled_target.shape}")
            return scaled_target, scale_info
            
        except Exception as e:
            raise ImageProcessingError(f"Failed to scale image to template: {e}")
    
    @staticmethod
    def scale_template_to_target(
        template_image: np.ndarray,
        target_image: np.ndarray,
        scale_factors: Dict[str, float]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        将模板图像缩放到目标图像的尺寸
        
        Args:
            template_image: 模板图像
            target_image: 目标图像
            scale_factors: 缩放因子
            
        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: 缩放后的图像和缩放信息
        """
        try:
            template_height, template_width = template_image.shape[:2]
            target_height, target_width = target_image.shape[:2]
            
            # 计算缩放比例
            scale_x = target_width / template_width
            scale_y = target_height / template_height
            
            # 应用缩放
            scaled_template, scale_info = ScaleUtils.apply_scale(
                template_image, (scale_x, scale_y)
            )
            
            # 更新缩放信息
            scale_info.update({
                "template_size": (template_width, template_height),
                "target_size": (target_width, target_height),
                "scale_to_target": True
            })
            
            logger.debug(f"Template image scaled to target size: {scaled_template.shape}")
            return scaled_template, scale_info
            
        except Exception as e:
            raise ImageProcessingError(f"Failed to scale template to target: {e}")
    
    @staticmethod
    def calculate_optimal_scale(
        image1: np.ndarray,
        image2: np.ndarray,
        max_scale: float = 2.0,
        min_scale: float = 0.5
    ) -> float:
        """
        计算两个图像之间的最优缩放比例
        
        Args:
            image1: 第一个图像
            image2: 第二个图像
            max_scale: 最大缩放比例
            min_scale: 最小缩放比例
            
        Returns:
            float: 最优缩放比例
        """
        try:
            h1, w1 = image1.shape[:2]
            h2, w2 = image2.shape[:2]
            
            # 计算宽高比
            aspect1 = w1 / h1
            aspect2 = w2 / h2
            
            # 计算缩放比例
            scale_by_width = w2 / w1
            scale_by_height = h2 / h1
            
            # 选择更保守的缩放比例
            optimal_scale = min(scale_by_width, scale_by_height)
            
            # 限制在指定范围内
            optimal_scale = max(min_scale, min(max_scale, optimal_scale))
            
            logger.debug(f"Optimal scale calculated: {optimal_scale:.3f}")
            return optimal_scale
            
        except Exception as e:
            raise ImageProcessingError(f"Failed to calculate optimal scale: {e}")
    
    @staticmethod
    def create_scale_pyramid(
        image: np.ndarray,
        levels: int = 4,
        scale_factor: float = 0.5
    ) -> List[np.ndarray]:
        """
        创建图像缩放金字塔
        
        Args:
            image: 输入图像
            levels: 金字塔层数
            scale_factor: 每层缩放因子
            
        Returns:
            List[np.ndarray]: 金字塔图像列表
        """
        try:
            pyramid = [image.copy()]
            current_image = image.copy()
            
            for i in range(1, levels):
                # 计算当前层的尺寸
                height, width = current_image.shape[:2]
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                
                # 缩放图像
                scaled = cv2.resize(
                    current_image, 
                    (new_width, new_height), 
                    interpolation=cv2.INTER_LINEAR
                )
                
                pyramid.append(scaled)
                current_image = scaled
                
                # 如果图像太小，停止
                if new_width < 10 or new_height < 10:
                    break
            
            logger.debug(f"Scale pyramid created with {len(pyramid)} levels")
            return pyramid
            
        except Exception as e:
            raise ImageProcessingError(f"Failed to create scale pyramid: {e}")
    
    @staticmethod
    def create_scale_record(
        template_image: np.ndarray,
        target_image: np.ndarray,
        scale_factors: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        创建缩放记录，用于后续坐标转换
        
        Args:
            template_image: 模板图像
            target_image: 目标图像
            scale_factors: 缩放因子
            
        Returns:
            Dict[str, Any]: 缩放记录
        """
        try:
            template_h, template_w = template_image.shape[:2]
            target_h, target_w = target_image.shape[:2]
            
            # 计算基础缩放因子
            scale_x = template_w / target_w
            scale_y = template_h / target_h
            
            # 创建缩放记录
            scale_record = {
                "template_size": (template_w, template_h),
                "target_size": (target_w, target_h),
                "scale_x": scale_x,
                "scale_y": scale_y,
                "scale_factor": (scale_x + scale_y) / 2.0,
                "scale_factors": scale_factors or {},
                "timestamp": np.datetime64('now'),
                "coordinate_mapping": {
                    "original_size": (target_h, target_w),
                    "scaled_size": (template_h, template_w),
                    "scale_x": scale_x,
                    "scale_y": scale_y,
                    "inverse_scale_x": 1.0 / scale_x,
                    "inverse_scale_y": 1.0 / scale_y
                }
            }
            
            logger.debug(f"Scale record created: {scale_record}")
            return scale_record
            
        except Exception as e:
            raise ImageProcessingError(f"Failed to create scale record: {e}")
    
    @staticmethod
    def convert_coordinates_with_scale_record(
        coordinates: Union[Tuple[int, int], List[Tuple[int, int]]],
        scale_record: Dict[str, Any],
        direction: str = "to_scaled"
    ) -> Union[Tuple[int, int], List[Tuple[int, int]]]:
        """
        使用缩放记录转换坐标
        
        Args:
            coordinates: 要转换的坐标
            scale_record: 缩放记录
            direction: 转换方向 ("to_scaled" 或 "to_original")
            
        Returns:
            转换后的坐标
        """
        try:
            mapping = scale_record.get("coordinate_mapping", {})
            
            if direction == "to_scaled":
                scale_x = mapping.get("scale_x", 1.0)
                scale_y = mapping.get("scale_y", 1.0)
            else:  # to_original
                scale_x = mapping.get("inverse_scale_x", 1.0)
                scale_y = mapping.get("inverse_scale_y", 1.0)
            
            # 处理单个坐标
            if isinstance(coordinates, (tuple, list)) and len(coordinates) == 2 and isinstance(coordinates[0], (int, float)):
                x, y = coordinates
                new_x = int(round(x * scale_x))
                new_y = int(round(y * scale_y))
                return (new_x, new_y)
            
            # 处理坐标列表
            if isinstance(coordinates, list):
                return [
                    (int(round(x * scale_x)), int(round(y * scale_y)))
                    for x, y in coordinates
                ]
            
            return coordinates
            
        except Exception as e:
            raise ImageProcessingError(f"Failed to convert coordinates with scale record: {e}")
    
    @staticmethod
    def validate_scale_record(scale_record: Dict[str, Any]) -> bool:
        """
        验证缩放记录的有效性
        
        Args:
            scale_record: 缩放记录
            
        Returns:
            bool: 记录是否有效
        """
        try:
            required_keys = [
                "template_size", "target_size", "scale_x", "scale_y",
                "coordinate_mapping"
            ]
            
            for key in required_keys:
                if key not in scale_record:
                    logger.warning(f"Missing required key in scale record: {key}")
                    return False
            
            # 验证坐标映射
            mapping = scale_record.get("coordinate_mapping", {})
            mapping_keys = [
                "original_size", "scaled_size", "scale_x", "scale_y",
                "inverse_scale_x", "inverse_scale_y"
            ]
            
            for key in mapping_keys:
                if key not in mapping:
                    logger.warning(f"Missing required key in coordinate mapping: {key}")
                    return False
            
            # 验证缩放因子的一致性
            scale_x = scale_record.get("scale_x", 1.0)
            scale_y = scale_record.get("scale_y", 1.0)
            inv_scale_x = mapping.get("inverse_scale_x", 1.0)
            inv_scale_y = mapping.get("inverse_scale_y", 1.0)
            
            if abs(scale_x * inv_scale_x - 1.0) > 0.001:
                logger.warning("Scale factor inconsistency detected")
                return False
            
            if abs(scale_y * inv_scale_y - 1.0) > 0.001:
                logger.warning("Scale factor inconsistency detected")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to validate scale record: {e}")
            return False
    
    @staticmethod
    def get_optimal_scale_for_matching(
        template: np.ndarray,
        target: np.ndarray,
        max_scale: float = 2.0,
        min_scale: float = 0.5
    ) -> Tuple[float, Dict[str, Any]]:
        """
        获取用于匹配的最优缩放比例
        
        Args:
            template: 模板图像
            target: 目标图像
            max_scale: 最大缩放比例
            min_scale: 最小缩放比例
            
        Returns:
            Tuple[float, Dict[str, Any]]: 最优缩放比例和缩放信息
        """
        try:
            template_h, template_w = template.shape[:2]
            target_h, target_w = target.shape[:2]
            
            # 计算多种缩放策略
            strategies = {
                "match_width": target_w / template_w,
                "match_height": target_h / template_h,
                "match_area": np.sqrt((target_w * target_h) / (template_w * template_h)),
                "match_diagonal": np.sqrt((target_w**2 + target_h**2) / (template_w**2 + template_h**2))
            }
            
            # 选择最合适的策略
            best_strategy = "match_area"  # 默认使用面积匹配
            best_scale = strategies[best_strategy]
            
            # 限制在指定范围内
            best_scale = max(min_scale, min(max_scale, best_scale))
            
            scale_info = {
                "strategies": strategies,
                "best_strategy": best_strategy,
                "best_scale": best_scale,
                "template_size": (template_w, template_h),
                "target_size": (target_w, target_h)
            }
            
            logger.debug(f"Optimal scale for matching: {best_scale:.3f} (strategy: {best_strategy})")
            return best_scale, scale_info
            
        except Exception as e:
            raise ImageProcessingError(f"Failed to get optimal scale for matching: {e}")
    
    @staticmethod
    def create_multi_scale_pyramid(
        image: np.ndarray,
        scales: List[float],
        interpolation: int = cv2.INTER_LINEAR
    ) -> List[Tuple[float, np.ndarray]]:
        """
        创建多尺度金字塔
        
        Args:
            image: 输入图像
            scales: 缩放比例列表
            interpolation: 插值方法
            
        Returns:
            List[Tuple[float, np.ndarray]]: (缩放比例, 图像) 元组列表
        """
        try:
            pyramid = []
            
            for scale in scales:
                if scale <= 0:
                    continue
                
                scaled_image, _ = ScaleUtils.apply_scale(image, scale, interpolation)
                pyramid.append((scale, scaled_image))
            
            logger.debug(f"Multi-scale pyramid created with {len(pyramid)} scales")
            return pyramid
            
        except Exception as e:
            raise ImageProcessingError(f"Failed to create multi-scale pyramid: {e}")


# 便捷函数
def calculate_scale_factors(current_resolution, current_dpi, **kwargs):
    """便捷的缩放因子计算函数"""
    return ScaleUtils.calculate_scale_factors(current_resolution, current_dpi, **kwargs)


def apply_scale(image, scale_factor, **kwargs):
    """便捷的图像缩放函数"""
    return ScaleUtils.apply_scale(image, scale_factor, **kwargs)


def get_system_scale():
    """便捷的系统缩放获取函数"""
    return ScaleUtils.get_system_scale()


def get_resolution_scale(current_resolution, base_resolution=(1920, 1080)):
    """便捷的分辨率缩放获取函数"""
    return ScaleUtils.get_resolution_scale(current_resolution, base_resolution)


def normalize_coordinates(coordinates, scale_factors, **kwargs):
    """便捷的坐标标准化函数"""
    return ScaleUtils.normalize_coordinates(coordinates, scale_factors, **kwargs)
