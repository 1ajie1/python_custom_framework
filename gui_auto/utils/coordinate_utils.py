"""
坐标工具模块
提供坐标转换、缩放、验证等基础功能
"""

import numpy as np
from typing import Union, Tuple, List, Dict, Any, Optional
import logging

from ..core.exceptions import ImageProcessingError

logger = logging.getLogger(__name__)


class CoordinateUtils:
    """坐标处理工具类"""
    
    @staticmethod
    def convert_coordinates(
        coordinates: Union[Tuple[int, int], List[Tuple[int, int]]],
        from_system: str,
        to_system: str,
        scale_factors: Optional[Dict[str, float]] = None
    ) -> Union[Tuple[int, int], List[Tuple[int, int]]]:
        """
        转换坐标系统
        
        Args:
            coordinates: 要转换的坐标（单个坐标或坐标列表）
            from_system: 源坐标系统 ("system", "base", "template", "target")
            to_system: 目标坐标系统 ("system", "base", "template", "target")
            scale_factors: 缩放因子字典
            
        Returns:
            转换后的坐标
        """
        try:
            if scale_factors is None:
                scale_factors = {}
            
            # 如果源和目标相同，直接返回
            if from_system == to_system:
                return coordinates
            
            # 处理单个坐标
            if isinstance(coordinates, (tuple, list)) and len(coordinates) == 2 and isinstance(coordinates[0], (int, float)):
                return CoordinateUtils._convert_single_coordinate(
                    coordinates, from_system, to_system, scale_factors
                )
            
            # 处理坐标列表
            if isinstance(coordinates, list):
                return [
                    CoordinateUtils._convert_single_coordinate(
                        coord, from_system, to_system, scale_factors
                    ) for coord in coordinates
                ]
            
            raise ImageProcessingError(f"Invalid coordinates format: {type(coordinates)}")
            
        except Exception as e:
            raise ImageProcessingError(f"Failed to convert coordinates: {e}")
    
    @staticmethod
    def _convert_single_coordinate(
        coord: Tuple[int, int],
        from_system: str,
        to_system: str,
        scale_factors: Dict[str, float]
    ) -> Tuple[int, int]:
        """转换单个坐标"""
        x, y = coord
        
        # 定义转换矩阵
        conversions = {
            ("system", "base"): lambda x, y: (x / scale_factors.get("dpi_factor", 1.0), y / scale_factors.get("dpi_factor", 1.0)),
            ("base", "system"): lambda x, y: (x * scale_factors.get("dpi_factor", 1.0), y * scale_factors.get("dpi_factor", 1.0)),
            ("template", "base"): lambda x, y: (x * scale_factors.get("template_scale", 1.0), y * scale_factors.get("template_scale", 1.0)),
            ("base", "template"): lambda x, y: (x / scale_factors.get("template_scale", 1.0), y / scale_factors.get("template_scale", 1.0)),
            ("target", "base"): lambda x, y: (x * scale_factors.get("target_scale", 1.0), y * scale_factors.get("target_scale", 1.0)),
            ("base", "target"): lambda x, y: (x / scale_factors.get("target_scale", 1.0), y / scale_factors.get("target_scale", 1.0)),
            ("system", "template"): lambda x, y: CoordinateUtils._convert_single_coordinate(
                CoordinateUtils._convert_single_coordinate((x, y), "system", "base", scale_factors),
                "base", "template", scale_factors
            ),
            ("template", "system"): lambda x, y: CoordinateUtils._convert_single_coordinate(
                CoordinateUtils._convert_single_coordinate((x, y), "template", "base", scale_factors),
                "base", "system", scale_factors
            ),
            ("system", "target"): lambda x, y: CoordinateUtils._convert_single_coordinate(
                CoordinateUtils._convert_single_coordinate((x, y), "system", "base", scale_factors),
                "base", "target", scale_factors
            ),
            ("target", "system"): lambda x, y: CoordinateUtils._convert_single_coordinate(
                CoordinateUtils._convert_single_coordinate((x, y), "target", "base", scale_factors),
                "base", "system", scale_factors
            ),
            ("template", "target"): lambda x, y: CoordinateUtils._convert_single_coordinate(
                CoordinateUtils._convert_single_coordinate((x, y), "template", "base", scale_factors),
                "base", "target", scale_factors
            ),
            ("target", "template"): lambda x, y: CoordinateUtils._convert_single_coordinate(
                CoordinateUtils._convert_single_coordinate((x, y), "target", "base", scale_factors),
                "base", "template", scale_factors
            )
        }
        
        conversion_key = (from_system, to_system)
        if conversion_key not in conversions:
            raise ImageProcessingError(f"Unsupported coordinate conversion: {from_system} -> {to_system}")
        
        new_x, new_y = conversions[conversion_key](x, y)
        return (int(round(new_x)), int(round(new_y)))
    
    @staticmethod
    def scale_coordinates(
        coordinates: Union[Tuple[int, int], List[Tuple[int, int]]],
        scale_x: float,
        scale_y: Optional[float] = None
    ) -> Union[Tuple[int, int], List[Tuple[int, int]]]:
        """
        缩放坐标
        
        Args:
            coordinates: 要缩放的坐标
            scale_x: X轴缩放因子
            scale_y: Y轴缩放因子，None表示与X轴相同
            
        Returns:
            缩放后的坐标
        """
        try:
            if scale_y is None:
                scale_y = scale_x
            
            # 处理单个坐标
            if isinstance(coordinates, (tuple, list)) and len(coordinates) == 2 and isinstance(coordinates[0], (int, float)):
                x, y = coordinates
                return (int(round(x * scale_x)), int(round(y * scale_y)))
            
            # 处理坐标列表
            if isinstance(coordinates, list):
                return [
                    (int(round(x * scale_x)), int(round(y * scale_y)))
                    for x, y in coordinates
                ]
            
            raise ImageProcessingError(f"Invalid coordinates format: {type(coordinates)}")
            
        except Exception as e:
            raise ImageProcessingError(f"Failed to scale coordinates: {e}")
    
    @staticmethod
    def validate_coordinates(
        coordinates: Union[Tuple[int, int], List[Tuple[int, int]]],
        bounds: Optional[Tuple[int, int, int, int]] = None
    ) -> bool:
        """
        验证坐标是否有效
        
        Args:
            coordinates: 要验证的坐标
            bounds: 边界 (x_min, y_min, x_max, y_max)，None表示不检查边界
            
        Returns:
            bool: 坐标是否有效
        """
        try:
            # 处理单个坐标
            if isinstance(coordinates, (tuple, list)) and len(coordinates) == 2 and isinstance(coordinates[0], (int, float)):
                x, y = coordinates
                
                # 检查坐标类型
                if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                    return False
                
                # 检查坐标值
                if not np.isfinite(x) or not np.isfinite(y):
                    return False
                
                # 检查边界
                if bounds is not None:
                    x_min, y_min, x_max, y_max = bounds
                    if not (x_min <= x <= x_max and y_min <= y <= y_max):
                        return False
                
                return True
            
            # 处理坐标列表
            if isinstance(coordinates, list):
                return all(
                    CoordinateUtils.validate_coordinates(coord, bounds)
                    for coord in coordinates
                )
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to validate coordinates: {e}")
            return False
    
    @staticmethod
    def get_center_point(
        coordinates: Union[Tuple[int, int], List[Tuple[int, int]]]
    ) -> Tuple[int, int]:
        """
        获取坐标的中心点
        
        Args:
            coordinates: 坐标列表
            
        Returns:
            Tuple[int, int]: 中心点坐标
        """
        try:
            if isinstance(coordinates, (tuple, list)) and len(coordinates) == 2 and isinstance(coordinates[0], (int, float)):
                # 单个坐标，直接返回
                return coordinates
            
            if isinstance(coordinates, list) and len(coordinates) > 0:
                # 坐标列表，计算中心点
                x_coords = [coord[0] for coord in coordinates]
                y_coords = [coord[1] for coord in coordinates]
                
                center_x = int(round(np.mean(x_coords)))
                center_y = int(round(np.mean(y_coords)))
                
                return (center_x, center_y)
            
            raise ImageProcessingError(f"Invalid coordinates format: {type(coordinates)}")
            
        except Exception as e:
            raise ImageProcessingError(f"Failed to get center point: {e}")
    
    @staticmethod
    def get_rectangle_bounds(
        coordinates: List[Tuple[int, int]]
    ) -> Tuple[int, int, int, int]:
        """
        获取坐标列表的矩形边界
        
        Args:
            coordinates: 坐标列表
            
        Returns:
            Tuple[int, int, int, int]: 边界 (x_min, y_min, x_max, y_max)
        """
        try:
            if not coordinates:
                raise ImageProcessingError("Empty coordinates list")
            
            x_coords = [coord[0] for coord in coordinates]
            y_coords = [coord[1] for coord in coordinates]
            
            x_min = min(x_coords)
            y_min = min(y_coords)
            x_max = max(x_coords)
            y_max = max(y_coords)
            
            return (x_min, y_min, x_max, y_max)
            
        except Exception as e:
            raise ImageProcessingError(f"Failed to get rectangle bounds: {e}")
    
    @staticmethod
    def distance_between_points(
        point1: Tuple[int, int],
        point2: Tuple[int, int]
    ) -> float:
        """
        计算两点之间的距离
        
        Args:
            point1: 第一个点
            point2: 第二个点
            
        Returns:
            float: 距离
        """
        try:
            x1, y1 = point1
            x2, y2 = point2
            
            distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            return float(distance)
            
        except Exception as e:
            raise ImageProcessingError(f"Failed to calculate distance: {e}")
    
    @staticmethod
    def is_point_in_rectangle(
        point: Tuple[int, int],
        rectangle: Tuple[int, int, int, int]
    ) -> bool:
        """
        检查点是否在矩形内
        
        Args:
            point: 要检查的点
            rectangle: 矩形 (x, y, width, height)
            
        Returns:
            bool: 点是否在矩形内
        """
        try:
            px, py = point
            rx, ry, rw, rh = rectangle
            
            return rx <= px <= rx + rw and ry <= py <= ry + rh
            
        except Exception as e:
            raise ImageProcessingError(f"Failed to check point in rectangle: {e}")
    
    @staticmethod
    def normalize_coordinates(
        coordinates: Union[Tuple[int, int], List[Tuple[int, int]]],
        image_size: Tuple[int, int]
    ) -> Union[Tuple[float, float], List[Tuple[float, float]]]:
        """
        将坐标标准化到0-1范围
        
        Args:
            coordinates: 要标准化的坐标
            image_size: 图像尺寸 (width, height)
            
        Returns:
            标准化后的坐标（0-1范围）
        """
        try:
            width, height = image_size
            
            # 处理单个坐标
            if isinstance(coordinates, (tuple, list)) and len(coordinates) == 2 and isinstance(coordinates[0], (int, float)):
                x, y = coordinates
                norm_x = x / width
                norm_y = y / height
                return (norm_x, norm_y)
            
            # 处理坐标列表
            if isinstance(coordinates, list):
                return [
                    (x / width, y / height)
                    for x, y in coordinates
                ]
            
            raise ImageProcessingError(f"Invalid coordinates format: {type(coordinates)}")
            
        except Exception as e:
            raise ImageProcessingError(f"Failed to normalize coordinates: {e}")


# 便捷函数
def convert_coordinates(coordinates, from_system, to_system, **kwargs):
    """便捷的坐标转换函数"""
    return CoordinateUtils.convert_coordinates(coordinates, from_system, to_system, **kwargs)


def scale_coordinates(coordinates, scale_x, scale_y=None):
    """便捷的坐标缩放函数"""
    return CoordinateUtils.scale_coordinates(coordinates, scale_x, scale_y)


def validate_coordinates(coordinates, bounds=None):
    """便捷的坐标验证函数"""
    return CoordinateUtils.validate_coordinates(coordinates, bounds)


def get_center_point(coordinates):
    """便捷的中心点获取函数"""
    return CoordinateUtils.get_center_point(coordinates)


def get_rectangle_bounds(coordinates):
    """便捷的矩形边界获取函数"""
    return CoordinateUtils.get_rectangle_bounds(coordinates)


# 坐标系统简化功能
class CoordinateSystemSimplifier:
    """坐标系统简化器"""
    
    @staticmethod
    def simplify_coordinate_system(
        coordinates: Union[Tuple[int, int], List[Tuple[int, int]]],
        source_system: str = "base",
        target_system: str = "system",
        scale_factors: Optional[Dict[str, float]] = None
    ) -> Union[Tuple[int, int], List[Tuple[int, int]]]:
        """
        简化坐标系统转换
        
        Args:
            coordinates: 要转换的坐标
            source_system: 源坐标系统
            target_system: 目标坐标系统
            scale_factors: 缩放因子
            
        Returns:
            转换后的坐标
        """
        try:
            # 直接使用坐标工具类进行转换
            return CoordinateUtils.convert_coordinates(
                coordinates, source_system, target_system, scale_factors or {}
            )
        except Exception as e:
            logger.error(f"Coordinate system simplification failed: {e}")
            return coordinates
    
    @staticmethod
    def get_system_coordinates_directly(
        coordinates: Union[Tuple[int, int], List[Tuple[int, int]]],
        scale_factors: Optional[Dict[str, float]] = None
    ) -> Union[Tuple[int, int], List[Tuple[int, int]]]:
        """
        直接获取系统坐标（简化版本）
        
        Args:
            coordinates: 要转换的坐标
            scale_factors: 缩放因子
            
        Returns:
            系统坐标
        """
        try:
            # 从基准坐标直接转换为系统坐标
            return CoordinateUtils.convert_coordinates(
                coordinates, "base", "system", scale_factors or {}
            )
        except Exception as e:
            logger.error(f"Direct system coordinate conversion failed: {e}")
            return coordinates
    
    @staticmethod
    def convert_to_original_target_coordinates(
        coordinates: Union[Tuple[int, int], List[Tuple[int, int]]],
        scale_record: Dict[str, Any]
    ) -> Union[Tuple[int, int], List[Tuple[int, int]]]:
        """
        转换到原始目标图像坐标系
        
        Args:
            coordinates: 要转换的坐标
            scale_record: 缩放记录
            
        Returns:
            原始目标图像坐标系中的坐标
        """
        try:
            from .scale_utils import ScaleUtils
            
            # 使用缩放记录进行转换
            return ScaleUtils.convert_coordinates_with_scale_record(
                coordinates, scale_record, "to_original"
            )
        except Exception as e:
            logger.error(f"Conversion to original target coordinates failed: {e}")
            return coordinates
    
    @staticmethod
    def create_simplified_coordinate_mapping(
        scale_factors: Dict[str, float],
        scale_record: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        创建简化的坐标映射
        
        Args:
            scale_factors: 缩放因子
            scale_record: 缩放记录
            
        Returns:
            简化的坐标映射
        """
        try:
            mapping = {
                "system_scale": scale_factors.get("dpi_factor", 1.0),
                "resolution_scale": scale_factors.get("res_factor", 1.0),
                "total_scale": scale_factors.get("total_factor", 1.0)
            }
            
            if scale_record:
                coord_mapping = scale_record.get("coordinate_mapping", {})
                mapping.update({
                    "template_to_target_scale": coord_mapping.get("scale_x", 1.0),
                    "target_to_template_scale": coord_mapping.get("inverse_scale_x", 1.0)
                })
            
            return mapping
        except Exception as e:
            logger.error(f"Failed to create simplified coordinate mapping: {e}")
            return {}


# 便捷函数
def simplify_coordinate_system(coordinates, source_system="base", target_system="system", **kwargs):
    """便捷的坐标系统简化函数"""
    return CoordinateSystemSimplifier.simplify_coordinate_system(
        coordinates, source_system, target_system, **kwargs
    )


def get_system_coordinates_directly(coordinates, **kwargs):
    """便捷的直接获取系统坐标函数"""
    return CoordinateSystemSimplifier.get_system_coordinates_directly(coordinates, **kwargs)


def convert_to_original_target_coordinates(coordinates, scale_record):
    """便捷的转换到原始目标图像坐标系函数"""
    return CoordinateSystemSimplifier.convert_to_original_target_coordinates(coordinates, scale_record)
