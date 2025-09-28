"""
算法基类模块
定义算法接口和基础功能
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union, Tuple, Optional, Dict, Any, List
import numpy as np
import logging

from ..core.exceptions import AlgorithmError

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """匹配结果类"""
    found: bool
    confidence: float
    center: Tuple[int, int]
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    scale_factor: float = 1.0
    algorithm_name: str = ""
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """验证结果数据"""
        if not 0.0 <= self.confidence <= 1.0:
            raise AlgorithmError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")
        
        if self.found and (self.center[0] < 0 or self.center[1] < 0):
            raise AlgorithmError(f"Invalid center coordinates: {self.center}")
    
    def to_system_coordinates(self, scale_factors: Optional[Dict[str, float]] = None) -> 'MatchResult':
        """
        将匹配结果转换为系统坐标
        
        Args:
            scale_factors: 缩放因子
            
        Returns:
            MatchResult: 转换后的结果
        """
        if not self.found:
            return self
        
        try:
            from ..utils.coordinate_utils import CoordinateUtils
            
            # 转换中心点坐标
            system_center = CoordinateUtils.convert_coordinates(
                self.center, "base", "system", scale_factors or {}
            )
            
            # 转换边界框坐标
            x, y, w, h = self.bbox
            # 避免重复调用，先计算转换后的坐标
            converted_coords = CoordinateUtils.convert_coordinates((x, y), "base", "system", scale_factors or {})
            dpi_factor = scale_factors.get("dpi_factor", 1.0) if scale_factors else 1.0
            system_bbox = (
                converted_coords[0],
                converted_coords[1],
                int(w * dpi_factor),
                int(h * dpi_factor)
            )
            
            return MatchResult(
                found=self.found,
                confidence=self.confidence,
                center=system_center,
                bbox=system_bbox,
                scale_factor=self.scale_factor,
                algorithm_name=self.algorithm_name,
                metadata=self.metadata
            )
            
        except Exception as e:
            logger.error(f"Failed to convert to system coordinates: {e}")
            return self
    
    def to_original_coordinates(self, scale_record: Optional[Dict[str, Any]] = None) -> 'MatchResult':
        """
        将匹配结果转换为原始目标图像坐标系
        
        Args:
            scale_record: 缩放记录
            
        Returns:
            MatchResult: 转换后的结果
        """
        if not self.found or not scale_record:
            return self
        
        try:
            from ..utils.scale_utils import ScaleUtils
            
            # 转换中心点坐标
            original_center = ScaleUtils.convert_coordinates_with_scale_record(
                self.center, scale_record, "to_original"
            )
            
            # 转换边界框坐标
            x, y, w, h = self.bbox
            # 避免重复调用，先计算转换后的坐标
            converted_coords = ScaleUtils.convert_coordinates_with_scale_record((x, y), scale_record, "to_original")
            scale_x = scale_record.get("coordinate_mapping", {}).get("scale_x", 1.0)
            scale_y = scale_record.get("coordinate_mapping", {}).get("scale_y", 1.0)
            original_bbox = (
                converted_coords[0],
                converted_coords[1],
                int(w / scale_x),
                int(h / scale_y)
            )
            
            return MatchResult(
                found=self.found,
                confidence=self.confidence,
                center=original_center,
                bbox=original_bbox,
                scale_factor=self.scale_factor,
                algorithm_name=self.algorithm_name,
                metadata=self.metadata
            )
            
        except Exception as e:
            logger.error(f"Failed to convert to original coordinates: {e}")
            return self


class MatchingAlgorithm(ABC):
    """图像匹配算法基类"""
    
    def __init__(self, config: Optional[Any] = None):
        """
        初始化算法
        
        Args:
            config: 算法配置
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def match(self, template: np.ndarray, target: np.ndarray, 
              config: Optional[Any] = None) -> MatchResult:
        """
        执行图像匹配
        
        Args:
            template: 模板图像
            target: 目标图像
            config: 匹配配置
            
        Returns:
            MatchResult: 匹配结果
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """
        获取算法名称
        
        Returns:
            str: 算法名称
        """
        pass
    
    def validate_input(self, template: np.ndarray, target: np.ndarray) -> bool:
        """
        验证输入图像
        
        Args:
            template: 模板图像
            target: 目标图像
            
        Returns:
            bool: 输入是否有效
        """
        try:
            if template is None or target is None:
                return False
            
            if not isinstance(template, np.ndarray) or not isinstance(target, np.ndarray):
                return False
            
            if len(template.shape) < 2 or len(target.shape) < 2:
                return False
            
            if template.size == 0 or target.size == 0:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Input validation failed: {e}")
            return False
    
    def preprocess_images(self, template: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        预处理图像
        
        Args:
            template: 模板图像
            target: 目标图像
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: 预处理后的图像
        """
        try:
            from ..utils.format_utils import FormatUtils
            
            # 确保图像格式一致
            template_bgr = FormatUtils.ensure_bgr_format(template)
            target_bgr = FormatUtils.ensure_bgr_format(target)
            
            return template_bgr, target_bgr
            
        except Exception as e:
            self.logger.error(f"Image preprocessing failed: {e}")
            return template, target
    
    def safe_match(self, template: np.ndarray, target: np.ndarray, 
                   config: Optional[Any] = None) -> MatchResult:
        """
        安全执行匹配（包含错误处理）
        
        Args:
            template: 模板图像
            target: 目标图像
            config: 匹配配置
            
        Returns:
            MatchResult: 匹配结果
        """
        try:
            # 验证输入
            if not self.validate_input(template, target):
                error_msg = f"输入验证失败: template={template is not None}, target={target is not None}"
                if template is not None:
                    error_msg += f", template_shape={template.shape if hasattr(template, 'shape') else 'unknown'}"
                if target is not None:
                    error_msg += f", target_shape={target.shape if hasattr(target, 'shape') else 'unknown'}"
                
                self.logger.error(error_msg)
                return MatchResult(
                    found=False,
                    confidence=0.0,
                    center=(0, 0),
                    bbox=(0, 0, 0, 0),
                    algorithm_name=self.get_name(),
                    metadata={
                        "error": "input_validation_failed",
                        "error_details": error_msg,
                        "template_info": {
                            "is_none": template is None,
                            "shape": template.shape if template is not None and hasattr(template, 'shape') else None,
                            "size": template.size if template is not None and hasattr(template, 'size') else None
                        },
                        "target_info": {
                            "is_none": target is None,
                            "shape": target.shape if target is not None and hasattr(target, 'shape') else None,
                            "size": target.size if target is not None and hasattr(target, 'size') else None
                        }
                    }
                )
            
            # 预处理图像
            template_processed, target_processed = self.preprocess_images(template, target)
            
            # 执行匹配
            result = self.match(template_processed, target_processed, config)
            
            # 设置算法名称
            result.algorithm_name = self.get_name()
            
            # 添加调试信息
            if result.metadata is None:
                result.metadata = {}
            
            result.metadata.update({
                "template_original_shape": template.shape,
                "target_original_shape": target.shape,
                "template_processed_shape": template_processed.shape,
                "target_processed_shape": target_processed.shape,
                "config_info": {
                    "has_config": config is not None,
                    "config_type": type(config).__name__ if config else None
                }
            })
            
            return result
            
        except Exception as e:
            error_msg = f"Safe match failed: {e}"
            self.logger.error(error_msg, exc_info=True)
            return MatchResult(
                found=False,
                confidence=0.0,
                center=(0, 0),
                bbox=(0, 0, 0, 0),
                algorithm_name=self.get_name(),
                metadata={
                    "error": "safe_match_exception",
                    "error_details": error_msg,
                    "exception_type": type(e).__name__,
                    "template_info": {
                        "is_none": template is None,
                        "shape": template.shape if template is not None and hasattr(template, 'shape') else None,
                        "dtype": template.dtype if template is not None and hasattr(template, 'dtype') else None
                    },
                    "target_info": {
                        "is_none": target is None,
                        "shape": target.shape if target is not None and hasattr(target, 'shape') else None,
                        "dtype": target.dtype if target is not None and hasattr(target, 'dtype') else None
                    }
                }
            )


class AlgorithmFactory:
    """算法工厂类"""
    
    _algorithms = {}
    
    # 注册默认算法
    @classmethod
    def _register_default_algorithms(cls):
        """注册默认算法"""
        from .template_matching import TemplateMatching
        from .feature_matching import FeatureMatching
        from .hybrid_matching import HybridMatching
        from .pyramid_matching import PyramidMatching
        
        cls.register_algorithm("template", TemplateMatching)
        cls.register_algorithm("feature", FeatureMatching)
        cls.register_algorithm("hybrid", HybridMatching)
        cls.register_algorithm("pyramid", PyramidMatching)
    
    @classmethod
    def _ensure_default_algorithms(cls):
        """确保默认算法已注册"""
        if not cls._algorithms:
            cls._register_default_algorithms()
    
    @classmethod
    def register_algorithm(cls, name: str, algorithm_class: type) -> None:
        """
        注册算法
        
        Args:
            name: 算法名称
            algorithm_class: 算法类
        """
        cls._algorithms[name] = algorithm_class
        logger.debug(f"Algorithm registered: {name}")
    
    @classmethod
    def create_algorithm(cls, name: str, config: Optional[Any] = None) -> MatchingAlgorithm:
        """
        创建算法实例
        
        Args:
            name: 算法名称
            config: 算法配置
            
        Returns:
            MatchingAlgorithm: 算法实例
        """
        cls._ensure_default_algorithms()
        
        if name not in cls._algorithms:
            raise AlgorithmError(f"Unknown algorithm: {name}")
        
        algorithm_class = cls._algorithms[name]
        return algorithm_class(config)
    
    @classmethod
    def get_available_algorithms(cls) -> List[str]:
        """
        获取可用的算法列表
        
        Returns:
            List[str]: 算法名称列表
        """
        cls._ensure_default_algorithms()
        return list(cls._algorithms.keys())
    
    @classmethod
    def create_matcher(cls, config: Optional[Any] = None) -> 'Matcher':
        """
        创建匹配器
        
        Args:
            config: 配置
            
        Returns:
            Matcher: 匹配器实例
        """
        return Matcher(config)


class Matcher:
    """匹配器类 - 简化坐标系统的核心"""
    
    def __init__(self, config: Optional[Any] = None):
        """
        初始化匹配器
        
        Args:
            config: 配置
        """
        self.config = config
        self.logger = logging.getLogger("Matcher")
        self._current_scale_record = None
    
    def find(self, template: Union[str, np.ndarray], 
             target: Union[str, np.ndarray] = None,
             algorithm: str = "template",
             return_system_coordinates: bool = True,
             confidence: float = 0.8,
             **kwargs) -> Optional[MatchResult]:
        """
        查找图像（简化坐标系统）
        
        Args:
            template: 模板图像
            target: 目标图像，None表示使用屏幕截图
            algorithm: 使用的算法
            return_system_coordinates: 是否返回系统坐标
            confidence: 置信度阈值
            **kwargs: 其他参数
            
        Returns:
            Optional[MatchResult]: 匹配结果，None表示未找到
        """
        try:
            from ..utils.image_utils import ImageUtils
            from ..operations import ImageOperations
            
            # 加载图像
            image_ops = ImageOperations()
            
            # 加载模板图像
            template_result = image_ops.execute("load", template)
            if not template_result.success:
                self.logger.error(f"Failed to load template: {template_result.error}")
                return None
            
            template_image = template_result.data
            
            # 加载目标图像
            if target is None:
                # 使用屏幕截图
                target_image = ImageUtils.capture_screen()
            else:
                target_result = image_ops.execute("load", target)
                if not target_result.success:
                    self.logger.error(f"Failed to load target: {target_result.error}")
                    return None
                target_image = target_result.data
            
            # 直接使用原始目标图像进行匹配，不进行缩放调整
            # 模板匹配应该在原始目标图像中寻找模板
            adjusted_target = target_image
            self._current_scale_record = {
                "template_size": template_image.shape[:2],
                "target_size": target_image.shape[:2],
                "adjusted_size": target_image.shape[:2],
                "scale_factors": {"dpi_factor": 1.0, "res_factor": 1.0, "total_factor": 1.0},
                "coordinate_mapping": {
                    "scale_x": 1.0,
                    "scale_y": 1.0
                }
            }
            
            # 创建包含confidence和其他参数的配置
            config_dict = {
                'confidence': confidence,
                'method': getattr(self.config, 'method', 'TM_CCOEFF_NORMED') if self.config else 'TM_CCOEFF_NORMED'
            }
            # 添加kwargs中的额外参数
            config_dict.update(kwargs)
            match_config = type('Config', (), config_dict)()
            
            # 执行匹配 - 使用新的配置创建算法实例
            matcher = AlgorithmFactory.create_algorithm(algorithm, match_config)
            result = matcher.safe_match(template_image, adjusted_target, match_config)
            
            if not result.found:
                return None
            
            # 根据配置返回坐标系统
            if return_system_coordinates:
                # 直接返回系统坐标，传递缩放因子
                scale_factors = self._current_scale_record.get("scale_factors", {}) if self._current_scale_record else {}
                return result.to_system_coordinates(scale_factors)
            else:
                # 返回原始目标图像坐标系
                return result.to_original_coordinates(self._current_scale_record)
            
        except Exception as e:
            self.logger.error(f"Find operation failed: {e}")
            return None
    
    def get_scale_record(self) -> Optional[Dict[str, Any]]:
        """
        获取当前的缩放记录
        
        Returns:
            Optional[Dict[str, Any]]: 缩放记录
        """
        return self._current_scale_record
