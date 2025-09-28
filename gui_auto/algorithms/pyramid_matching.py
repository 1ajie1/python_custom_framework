"""
金字塔匹配算法
使用图像金字塔进行多尺度匹配
"""

import cv2
import numpy as np
from typing import Optional, Any, List, Tuple
import logging

from .base import MatchingAlgorithm, MatchResult
from ..utils.scale_utils import ScaleUtils

logger = logging.getLogger(__name__)


class PyramidMatching(MatchingAlgorithm):
    """金字塔匹配算法"""
    
    def __init__(self, config: Optional[Any] = None):
        super().__init__(config)
        self.levels = getattr(config, 'pyramid_levels', 4) if config else 4
        self.scale_factor = getattr(config, 'pyramid_scale_factor', 0.5) if config else 0.5
        self.confidence = getattr(config, 'confidence', 0.8) if config else 0.8
    
    def match(self, template: np.ndarray, target: np.ndarray, 
              config: Optional[Any] = None) -> MatchResult:
        """
        执行金字塔匹配
        
        Args:
            template: 模板图像
            target: 目标图像
            config: 匹配配置
            
        Returns:
            MatchResult: 匹配结果
        """
        try:
            levels = getattr(config, 'pyramid_levels', self.levels) if config else self.levels
            scale_factor = getattr(config, 'pyramid_scale_factor', self.scale_factor) if config else self.scale_factor
            confidence_threshold = getattr(config, 'confidence', self.confidence) if config else self.confidence
            
            logger.info(f"开始金字塔匹配: levels={levels}, scale_factor={scale_factor}, confidence_threshold={confidence_threshold}")
            
            # 创建模板金字塔
            template_pyramid = ScaleUtils.create_scale_pyramid(template, levels, scale_factor)
            
            # 创建目标金字塔
            target_pyramid = ScaleUtils.create_scale_pyramid(target, levels, scale_factor)
            
            logger.info(f"金字塔创建完成: 模板层数={len(template_pyramid)}, 目标层数={len(target_pyramid)}")
            
            best_result = None
            best_confidence = 0.0
            level_results = []
            
            # 在每一层进行匹配
            for i, (template_level, target_level) in enumerate(zip(template_pyramid, target_pyramid)):
                logger.info(f"在第 {i} 层进行匹配: 模板尺寸={template_level.shape}, 目标尺寸={target_level.shape}")
                
                # 执行模板匹配
                result = self._match_at_level(template_level, target_level, confidence_threshold)
                level_results.append({
                    "level": i,
                    "found": result.found,
                    "confidence": result.confidence,
                    "template_shape": template_level.shape,
                    "target_shape": target_level.shape
                })
                
                if result.found and result.confidence > best_confidence:
                    # 将结果缩放回原始尺寸
                    scale = scale_factor ** i
                    scaled_result = self._scale_result(result, scale)
                    best_result = scaled_result
                    best_confidence = result.confidence
                    logger.info(f"第 {i} 层找到更好匹配: 置信度={result.confidence:.3f}, 缩放因子={scale}")
            
            if best_result is None:
                error_msg = f"在所有 {levels} 层中都未找到匹配"
                logger.warning(error_msg)
                return MatchResult(
                    found=False,
                    confidence=0.0,
                    center=(0, 0),
                    bbox=(0, 0, 0, 0),
                    algorithm_name=self.get_name(),
                    metadata={
                        "error": "no_match_at_any_level",
                        "error_details": error_msg,
                        "levels": levels,
                        "scale_factor": scale_factor,
                        "confidence_threshold": confidence_threshold,
                        "level_results": level_results,
                        "template_original_shape": template.shape,
                        "target_original_shape": target.shape
                    }
                )
            
            logger.info(f"金字塔匹配成功: 最佳置信度={best_confidence:.3f}, 位置={best_result.center}")
            best_result.algorithm_name = self.get_name()
            
            # 添加金字塔匹配的元数据
            if best_result.metadata is None:
                best_result.metadata = {}
            best_result.metadata.update({
                "pyramid_info": {
                    "levels": levels,
                    "scale_factor": scale_factor,
                    "confidence_threshold": confidence_threshold,
                    "level_results": level_results,
                    "best_level": next(i for i, r in enumerate(level_results) if r["found"] and r["confidence"] == best_confidence)
                }
            })
            
            return best_result
            
        except Exception as e:
            error_msg = f"Pyramid matching failed: {e}"
            logger.error(error_msg, exc_info=True)
            return MatchResult(
                found=False,
                confidence=0.0,
                center=(0, 0),
                bbox=(0, 0, 0, 0),
                algorithm_name=self.get_name(),
                metadata={
                    "error": "pyramid_matching_exception",
                    "error_details": error_msg,
                    "exception_type": type(e).__name__,
                    "template_shape": template.shape if template is not None else None,
                    "target_shape": target.shape if target is not None else None,
                    "levels": levels if 'levels' in locals() else None,
                    "scale_factor": scale_factor if 'scale_factor' in locals() else None
                }
            )
    
    def _match_at_level(self, template: np.ndarray, target: np.ndarray, 
                       confidence_threshold: float) -> MatchResult:
        """
        在指定层级进行匹配
        
        Args:
            template: 模板图像
            target: 目标图像
            confidence_threshold: 置信度阈值
            
        Returns:
            MatchResult: 匹配结果
        """
        try:
            # 检查模板尺寸
            if template.shape[0] > target.shape[0] or template.shape[1] > target.shape[1]:
                error_msg = f"层级模板尺寸 {template.shape} 大于目标图像尺寸 {target.shape}"
                logger.warning(error_msg)
                return MatchResult(
                    found=False,
                    confidence=0.0,
                    center=(0, 0),
                    bbox=(0, 0, 0, 0),
                    algorithm_name=self.get_name(),
                    metadata={
                        "error": "template_larger_than_target_at_level",
                        "error_details": error_msg,
                        "template_shape": template.shape,
                        "target_shape": target.shape,
                        "confidence_threshold": confidence_threshold
                    }
                )
            
            # 使用模板匹配
            result = cv2.matchTemplate(target, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            if max_val < confidence_threshold:
                logger.debug(f"层级匹配置信度 {max_val:.3f} 低于阈值 {confidence_threshold:.3f}")
                return MatchResult(
                    found=False,
                    confidence=max_val,
                    center=(0, 0),
                    bbox=(0, 0, 0, 0),
                    algorithm_name=self.get_name(),
                    metadata={
                        "error": "low_confidence_at_level",
                        "error_details": f"层级匹配置信度 {max_val:.3f} 低于阈值 {confidence_threshold:.3f}",
                        "match_confidence": max_val,
                        "confidence_threshold": confidence_threshold,
                        "template_shape": template.shape,
                        "target_shape": target.shape,
                        "match_location": max_loc,
                        "raw_min_val": min_val,
                        "raw_max_val": max_val
                    }
                )
            
            # 计算中心点和边界框
            h, w = template.shape[:2]
            center_x = max_loc[0] + w // 2
            center_y = max_loc[1] + h // 2
            
            logger.debug(f"层级匹配成功: 置信度={max_val:.3f}, 位置=({center_x}, {center_y})")
            return MatchResult(
                found=True,
                confidence=max_val,
                center=(center_x, center_y),
                bbox=(max_loc[0], max_loc[1], w, h),
                algorithm_name=self.get_name(),
                metadata={
                    "template_shape": template.shape,
                    "target_shape": target.shape,
                    "match_location": max_loc,
                    "raw_min_val": min_val,
                    "raw_max_val": max_val,
                    "confidence_threshold": confidence_threshold
                }
            )
            
        except Exception as e:
            error_msg = f"Level matching failed: {e}"
            logger.error(error_msg, exc_info=True)
            return MatchResult(
                found=False,
                confidence=0.0,
                center=(0, 0),
                bbox=(0, 0, 0, 0),
                algorithm_name=self.get_name(),
                metadata={
                    "error": "level_matching_exception",
                    "error_details": error_msg,
                    "exception_type": type(e).__name__,
                    "template_shape": template.shape if template is not None else None,
                    "target_shape": target.shape if target is not None else None,
                    "confidence_threshold": confidence_threshold
                }
            )
    
    def _scale_result(self, result: MatchResult, scale: float) -> MatchResult:
        """
        缩放匹配结果
        
        Args:
            result: 匹配结果
            scale: 缩放因子
            
        Returns:
            MatchResult: 缩放后的结果
        """
        try:
            if not result.found:
                return result
            
            # 缩放中心点
            scaled_center = (
                int(result.center[0] / scale),
                int(result.center[1] / scale)
            )
            
            # 缩放边界框
            x, y, w, h = result.bbox
            scaled_bbox = (
                int(x / scale),
                int(y / scale),
                int(w / scale),
                int(h / scale)
            )
            
            return MatchResult(
                found=result.found,
                confidence=result.confidence,
                center=scaled_center,
                bbox=scaled_bbox,
                scale_factor=scale,
                algorithm_name=result.algorithm_name,
                metadata=result.metadata
            )
            
        except Exception as e:
            logger.error(f"Result scaling failed: {e}")
            return result
    
    def get_name(self) -> str:
        """获取算法名称"""
        return "PyramidMatching"
