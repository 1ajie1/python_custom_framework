"""
模板匹配算法
"""

import cv2
import numpy as np
from typing import Optional, Any
import logging

from .base import MatchingAlgorithm, MatchResult

logger = logging.getLogger(__name__)


class TemplateMatching(MatchingAlgorithm):
    """模板匹配算法"""
    
    def __init__(self, config: Optional[Any] = None):
        super().__init__(config)
        self.method = getattr(config, 'method', 'TM_CCOEFF_NORMED') if config else 'TM_CCOEFF_NORMED'
        self.confidence = getattr(config, 'confidence', 0.8) if config else 0.8
    
    def match(self, template: np.ndarray, target: np.ndarray, 
              config: Optional[Any] = None) -> MatchResult:
        """
        执行模板匹配
        
        Args:
            template: 模板图像
            target: 目标图像
            config: 匹配配置
            
        Returns:
            MatchResult: 匹配结果
        """
        try:
            # 获取匹配方法
            method = getattr(config, 'method', self.method) if config else self.method
            confidence_threshold = getattr(config, 'confidence', self.confidence) if config else self.confidence
            
            # 验证模板尺寸
            if template.shape[0] > target.shape[0] or template.shape[1] > target.shape[1]:
                error_msg = f"模板尺寸 {template.shape} 大于目标图像尺寸 {target.shape}"
                logger.warning(error_msg)
                return MatchResult(
                    found=False,
                    confidence=0.0,
                    center=(0, 0),
                    bbox=(0, 0, 0, 0),
                    algorithm_name=self.get_name(),
                    metadata={
                        "error": "template_larger_than_target",
                        "error_details": error_msg,
                        "template_shape": template.shape,
                        "target_shape": target.shape,
                        "method": method,
                        "confidence_threshold": confidence_threshold
                    }
                )
            
            # 转换为OpenCV方法
            cv_method = getattr(cv2, method, cv2.TM_CCOEFF_NORMED)
            if not hasattr(cv2, method):
                error_msg = f"无效的匹配方法: {method}"
                logger.error(error_msg)
                return MatchResult(
                    found=False,
                    confidence=0.0,
                    center=(0, 0),
                    bbox=(0, 0, 0, 0),
                    algorithm_name=self.get_name(),
                    metadata={
                        "error": "invalid_method",
                        "error_details": error_msg,
                        "requested_method": method,
                        "available_methods": [attr for attr in dir(cv2) if attr.startswith('TM_')]
                    }
                )
            
            # 执行模板匹配
            result = cv2.matchTemplate(target, template, cv_method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            # 根据方法选择最佳匹配位置和置信度
            logger.debug(f"匹配方法: {cv_method}, max_val: {max_val}, min_val: {min_val}")
            
            if cv_method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                match_loc = min_loc
                # 对于SQDIFF方法，值越小表示匹配越好
                # 将距离转换为相似度：相似度 = 1 - 距离
                match_val = 1 - min_val if min_val <= 1.0 else 0.0
                logger.debug(f"使用SQDIFF方法，match_val: {match_val}")
            elif cv_method in [cv2.TM_CCOEFF, cv2.TM_CCORR]:
                match_loc = max_loc
                # 对于CCOEFF和CCORR方法，可能返回负值
                # 需要将值标准化到0-1范围
                if cv_method == cv2.TM_CCOEFF:
                    # CCOEFF: 值范围通常在[-1, 1]，需要转换到[0, 1]
                    match_val = (max_val + 1) / 2
                    logger.debug(f"使用CCOEFF方法，match_val: {match_val}")
                else:  # TM_CCORR
                    # CCORR: 值范围很大，需要标准化
                    # 使用min-max标准化，但需要先获取全局范围
                    result_min, result_max, _, _ = cv2.minMaxLoc(result)
                    if result_max > result_min:
                        match_val = (max_val - result_min) / (result_max - result_min)
                    else:
                        match_val = 0.0
                    logger.debug(f"使用CCORR方法，match_val: {match_val}")
            else:
                # TM_CCOEFF_NORMED, TM_CCORR_NORMED 返回 [-1, 1] 范围的值
                # 需要转换到 [0, 1] 范围
                match_loc = max_loc
                match_val = (max_val + 1) / 2  # 将 [-1, 1] 转换到 [0, 1]
                logger.debug(f"使用NORMED方法，原始max_val: {max_val}, 转换后match_val: {match_val}")
            
            # 检查置信度
            if match_val < confidence_threshold:
                logger.info(f"匹配置信度 {match_val:.3f} 低于阈值 {confidence_threshold:.3f}")
                return MatchResult(
                    found=False,
                    confidence=match_val,
                    center=(0, 0),
                    bbox=(0, 0, 0, 0),
                    algorithm_name=self.get_name(),
                    metadata={
                        "error": "low_confidence",
                        "error_details": f"匹配置信度 {match_val:.3f} 低于阈值 {confidence_threshold:.3f}",
                        "match_confidence": match_val,
                        "confidence_threshold": confidence_threshold,
                        "method": method,
                        "template_shape": template.shape,
                        "target_shape": target.shape,
                        "match_location": match_loc,
                        "raw_min_val": min_val,
                        "raw_max_val": max_val
                    }
                )
            
            # 计算中心点和边界框
            h, w = template.shape[:2]
            center_x = match_loc[0] + w // 2
            center_y = match_loc[1] + h // 2
            
            logger.info(f"模板匹配成功: 置信度={match_val:.3f}, 位置=({center_x}, {center_y})")
            return MatchResult(
                found=True,
                confidence=match_val,
                center=(center_x, center_y),
                bbox=(match_loc[0], match_loc[1], w, h),
                algorithm_name=self.get_name(),
                metadata={
                    "method": method,
                    "template_shape": template.shape,
                    "target_shape": target.shape,
                    "match_location": match_loc,
                    "raw_min_val": min_val,
                    "raw_max_val": max_val,
                    "confidence_threshold": confidence_threshold
                }
            )
            
        except Exception as e:
            error_msg = f"Template matching failed: {e}"
            logger.error(error_msg, exc_info=True)
            return MatchResult(
                found=False,
                confidence=0.0,
                center=(0, 0),
                bbox=(0, 0, 0, 0),
                algorithm_name=self.get_name(),
                metadata={
                    "error": "template_matching_exception",
                    "error_details": error_msg,
                    "exception_type": type(e).__name__,
                    "template_shape": template.shape if template is not None else None,
                    "target_shape": target.shape if target is not None else None,
                    "method": method if 'method' in locals() else None,
                    "confidence_threshold": confidence_threshold if 'confidence_threshold' in locals() else None
                }
            )
    
    def get_name(self) -> str:
        """获取算法名称"""
        return "TemplateMatching"
