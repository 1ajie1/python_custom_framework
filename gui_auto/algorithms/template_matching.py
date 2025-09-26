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
            
            # 转换为OpenCV方法
            cv_method = getattr(cv2, method, cv2.TM_CCOEFF_NORMED)
            
            # 执行模板匹配
            result = cv2.matchTemplate(target, template, cv_method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            # 根据方法选择最佳匹配位置
            if cv_method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                match_loc = min_loc
                match_val = 1 - min_val  # 转换为相似度
            else:
                match_loc = max_loc
                match_val = max_val
            
            # 检查置信度
            if match_val < confidence_threshold:
                return MatchResult(
                    found=False,
                    confidence=match_val,
                    center=(0, 0),
                    bbox=(0, 0, 0, 0),
                    algorithm_name=self.get_name()
                )
            
            # 计算中心点和边界框
            h, w = template.shape[:2]
            center_x = match_loc[0] + w // 2
            center_y = match_loc[1] + h // 2
            
            return MatchResult(
                found=True,
                confidence=match_val,
                center=(center_x, center_y),
                bbox=(match_loc[0], match_loc[1], w, h),
                algorithm_name=self.get_name()
            )
            
        except Exception as e:
            logger.error(f"Template matching failed: {e}")
            return MatchResult(
                found=False,
                confidence=0.0,
                center=(0, 0),
                bbox=(0, 0, 0, 0),
                algorithm_name=self.get_name(),
                metadata={"error": str(e)}
            )
    
    def get_name(self) -> str:
        """获取算法名称"""
        return "TemplateMatching"
