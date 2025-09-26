"""
混合匹配算法
结合模板匹配和特征匹配
"""

import numpy as np
from typing import Optional, Any
import logging

from .base import MatchingAlgorithm, MatchResult
from .template_matching import TemplateMatching
from .feature_matching import FeatureMatching

logger = logging.getLogger(__name__)


class HybridMatching(MatchingAlgorithm):
    """混合匹配算法"""
    
    def __init__(self, config: Optional[Any] = None):
        super().__init__(config)
        self.template_matcher = TemplateMatching(config)
        self.feature_matcher = FeatureMatching(config)
        self.template_weight = getattr(config, 'template_weight', 0.6) if config else 0.6
        self.feature_weight = getattr(config, 'feature_weight', 0.4) if config else 0.4
    
    def match(self, template: np.ndarray, target: np.ndarray, 
              config: Optional[Any] = None) -> MatchResult:
        """
        执行混合匹配
        
        Args:
            template: 模板图像
            target: 目标图像
            config: 匹配配置
            
        Returns:
            MatchResult: 匹配结果
        """
        try:
            # 执行模板匹配
            template_result = self.template_matcher.match(template, target, config)
            
            # 执行特征匹配
            feature_result = self.feature_matcher.match(template, target, config)
            
            # 如果两个算法都失败，返回失败结果
            if not template_result.found and not feature_result.found:
                return MatchResult(
                    found=False,
                    confidence=0.0,
                    center=(0, 0),
                    bbox=(0, 0, 0, 0),
                    algorithm_name=self.get_name()
                )
            
            # 如果只有一个算法成功，返回成功的结果
            if template_result.found and not feature_result.found:
                template_result.algorithm_name = self.get_name()
                return template_result
            
            if feature_result.found and not template_result.found:
                feature_result.algorithm_name = self.get_name()
                return feature_result
            
            # 两个算法都成功，进行结果融合
            return self._fuse_results(template_result, feature_result)
            
        except Exception as e:
            logger.error(f"Hybrid matching failed: {e}")
            return MatchResult(
                found=False,
                confidence=0.0,
                center=(0, 0),
                bbox=(0, 0, 0, 0),
                algorithm_name=self.get_name(),
                metadata={"error": str(e)}
            )
    
    def _fuse_results(self, template_result: MatchResult, feature_result: MatchResult) -> MatchResult:
        """
        融合两个匹配结果
        
        Args:
            template_result: 模板匹配结果
            feature_result: 特征匹配结果
            
        Returns:
            MatchResult: 融合后的结果
        """
        try:
            # 计算加权置信度
            weighted_confidence = (
                template_result.confidence * self.template_weight +
                feature_result.confidence * self.feature_weight
            )
            
            # 计算加权中心点
            center_x = int(
                template_result.center[0] * self.template_weight +
                feature_result.center[0] * self.feature_weight
            )
            center_y = int(
                template_result.center[1] * self.template_weight +
                feature_result.center[1] * self.feature_weight
            )
            
            # 计算加权边界框
            bbox_x = int(
                template_result.bbox[0] * self.template_weight +
                feature_result.bbox[0] * self.feature_weight
            )
            bbox_y = int(
                template_result.bbox[1] * self.template_weight +
                feature_result.bbox[1] * self.feature_weight
            )
            bbox_w = int(
                template_result.bbox[2] * self.template_weight +
                feature_result.bbox[2] * self.feature_weight
            )
            bbox_h = int(
                template_result.bbox[3] * self.template_weight +
                feature_result.bbox[3] * self.feature_weight
            )
            
            return MatchResult(
                found=True,
                confidence=weighted_confidence,
                center=(center_x, center_y),
                bbox=(bbox_x, bbox_y, bbox_w, bbox_h),
                algorithm_name=self.get_name(),
                metadata={
                    "template_confidence": template_result.confidence,
                    "feature_confidence": feature_result.confidence,
                    "template_weight": self.template_weight,
                    "feature_weight": self.feature_weight
                }
            )
            
        except Exception as e:
            logger.error(f"Result fusion failed: {e}")
            # 如果融合失败，返回置信度更高的结果
            if template_result.confidence >= feature_result.confidence:
                template_result.algorithm_name = self.get_name()
                return template_result
            else:
                feature_result.algorithm_name = self.get_name()
                return feature_result
    
    def get_name(self) -> str:
        """获取算法名称"""
        return "HybridMatching"
