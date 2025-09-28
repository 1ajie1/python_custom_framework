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
        # 为特征匹配创建更宽松的配置
        feature_config = type('Config', (), {
            'min_matches': 3,  # 进一步降低最小匹配数
            'method': getattr(config, 'method', 'TM_CCOEFF_NORMED') if config else 'TM_CCOEFF_NORMED',
            'confidence': getattr(config, 'confidence', 0.5) if config else 0.5
        })()
        
        self.template_matcher = TemplateMatching(config)
        self.feature_matcher = FeatureMatching(feature_config)
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
            logger.info("开始执行模板匹配...")
            template_result = self.template_matcher.match(template, target, config)
            
            # 执行特征匹配
            logger.info("开始执行特征匹配...")
            feature_result = self.feature_matcher.match(template, target, config)
            
            # 记录两个算法的结果
            logger.info(f"模板匹配结果: found={template_result.found}, confidence={template_result.confidence:.3f}")
            logger.info(f"特征匹配结果: found={feature_result.found}, confidence={feature_result.confidence:.3f}")
            
            # 如果两个算法都失败，返回失败结果
            if not template_result.found and not feature_result.found:
                error_msg = "模板匹配和特征匹配都失败"
                logger.warning(error_msg)
                return MatchResult(
                    found=False,
                    confidence=0.0,
                    center=(0, 0),
                    bbox=(0, 0, 0, 0),
                    algorithm_name=self.get_name(),
                    metadata={
                        "error": "both_algorithms_failed",
                        "error_details": error_msg,
                        "template_result": {
                            "found": template_result.found,
                            "confidence": template_result.confidence,
                            "error": template_result.metadata.get("error") if template_result.metadata else None
                        },
                        "feature_result": {
                            "found": feature_result.found,
                            "confidence": feature_result.confidence,
                            "error": feature_result.metadata.get("error") if feature_result.metadata else None
                        },
                        "template_weight": self.template_weight,
                        "feature_weight": self.feature_weight
                    }
                )
            
            # 如果只有一个算法成功，返回成功的结果
            if template_result.found and not feature_result.found:
                logger.info("仅模板匹配成功，使用模板匹配结果")
                template_result.algorithm_name = self.get_name()
                if template_result.metadata is None:
                    template_result.metadata = {}
                template_result.metadata.update({
                    "hybrid_info": {
                        "template_weight": self.template_weight,
                        "feature_weight": self.feature_weight,
                        "feature_failed": True,
                        "feature_error": feature_result.metadata.get("error") if feature_result.metadata else None
                    }
                })
                return template_result
            
            if feature_result.found and not template_result.found:
                logger.info("仅特征匹配成功，使用特征匹配结果")
                feature_result.algorithm_name = self.get_name()
                if feature_result.metadata is None:
                    feature_result.metadata = {}
                feature_result.metadata.update({
                    "hybrid_info": {
                        "template_weight": self.template_weight,
                        "feature_weight": self.feature_weight,
                        "template_failed": True,
                        "template_error": template_result.metadata.get("error") if template_result.metadata else None
                    }
                })
                return feature_result
            
            # 两个算法都成功，进行结果融合
            logger.info("两个算法都成功，开始融合结果")
            return self._fuse_results(template_result, feature_result)
            
        except Exception as e:
            error_msg = f"Hybrid matching failed: {e}"
            logger.error(error_msg, exc_info=True)
            return MatchResult(
                found=False,
                confidence=0.0,
                center=(0, 0),
                bbox=(0, 0, 0, 0),
                algorithm_name=self.get_name(),
                metadata={
                    "error": "hybrid_matching_exception",
                    "error_details": error_msg,
                    "exception_type": type(e).__name__,
                    "template_weight": self.template_weight,
                    "feature_weight": self.feature_weight,
                    "template_shape": template.shape if template is not None else None,
                    "target_shape": target.shape if target is not None else None
                }
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
            
            logger.info(f"结果融合成功: 加权置信度={weighted_confidence:.3f}, 中心=({center_x}, {center_y})")
            
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
                    "feature_weight": self.feature_weight,
                    "template_center": template_result.center,
                    "feature_center": feature_result.center,
                    "template_bbox": template_result.bbox,
                    "feature_bbox": feature_result.bbox,
                    "fusion_method": "weighted_average"
                }
            )
            
        except Exception as e:
            error_msg = f"Result fusion failed: {e}"
            logger.error(error_msg, exc_info=True)
            
            # 如果融合失败，返回置信度更高的结果
            if template_result.confidence >= feature_result.confidence:
                logger.info(f"融合失败，使用模板匹配结果 (置信度: {template_result.confidence:.3f})")
                template_result.algorithm_name = self.get_name()
                if template_result.metadata is None:
                    template_result.metadata = {}
                template_result.metadata.update({
                    "fusion_failed": True,
                    "fusion_error": error_msg,
                    "fallback_reason": "template_higher_confidence",
                    "template_confidence": template_result.confidence,
                    "feature_confidence": feature_result.confidence
                })
                return template_result
            else:
                logger.info(f"融合失败，使用特征匹配结果 (置信度: {feature_result.confidence:.3f})")
                feature_result.algorithm_name = self.get_name()
                if feature_result.metadata is None:
                    feature_result.metadata = {}
                feature_result.metadata.update({
                    "fusion_failed": True,
                    "fusion_error": error_msg,
                    "fallback_reason": "feature_higher_confidence",
                    "template_confidence": template_result.confidence,
                    "feature_confidence": feature_result.confidence
                })
                return feature_result
    
    def get_name(self) -> str:
        """获取算法名称"""
        return "HybridMatching"
