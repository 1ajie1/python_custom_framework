"""
特征匹配算法
"""

import cv2
import numpy as np
from typing import Optional, Any
import logging

from .base import MatchingAlgorithm, MatchResult

logger = logging.getLogger(__name__)


class FeatureMatching(MatchingAlgorithm):
    """特征匹配算法"""
    
    def __init__(self, config: Optional[Any] = None):
        super().__init__(config)
        self.detector = cv2.ORB_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.min_matches = getattr(config, 'min_matches', 10) if config else 10
    
    def match(self, template: np.ndarray, target: np.ndarray, 
              config: Optional[Any] = None) -> MatchResult:
        """
        执行特征匹配
        
        Args:
            template: 模板图像
            target: 目标图像
            config: 匹配配置
            
        Returns:
            MatchResult: 匹配结果
        """
        try:
            min_matches = getattr(config, 'min_matches', self.min_matches) if config else self.min_matches
            
            # 检测关键点和描述符
            kp1, des1 = self.detector.detectAndCompute(template, None)
            kp2, des2 = self.detector.detectAndCompute(target, None)
            
            if des1 is None or des2 is None:
                return MatchResult(
                    found=False,
                    confidence=0.0,
                    center=(0, 0),
                    bbox=(0, 0, 0, 0),
                    algorithm_name=self.get_name()
                )
            
            # 匹配特征
            matches = self.matcher.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            
            if len(matches) < min_matches:
                return MatchResult(
                    found=False,
                    confidence=len(matches) / min_matches,
                    center=(0, 0),
                    bbox=(0, 0, 0, 0),
                    algorithm_name=self.get_name()
                )
            
            # 计算匹配点
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            
            # 计算单应性矩阵
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            if M is None:
                return MatchResult(
                    found=False,
                    confidence=0.0,
                    center=(0, 0),
                    bbox=(0, 0, 0, 0),
                    algorithm_name=self.get_name()
                )
            
            # 计算模板在目标图像中的位置
            h, w = template.shape[:2]
            pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            
            # 计算边界框
            x_coords = dst[:, 0, 0]
            y_coords = dst[:, 0, 1]
            
            x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
            y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))
            
            center_x = (x_min + x_max) // 2
            center_y = (y_min + y_max) // 2
            
            # 计算置信度
            confidence = len(matches) / max(len(kp1), len(kp2))
            
            return MatchResult(
                found=True,
                confidence=confidence,
                center=(center_x, center_y),
                bbox=(x_min, y_min, x_max - x_min, y_max - y_min),
                algorithm_name=self.get_name()
            )
            
        except Exception as e:
            logger.error(f"Feature matching failed: {e}")
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
        return "FeatureMatching"
