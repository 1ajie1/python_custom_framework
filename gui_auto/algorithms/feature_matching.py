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
        # 使用更宽松的ORB参数以提高关键点检测成功率，特别针对小图像优化
        self.detector = cv2.ORB_create(
            nfeatures=500,  # 减少特征点数量，避免过度检测
            scaleFactor=1.1,  # 进一步降低缩放因子
            nlevels=4,  # 减少金字塔层数，适合小图像
            edgeThreshold=5,  # 大幅降低边缘阈值
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=15,  # 减小patch大小，适合小图像
            fastThreshold=10  # 降低FAST阈值
        )
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.min_matches = getattr(config, 'min_matches', 5) if config else 5  # 降低最小匹配数
    
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
            
            # 确保图像是灰度图
            if len(template.shape) == 3:
                template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            else:
                template_gray = template
                
            if len(target.shape) == 3:
                target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
            else:
                target_gray = target
            
            # 检查图像尺寸，如果模板太小则跳过特征匹配
            template_h, template_w = template_gray.shape[:2]
            if template_h < 50 or template_w < 50:
                logger.info(f"模板图像太小 ({template_w}x{template_h})，跳过特征匹配，建议使用模板匹配算法")
                return MatchResult(
                    found=False,
                    confidence=0.0,
                    center=(0, 0),
                    bbox=(0, 0, 0, 0),
                    algorithm_name=self.get_name(),
                    metadata={
                        "error": "template_too_small",
                        "error_details": f"模板图像尺寸 {template_w}x{template_h} 太小，无法进行有效的特征匹配",
                        "template_shape": template.shape,
                        "target_shape": target.shape,
                        "suggestion": "建议使用模板匹配算法"
                    }
                )
            
            # 检测关键点和描述符
            kp1, des1 = self.detector.detectAndCompute(template_gray, None)
            kp2, des2 = self.detector.detectAndCompute(target_gray, None)
            
            # 添加调试信息
            logger.debug(f"特征检测结果: template_kp={len(kp1) if kp1 is not None else 0}, target_kp={len(kp2) if kp2 is not None else 0}")
            logger.debug(f"描述符结果: template_des={des1 is not None}, target_des={des2 is not None}")
            if des1 is not None:
                logger.debug(f"模板描述符形状: {des1.shape}")
            if des2 is not None:
                logger.debug(f"目标描述符形状: {des2.shape}")
            
            # 检查关键点检测结果
            if des1 is None or des2 is None:
                error_msg = f"关键点检测失败: template_descriptors={des1 is not None}, target_descriptors={des2 is not None}"
                logger.warning(error_msg)
                return MatchResult(
                    found=False,
                    confidence=0.0,
                    center=(0, 0),
                    bbox=(0, 0, 0, 0),
                    algorithm_name=self.get_name(),
                    metadata={
                        "error": "keypoint_detection_failed",
                        "error_details": error_msg,
                        "template_keypoints": len(kp1) if kp1 is not None else 0,
                        "target_keypoints": len(kp2) if kp2 is not None else 0,
                        "template_descriptors_shape": des1.shape if des1 is not None else None,
                        "target_descriptors_shape": des2.shape if des2 is not None else None,
                        "template_shape": template.shape,
                        "target_shape": target.shape,
                        "min_matches": min_matches
                    }
                )
            
            # 检查关键点数量
            if len(kp1) < 4 or len(kp2) < 4:
                error_msg = f"关键点数量不足: template={len(kp1)}, target={len(kp2)} (至少需要4个)"
                logger.warning(error_msg)
                return MatchResult(
                    found=False,
                    confidence=0.0,
                    center=(0, 0),
                    bbox=(0, 0, 0, 0),
                    algorithm_name=self.get_name(),
                    metadata={
                        "error": "insufficient_keypoints",
                        "error_details": error_msg,
                        "template_keypoints": len(kp1),
                        "target_keypoints": len(kp2),
                        "min_required": 4,
                        "template_shape": template.shape,
                        "target_shape": target.shape
                    }
                )
            
            # 匹配特征
            matches = self.matcher.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            
            logger.info(f"特征匹配: 模板关键点={len(kp1)}, 目标关键点={len(kp2)}, 匹配数={len(matches)}")
            
            if len(matches) < min_matches:
                error_msg = f"匹配数量不足: {len(matches)} < {min_matches}"
                logger.info(error_msg)
                return MatchResult(
                    found=False,
                    confidence=len(matches) / min_matches,
                    center=(0, 0),
                    bbox=(0, 0, 0, 0),
                    algorithm_name=self.get_name(),
                    metadata={
                        "error": "insufficient_matches",
                        "error_details": error_msg,
                        "matches_found": len(matches),
                        "min_matches_required": min_matches,
                        "template_keypoints": len(kp1),
                        "target_keypoints": len(kp2),
                        "confidence": len(matches) / min_matches
                    }
                )
            
            # 计算匹配点
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            
            # 计算单应性矩阵
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            if M is None:
                error_msg = "单应性矩阵计算失败"
                logger.warning(error_msg)
                return MatchResult(
                    found=False,
                    confidence=0.0,
                    center=(0, 0),
                    bbox=(0, 0, 0, 0),
                    algorithm_name=self.get_name(),
                    metadata={
                        "error": "homography_failed",
                        "error_details": error_msg,
                        "matches_count": len(matches),
                        "template_keypoints": len(kp1),
                        "target_keypoints": len(kp2),
                        "src_points_shape": src_pts.shape,
                        "dst_points_shape": dst_pts.shape
                    }
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
            
            logger.info(f"特征匹配成功: 置信度={confidence:.3f}, 位置=({center_x}, {center_y}), 匹配数={len(matches)}")
            return MatchResult(
                found=True,
                confidence=confidence,
                center=(center_x, center_y),
                bbox=(x_min, y_min, x_max - x_min, y_max - y_min),
                algorithm_name=self.get_name(),
                metadata={
                    "matches_count": len(matches),
                    "template_keypoints": len(kp1),
                    "target_keypoints": len(kp2),
                    "template_shape": template.shape,
                    "target_shape": target.shape,
                    "min_matches": min_matches,
                    "homography_matrix_shape": M.shape if M is not None else None,
                    "inlier_count": np.sum(mask) if mask is not None else None
                }
            )
            
        except Exception as e:
            error_msg = f"Feature matching failed: {e}"
            logger.error(error_msg, exc_info=True)
            return MatchResult(
                found=False,
                confidence=0.0,
                center=(0, 0),
                bbox=(0, 0, 0, 0),
                algorithm_name=self.get_name(),
                metadata={
                    "error": "feature_matching_exception",
                    "error_details": error_msg,
                    "exception_type": type(e).__name__,
                    "template_shape": template.shape if template is not None else None,
                    "target_shape": target.shape if target is not None else None,
                    "min_matches": min_matches if 'min_matches' in locals() else None
                }
            )
    
    def get_name(self) -> str:
        """获取算法名称"""
        return "FeatureMatching"
