"""
算法模块
提供图像匹配算法、特征检测算法等功能
"""

from .base import (
    MatchingAlgorithm,
    MatchResult,
    AlgorithmFactory,
    AlgorithmError
)
from .template_matching import TemplateMatching
from .feature_matching import FeatureMatching
from .hybrid_matching import HybridMatching
from .pyramid_matching import PyramidMatching

__all__ = [
    # 基础类
    'MatchingAlgorithm',
    'MatchResult',
    'AlgorithmFactory',
    'AlgorithmError',
    
    # 具体算法
    'TemplateMatching',
    'FeatureMatching',
    'HybridMatching',
    'PyramidMatching'
]
