"""
工具模块
提供图像处理、坐标转换、缩放、格式转换、重试等基础工具功能
"""

from .image_utils import (
    ImageUtils,
    load_image,
    save_image,
    convert_image_format,
    validate_image,
    analyze_image_quality,
    enhance_image,
    resize_image,
    crop_image
)
from .coordinate_utils import (
    CoordinateUtils,
    convert_coordinates,
    scale_coordinates,
    validate_coordinates,
    get_center_point,
    get_rectangle_bounds
)
from .scale_utils import (
    ScaleUtils,
    calculate_scale_factors,
    apply_scale,
    get_system_scale,
    get_resolution_scale,
    normalize_coordinates
)
from .format_utils import (
    FormatUtils,
    detect_image_format,
    convert_rgb_to_bgr,
    convert_bgr_to_rgb,
    ensure_bgr_format,
    ensure_rgb_format,
    get_image_info
)
from .retry_utils import (
    RetryUtils,
    retry_on_failure,
    retry_with_backoff,
    retry_on_confidence,
    RetryConfig as UtilsRetryConfig
)

__all__ = [
    # 图像工具
    'ImageUtils',
    'load_image',
    'save_image',
    'convert_image_format',
    'validate_image',
    'analyze_image_quality',
    'enhance_image',
    'resize_image',
    'crop_image',
    
    # 坐标工具
    'CoordinateUtils',
    'convert_coordinates',
    'scale_coordinates',
    'validate_coordinates',
    'get_center_point',
    'get_rectangle_bounds',
    
    # 缩放工具
    'ScaleUtils',
    'calculate_scale_factors',
    'apply_scale',
    'get_system_scale',
    'get_resolution_scale',
    'normalize_coordinates',
    
    # 格式工具
    'FormatUtils',
    'detect_image_format',
    'convert_rgb_to_bgr',
    'convert_bgr_to_rgb',
    'ensure_bgr_format',
    'ensure_rgb_format',
    'get_image_info',
    
    # 重试工具
    'RetryUtils',
    'retry_on_failure',
    'retry_with_backoff',
    'retry_on_confidence',
    'UtilsRetryConfig'
]
