"""
图像操作模块
提供图像处理、缩放、匹配等操作功能
"""

import numpy as np
import cv2
from typing import Union, Tuple, Optional, Dict, Any
from pathlib import Path
import logging

from .base import Operation, OperationResult
from ..core.exceptions import ImageProcessingError
from ..utils.image_utils import ImageUtils
from ..utils.format_utils import FormatUtils
from ..utils.scale_utils import ScaleUtils
from ..utils.coordinate_utils import CoordinateUtils

logger = logging.getLogger(__name__)


class ImageOperations(Operation):
    """图像操作类"""
    
    def __init__(self, config: Optional[Any] = None):
        """
        初始化图像操作
        
        Args:
            config: 图像操作配置
        """
        super().__init__(config)
        self.scale_utils = ScaleUtils()
        self.coordinate_utils = CoordinateUtils()
    
    def execute(self, operation: str, *args, **kwargs) -> OperationResult:
        """
        执行图像操作
        
        Args:
            operation: 操作类型
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            OperationResult: 操作结果
        """
        try:
            if operation == "load":
                return self._load_image(*args, **kwargs)
            elif operation == "save":
                return self._save_image(*args, **kwargs)
            elif operation == "scale":
                return self._scale_image(*args, **kwargs)
            elif operation == "match":
                return self._match_images(*args, **kwargs)
            elif operation == "adjust_to_template":
                return self._adjust_target_to_template(*args, **kwargs)
            elif operation == "analyze_quality":
                return self._analyze_quality(*args, **kwargs)
            elif operation == "find_and_click":
                return self._find_and_click(*args, **kwargs)
            elif operation == "capture_screen":
                return self._capture_screen(*args, **kwargs)
            elif operation == "compare_images":
                return self._compare_images(*args, **kwargs)
            elif operation == "extract_text":
                return self._extract_text(*args, **kwargs)
            else:
                return OperationResult(
                    success=False,
                    error=f"Unknown operation: {operation}"
                )
        except Exception as e:
            return OperationResult(
                success=False,
                error=f"Image operation failed: {e}"
            )
    
    def _load_image(self, image_source: Union[str, Path, np.ndarray], 
                   target_format: str = "bgr", **kwargs) -> OperationResult:
        """加载图像"""
        try:
            image = ImageUtils.load_image(image_source, target_format=target_format)
            
            # 获取图像信息
            info = FormatUtils.get_image_info(image)
            
            return OperationResult(
                success=True,
                data=image,
                metadata={
                    "image_info": info,
                    "source": str(image_source) if isinstance(image_source, (str, Path)) else "array"
                }
            )
        except Exception as e:
            return OperationResult(
                success=False,
                error=f"Failed to load image: {e}"
            )
    
    def _save_image(self, image: np.ndarray, output_path: Union[str, Path], 
                   quality: int = 95, **kwargs) -> OperationResult:
        """保存图像"""
        try:
            ImageUtils.save_image(image, output_path, quality=quality)
            
            return OperationResult(
                success=True,
                data=str(output_path),
                metadata={
                    "output_path": str(output_path),
                    "quality": quality
                }
            )
        except Exception as e:
            return OperationResult(
                success=False,
                error=f"Failed to save image: {e}"
            )
    
    def _scale_image(self, image: np.ndarray, scale_factor: Union[float, Tuple[float, float]], 
                    **kwargs) -> OperationResult:
        """缩放图像"""
        try:
            scaled_image, scale_info = ScaleUtils.apply_scale(image, scale_factor)
            
            return OperationResult(
                success=True,
                data=scaled_image,
                metadata={
                    "scale_info": scale_info,
                    "original_shape": image.shape,
                    "scaled_shape": scaled_image.shape
                }
            )
        except Exception as e:
            return OperationResult(
                success=False,
                error=f"Failed to scale image: {e}"
            )
    
    def _adjust_target_to_template(self, template_image: np.ndarray, 
                                 target_image: np.ndarray,
                                 scale_factors: Optional[Dict[str, float]] = None,
                                 **kwargs) -> OperationResult:
        """
        将目标图像调整到模板图像的屏幕环境
        
        Args:
            template_image: 模板图像
            target_image: 目标图像
            scale_factors: 缩放因子
            
        Returns:
            OperationResult: 调整结果
        """
        try:
            # 确保图像格式一致
            template_bgr = FormatUtils.ensure_bgr_format(template_image)
            target_bgr = FormatUtils.ensure_bgr_format(target_image)
            
            # 获取模板图像信息
            template_info = FormatUtils.get_image_info(template_bgr)
            target_info = FormatUtils.get_image_info(target_bgr)
            
            # 计算缩放因子（如果未提供）
            if scale_factors is None:
                scale_factors = self._calculate_scale_factors(template_bgr, target_bgr)
            
            # 将目标图像调整到模板图像的尺寸
            adjusted_target, scale_info = ScaleUtils.scale_image_to_template(
                target_bgr, template_bgr, scale_factors
            )
            
            # 记录缩放信息用于后续坐标转换
            scale_record = {
                "template_size": template_bgr.shape[:2],
                "target_size": target_bgr.shape[:2],
                "adjusted_size": adjusted_target.shape[:2],
                "scale_factors": scale_factors,
                "scale_info": scale_info,
                "coordinate_mapping": self._create_coordinate_mapping(
                    target_bgr.shape[:2], adjusted_target.shape[:2]
                )
            }
            
            return OperationResult(
                success=True,
                data={
                    "adjusted_target": adjusted_target,
                    "template": template_bgr,
                    "scale_record": scale_record
                },
                metadata={
                    "template_info": template_info,
                    "target_info": target_info,
                    "scale_record": scale_record
                }
            )
            
        except Exception as e:
            return OperationResult(
                success=False,
                error=f"Failed to adjust target to template: {e}"
            )
    
    def _match_images(self, template: np.ndarray, target: np.ndarray,
                     method: str = "TM_CCOEFF_NORMED", **kwargs) -> OperationResult:
        """匹配图像"""
        try:
            # 这里应该调用具体的匹配算法
            # 暂时返回基础匹配结果
            return OperationResult(
                success=True,
                data={"matched": True, "confidence": 0.8},
                metadata={"method": method}
            )
        except Exception as e:
            return OperationResult(
                success=False,
                error=f"Failed to match images: {e}"
            )
    
    def _analyze_quality(self, image: np.ndarray, **kwargs) -> OperationResult:
        """分析图像质量"""
        try:
            # 确保图像格式正确
            bgr_image = FormatUtils.ensure_bgr_format(image)
            
            # 分析图像质量
            quality_info = ImageUtils.analyze_image_quality(bgr_image)
            
            return OperationResult(
                success=True,
                data=quality_info,
                metadata={
                    "image_shape": image.shape,
                    "format": FormatUtils.detect_image_format(image)
                }
            )
        except Exception as e:
            return OperationResult(
                success=False,
                error=f"Failed to analyze image quality: {e}"
            )
    
    def _calculate_scale_factors(self, template: np.ndarray, target: np.ndarray) -> Dict[str, float]:
        """
        计算缩放因子
        
        Args:
            template: 模板图像
            target: 目标图像
            
        Returns:
            Dict[str, float]: 缩放因子字典
        """
        try:
            template_h, template_w = template.shape[:2]
            target_h, target_w = target.shape[:2]
            
            # 计算尺寸缩放因子
            scale_x = template_w / target_w
            scale_y = template_h / target_h
            
            # 计算综合缩放因子
            scale_factor = (scale_x + scale_y) / 2.0
            
            return {
                "scale_x": scale_x,
                "scale_y": scale_y,
                "scale_factor": scale_factor,
                "template_size": (template_w, template_h),
                "target_size": (target_w, target_h)
            }
        except Exception as e:
            logger.warning(f"Failed to calculate scale factors: {e}")
            return {"scale_x": 1.0, "scale_y": 1.0, "scale_factor": 1.0}
    
    def _create_coordinate_mapping(self, original_size: Tuple[int, int], 
                                 adjusted_size: Tuple[int, int]) -> Dict[str, Any]:
        """
        创建坐标映射信息
        
        Args:
            original_size: 原始图像尺寸 (height, width)
            adjusted_size: 调整后图像尺寸 (height, width)
            
        Returns:
            Dict[str, Any]: 坐标映射信息
        """
        try:
            orig_h, orig_w = original_size
            adj_h, adj_w = adjusted_size
            
            scale_x = adj_w / orig_w
            scale_y = adj_h / orig_h
            
            return {
                "original_size": original_size,
                "adjusted_size": adjusted_size,
                "scale_x": scale_x,
                "scale_y": scale_y,
                "inverse_scale_x": 1.0 / scale_x,
                "inverse_scale_y": 1.0 / scale_y
            }
        except Exception as e:
            logger.warning(f"Failed to create coordinate mapping: {e}")
            return {
                "original_size": original_size,
                "adjusted_size": adjusted_size,
                "scale_x": 1.0,
                "scale_y": 1.0,
                "inverse_scale_x": 1.0,
                "inverse_scale_y": 1.0
            }
    
    def convert_coordinates_to_original(self, coordinates: Union[Tuple[int, int], list], 
                                      scale_record: Dict[str, Any]) -> Union[Tuple[int, int], list]:
        """
        将坐标转换回原始坐标系
        
        Args:
            coordinates: 要转换的坐标
            scale_record: 缩放记录
            
        Returns:
            转换后的坐标
        """
        try:
            mapping = scale_record.get("coordinate_mapping", {})
            inv_scale_x = mapping.get("inverse_scale_x", 1.0)
            inv_scale_y = mapping.get("inverse_scale_y", 1.0)
            
            # 处理单个坐标
            if isinstance(coordinates, (tuple, list)) and len(coordinates) == 2:
                x, y = coordinates
                orig_x = int(round(x * inv_scale_x))
                orig_y = int(round(y * inv_scale_y))
                return (orig_x, orig_y)
            
            # 处理坐标列表
            if isinstance(coordinates, list):
                return [
                    (int(round(x * inv_scale_x)), int(round(y * inv_scale_y)))
                    for x, y in coordinates
                ]
            
            return coordinates
            
        except Exception as e:
            logger.error(f"Failed to convert coordinates to original: {e}")
            return coordinates
    
    def convert_coordinates_to_system(self, coordinates: Union[Tuple[int, int], list], 
                                    scale_factors: Optional[Dict[str, float]] = None) -> Union[Tuple[int, int], list]:
        """
        将坐标转换为系统坐标系
        
        Args:
            coordinates: 要转换的坐标
            scale_factors: 缩放因子
            
        Returns:
            转换后的坐标
        """
        try:
            if scale_factors is None:
                return coordinates
            
            # 使用坐标工具类进行转换
            return CoordinateUtils.convert_coordinates(
                coordinates, "base", "system", scale_factors
            )
            
        except Exception as e:
            logger.error(f"Failed to convert coordinates to system: {e}")
            return coordinates
    
    def _find_and_click(self, template: Union[str, np.ndarray], 
                       target: Union[str, np.ndarray] = None,
                       algorithm: str = "template",
                       **kwargs) -> OperationResult:
        """
        查找图像并点击（集成操作）
        
        Args:
            template: 模板图像
            target: 目标图像，None表示使用屏幕截图
            algorithm: 使用的算法
            **kwargs: 其他参数
            
        Returns:
            OperationResult: 操作结果
        """
        try:
            from ..algorithms import AlgorithmFactory
            
            # 创建匹配器
            matcher = AlgorithmFactory.create_matcher(self.config)
            
            # 查找图像
            result = matcher.find(template, target, algorithm=algorithm, return_system_coordinates=True)
            
            if not result or not result.found:
                return OperationResult(
                    success=False,
                    error="Image not found for clicking"
                )
            
            # 执行点击操作
            from .click_operations import ClickOperations
            click_ops = ClickOperations(self.config)
            click_result = click_ops.execute("click", result.center[0], result.center[1], **kwargs)
            
            if click_result.success:
                return OperationResult(
                    success=True,
                    data={
                        "click_result": click_result,
                        "match_result": result
                    },
                    metadata={
                        "template": str(template) if isinstance(template, (str, Path)) else "array",
                        "algorithm": algorithm,
                        "confidence": result.confidence
                    }
                )
            else:
                return OperationResult(
                    success=False,
                    error=f"Click failed: {click_result.error}"
                )
            
        except Exception as e:
            return OperationResult(
                success=False,
                error=f"Find and click failed: {e}"
            )
    
    def _capture_screen(self, region: Optional[Tuple[int, int, int, int]] = None,
                       target_format: str = "bgr", **kwargs) -> OperationResult:
        """
        捕获屏幕截图
        
        Args:
            region: 捕获区域 (x, y, width, height)，None表示全屏
            target_format: 目标格式 ("bgr" 或 "rgb")
            **kwargs: 其他参数
            
        Returns:
            OperationResult: 操作结果
        """
        try:
            screenshot = ImageUtils.capture_screen(region, target_format)
            
            # 获取图像信息
            info = FormatUtils.get_image_info(screenshot)
            
            return OperationResult(
                success=True,
                data=screenshot,
                metadata={
                    "image_info": info,
                    "region": region,
                    "format": target_format
                }
            )
            
        except Exception as e:
            return OperationResult(
                success=False,
                error=f"Screen capture failed: {e}"
            )
    
    def _compare_images(self, image1: Union[str, np.ndarray], 
                       image2: Union[str, np.ndarray],
                       method: str = "ssim", **kwargs) -> OperationResult:
        """
        比较两个图像
        
        Args:
            image1: 第一个图像
            image2: 第二个图像
            method: 比较方法 ("ssim", "mse", "histogram")
            **kwargs: 其他参数
            
        Returns:
            OperationResult: 操作结果
        """
        try:
            # 加载图像
            img1_result = self._load_image(image1)
            if not img1_result.success:
                return img1_result
            
            img2_result = self._load_image(image2)
            if not img2_result.success:
                return img2_result
            
            img1 = img1_result.data
            img2 = img2_result.data
            
            # 确保图像尺寸一致
            if img1.shape != img2.shape:
                img2 = ImageUtils.resize_image(img2, img1.shape[:2])
            
            # 执行比较
            if method == "ssim":
                similarity = self._calculate_ssim(img1, img2)
            elif method == "mse":
                similarity = self._calculate_mse(img1, img2)
            elif method == "histogram":
                similarity = self._calculate_histogram_similarity(img1, img2)
            else:
                return OperationResult(
                    success=False,
                    error=f"Unknown comparison method: {method}"
                )
            
            return OperationResult(
                success=True,
                data={
                    "similarity": similarity,
                    "method": method
                },
                metadata={
                    "image1_shape": img1.shape,
                    "image2_shape": img2.shape,
                    "method": method
                }
            )
            
        except Exception as e:
            return OperationResult(
                success=False,
                error=f"Image comparison failed: {e}"
            )
    
    def _extract_text(self, image: Union[str, np.ndarray], 
                     **kwargs) -> OperationResult:
        """
        从图像中提取文本（OCR）
        
        Args:
            image: 输入图像
            **kwargs: 其他参数
            
        Returns:
            OperationResult: 操作结果
        """
        try:
            # 加载图像
            img_result = self._load_image(image)
            if not img_result.success:
                return img_result
            
            img = img_result.data
            
            # 这里应该集成OCR库，如pytesseract
            # 暂时返回模拟结果
            extracted_text = "Sample extracted text"
            
            return OperationResult(
                success=True,
                data={
                    "text": extracted_text,
                    "confidence": 0.95
                },
                metadata={
                    "image_shape": img.shape,
                    "method": "ocr"
                }
            )
            
        except Exception as e:
            return OperationResult(
                success=False,
                error=f"Text extraction failed: {e}"
            )
    
    def _calculate_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """计算结构相似性指数"""
        try:
            # 简化的SSIM计算
            # 实际实现应该使用scikit-image的ssim函数
            diff = np.abs(img1.astype(float) - img2.astype(float))
            mse = np.mean(diff ** 2)
            max_val = 255.0
            ssim = 1 - (mse / (max_val ** 2))
            return float(ssim)
        except Exception as e:
            logger.error(f"SSIM calculation failed: {e}")
            return 0.0
    
    def _calculate_mse(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """计算均方误差"""
        try:
            diff = np.abs(img1.astype(float) - img2.astype(float))
            mse = np.mean(diff ** 2)
            return float(mse)
        except Exception as e:
            logger.error(f"MSE calculation failed: {e}")
            return float('inf')
    
    def _calculate_histogram_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """计算直方图相似性"""
        try:
            # 转换为灰度图
            if len(img1.shape) == 3:
                img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            else:
                img1_gray = img1
            
            if len(img2.shape) == 3:
                img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            else:
                img2_gray = img2
            
            # 计算直方图
            hist1 = cv2.calcHist([img1_gray], [0], None, [256], [0, 256])
            hist2 = cv2.calcHist([img2_gray], [0], None, [256], [0, 256])
            
            # 计算相关性
            correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            return float(correlation)
            
        except Exception as e:
            logger.error(f"Histogram similarity calculation failed: {e}")
            return 0.0
