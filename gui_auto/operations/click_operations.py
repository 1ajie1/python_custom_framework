"""
点击操作模块
提供鼠标点击、拖拽等操作功能
优化坐标转换复杂性，直接使用系统坐标
"""

import pyautogui
import numpy as np
from typing import Union, Tuple, Optional, Dict, Any, List
import logging

from .base import Operation, OperationResult
from ..core.exceptions import ClickError
from ..utils.coordinate_utils import CoordinateSystemSimplifier

logger = logging.getLogger(__name__)


class ClickOperations(Operation):
    """点击操作类"""
    
    def __init__(self, config: Optional[Any] = None):
        """
        初始化点击操作
        
        Args:
            config: 点击操作配置
        """
        super().__init__(config)
        self.click_delay = getattr(config, 'click_delay', 0.1) if config else 0.1
    
    def execute(self, operation: str, *args, **kwargs) -> OperationResult:
        """
        执行点击操作
        
        Args:
            operation: 操作类型
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            OperationResult: 操作结果
        """
        try:
            if operation == "click":
                return self._click(*args, **kwargs)
            elif operation == "double_click":
                return self._double_click(*args, **kwargs)
            elif operation == "right_click":
                return self._right_click(*args, **kwargs)
            elif operation == "drag":
                return self._drag(*args, **kwargs)
            elif operation == "click_image":
                return self._click_image(*args, **kwargs)
            elif operation == "click_with_coordinate_conversion":
                return self._click_with_coordinate_conversion(*args, **kwargs)
            else:
                return OperationResult(
                    success=False,
                    error=f"Unknown click operation: {operation}"
                )
        except Exception as e:
            return OperationResult(
                success=False,
                error=f"Click operation failed: {e}"
            )
    
    def _click(self, x: int, y: int, button: str = "left", **kwargs) -> OperationResult:
        """执行点击操作"""
        try:
            pyautogui.click(x, y, button=button)
            return OperationResult(
                success=True,
                data={"x": x, "y": y, "button": button}
            )
        except Exception as e:
            return OperationResult(
                success=False,
                error=f"Click failed: {e}"
            )
    
    def _double_click(self, x: int, y: int, **kwargs) -> OperationResult:
        """执行双击操作"""
        try:
            pyautogui.doubleClick(x, y)
            return OperationResult(
                success=True,
                data={"x": x, "y": y, "action": "double_click"}
            )
        except Exception as e:
            return OperationResult(
                success=False,
                error=f"Double click failed: {e}"
            )
    
    def _right_click(self, x: int, y: int, **kwargs) -> OperationResult:
        """执行右键点击操作"""
        try:
            pyautogui.rightClick(x, y)
            return OperationResult(
                success=True,
                data={"x": x, "y": y, "action": "right_click"}
            )
        except Exception as e:
            return OperationResult(
                success=False,
                error=f"Right click failed: {e}"
            )
    
    def _drag(self, start_x: int, start_y: int, end_x: int, end_y: int, **kwargs) -> OperationResult:
        """执行拖拽操作"""
        try:
            pyautogui.drag(end_x - start_x, end_y - start_y, duration=kwargs.get('duration', 1.0))
            return OperationResult(
                success=True,
                data={
                    "start": (start_x, start_y),
                    "end": (end_x, end_y)
                }
            )
        except Exception as e:
            return OperationResult(
                success=False,
                error=f"Drag failed: {e}"
            )
    
    def _click_image(self, template: Union[str, np.ndarray], 
                    target: Union[str, np.ndarray] = None,
                    **kwargs) -> OperationResult:
        """
        点击图像（简化坐标系统）
        
        Args:
            template: 模板图像
            target: 目标图像，None表示使用屏幕截图
            **kwargs: 其他参数
            
        Returns:
            OperationResult: 操作结果
        """
        try:
            from ..algorithms import AlgorithmFactory
            
            # 创建匹配器
            matcher = AlgorithmFactory.create_matcher(self.config)
            
            # 查找图像
            result = matcher.find(template, target, return_system_coordinates=True)
            
            if not result or not result.found:
                return OperationResult(
                    success=False,
                    error="Image not found"
                )
            
            # 直接使用系统坐标进行点击
            return self._click(result.center[0], result.center[1], **kwargs)
            
        except Exception as e:
            return OperationResult(
                success=False,
                error=f"Click image failed: {e}"
            )
    
    def _click_with_coordinate_conversion(self, coordinates: Union[Tuple[int, int], List[Tuple[int, int]]],
                                        source_system: str = "base",
                                        scale_factors: Optional[Dict[str, float]] = None,
                                        **kwargs) -> OperationResult:
        """
        带坐标转换的点击操作
        
        Args:
            coordinates: 坐标
            source_system: 源坐标系统
            scale_factors: 缩放因子
            **kwargs: 其他参数
            
        Returns:
            OperationResult: 操作结果
        """
        try:
            # 简化坐标转换
            system_coords = CoordinateSystemSimplifier.simplify_coordinate_system(
                coordinates, source_system, "system", scale_factors
            )
            
            # 处理单个坐标
            if isinstance(system_coords, (tuple, list)) and len(system_coords) == 2:
                return self._click(system_coords[0], system_coords[1], **kwargs)
            
            # 处理坐标列表
            if isinstance(system_coords, list):
                results = []
                for coord in system_coords:
                    result = self._click(coord[0], coord[1], **kwargs)
                    results.append(result)
                
                # 检查所有点击是否成功
                all_success = all(r.success for r in results)
                return OperationResult(
                    success=all_success,
                    data={"results": results},
                    error=None if all_success else "Some clicks failed"
                )
            
            return OperationResult(
                success=False,
                error="Invalid coordinates format"
            )
            
        except Exception as e:
            return OperationResult(
                success=False,
                error=f"Click with coordinate conversion failed: {e}"
            )
