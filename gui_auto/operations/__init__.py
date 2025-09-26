"""
操作模块
提供图像操作、点击操作、键盘操作、窗口操作等功能
"""

from .base import (
    Operation, 
    OperationResult, 
    OperationError,
    OperationFactory,
    OperationManager
)
from .image_operations import ImageOperations
from .click_operations import ClickOperations
from .keyboard_operations import KeyboardOperations
from .window_operations import WindowOperations

__all__ = [
    # 基础类
    'Operation',
    'OperationResult', 
    'OperationError',
    'OperationFactory',
    'OperationManager',
    
    # 具体操作类
    'ImageOperations',
    'ClickOperations',
    'KeyboardOperations',
    'WindowOperations'
]
