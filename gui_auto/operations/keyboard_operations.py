"""
键盘操作模块
提供键盘输入、快捷键等操作功能
"""

import pyautogui
from typing import Union, List, Optional, Dict, Any
import logging

from .base import Operation, OperationResult
from ..core.exceptions import KeyboardError

logger = logging.getLogger(__name__)


class KeyboardOperations(Operation):
    """键盘操作类"""
    
    def __init__(self, config: Optional[Any] = None):
        """
        初始化键盘操作
        
        Args:
            config: 键盘操作配置
        """
        super().__init__(config)
        self.key_delay = getattr(config, 'key_delay', 0.1) if config else 0.1
        self.text_delay = getattr(config, 'text_delay', 0.05) if config else 0.05
    
    def execute(self, operation: str, *args, **kwargs) -> OperationResult:
        """
        执行键盘操作
        
        Args:
            operation: 操作类型
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            OperationResult: 操作结果
        """
        try:
            if operation == "type":
                return self._type_text(*args, **kwargs)
            elif operation == "press":
                return self._press_key(*args, **kwargs)
            elif operation == "hotkey":
                return self._hotkey(*args, **kwargs)
            elif operation == "type_with_delay":
                return self._type_with_delay(*args, **kwargs)
            elif operation == "key_combination":
                return self._key_combination(*args, **kwargs)
            elif operation == "clear_text":
                return self._clear_text(*args, **kwargs)
            else:
                return OperationResult(
                    success=False,
                    error=f"Unknown keyboard operation: {operation}"
                )
        except Exception as e:
            return OperationResult(
                success=False,
                error=f"Keyboard operation failed: {e}"
            )
    
    def _type_text(self, text: str, **kwargs) -> OperationResult:
        """输入文本"""
        try:
            pyautogui.typewrite(text, interval=self.text_delay)
            return OperationResult(
                success=True,
                data={"text": text}
            )
        except Exception as e:
            return OperationResult(
                success=False,
                error=f"Type text failed: {e}"
            )
    
    def _press_key(self, key: str, **kwargs) -> OperationResult:
        """按下按键"""
        try:
            pyautogui.press(key)
            return OperationResult(
                success=True,
                data={"key": key}
            )
        except Exception as e:
            return OperationResult(
                success=False,
                error=f"Press key failed: {e}"
            )
    
    def _hotkey(self, *keys, **kwargs) -> OperationResult:
        """按下快捷键组合"""
        try:
            pyautogui.hotkey(*keys)
            return OperationResult(
                success=True,
                data={"keys": keys}
            )
        except Exception as e:
            return OperationResult(
                success=False,
                error=f"Hotkey failed: {e}"
            )
    
    def _type_with_delay(self, text: str, delay: float = None, **kwargs) -> OperationResult:
        """带延迟的文本输入"""
        try:
            delay = delay or self.text_delay
            pyautogui.typewrite(text, interval=delay)
            return OperationResult(
                success=True,
                data={"text": text, "delay": delay}
            )
        except Exception as e:
            return OperationResult(
                success=False,
                error=f"Type with delay failed: {e}"
            )
    
    def _key_combination(self, keys: list, **kwargs) -> OperationResult:
        """按键组合"""
        try:
            pyautogui.hotkey(*keys)
            return OperationResult(
                success=True,
                data={"keys": keys}
            )
        except Exception as e:
            return OperationResult(
                success=False,
                error=f"Key combination failed: {e}"
            )
    
    def _clear_text(self, **kwargs) -> OperationResult:
        """清除文本（选中全部并删除）"""
        try:
            # 选中全部文本
            pyautogui.hotkey('ctrl', 'a')
            # 删除选中文本
            pyautogui.press('delete')
            
            return OperationResult(
                success=True,
                data={"action": "clear_text"}
            )
        except Exception as e:
            return OperationResult(
                success=False,
                error=f"Clear text failed: {e}"
            )
