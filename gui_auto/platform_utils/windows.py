"""
Windows平台实现
提供Windows特定的GUI自动化功能
"""

import pyautogui
import win32gui
import win32con
import win32api
import win32ui
from PIL import ImageGrab
from typing import Union, Tuple, Optional, Dict, Any, List
import logging

from .base import PlatformBase, PlatformInfo

logger = logging.getLogger(__name__)


class WindowsPlatform(PlatformBase):
    """Windows平台实现"""
    
    def _detect_platform(self) -> PlatformInfo:
        """
        检测Windows平台信息
        
        Returns:
            PlatformInfo: 平台信息
        """
        try:
            import platform as sys_platform
            
            # 获取Windows版本信息
            version_info = sys_platform.version()
            architecture = sys_platform.architecture()[0]
            
            # 检查支持的功能
            features = [
                "screen_capture",
                "mouse_click",
                "keyboard_input",
                "window_management",
                "dpi_scaling",
                "multi_monitor"
            ]
            
            # 检查高级功能
            try:
                import win32gui
                features.append("window_enumeration")
                features.append("window_control")
            except ImportError:
                logger.warning("win32gui not available, some features disabled")
            
            return PlatformInfo(
                name="Windows",
                version=version_info,
                architecture=architecture,
                is_supported=True,
                features=features,
                metadata={
                    "platform": "windows",
                    "win32_available": "win32gui" in globals()
                }
            )
            
        except Exception as e:
            logger.error(f"Windows platform detection failed: {e}")
            return PlatformInfo(
                name="Windows",
                version="Unknown",
                architecture="Unknown",
                is_supported=False,
                features=[],
                metadata={"error": str(e)}
            )
    
    def get_screen_size(self) -> Tuple[int, int]:
        """
        获取屏幕尺寸
        
        Returns:
            Tuple[int, int]: 屏幕尺寸 (width, height)
        """
        try:
            return pyautogui.size()
        except Exception as e:
            logger.error(f"Failed to get screen size: {e}")
            return (1920, 1080)  # 默认尺寸
    
    def get_dpi_scale(self) -> float:
        """
        获取DPI缩放比例
        
        Returns:
            float: DPI缩放比例
        """
        try:
            import win32api
            import win32con
            
            # 获取系统DPI
            hdc = win32api.GetDC(0)
            dpi_x = win32api.GetDeviceCaps(hdc, win32con.LOGPIXELSX)
            win32api.ReleaseDC(0, hdc)
            
            # 计算缩放比例 (96 DPI = 100%)
            scale = dpi_x / 96.0
            return scale
            
        except Exception as e:
            logger.warning(f"Failed to get DPI scale: {e}")
            return 1.0  # 默认无缩放
    
    def capture_screen(self, region: Optional[Tuple[int, int, int, int]] = None) -> Any:
        """
        捕获屏幕截图
        
        Args:
            region: 捕获区域 (x, y, width, height)，None表示全屏
            
        Returns:
            Any: 屏幕截图
        """
        try:
            if region:
                # 捕获指定区域
                bbox = (region[0], region[1], region[0] + region[2], region[1] + region[3])
                screenshot = ImageGrab.grab(bbox)
            else:
                # 捕获全屏
                screenshot = ImageGrab.grab()
            
            # 转换为BGR格式（OpenCV格式）
            import cv2
            import numpy as np
            
            # PIL Image -> numpy array
            img_array = np.array(screenshot)
            
            # RGB -> BGR
            if len(img_array.shape) == 3:
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else:
                img_bgr = img_array
            
            return img_bgr
            
        except Exception as e:
            logger.error(f"Screen capture failed: {e}")
            raise
    
    def click(self, x: int, y: int, button: str = "left") -> bool:
        """
        执行点击操作
        
        Args:
            x: X坐标
            y: Y坐标
            button: 鼠标按钮 ("left", "right", "middle")
            
        Returns:
            bool: 操作是否成功
        """
        try:
            if button == "left":
                pyautogui.click(x, y)
            elif button == "right":
                pyautogui.rightClick(x, y)
            elif button == "middle":
                pyautogui.middleClick(x, y)
            else:
                raise ValueError(f"Unsupported button: {button}")
            
            return True
            
        except Exception as e:
            logger.error(f"Click failed: {e}")
            return False
    
    def type_text(self, text: str, delay: float = 0.05) -> bool:
        """
        输入文本
        
        Args:
            text: 要输入的文本
            delay: 按键延迟
            
        Returns:
            bool: 操作是否成功
        """
        try:
            pyautogui.typewrite(text, interval=delay)
            return True
            
        except Exception as e:
            logger.error(f"Type text failed: {e}")
            return False
    
    def press_key(self, key: str) -> bool:
        """
        按下按键
        
        Args:
            key: 按键名称
            
        Returns:
            bool: 操作是否成功
        """
        try:
            pyautogui.press(key)
            return True
            
        except Exception as e:
            logger.error(f"Press key failed: {e}")
            return False
    
    def hotkey(self, *keys: str) -> bool:
        """
        按下快捷键组合
        
        Args:
            *keys: 按键组合
            
        Returns:
            bool: 操作是否成功
        """
        try:
            pyautogui.hotkey(*keys)
            return True
            
        except Exception as e:
            logger.error(f"Hotkey failed: {e}")
            return False
    
    def find_window(self, title: str) -> Optional[str]:
        """
        查找窗口
        
        Args:
            title: 窗口标题
            
        Returns:
            Optional[str]: 窗口句柄，未找到返回None
        """
        try:
            def enum_windows_callback(hwnd, windows):
                if win32gui.IsWindowVisible(hwnd):
                    window_title = win32gui.GetWindowText(hwnd)
                    if title.lower() in window_title.lower():
                        windows.append((hwnd, window_title))
                return True
            
            windows = []
            win32gui.EnumWindows(enum_windows_callback, windows)
            
            if windows:
                hwnd, window_title = windows[0]
                return str(hwnd)
            else:
                return None
                
        except Exception as e:
            logger.error(f"Find window failed: {e}")
            return None
    
    def activate_window(self, window_id: str) -> bool:
        """
        激活窗口
        
        Args:
            window_id: 窗口句柄
            
        Returns:
            bool: 操作是否成功
        """
        try:
            hwnd = int(window_id)
            
            # 检查窗口是否存在
            if not win32gui.IsWindow(hwnd):
                return False
            
            # 激活窗口
            win32gui.SetForegroundWindow(hwnd)
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            
            return True
            
        except Exception as e:
            logger.error(f"Activate window failed: {e}")
            return False
    
    def get_window_info(self, window_id: str) -> Optional[Dict[str, Any]]:
        """
        获取窗口信息
        
        Args:
            window_id: 窗口句柄
            
        Returns:
            Optional[Dict[str, Any]]: 窗口信息
        """
        try:
            hwnd = int(window_id)
            
            if not win32gui.IsWindow(hwnd):
                return None
            
            # 获取窗口信息
            title = win32gui.GetWindowText(hwnd)
            rect = win32gui.GetWindowRect(hwnd)
            
            return {
                "hwnd": hwnd,
                "title": title,
                "position": (rect[0], rect[1]),
                "size": (rect[2] - rect[0], rect[3] - rect[1]),
                "is_visible": win32gui.IsWindowVisible(hwnd),
                "is_minimized": win32gui.IsIconic(hwnd)
            }
            
        except Exception as e:
            logger.error(f"Get window info failed: {e}")
            return None
    
    def minimize_window(self, window_id: str) -> bool:
        """
        最小化窗口
        
        Args:
            window_id: 窗口句柄
            
        Returns:
            bool: 操作是否成功
        """
        try:
            hwnd = int(window_id)
            win32gui.ShowWindow(hwnd, win32con.SW_MINIMIZE)
            return True
            
        except Exception as e:
            logger.error(f"Minimize window failed: {e}")
            return False
    
    def maximize_window(self, window_id: str) -> bool:
        """
        最大化窗口
        
        Args:
            window_id: 窗口句柄
            
        Returns:
            bool: 操作是否成功
        """
        try:
            hwnd = int(window_id)
            win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE)
            return True
            
        except Exception as e:
            logger.error(f"Maximize window failed: {e}")
            return False
    
    def close_window(self, window_id: str) -> bool:
        """
        关闭窗口
        
        Args:
            window_id: 窗口句柄
            
        Returns:
            bool: 操作是否成功
        """
        try:
            hwnd = int(window_id)
            win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)
            return True
            
        except Exception as e:
            logger.error(f"Close window failed: {e}")
            return False
    
    def move_window(self, window_id: str, x: int, y: int) -> bool:
        """
        移动窗口
        
        Args:
            window_id: 窗口句柄
            x: X坐标
            y: Y坐标
            
        Returns:
            bool: 操作是否成功
        """
        try:
            hwnd = int(window_id)
            
            # 获取当前窗口大小
            rect = win32gui.GetWindowRect(hwnd)
            width = rect[2] - rect[0]
            height = rect[3] - rect[1]
            
            # 移动窗口
            win32gui.SetWindowPos(hwnd, 0, x, y, width, height, win32con.SWP_NOZORDER)
            return True
            
        except Exception as e:
            logger.error(f"Move window failed: {e}")
            return False
    
    def resize_window(self, window_id: str, width: int, height: int) -> bool:
        """
        调整窗口大小
        
        Args:
            window_id: 窗口句柄
            width: 宽度
            height: 高度
            
        Returns:
            bool: 操作是否成功
        """
        try:
            hwnd = int(window_id)
            
            # 获取当前窗口位置
            rect = win32gui.GetWindowRect(hwnd)
            x = rect[0]
            y = rect[1]
            
            # 调整窗口大小
            win32gui.SetWindowPos(hwnd, 0, x, y, width, height, win32con.SWP_NOZORDER)
            return True
            
        except Exception as e:
            logger.error(f"Resize window failed: {e}")
            return False
    
    def _validate_operation_platform(self, operation: str, **kwargs) -> bool:
        """
        Windows特定的操作验证
        
        Args:
            operation: 操作名称
            **kwargs: 操作参数
            
        Returns:
            bool: 是否支持
        """
        try:
            if operation == "find_window":
                return "title" in kwargs and isinstance(kwargs["title"], str)
            elif operation == "activate_window":
                return "window_id" in kwargs and isinstance(kwargs["window_id"], str)
            elif operation in ["minimize_window", "maximize_window", "close_window"]:
                return "window_id" in kwargs and isinstance(kwargs["window_id"], str)
            elif operation == "move_window":
                return all(key in kwargs for key in ["window_id", "x", "y"])
            elif operation == "resize_window":
                return all(key in kwargs for key in ["window_id", "width", "height"])
            
            return True
            
        except Exception as e:
            logger.error(f"Platform validation failed: {e}")
            return False
