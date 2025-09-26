"""
Linux平台实现
提供Linux特定的GUI自动化功能
"""

import pyautogui
from typing import Union, Tuple, Optional, Dict, Any, List
import logging
import subprocess
import os

from .base import PlatformBase, PlatformInfo

logger = logging.getLogger(__name__)


class LinuxPlatform(PlatformBase):
    """Linux平台实现"""
    
    def _detect_platform(self) -> PlatformInfo:
        """
        检测Linux平台信息
        
        Returns:
            PlatformInfo: 平台信息
        """
        try:
            import platform as sys_platform
            
            # 获取Linux版本信息
            version_info = sys_platform.version()
            architecture = sys_platform.architecture()[0]
            
            # 检查支持的功能
            features = [
                "screen_capture",
                "mouse_click",
                "keyboard_input"
            ]
            
            # 检查X11相关功能
            if self._check_x11_available():
                features.extend([
                    "window_management",
                    "dpi_scaling",
                    "multi_monitor"
                ])
            
            # 检查Wayland支持
            if self._check_wayland_available():
                features.append("wayland_support")
            
            # 检查窗口管理器
            wm_info = self._detect_window_manager()
            if wm_info:
                features.append("window_manager")
            
            return PlatformInfo(
                name="Linux",
                version=version_info,
                architecture=architecture,
                is_supported=True,
                features=features,
                metadata={
                    "platform": "linux",
                    "x11_available": self._check_x11_available(),
                    "wayland_available": self._check_wayland_available(),
                    "window_manager": wm_info
                }
            )
            
        except Exception as e:
            logger.error(f"Linux platform detection failed: {e}")
            return PlatformInfo(
                name="Linux",
                version="Unknown",
                architecture="Unknown",
                is_supported=False,
                features=[],
                metadata={"error": str(e)}
            )
    
    def _check_x11_available(self) -> bool:
        """检查X11是否可用"""
        try:
            return os.environ.get('DISPLAY') is not None
        except Exception:
            return False
    
    def _check_wayland_available(self) -> bool:
        """检查Wayland是否可用"""
        try:
            return os.environ.get('WAYLAND_DISPLAY') is not None
        except Exception:
            return False
    
    def _detect_window_manager(self) -> Optional[str]:
        """检测窗口管理器"""
        try:
            # 检查常见的窗口管理器
            wm_commands = [
                'echo $XDG_CURRENT_DESKTOP',
                'echo $DESKTOP_SESSION',
                'ps -e | grep -E "(gnome|kde|xfce|lxde|mate|i3|openbox)" | head -1'
            ]
            
            for cmd in wm_commands:
                try:
                    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
                    if result.returncode == 0 and result.stdout.strip():
                        return result.stdout.strip()
                except Exception:
                    continue
            
            return None
            
        except Exception as e:
            logger.warning(f"Window manager detection failed: {e}")
            return None
    
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
            # 尝试从环境变量获取缩放比例
            scale = os.environ.get('GDK_SCALE', '1.0')
            return float(scale)
            
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
                screenshot = pyautogui.screenshot(region=bbox)
            else:
                # 捕获全屏
                screenshot = pyautogui.screenshot()
            
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
            Optional[str]: 窗口ID，未找到返回None
        """
        try:
            # 使用xdotool查找窗口
            if self._check_command_available('xdotool'):
                cmd = f"xdotool search --name '{title}'"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0 and result.stdout.strip():
                    window_ids = result.stdout.strip().split('\n')
                    return window_ids[0]  # 返回第一个匹配的窗口
            
            # 使用wmctrl查找窗口
            elif self._check_command_available('wmctrl'):
                cmd = f"wmctrl -l | grep -i '{title}'"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0 and result.stdout.strip():
                    lines = result.stdout.strip().split('\n')
                    if lines:
                        # 解析wmctrl输出格式: 0x02000002  0 desktop
                        parts = lines[0].split()
                        if len(parts) >= 1:
                            return parts[0]
            
            return None
            
        except Exception as e:
            logger.error(f"Find window failed: {e}")
            return None
    
    def activate_window(self, window_id: str) -> bool:
        """
        激活窗口
        
        Args:
            window_id: 窗口ID
            
        Returns:
            bool: 操作是否成功
        """
        try:
            # 使用xdotool激活窗口
            if self._check_command_available('xdotool'):
                cmd = f"xdotool windowactivate {window_id}"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
                return result.returncode == 0
            
            # 使用wmctrl激活窗口
            elif self._check_command_available('wmctrl'):
                cmd = f"wmctrl -i -a {window_id}"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
                return result.returncode == 0
            
            return False
            
        except Exception as e:
            logger.error(f"Activate window failed: {e}")
            return False
    
    def get_window_info(self, window_id: str) -> Optional[Dict[str, Any]]:
        """
        获取窗口信息
        
        Args:
            window_id: 窗口ID
            
        Returns:
            Optional[Dict[str, Any]]: 窗口信息
        """
        try:
            window_info = {
                "window_id": window_id,
                "title": "Unknown",
                "position": (0, 0),
                "size": (800, 600),
                "is_visible": True,
                "is_minimized": False
            }
            
            # 使用xdotool获取窗口信息
            if self._check_command_available('xdotool'):
                # 获取窗口标题
                cmd = f"xdotool getwindowname {window_id}"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    window_info["title"] = result.stdout.strip()
                
                # 获取窗口几何信息
                cmd = f"xdotool getwindowgeometry {window_id}"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    # 解析几何信息
                    for line in result.stdout.split('\n'):
                        if 'Position:' in line:
                            pos_str = line.split('Position:')[1].strip()
                            x, y = map(int, pos_str.split(','))
                            window_info["position"] = (x, y)
                        elif 'Geometry:' in line:
                            geo_str = line.split('Geometry:')[1].strip()
                            w, h = map(int, geo_str.split('x'))
                            window_info["size"] = (w, h)
            
            return window_info
            
        except Exception as e:
            logger.error(f"Get window info failed: {e}")
            return None
    
    def minimize_window(self, window_id: str) -> bool:
        """
        最小化窗口
        
        Args:
            window_id: 窗口ID
            
        Returns:
            bool: 操作是否成功
        """
        try:
            if self._check_command_available('xdotool'):
                cmd = f"xdotool windowminimize {window_id}"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
                return result.returncode == 0
            
            return False
            
        except Exception as e:
            logger.error(f"Minimize window failed: {e}")
            return False
    
    def maximize_window(self, window_id: str) -> bool:
        """
        最大化窗口
        
        Args:
            window_id: 窗口ID
            
        Returns:
            bool: 操作是否成功
        """
        try:
            if self._check_command_available('xdotool'):
                cmd = f"xdotool windowstate {window_id} add maximized_vert maximized_horz"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
                return result.returncode == 0
            
            return False
            
        except Exception as e:
            logger.error(f"Maximize window failed: {e}")
            return False
    
    def close_window(self, window_id: str) -> bool:
        """
        关闭窗口
        
        Args:
            window_id: 窗口ID
            
        Returns:
            bool: 操作是否成功
        """
        try:
            if self._check_command_available('xdotool'):
                cmd = f"xdotool windowclose {window_id}"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
                return result.returncode == 0
            
            return False
            
        except Exception as e:
            logger.error(f"Close window failed: {e}")
            return False
    
    def move_window(self, window_id: str, x: int, y: int) -> bool:
        """
        移动窗口
        
        Args:
            window_id: 窗口ID
            x: X坐标
            y: Y坐标
            
        Returns:
            bool: 操作是否成功
        """
        try:
            if self._check_command_available('xdotool'):
                cmd = f"xdotool windowmove {window_id} {x} {y}"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
                return result.returncode == 0
            
            return False
            
        except Exception as e:
            logger.error(f"Move window failed: {e}")
            return False
    
    def resize_window(self, window_id: str, width: int, height: int) -> bool:
        """
        调整窗口大小
        
        Args:
            window_id: 窗口ID
            width: 宽度
            height: 高度
            
        Returns:
            bool: 操作是否成功
        """
        try:
            if self._check_command_available('xdotool'):
                cmd = f"xdotool windowsize {window_id} {width} {height}"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
                return result.returncode == 0
            
            return False
            
        except Exception as e:
            logger.error(f"Resize window failed: {e}")
            return False
    
    def _check_command_available(self, command: str) -> bool:
        """
        检查命令是否可用
        
        Args:
            command: 命令名称
            
        Returns:
            bool: 是否可用
        """
        try:
            result = subprocess.run(f"which {command}", shell=True, capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except Exception:
            return False
    
    def _validate_operation_platform(self, operation: str, **kwargs) -> bool:
        """
        Linux特定的操作验证
        
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
