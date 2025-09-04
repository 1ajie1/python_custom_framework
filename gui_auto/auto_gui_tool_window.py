"""
Windows 窗口操作自动化工具
提供常用的窗口操作功能，如激活、最大化、置顶、等待等
"""

import time
import logging
import atexit
from typing import Optional, List, Tuple, Callable, Any
from ctypes import windll, wintypes, byref, create_unicode_buffer, sizeof
from ctypes.wintypes import HWND, LPARAM, DWORD, RECT, POINT
import ctypes


# Windows API 常量
SW_HIDE = 0
SW_SHOWNORMAL = 1
SW_SHOWMINIMIZED = 2
SW_SHOWMAXIMIZED = 3
SW_SHOWNOACTIVATE = 4
SW_SHOW = 5
SW_MINIMIZE = 6
SW_SHOWMINNOACTIVE = 7
SW_SHOWNA = 8
SW_RESTORE = 9
SW_SHOWDEFAULT = 10

HWND_TOP = 0
HWND_BOTTOM = 1
HWND_TOPMOST = -1
HWND_NOTOPMOST = -2

SWP_NOSIZE = 0x0001
SWP_NOMOVE = 0x0002
SWP_NOZORDER = 0x0004
SWP_NOREDRAW = 0x0008
SWP_NOACTIVATE = 0x0010
SWP_FRAMECHANGED = 0x0020
SWP_SHOWWINDOW = 0x0040
SWP_HIDEWINDOW = 0x0080

# 电源管理相关常量
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002
ES_AWAYMODE_REQUIRED = 0x00000040
ES_USER_PRESENT = 0x00000004


class WindowInfo:
    """窗口信息类"""
    
    def __init__(self, hwnd: int, title: str = "", class_name: str = "", 
                 process_id: int = 0, thread_id: int = 0):
        self.hwnd = hwnd
        self.title = title
        self.class_name = class_name
        self.process_id = process_id
        self.thread_id = thread_id
    
    def __str__(self):
        return f"WindowInfo(hwnd={self.hwnd}, title='{self.title}', class='{self.class_name}', pid={self.process_id})"


class WindowTool:
    """Windows 窗口操作工具类"""
    
    def __init__(self, enable_logging: bool = True):
        """
        初始化窗口工具
        
        Args:
            enable_logging: 是否启用日志记录
        """
        self.logger = self._setup_logger(enable_logging)
        
        # 加载 Windows API
        self.user32 = windll.user32
        self.kernel32 = windll.kernel32
        
        # 设置函数原型
        self._setup_api_prototypes()
        
        # 屏幕常亮状态管理
        self._screen_always_on = False
        self._original_thread_execution_state = None
        
        # 注册程序退出时的清理函数
        atexit.register(self._cleanup_on_exit)
    
    def _setup_logger(self, enable_logging: bool) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger("WindowTool")
        if enable_logging and not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _setup_api_prototypes(self):
        """设置 Windows API 函数原型"""
        # EnumWindows 回调函数类型
        self.WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, HWND, LPARAM)
        
        # 设置函数参数和返回值类型
        self.user32.EnumWindows.argtypes = [self.WNDENUMPROC, LPARAM]
        self.user32.EnumWindows.restype = ctypes.c_bool
        
        self.user32.GetWindowTextW.argtypes = [HWND, ctypes.c_wchar_p, ctypes.c_int]
        self.user32.GetWindowTextW.restype = ctypes.c_int
        
        self.user32.GetClassNameW.argtypes = [HWND, ctypes.c_wchar_p, ctypes.c_int]
        self.user32.GetClassNameW.restype = ctypes.c_int
        
        self.user32.GetWindowThreadProcessId.argtypes = [HWND, ctypes.POINTER(DWORD)]
        self.user32.GetWindowThreadProcessId.restype = DWORD
        
        # 电源管理 API
        self.kernel32.SetThreadExecutionState.argtypes = [ctypes.c_uint32]
        self.kernel32.SetThreadExecutionState.restype = ctypes.c_uint32
    
    def get_window_text(self, hwnd: int) -> str:
        """获取窗口标题"""
        try:
            length = self.user32.GetWindowTextLengthW(hwnd)
            if length == 0:
                return ""
            
            buffer = create_unicode_buffer(length + 1)
            self.user32.GetWindowTextW(hwnd, buffer, length + 1)
            return buffer.value
        except Exception as e:
            self.logger.error(f"获取窗口标题失败: {e}")
            return ""
    
    def get_class_name(self, hwnd: int) -> str:
        """获取窗口类名"""
        try:
            buffer = create_unicode_buffer(256)
            self.user32.GetClassNameW(hwnd, buffer, 256)
            return buffer.value
        except Exception as e:
            self.logger.error(f"获取窗口类名失败: {e}")
            return ""
    
    def get_window_process_id(self, hwnd: int) -> Tuple[int, int]:
        """获取窗口的进程ID和线程ID"""
        try:
            process_id = DWORD()
            thread_id = self.user32.GetWindowThreadProcessId(hwnd, byref(process_id))
            return process_id.value, thread_id
        except Exception as e:
            self.logger.error(f"获取窗口进程ID失败: {e}")
            return 0, 0
    
    def is_window_visible(self, hwnd: int) -> bool:
        """检查窗口是否可见"""
        try:
            return bool(self.user32.IsWindowVisible(hwnd))
        except Exception as e:
            self.logger.error(f"检查窗口可见性失败: {e}")
            return False
    
    def is_window_maximized(self, hwnd: int) -> bool:
        """检查窗口是否最大化"""
        try:
            return bool(self.user32.IsZoomed(hwnd))
        except Exception as e:
            self.logger.error(f"检查窗口最大化状态失败: {e}")
            return False
    
    def is_window_minimized(self, hwnd: int) -> bool:
        """检查窗口是否最小化"""
        try:
            return bool(self.user32.IsIconic(hwnd))
        except Exception as e:
            self.logger.error(f"检查窗口最小化状态失败: {e}")
            return False
    
    def get_window_rect(self, hwnd: int) -> Optional[Tuple[int, int, int, int]]:
        """获取窗口位置和大小"""
        try:
            rect = RECT()
            if self.user32.GetWindowRect(hwnd, byref(rect)):
                return rect.left, rect.top, rect.right, rect.bottom
            return None
        except Exception as e:
            self.logger.error(f"获取窗口位置失败: {e}")
            return None
    
    def get_all_windows(self, visible_only: bool = True) -> List[WindowInfo]:
        """获取所有窗口信息"""
        windows = []
        
        def enum_proc(hwnd, lParam):
            try:
                if visible_only and not self.is_window_visible(hwnd):
                    return True
                
                title = self.get_window_text(hwnd)
                class_name = self.get_class_name(hwnd)
                process_id, thread_id = self.get_window_process_id(hwnd)
                
                window_info = WindowInfo(hwnd, title, class_name, process_id, thread_id)
                windows.append(window_info)
            except Exception as e:
                self.logger.error(f"枚举窗口时出错: {e}")
            
            return True
        
        try:
            callback = self.WNDENUMPROC(enum_proc)
            self.user32.EnumWindows(callback, 0)
        except Exception as e:
            self.logger.error(f"枚举窗口失败: {e}")
        
        return windows
    
    def find_windows_by_title(self, title: str, exact_match: bool = False) -> List[WindowInfo]:
        """根据标题查找窗口"""
        windows = self.get_all_windows()
        result = []
        
        for window in windows:
            if exact_match:
                if window.title == title:
                    result.append(window)
            else:
                if title.lower() in window.title.lower():
                    result.append(window)
        
        return result
    
    def find_windows_by_class(self, class_name: str, exact_match: bool = True) -> List[WindowInfo]:
        """根据类名查找窗口"""
        windows = self.get_all_windows()
        result = []
        
        for window in windows:
            if exact_match:
                if window.class_name == class_name:
                    result.append(window)
            else:
                if class_name.lower() in window.class_name.lower():
                    result.append(window)
        
        return result
    
    def find_windows_by_process_id(self, process_id: int) -> List[WindowInfo]:
        """根据进程ID查找窗口"""
        windows = self.get_all_windows()
        return [window for window in windows if window.process_id == process_id]
    
    def activate_window(self, hwnd: int) -> bool:
        """激活窗口"""
        try:
            # 如果窗口最小化，先还原
            if self.is_window_minimized(hwnd):
                self.user32.ShowWindow(hwnd, SW_RESTORE)
            
            # 激活窗口
            result = self.user32.SetForegroundWindow(hwnd)
            if result:
                self.logger.info(f"成功激活窗口: {hwnd}")
                return True
            else:
                self.logger.warning(f"激活窗口失败: {hwnd}")
                return False
        except Exception as e:
            self.logger.error(f"激活窗口时出错: {e}")
            return False
    
    def maximize_window(self, hwnd: int) -> bool:
        """最大化窗口"""
        try:
            result = self.user32.ShowWindow(hwnd, SW_SHOWMAXIMIZED)
            if result or self.is_window_maximized(hwnd):
                self.logger.info(f"成功最大化窗口: {hwnd}")
                return True
            else:
                self.logger.warning(f"最大化窗口失败: {hwnd}")
                return False
        except Exception as e:
            self.logger.error(f"最大化窗口时出错: {e}")
            return False
    
    def minimize_window(self, hwnd: int) -> bool:
        """最小化窗口"""
        try:
            result = self.user32.ShowWindow(hwnd, SW_SHOWMINIMIZED)
            if result:
                self.logger.info(f"成功最小化窗口: {hwnd}")
                return True
            else:
                self.logger.warning(f"最小化窗口失败: {hwnd}")
                return False
        except Exception as e:
            self.logger.error(f"最小化窗口时出错: {e}")
            return False
    
    def restore_window(self, hwnd: int) -> bool:
        """还原窗口"""
        try:
            result = self.user32.ShowWindow(hwnd, SW_RESTORE)
            if result:
                self.logger.info(f"成功还原窗口: {hwnd}")
                return True
            else:
                self.logger.warning(f"还原窗口失败: {hwnd}")
                return False
        except Exception as e:
            self.logger.error(f"还原窗口时出错: {e}")
            return False
    
    def set_window_topmost(self, hwnd: int, topmost: bool = True) -> bool:
        """设置窗口置顶"""
        try:
            z_order = HWND_TOPMOST if topmost else HWND_NOTOPMOST
            result = self.user32.SetWindowPos(
                hwnd, z_order, 0, 0, 0, 0,
                SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE
            )
            if result:
                action = "置顶" if topmost else "取消置顶"
                self.logger.info(f"成功{action}窗口: {hwnd}")
                return True
            else:
                action = "置顶" if topmost else "取消置顶"
                self.logger.warning(f"{action}窗口失败: {hwnd}")
                return False
        except Exception as e:
            self.logger.error(f"设置窗口置顶时出错: {e}")
            return False
    
    def move_window(self, hwnd: int, x: int, y: int, width: int = None, height: int = None) -> bool:
        """移动和调整窗口大小"""
        try:
            if width is None or height is None:
                # 只移动位置，不改变大小
                result = self.user32.SetWindowPos(
                    hwnd, 0, x, y, 0, 0,
                    SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE
                )
            else:
                # 同时移动位置和调整大小
                result = self.user32.SetWindowPos(
                    hwnd, 0, x, y, width, height,
                    SWP_NOZORDER | SWP_NOACTIVATE
                )
            
            if result:
                self.logger.info(f"成功移动窗口: {hwnd} 到位置 ({x}, {y})")
                return True
            else:
                self.logger.warning(f"移动窗口失败: {hwnd}")
                return False
        except Exception as e:
            self.logger.error(f"移动窗口时出错: {e}")
            return False
    
    def close_window(self, hwnd: int) -> bool:
        """关闭窗口"""
        try:
            WM_CLOSE = 0x0010
            result = self.user32.SendMessageW(hwnd, WM_CLOSE, 0, 0)
            self.logger.info(f"发送关闭消息到窗口: {hwnd}")
            return True
        except Exception as e:
            self.logger.error(f"关闭窗口时出错: {e}")
            return False
    
    def wait_for_window(self, title: str = None, class_name: str = None, 
                       check_interval: float = 0.5, timeout: float = None) -> Optional[WindowInfo]:
        """等待窗口出现"""
        start_time = time.time()
        
        if timeout is None:
            while True:
                if title:
                    windows = self.find_windows_by_title(title, exact_match=False)
                elif class_name:
                    windows = self.find_windows_by_class(class_name, exact_match=True)
                else:
                    raise ValueError("必须指定 title 或 class_name 参数")

                if windows:
                    self.logger.info(f"找到目标窗口: {windows[0]}")
                    return windows[0]
                
                time.sleep(check_interval)
        else:
            while time.time() - start_time < timeout:
                if title:
                    windows = self.find_windows_by_title(title, exact_match=False)
                elif class_name:
                    windows = self.find_windows_by_class(class_name, exact_match=True)
                else:
                    raise ValueError("必须指定 title 或 class_name 参数")
                
                if windows:
                    self.logger.info(f"找到目标窗口: {windows[0]}")
                    return windows[0]
            
                time.sleep(check_interval)
        
        self.logger.warning(f"等待窗口超时: title={title}, class_name={class_name}")
        return None
    
    def wait_for_window_close(self, hwnd: int, timeout: float = 30.0, 
                             check_interval: float = 0.5) -> bool:
        """等待窗口关闭"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if not self.user32.IsWindow(hwnd):
                self.logger.info(f"窗口已关闭: {hwnd}")
                return True
            
            time.sleep(check_interval)
        
        self.logger.warning(f"等待窗口关闭超时: {hwnd}")
        return False
    
    def monitor_window_state(self, hwnd: int, callback: Callable[[WindowInfo], Any], 
                           interval: float = 1.0, duration: float = None) -> None:
        """监控窗口状态变化"""
        start_time = time.time()
        last_state = None
        
        while True:
            if duration and time.time() - start_time >= duration:
                break
            
            if not self.user32.IsWindow(hwnd):
                self.logger.info(f"窗口已不存在，停止监控: {hwnd}")
                break
            
            # 获取当前窗口状态
            title = self.get_window_text(hwnd)
            class_name = self.get_class_name(hwnd)
            process_id, thread_id = self.get_window_process_id(hwnd)
            
            current_state = WindowInfo(hwnd, title, class_name, process_id, thread_id)
            
            # 检查状态是否发生变化
            if last_state is None or (
                current_state.title != last_state.title or
                current_state.class_name != last_state.class_name
            ):
                try:
                    callback(current_state)
                except Exception as e:
                    self.logger.error(f"回调函数执行出错: {e}")
            
            last_state = current_state
            time.sleep(interval)
    
    def get_desktop_window(self) -> int:
        """获取桌面窗口句柄"""
        return self.user32.GetDesktopWindow()
    
    def get_foreground_window(self) -> Optional[WindowInfo]:
        """获取当前前台窗口"""
        try:
            hwnd = self.user32.GetForegroundWindow()
            if hwnd:
                title = self.get_window_text(hwnd)
                class_name = self.get_class_name(hwnd)
                process_id, thread_id = self.get_window_process_id(hwnd)
                return WindowInfo(hwnd, title, class_name, process_id, thread_id)
            return None
        except Exception as e:
            self.logger.error(f"获取前台窗口失败: {e}")
            return None
    
    def _cleanup_on_exit(self):
        """程序退出时的清理函数"""
        if self._screen_always_on:
            self.restore_screen_power_settings()
            self.logger.info("程序退出时自动恢复屏幕电源设置")
    
    def set_screen_always_on(self) -> bool:
        """
        设置屏幕常亮，阻止系统进入休眠和关闭显示器
        
        Returns:
            bool: 设置是否成功
        """
        try:
            # 保存当前的执行状态
            if self._original_thread_execution_state is None:
                self._original_thread_execution_state = self.kernel32.SetThreadExecutionState(
                    ES_CONTINUOUS
                )
            
            # 设置屏幕常亮：阻止系统休眠和显示器关闭
            new_state = ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
            result = self.kernel32.SetThreadExecutionState(new_state)
            
            if result != 0:
                self._screen_always_on = True
                self.logger.info("已启用屏幕常亮模式")
                return True
            else:
                self.logger.error("设置屏幕常亮失败")
                return False
                
        except Exception as e:
            self.logger.error(f"设置屏幕常亮时出错: {e}")
            return False
    
    def restore_screen_power_settings(self) -> bool:
        """
        恢复屏幕电源设置到原始状态
        
        Returns:
            bool: 恢复是否成功
        """
        try:
            if not self._screen_always_on:
                self.logger.info("屏幕常亮未启用，无需恢复")
                return True
            
            # 恢复到原始的执行状态
            result = self.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
            
            if result != 0:
                self._screen_always_on = False
                self.logger.info("已恢复屏幕电源设置")
                return True
            else:
                self.logger.error("恢复屏幕电源设置失败")
                return False
                
        except Exception as e:
            self.logger.error(f"恢复屏幕电源设置时出错: {e}")
            return False
    
    def is_screen_always_on(self) -> bool:
        """
        检查屏幕常亮模式是否启用
        
        Returns:
            bool: 屏幕常亮模式是否启用
        """
        return self._screen_always_on
    
    def toggle_screen_always_on(self) -> bool:
        """
        切换屏幕常亮模式
        
        Returns:
            bool: 操作是否成功
        """
        if self._screen_always_on:
            return self.restore_screen_power_settings()
        else:
            return self.set_screen_always_on()
