"""
平台基类模块
定义平台接口和基础功能
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union, Tuple, Optional, Dict, Any, List
import logging
import platform as sys_platform

from ..core.exceptions import PlatformError

logger = logging.getLogger(__name__)


@dataclass
class PlatformInfo:
    """平台信息类"""
    name: str
    version: str
    architecture: str
    is_supported: bool
    features: List[str]
    metadata: Optional[Dict[str, Any]] = None


class PlatformBase(ABC):
    """平台基类"""
    
    def __init__(self, config: Optional[Any] = None):
        """
        初始化平台
        
        Args:
            config: 平台配置
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self._platform_info = None
        self._init_platform()
    
    def _init_platform(self):
        """初始化平台"""
        try:
            self._platform_info = self._detect_platform()
            self.logger.debug(f"Platform initialized: {self._platform_info.name}")
        except Exception as e:
            self.logger.error(f"Platform initialization failed: {e}")
            raise PlatformError(f"Platform initialization failed: {e}")
    
    @abstractmethod
    def _detect_platform(self) -> PlatformInfo:
        """
        检测平台信息
        
        Returns:
            PlatformInfo: 平台信息
        """
        pass
    
    @abstractmethod
    def get_screen_size(self) -> Tuple[int, int]:
        """
        获取屏幕尺寸
        
        Returns:
            Tuple[int, int]: 屏幕尺寸 (width, height)
        """
        pass
    
    @abstractmethod
    def get_dpi_scale(self) -> float:
        """
        获取DPI缩放比例
        
        Returns:
            float: DPI缩放比例
        """
        pass
    
    @abstractmethod
    def capture_screen(self, region: Optional[Tuple[int, int, int, int]] = None) -> Any:
        """
        捕获屏幕截图
        
        Args:
            region: 捕获区域 (x, y, width, height)，None表示全屏
            
        Returns:
            Any: 屏幕截图
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def type_text(self, text: str, delay: float = 0.05) -> bool:
        """
        输入文本
        
        Args:
            text: 要输入的文本
            delay: 按键延迟
            
        Returns:
            bool: 操作是否成功
        """
        pass
    
    @abstractmethod
    def press_key(self, key: str) -> bool:
        """
        按下按键
        
        Args:
            key: 按键名称
            
        Returns:
            bool: 操作是否成功
        """
        pass
    
    @abstractmethod
    def hotkey(self, *keys: str) -> bool:
        """
        按下快捷键组合
        
        Args:
            *keys: 按键组合
            
        Returns:
            bool: 操作是否成功
        """
        pass
    
    @abstractmethod
    def find_window(self, title: str) -> Optional[str]:
        """
        查找窗口
        
        Args:
            title: 窗口标题
            
        Returns:
            Optional[str]: 窗口ID，未找到返回None
        """
        pass
    
    @abstractmethod
    def activate_window(self, window_id: str) -> bool:
        """
        激活窗口
        
        Args:
            window_id: 窗口ID
            
        Returns:
            bool: 操作是否成功
        """
        pass
    
    def get_platform_info(self) -> PlatformInfo:
        """
        获取平台信息
        
        Returns:
            PlatformInfo: 平台信息
        """
        return self._platform_info
    
    def is_feature_supported(self, feature: str) -> bool:
        """
        检查功能是否支持
        
        Args:
            feature: 功能名称
            
        Returns:
            bool: 是否支持
        """
        return feature in self._platform_info.features
    
    def get_supported_features(self) -> List[str]:
        """
        获取支持的功能列表
        
        Returns:
            List[str]: 功能列表
        """
        return self._platform_info.features.copy()
    
    def validate_operation(self, operation: str, **kwargs) -> bool:
        """
        验证操作是否支持
        
        Args:
            operation: 操作名称
            **kwargs: 操作参数
            
        Returns:
            bool: 是否支持
        """
        try:
            # 基础验证
            if not self.is_feature_supported(operation):
                return False
            
            # 平台特定验证
            return self._validate_operation_platform(operation, **kwargs)
        except Exception as e:
            self.logger.error(f"Operation validation failed: {e}")
            return False
    
    def _validate_operation_platform(self, operation: str, **kwargs) -> bool:
        """
        平台特定的操作验证
        
        Args:
            operation: 操作名称
            **kwargs: 操作参数
            
        Returns:
            bool: 是否支持
        """
        return True
    
    def safe_execute(self, operation: str, *args, **kwargs) -> Any:
        """
        安全执行操作
        
        Args:
            operation: 操作名称
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            Any: 操作结果
        """
        try:
            # 验证操作
            if not self.validate_operation(operation, **kwargs):
                raise PlatformError(f"Operation not supported: {operation}")
            
            # 执行操作
            return getattr(self, operation)(*args, **kwargs)
            
        except Exception as e:
            self.logger.error(f"Safe execution failed: {e}")
            raise PlatformError(f"Operation execution failed: {e}")


class PlatformFactory:
    """平台工厂类"""
    
    _platforms = {}
    
    # 注册默认平台
    @classmethod
    def _register_default_platforms(cls):
        """注册默认平台"""
        from .windows import WindowsPlatform
        from .linux import LinuxPlatform
        
        cls.register_platform("windows", WindowsPlatform)
        cls.register_platform("linux", LinuxPlatform)
    
    @classmethod
    def _ensure_default_platforms(cls):
        """确保默认平台已注册"""
        if not cls._platforms:
            cls._register_default_platforms()
    
    @classmethod
    def register_platform(cls, name: str, platform_class: type) -> None:
        """
        注册平台
        
        Args:
            name: 平台名称
            platform_class: 平台类
        """
        cls._platforms[name] = platform_class
        logger.debug(f"Platform registered: {name}")
    
    @classmethod
    def create_platform(cls, name: Optional[str] = None, config: Optional[Any] = None) -> PlatformBase:
        """
        创建平台实例
        
        Args:
            name: 平台名称，None表示自动检测
            config: 平台配置
            
        Returns:
            PlatformBase: 平台实例
        """
        cls._ensure_default_platforms()
        
        if name is None:
            name = cls._detect_current_platform()
        
        if name not in cls._platforms:
            raise PlatformError(f"Unknown platform: {name}")
        
        platform_class = cls._platforms[name]
        return platform_class(config)
    
    @classmethod
    def _detect_current_platform(cls) -> str:
        """
        检测当前平台
        
        Returns:
            str: 平台名称
        """
        system = sys_platform.system().lower()
        if system == "windows":
            return "windows"
        elif system == "linux":
            return "linux"
        else:
            raise PlatformError(f"Unsupported platform: {system}")
    
    @classmethod
    def get_available_platforms(cls) -> List[str]:
        """
        获取可用的平台列表
        
        Returns:
            List[str]: 平台名称列表
        """
        cls._ensure_default_platforms()
        return list(cls._platforms.keys())
    
    @classmethod
    def create_platform_manager(cls, config: Optional[Any] = None) -> 'PlatformManager':
        """
        创建平台管理器
        
        Args:
            config: 配置
            
        Returns:
            PlatformManager: 平台管理器实例
        """
        return PlatformManager(config)


class PlatformManager:
    """平台管理器类 - 统一管理平台功能"""
    
    def __init__(self, config: Optional[Any] = None):
        """
        初始化平台管理器
        
        Args:
            config: 配置
        """
        self.config = config
        self.logger = logging.getLogger("PlatformManager")
        self._platform = None
        self._init_platform()
    
    def _init_platform(self):
        """初始化平台"""
        try:
            self._platform = PlatformFactory.create_platform(config=self.config)
            self.logger.debug(f"Platform manager initialized with {self._platform.get_platform_info().name}")
        except Exception as e:
            self.logger.error(f"Platform manager initialization failed: {e}")
            raise PlatformError(f"Platform manager initialization failed: {e}")
    
    def get_platform(self) -> PlatformBase:
        """
        获取当前平台实例
        
        Returns:
            PlatformBase: 平台实例
        """
        return self._platform
    
    def get_platform_info(self) -> PlatformInfo:
        """
        获取平台信息
        
        Returns:
            PlatformInfo: 平台信息
        """
        return self._platform.get_platform_info()
    
    def is_feature_supported(self, feature: str) -> bool:
        """
        检查功能是否支持
        
        Args:
            feature: 功能名称
            
        Returns:
            bool: 是否支持
        """
        return self._platform.is_feature_supported(feature)
    
    def execute_operation(self, operation: str, *args, **kwargs) -> Any:
        """
        执行平台操作
        
        Args:
            operation: 操作名称
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            Any: 操作结果
        """
        try:
            # 直接调用平台方法，而不是通过safe_execute
            if hasattr(self._platform, operation):
                method = getattr(self._platform, operation)
                return method(*args, **kwargs)
            else:
                raise PlatformError(f"Operation not supported: {operation}")
        except Exception as e:
            self.logger.error(f"Platform operation failed: {e}")
            raise PlatformError(f"Platform operation failed: {e}")
    
    def switch_platform(self, platform_name: str) -> bool:
        """
        切换平台
        
        Args:
            platform_name: 平台名称
            
        Returns:
            bool: 是否成功
        """
        try:
            self._platform = PlatformFactory.create_platform(platform_name, self.config)
            self.logger.debug(f"Switched to platform: {platform_name}")
            return True
        except Exception as e:
            self.logger.error(f"Platform switch failed: {e}")
            return False
