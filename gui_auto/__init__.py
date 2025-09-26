"""
GUI自动化框架
提供跨平台的GUI自动化功能，支持图像识别、鼠标操作、键盘操作、窗口管理等
"""

from .core import GuiConfig, get_logger
from .algorithms import AlgorithmFactory, MatchResult
from .operations import OperationFactory, OperationManager
from .platform import PlatformFactory, PlatformManager
from .utils import ImageUtils, CoordinateUtils, ScaleUtils, FormatUtils

# 导入类型提示
from typing import Union, Tuple, Optional, Dict, Any
import numpy as np

# 版本信息
__version__ = "2.0.0"
__author__ = "GUI Auto Framework Team"
__email__ = "support@gui-auto-framework.com"

# 主类
class GuiAutoToolV2:
    """重构后的GUI自动化工具主类"""
    
    def __init__(self, config: GuiConfig = None):
        """
        初始化GUI自动化工具
        
        Args:
            config: 配置对象，None表示使用默认配置
        """
        self.config = config or GuiConfig()
        self.logger = get_logger("GuiAutoToolV2")
        
        # 初始化各个模块
        self._init_modules()
        
        self.logger.info(f"GuiAutoToolV2 initialized with version {__version__}")
    
    def _init_modules(self):
        """初始化各个功能模块"""
        try:
            # 初始化平台管理器
            self.platform_manager = PlatformFactory.create_platform_manager(self.config)
            
            # 初始化操作管理器
            self.operation_manager = OperationFactory.create_operation_manager(self.config)
            
            # 初始化算法工厂
            self.algorithm_factory = AlgorithmFactory
            
            # 初始化工具类
            self.image_utils = ImageUtils()
            self.coordinate_utils = CoordinateUtils()
            self.scale_utils = ScaleUtils()
            self.format_utils = FormatUtils()
            
            self.logger.debug("All modules initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Module initialization failed: {e}")
            raise
    
    # ==================== 图像操作API ====================
    
    def find_image(self, template: Union[str, np.ndarray], 
                   target: Union[str, np.ndarray] = None,
                   algorithm: str = "template",
                   confidence: float = 0.8,
                   **kwargs) -> Optional[MatchResult]:
        """
        查找图像
        
        Args:
            template: 模板图像（文件路径或numpy数组）
            target: 目标图像，None表示使用屏幕截图
            algorithm: 使用的算法 ("template", "feature", "hybrid", "pyramid")
            confidence: 置信度阈值
            **kwargs: 其他参数
            
        Returns:
            Optional[MatchResult]: 匹配结果，未找到返回None
        """
        try:
            # 创建匹配器
            matcher = self.algorithm_factory.create_matcher(self.config)
            
            # 执行匹配
            result = matcher.find(
                template=template,
                target=target,
                algorithm=algorithm,
                confidence=confidence,
                return_system_coordinates=True,
                **kwargs
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Find image failed: {e}")
            return None
    
    def click_image(self, template: Union[str, np.ndarray],
                    target: Union[str, np.ndarray] = None,
                    algorithm: str = "template",
                    confidence: float = 0.8,
                    **kwargs) -> bool:
        """
        点击图像
        
        Args:
            template: 模板图像
            target: 目标图像，None表示使用屏幕截图
            algorithm: 使用的算法
            confidence: 置信度阈值
            **kwargs: 其他参数
            
        Returns:
            bool: 操作是否成功
        """
        try:
            # 查找图像
            result = self.find_image(template, target, algorithm, confidence, **kwargs)
            
            if not result or not result.found:
                self.logger.warning("Image not found for clicking")
                return False
            
            # 执行点击
            click_result = self.operation_manager.execute_operation(
                "click", "click", result.center[0], result.center[1], **kwargs
            )
            
            return click_result.success
            
        except Exception as e:
            self.logger.error(f"Click image failed: {e}")
            return False
    
    def capture_screen(self, region: Optional[Tuple[int, int, int, int]] = None) -> Optional[np.ndarray]:
        """
        捕获屏幕截图
        
        Args:
            region: 捕获区域 (x, y, width, height)，None表示全屏
            
        Returns:
            Optional[np.ndarray]: 屏幕截图，失败返回None
        """
        try:
            result = self.operation_manager.execute_operation(
                "image", "capture_screen", region=region
            )
            
            if result.success:
                return result.data
            else:
                self.logger.error(f"Screen capture failed: {result.error}")
                return None
                
        except Exception as e:
            self.logger.error(f"Screen capture failed: {e}")
            return None
    
    # ==================== 鼠标操作API ====================
    
    def click(self, x: int, y: int, button: str = "left", **kwargs) -> bool:
        """
        点击指定坐标
        
        Args:
            x: X坐标
            y: Y坐标
            button: 鼠标按钮 ("left", "right", "middle")
            **kwargs: 其他参数
            
        Returns:
            bool: 操作是否成功
        """
        try:
            result = self.operation_manager.execute_operation(
                "click", "click", x, y, button=button, **kwargs
            )
            return result.success
            
        except Exception as e:
            self.logger.error(f"Click failed: {e}")
            return False
    
    def double_click(self, x: int, y: int, **kwargs) -> bool:
        """
        双击指定坐标
        
        Args:
            x: X坐标
            y: Y坐标
            **kwargs: 其他参数
            
        Returns:
            bool: 操作是否成功
        """
        try:
            result = self.operation_manager.execute_operation(
                "click", "double_click", x, y, **kwargs
            )
            return result.success
            
        except Exception as e:
            self.logger.error(f"Double click failed: {e}")
            return False
    
    def right_click(self, x: int, y: int, **kwargs) -> bool:
        """
        右键点击指定坐标
        
        Args:
            x: X坐标
            y: Y坐标
            **kwargs: 其他参数
            
        Returns:
            bool: 操作是否成功
        """
        try:
            result = self.operation_manager.execute_operation(
                "click", "right_click", x, y, **kwargs
            )
            return result.success
            
        except Exception as e:
            self.logger.error(f"Right click failed: {e}")
            return False
    
    def drag(self, start_x: int, start_y: int, end_x: int, end_y: int, 
             duration: float = 1.0, **kwargs) -> bool:
        """
        拖拽操作
        
        Args:
            start_x: 起始X坐标
            start_y: 起始Y坐标
            end_x: 结束X坐标
            end_y: 结束Y坐标
            duration: 拖拽持续时间
            **kwargs: 其他参数
            
        Returns:
            bool: 操作是否成功
        """
        try:
            result = self.operation_manager.execute_operation(
                "click", "drag", start_x, start_y, end_x, end_y, duration=duration, **kwargs
            )
            return result.success
            
        except Exception as e:
            self.logger.error(f"Drag failed: {e}")
            return False
    
    # ==================== 键盘操作API ====================
    
    def type_text(self, text: str, delay: float = 0.05, **kwargs) -> bool:
        """
        输入文本
        
        Args:
            text: 要输入的文本
            delay: 按键延迟
            **kwargs: 其他参数
            
        Returns:
            bool: 操作是否成功
        """
        try:
            result = self.operation_manager.execute_operation(
                "keyboard", "type", text, delay=delay, **kwargs
            )
            return result.success
            
        except Exception as e:
            self.logger.error(f"Type text failed: {e}")
            return False
    
    def press_key(self, key: str, **kwargs) -> bool:
        """
        按下按键
        
        Args:
            key: 按键名称
            **kwargs: 其他参数
            
        Returns:
            bool: 操作是否成功
        """
        try:
            result = self.operation_manager.execute_operation(
                "keyboard", "press", key, **kwargs
            )
            return result.success
            
        except Exception as e:
            self.logger.error(f"Press key failed: {e}")
            return False
    
    def hotkey(self, *keys: str, **kwargs) -> bool:
        """
        按下快捷键组合
        
        Args:
            *keys: 按键组合
            **kwargs: 其他参数
            
        Returns:
            bool: 操作是否成功
        """
        try:
            result = self.operation_manager.execute_operation(
                "keyboard", "hotkey", *keys, **kwargs
            )
            return result.success
            
        except Exception as e:
            self.logger.error(f"Hotkey failed: {e}")
            return False
    
    # ==================== 窗口操作API ====================
    
    def find_window(self, title: str, **kwargs) -> Optional[str]:
        """
        查找窗口
        
        Args:
            title: 窗口标题
            **kwargs: 其他参数
            
        Returns:
            Optional[str]: 窗口ID，未找到返回None
        """
        try:
            result = self.operation_manager.execute_operation(
                "window", "find_window", title, **kwargs
            )
            
            if result.success:
                return result.data
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Find window failed: {e}")
            return None
    
    def activate_window(self, window_id: str, **kwargs) -> bool:
        """
        激活窗口
        
        Args:
            window_id: 窗口ID
            **kwargs: 其他参数
            
        Returns:
            bool: 操作是否成功
        """
        try:
            result = self.operation_manager.execute_operation(
                "window", "activate", window_id, **kwargs
            )
            return result.success
            
        except Exception as e:
            self.logger.error(f"Activate window failed: {e}")
            return False
    
    def maximize_window(self, window_id: str, **kwargs) -> bool:
        """
        最大化窗口
        
        Args:
            window_id: 窗口ID
            **kwargs: 其他参数
            
        Returns:
            bool: 操作是否成功
        """
        try:
            result = self.operation_manager.execute_operation(
                "window", "maximize", window_id, **kwargs
            )
            return result.success
            
        except Exception as e:
            self.logger.error(f"Maximize window failed: {e}")
            return False
    
    def minimize_window(self, window_id: str, **kwargs) -> bool:
        """
        最小化窗口
        
        Args:
            window_id: 窗口ID
            **kwargs: 其他参数
            
        Returns:
            bool: 操作是否成功
        """
        try:
            result = self.operation_manager.execute_operation(
                "window", "minimize", window_id, **kwargs
            )
            return result.success
            
        except Exception as e:
            self.logger.error(f"Minimize window failed: {e}")
            return False
    
    # ==================== 高级功能API ====================
    
    def wait_for_image(self, template: Union[str, np.ndarray],
                       timeout: float = 10.0,
                       interval: float = 0.5,
                       **kwargs) -> Optional[MatchResult]:
        """
        等待图像出现
        
        Args:
            template: 模板图像
            timeout: 超时时间
            interval: 检查间隔
            **kwargs: 其他参数
            
        Returns:
            Optional[MatchResult]: 匹配结果，超时返回None
        """
        try:
            import time
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                result = self.find_image(template, **kwargs)
                if result and result.found:
                    return result
                
                time.sleep(interval)
            
            self.logger.warning(f"Image not found within {timeout} seconds")
            return None
            
        except Exception as e:
            self.logger.error(f"Wait for image failed: {e}")
            return None
    
    def compare_images(self, image1: Union[str, np.ndarray], 
                       image2: Union[str, np.ndarray],
                       method: str = "ssim", **kwargs) -> Optional[float]:
        """
        比较两个图像
        
        Args:
            image1: 第一个图像
            image2: 第二个图像
            method: 比较方法 ("ssim", "mse", "histogram")
            **kwargs: 其他参数
            
        Returns:
            Optional[float]: 相似度分数，失败返回None
        """
        try:
            result = self.operation_manager.execute_operation(
                "image", "compare_images", image1, image2, method=method, **kwargs
            )
            
            if result.success:
                return result.data["similarity"]
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Compare images failed: {e}")
            return None
    
    # ==================== 配置和状态API ====================
    
    def get_platform_info(self) -> Dict[str, Any]:
        """
        获取平台信息
        
        Returns:
            Dict[str, Any]: 平台信息
        """
        try:
            platform_info = self.platform_manager.get_platform_info()
            return {
                "name": platform_info.name,
                "version": platform_info.version,
                "architecture": platform_info.architecture,
                "is_supported": platform_info.is_supported,
                "features": platform_info.features,
                "metadata": platform_info.metadata
            }
        except Exception as e:
            self.logger.error(f"Get platform info failed: {e}")
            return {}
    
    def get_screen_size(self) -> Tuple[int, int]:
        """
        获取屏幕尺寸
        
        Returns:
            Tuple[int, int]: 屏幕尺寸 (width, height)
        """
        try:
            return self.platform_manager.execute_operation("get_screen_size")
        except Exception as e:
            self.logger.error(f"Get screen size failed: {e}")
            return (1920, 1080)  # 默认尺寸
    
    def get_dpi_scale(self) -> float:
        """
        获取DPI缩放比例
        
        Returns:
            float: DPI缩放比例
        """
        try:
            return self.platform_manager.execute_operation("get_dpi_scale")
        except Exception as e:
            self.logger.error(f"Get DPI scale failed: {e}")
            return 1.0  # 默认无缩放
    
    def set_config(self, config: GuiConfig) -> None:
        """
        设置配置
        
        Args:
            config: 新的配置
        """
        try:
            self.config = config
            # 重新初始化模块
            self._init_modules()
            self.logger.info("Configuration updated successfully")
        except Exception as e:
            self.logger.error(f"Set config failed: {e}")
            raise
    
    def get_config(self) -> GuiConfig:
        """
        获取当前配置
        
        Returns:
            GuiConfig: 当前配置
        """
        return self.config
    
    # ==================== 便捷属性 ====================
    
    @property
    def screen_size(self) -> Tuple[int, int]:
        """获取屏幕尺寸（便捷属性）"""
        return self.get_screen_size()
    
    @property
    def dpi_scale(self) -> float:
        """获取DPI缩放比例（便捷属性）"""
        return self.get_dpi_scale()

# 主类别名 - 简化API
GuiAutoTool = GuiAutoToolV2

# 便捷函数
def create_tool(config: GuiConfig = None) -> GuiAutoToolV2:
    """
    创建GUI自动化工具实例
    
    Args:
        config: 配置对象
        
    Returns:
        GuiAutoToolV2: 工具实例
    """
    return GuiAutoToolV2(config)


def get_version() -> str:
    """
    获取框架版本
    
    Returns:
        str: 版本号
    """
    return __version__


# 导出主要类和函数
__all__ = [
    # 主类
    'GuiAutoToolV2',
    'GuiAutoTool',  # 别名
    'create_tool',
    'get_version',
    
    # 核心模块
    'GuiConfig',
    'get_logger',
    
    # 算法模块
    'AlgorithmFactory',
    'MatchResult',
    
    # 操作模块
    'OperationFactory',
    'OperationManager',
    
    # 平台模块
    'PlatformFactory',
    'PlatformManager',
    
    # 工具模块
    'ImageUtils',
    'CoordinateUtils',
    'ScaleUtils',
    'FormatUtils',
    
    # 版本信息
    '__version__',
    '__author__',
    '__email__'
]
