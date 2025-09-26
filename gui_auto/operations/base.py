"""
操作基类模块
定义操作接口和基础功能
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Dict, Union
import logging

from ..core.exceptions import OperationError

logger = logging.getLogger(__name__)


@dataclass
class OperationResult:
    """操作结果类"""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """验证结果数据"""
        if not self.success and not self.error:
            self.error = "Operation failed without specific error message"


class Operation(ABC):
    """操作基类"""
    
    def __init__(self, config: Optional[Any] = None):
        """
        初始化操作
        
        Args:
            config: 操作配置
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> OperationResult:
        """
        执行操作
        
        Returns:
            OperationResult: 操作结果
        """
        pass
    
    def validate_input(self, *args, **kwargs) -> bool:
        """
        验证输入参数
        
        Args:
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            bool: 输入是否有效
        """
        return True
    
    def pre_execute(self, *args, **kwargs) -> None:
        """
        执行前的准备工作
        
        Args:
            *args: 位置参数
            **kwargs: 关键字参数
        """
        pass
    
    def post_execute(self, result: OperationResult, *args, **kwargs) -> OperationResult:
        """
        执行后的清理工作
        
        Args:
            result: 操作结果
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            OperationResult: 处理后的结果
        """
        return result
    
    def safe_execute(self, *args, **kwargs) -> OperationResult:
        """
        安全执行操作（包含错误处理）
        
        Args:
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            OperationResult: 操作结果
        """
        try:
            # 验证输入
            if not self.validate_input(*args, **kwargs):
                return OperationResult(
                    success=False,
                    error="Input validation failed"
                )
            
            # 执行前准备
            self.pre_execute(*args, **kwargs)
            
            # 执行操作
            result = self.execute(*args, **kwargs)
            
            # 执行后处理
            result = self.post_execute(result, *args, **kwargs)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Operation failed: {e}")
            return OperationResult(
                success=False,
                error=str(e)
            )
    
    def get_name(self) -> str:
        """
        获取操作名称
        
        Returns:
            str: 操作名称
        """
        return self.__class__.__name__
    
    def get_config(self) -> Optional[Any]:
        """
        获取操作配置
        
        Returns:
            Optional[Any]: 操作配置
        """
        return self.config
    
    def set_config(self, config: Any) -> None:
        """
        设置操作配置
        
        Args:
            config: 新的配置
        """
        self.config = config


class OperationFactory:
    """操作工厂类"""
    
    _operations = {}
    
    # 注册默认操作
    @classmethod
    def _register_default_operations(cls):
        """注册默认操作"""
        from .image_operations import ImageOperations
        from .click_operations import ClickOperations
        from .keyboard_operations import KeyboardOperations
        from .window_operations import WindowOperations
        
        cls.register_operation("image", ImageOperations)
        cls.register_operation("click", ClickOperations)
        cls.register_operation("keyboard", KeyboardOperations)
        cls.register_operation("window", WindowOperations)
    
    @classmethod
    def _ensure_default_operations(cls):
        """确保默认操作已注册"""
        if not cls._operations:
            cls._register_default_operations()
    
    @classmethod
    def register_operation(cls, name: str, operation_class: type) -> None:
        """
        注册操作
        
        Args:
            name: 操作名称
            operation_class: 操作类
        """
        cls._operations[name] = operation_class
        logger.debug(f"Operation registered: {name}")
    
    @classmethod
    def create_operation(cls, name: str, config: Optional[Any] = None) -> Operation:
        """
        创建操作实例
        
        Args:
            name: 操作名称
            config: 操作配置
            
        Returns:
            Operation: 操作实例
        """
        cls._ensure_default_operations()
        
        if name not in cls._operations:
            raise OperationError(f"Unknown operation: {name}")
        
        operation_class = cls._operations[name]
        return operation_class(config)
    
    @classmethod
    def get_available_operations(cls) -> list:
        """
        获取可用的操作列表
        
        Returns:
            list: 操作名称列表
        """
        cls._ensure_default_operations()
        return list(cls._operations.keys())
    
    @classmethod
    def create_operation_manager(cls, config: Optional[Any] = None) -> 'OperationManager':
        """
        创建操作管理器
        
        Args:
            config: 配置
            
        Returns:
            OperationManager: 操作管理器实例
        """
        return OperationManager(config)


class OperationManager:
    """操作管理器类 - 统一管理所有操作"""
    
    def __init__(self, config: Optional[Any] = None):
        """
        初始化操作管理器
        
        Args:
            config: 配置
        """
        self.config = config
        self.logger = logging.getLogger("OperationManager")
        self._operations = {}
        self._init_operations()
    
    def _init_operations(self):
        """初始化所有操作"""
        try:
            self._operations = {
                "image": OperationFactory.create_operation("image", self.config),
                "click": OperationFactory.create_operation("click", self.config),
                "keyboard": OperationFactory.create_operation("keyboard", self.config),
                "window": OperationFactory.create_operation("window", self.config)
            }
            self.logger.debug("All operations initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize operations: {e}")
            raise OperationError(f"Operation initialization failed: {e}")
    
    def execute_operation(self, operation_type: str, operation: str, *args, **kwargs) -> OperationResult:
        """
        执行操作
        
        Args:
            operation_type: 操作类型 (image, click, keyboard, window)
            operation: 具体操作名称
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            OperationResult: 操作结果
        """
        try:
            if operation_type not in self._operations:
                return OperationResult(
                    success=False,
                    error=f"Unknown operation type: {operation_type}"
                )
            
            op_instance = self._operations[operation_type]
            return op_instance.execute(operation, *args, **kwargs)
            
        except Exception as e:
            self.logger.error(f"Operation execution failed: {e}")
            return OperationResult(
                success=False,
                error=f"Operation execution failed: {e}"
            )
    
    def get_operation(self, operation_type: str) -> Optional[Operation]:
        """
        获取操作实例
        
        Args:
            operation_type: 操作类型
            
        Returns:
            Optional[Operation]: 操作实例
        """
        return self._operations.get(operation_type)
    
    def list_operations(self) -> Dict[str, Operation]:
        """
        列出所有操作
        
        Returns:
            Dict[str, Operation]: 操作字典
        """
        return self._operations.copy()
    
    def reload_operation(self, operation_type: str, config: Optional[Any] = None) -> bool:
        """
        重新加载操作
        
        Args:
            operation_type: 操作类型
            config: 新配置
            
        Returns:
            bool: 是否成功
        """
        try:
            new_config = config or self.config
            self._operations[operation_type] = OperationFactory.create_operation(
                operation_type, new_config
            )
            self.logger.debug(f"Operation {operation_type} reloaded successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to reload operation {operation_type}: {e}")
            return False
