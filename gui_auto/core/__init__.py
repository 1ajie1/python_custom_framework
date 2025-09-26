"""
GUI自动化框架核心模块
提供配置管理、日志管理、异常处理等核心功能
"""

from .config import (
    GuiConfig,
    MatchConfig,
    RetryConfig,
    ScaleConfig,
    ImageConfig,
    OperationConfig,
    ClickConfig,
    KeyboardConfig,
    WindowConfig,
    load_config,
    save_config,
    validate_config
)
from .logger import get_logger, setup_logging
from .exceptions import (
    GuiAutoException,
    ConfigError,
    ImageProcessingError,
    MatchingError,
    OperationError
)

__all__ = [
    # 配置相关
    'GuiConfig',
    'MatchConfig', 
    'RetryConfig',
    'ScaleConfig',
    'ImageConfig',
    'OperationConfig',
    'ClickConfig',
    'KeyboardConfig',
    'WindowConfig',
    'load_config',
    'save_config',
    'validate_config',
    
    # 日志相关
    'get_logger',
    'setup_logging',
    
    # 异常相关
    'GuiAutoException',
    'ConfigError',
    'ImageProcessingError',
    'MatchingError',
    'OperationError'
]
