"""
平台适配模块
提供跨平台的GUI自动化功能
"""

from .base import (
    PlatformBase,
    PlatformFactory,
    PlatformError,
    PlatformManager
)
from .windows import WindowsPlatform
from .linux import LinuxPlatform

__all__ = [
    # 基础类
    'PlatformBase',
    'PlatformFactory',
    'PlatformError',
    'PlatformManager',
    
    # 具体平台实现
    'WindowsPlatform',
    'LinuxPlatform'
]
