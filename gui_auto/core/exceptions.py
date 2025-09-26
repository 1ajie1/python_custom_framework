"""
自定义异常模块
定义GUI自动化框架中使用的各种异常类型
"""


class GuiAutoException(Exception):
    """GUI自动化框架基础异常类"""
    pass


class ConfigError(GuiAutoException):
    """配置相关异常"""
    pass


class ImageProcessingError(GuiAutoException):
    """图像处理相关异常"""
    pass


class MatchingError(GuiAutoException):
    """图像匹配相关异常"""
    pass


class OperationError(GuiAutoException):
    """操作执行相关异常"""
    pass


class ClickError(OperationError):
    """点击操作异常"""
    pass


class KeyboardError(OperationError):
    """键盘操作异常"""
    pass


class WindowError(OperationError):
    """窗口操作异常"""
    pass


class AlgorithmError(GuiAutoException):
    """算法相关异常"""
    pass


class PlatformError(GuiAutoException):
    """平台相关异常"""
    pass


class ValidationError(GuiAutoException):
    """验证相关异常"""
    pass
