"""
日志管理模块
提供统一的日志配置和管理功能
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union
from datetime import datetime


def setup_logging(
    level: Union[str, int] = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    log_format: Optional[str] = None,
    console_output: bool = True
) -> None:
    """
    设置日志配置
    
    Args:
        level: 日志级别
        log_file: 日志文件路径，None表示不输出到文件
        log_format: 日志格式，None使用默认格式
        console_output: 是否输出到控制台
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # 创建根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # 清除现有的处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 创建格式化器
    formatter = logging.Formatter(log_format)
    
    # 控制台处理器
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # 文件处理器
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    获取指定名称的日志器
    
    Args:
        name: 日志器名称
        
    Returns:
        logging.Logger: 日志器实例
    """
    return logging.getLogger(name)


def create_logger(
    name: str,
    level: Union[str, int] = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    log_format: Optional[str] = None
) -> logging.Logger:
    """
    创建独立的日志器
    
    Args:
        name: 日志器名称
        level: 日志级别
        log_file: 日志文件路径
        log_format: 日志格式
        
    Returns:
        logging.Logger: 日志器实例
    """
    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger  # 已经配置过，直接返回
    
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    
    logger.setLevel(level)
    
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    formatter = logging.Formatter(log_format)
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def setup_module_logging(module_name: str, level: Union[str, int] = logging.INFO) -> logging.Logger:
    """
    为模块设置日志
    
    Args:
        module_name: 模块名称
        level: 日志级别
        
    Returns:
        logging.Logger: 配置好的日志器
    """
    logger = get_logger(module_name)
    
    if not logger.handlers:
        if isinstance(level, str):
            level = getattr(logging, level.upper(), logging.INFO)
        
        logger.setLevel(level)
        
        # 使用父日志器的处理器
        parent_logger = logging.getLogger()
        if parent_logger.handlers:
            logger.parent = parent_logger
        else:
            # 如果没有父处理器，创建控制台处理器
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
    
    return logger


def log_function_call(func):
    """
    函数调用日志装饰器
    
    Args:
        func: 要装饰的函数
        
    Returns:
        装饰后的函数
    """
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} failed with error: {e}")
            raise
    return wrapper


def log_performance(func):
    """
    性能日志装饰器
    
    Args:
        func: 要装饰的函数
        
    Returns:
        装饰后的函数
    """
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = datetime.now()
        logger.debug(f"Starting {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.info(f"{func.__name__} completed in {duration:.3f}s")
            return result
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.error(f"{func.__name__} failed after {duration:.3f}s with error: {e}")
            raise
    return wrapper
