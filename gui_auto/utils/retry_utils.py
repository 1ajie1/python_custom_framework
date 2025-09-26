"""
重试工具模块
提供重试机制、退避策略、置信度重试等基础功能
"""

import time
import random
from typing import Callable, Any, Optional, Union, Dict, List
from dataclasses import dataclass
import logging

from ..core.exceptions import ImageProcessingError

logger = logging.getLogger(__name__)


@dataclass
class RetryConfig:
    """重试配置"""
    max_attempts: int = 3
    delay: float = 0.5
    backoff: float = 1.5
    max_delay: float = 10.0
    jitter: bool = True
    confidence_retry_enabled: bool = True
    confidence_retry_attempts: int = 3
    confidence_threshold: float = 0.8


class RetryUtils:
    """重试处理工具类"""
    
    @staticmethod
    def retry_on_failure(
        func: Callable,
        config: RetryConfig,
        exceptions: tuple = (Exception,),
        *args, **kwargs
    ) -> Any:
        """
        在失败时重试函数
        
        Args:
            func: 要重试的函数
            config: 重试配置
            exceptions: 要捕获的异常类型
            *args, **kwargs: 函数参数
            
        Returns:
            函数执行结果
            
        Raises:
            最后一次尝试的异常
        """
        last_exception = None
        
        for attempt in range(config.max_attempts):
            try:
                result = func(*args, **kwargs)
                if attempt > 0:
                    logger.debug(f"Function succeeded on attempt {attempt + 1}")
                return result
                
            except exceptions as e:
                last_exception = e
                logger.debug(f"Attempt {attempt + 1} failed: {e}")
                
                # 如果不是最后一次尝试，等待后重试
                if attempt < config.max_attempts - 1:
                    delay = RetryUtils._calculate_delay(config, attempt)
                    logger.debug(f"Waiting {delay:.2f}s before retry")
                    time.sleep(delay)
        
        # 所有尝试都失败了
        logger.error(f"All {config.max_attempts} attempts failed")
        raise last_exception
    
    @staticmethod
    def retry_with_backoff(
        func: Callable,
        max_attempts: int = 3,
        initial_delay: float = 0.5,
        backoff_factor: float = 1.5,
        max_delay: float = 10.0,
        jitter: bool = True,
        exceptions: tuple = (Exception,),
        *args, **kwargs
    ) -> Any:
        """
        使用指数退避策略重试函数
        
        Args:
            func: 要重试的函数
            max_attempts: 最大尝试次数
            initial_delay: 初始延迟时间
            backoff_factor: 退避因子
            max_delay: 最大延迟时间
            jitter: 是否添加随机抖动
            exceptions: 要捕获的异常类型
            *args, **kwargs: 函数参数
            
        Returns:
            函数执行结果
        """
        config = RetryConfig(
            max_attempts=max_attempts,
            delay=initial_delay,
            backoff=backoff_factor,
            max_delay=max_delay,
            jitter=jitter
        )
        
        return RetryUtils.retry_on_failure(func, config, exceptions, *args, **kwargs)
    
    @staticmethod
    def retry_on_confidence(
        func: Callable,
        config: RetryConfig,
        confidence_threshold: Optional[float] = None,
        *args, **kwargs
    ) -> Any:
        """
        基于置信度重试函数
        
        Args:
            func: 要重试的函数（应返回包含confidence字段的结果）
            config: 重试配置
            confidence_threshold: 置信度阈值
            *args, **kwargs: 函数参数
            
        Returns:
            函数执行结果
        """
        if not config.confidence_retry_enabled:
            return func(*args, **kwargs)
        
        threshold = confidence_threshold or config.confidence_threshold
        last_result = None
        last_exception = None
        
        for attempt in range(config.confidence_retry_attempts):
            try:
                result = func(*args, **kwargs)
                
                # 检查结果是否有置信度字段
                if hasattr(result, 'confidence'):
                    confidence = result.confidence
                elif isinstance(result, dict) and 'confidence' in result:
                    confidence = result['confidence']
                else:
                    # 没有置信度信息，直接返回结果
                    logger.debug("No confidence information in result, returning as-is")
                    return result
                
                # 检查置信度是否满足要求
                if confidence >= threshold:
                    if attempt > 0:
                        logger.debug(f"Confidence threshold met on attempt {attempt + 1}: {confidence:.3f}")
                    return result
                
                last_result = result
                logger.debug(f"Confidence {confidence:.3f} below threshold {threshold}, attempt {attempt + 1}")
                
                # 如果不是最后一次尝试，等待后重试
                if attempt < config.confidence_retry_attempts - 1:
                    delay = RetryUtils._calculate_delay(config, attempt)
                    logger.debug(f"Waiting {delay:.2f}s before confidence retry")
                    time.sleep(delay)
                    
            except Exception as e:
                last_exception = e
                logger.debug(f"Confidence retry attempt {attempt + 1} failed: {e}")
                
                # 如果不是最后一次尝试，等待后重试
                if attempt < config.confidence_retry_attempts - 1:
                    delay = RetryUtils._calculate_delay(config, attempt)
                    logger.debug(f"Waiting {delay:.2f}s before confidence retry")
                    time.sleep(delay)
        
        # 所有尝试都失败了
        if last_exception:
            logger.error(f"All {config.confidence_retry_attempts} confidence retry attempts failed")
            raise last_exception
        else:
            logger.warning(f"Confidence threshold not met after {config.confidence_retry_attempts} attempts")
            return last_result
    
    @staticmethod
    def _calculate_delay(config: RetryConfig, attempt: int) -> float:
        """
        计算重试延迟时间
        
        Args:
            config: 重试配置
            attempt: 当前尝试次数（从0开始）
            
        Returns:
            float: 延迟时间（秒）
        """
        # 基础延迟时间
        delay = config.delay * (config.backoff ** attempt)
        
        # 限制最大延迟时间
        delay = min(delay, config.max_delay)
        
        # 添加随机抖动
        if config.jitter:
            jitter_range = delay * 0.1  # 10%的抖动
            jitter = random.uniform(-jitter_range, jitter_range)
            delay += jitter
        
        # 确保延迟时间为正数
        delay = max(0.0, delay)
        
        return delay
    
    @staticmethod
    def retry_with_condition(
        func: Callable,
        condition: Callable[[Any], bool],
        config: RetryConfig,
        *args, **kwargs
    ) -> Any:
        """
        基于条件重试函数
        
        Args:
            func: 要重试的函数
            condition: 条件函数，返回True表示需要重试
            config: 重试配置
            *args, **kwargs: 函数参数
            
        Returns:
            函数执行结果
        """
        last_result = None
        last_exception = None
        
        for attempt in range(config.max_attempts):
            try:
                result = func(*args, **kwargs)
                
                # 检查条件
                if not condition(result):
                    if attempt > 0:
                        logger.debug(f"Condition satisfied on attempt {attempt + 1}")
                    return result
                
                last_result = result
                logger.debug(f"Condition not satisfied on attempt {attempt + 1}")
                
                # 如果不是最后一次尝试，等待后重试
                if attempt < config.max_attempts - 1:
                    delay = RetryUtils._calculate_delay(config, attempt)
                    logger.debug(f"Waiting {delay:.2f}s before condition retry")
                    time.sleep(delay)
                    
            except Exception as e:
                last_exception = e
                logger.debug(f"Condition retry attempt {attempt + 1} failed: {e}")
                
                # 如果不是最后一次尝试，等待后重试
                if attempt < config.max_attempts - 1:
                    delay = RetryUtils._calculate_delay(config, attempt)
                    logger.debug(f"Waiting {delay:.2f}s before condition retry")
                    time.sleep(delay)
        
        # 所有尝试都失败了
        if last_exception:
            logger.error(f"All {config.max_attempts} condition retry attempts failed")
            raise last_exception
        else:
            logger.warning(f"Condition not satisfied after {config.max_attempts} attempts")
            return last_result
    
    @staticmethod
    def create_retry_decorator(
        config: RetryConfig,
        exceptions: tuple = (Exception,)
    ) -> Callable:
        """
        创建重试装饰器
        
        Args:
            config: 重试配置
            exceptions: 要捕获的异常类型
            
        Returns:
            装饰器函数
        """
        def decorator(func: Callable) -> Callable:
            def wrapper(*args, **kwargs):
                return RetryUtils.retry_on_failure(
                    func, config, exceptions, *args, **kwargs
                )
            return wrapper
        return decorator
    
    @staticmethod
    def create_confidence_retry_decorator(
        config: RetryConfig,
        confidence_threshold: Optional[float] = None
    ) -> Callable:
        """
        创建置信度重试装饰器
        
        Args:
            config: 重试配置
            confidence_threshold: 置信度阈值
            
        Returns:
            装饰器函数
        """
        def decorator(func: Callable) -> Callable:
            def wrapper(*args, **kwargs):
                return RetryUtils.retry_on_confidence(
                    func, config, confidence_threshold, *args, **kwargs
                )
            return wrapper
        return decorator


# 便捷函数
def retry_on_failure(func, config, exceptions=(Exception,), *args, **kwargs):
    """便捷的失败重试函数"""
    return RetryUtils.retry_on_failure(func, config, exceptions, *args, **kwargs)


def retry_with_backoff(func, max_attempts=3, initial_delay=0.5, **kwargs):
    """便捷的退避重试函数"""
    return RetryUtils.retry_with_backoff(func, max_attempts, initial_delay, **kwargs)


def retry_on_confidence(func, config, confidence_threshold=None, *args, **kwargs):
    """便捷的置信度重试函数"""
    return RetryUtils.retry_on_confidence(func, config, confidence_threshold, *args, **kwargs)
