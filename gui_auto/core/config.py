"""
配置管理模块
提供统一的配置管理功能，包括配置类定义、验证、加载和保存
"""

import json
import yaml
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, Dict, Any, Union
from pathlib import Path
import logging

from .exceptions import ConfigError

logger = logging.getLogger(__name__)


@dataclass
class MatchConfig:
    """图像匹配配置"""
    confidence: float = 0.8
    method: str = "TM_CCOEFF_NORMED"
    use_multi_scale: bool = False
    enhancement_level: str = "light"
    use_pyramid: bool = False
    pyramid_levels: int = 4
    pyramid_scale_factor: float = 0.5
    timeout: float = 10.0
    
    def __post_init__(self):
        """配置验证"""
        if not 0.0 <= self.confidence <= 1.0:
            raise ConfigError(f"confidence must be between 0.0 and 1.0, got {self.confidence}")
        
        valid_methods = [
            "TM_CCOEFF_NORMED", "TM_CCORR_NORMED", "TM_SQDIFF_NORMED",
            "TM_CCOEFF", "TM_CCORR", "TM_SQDIFF"
        ]
        if self.method not in valid_methods:
            raise ConfigError(f"method must be one of {valid_methods}, got {self.method}")
        
        valid_enhancement_levels = ["light", "medium", "heavy", "none"]
        if self.enhancement_level not in valid_enhancement_levels:
            raise ConfigError(f"enhancement_level must be one of {valid_enhancement_levels}, got {self.enhancement_level}")
        
        if self.timeout <= 0:
            raise ConfigError(f"timeout must be positive, got {self.timeout}")
        
        if self.pyramid_levels < 1:
            raise ConfigError(f"pyramid_levels must be at least 1, got {self.pyramid_levels}")
        
        if not 0.0 < self.pyramid_scale_factor <= 1.0:
            raise ConfigError(f"pyramid_scale_factor must be between 0.0 and 1.0, got {self.pyramid_scale_factor}")


@dataclass
class RetryConfig:
    """重试配置"""
    max_attempts: int = 3
    delay: float = 0.5
    backoff: float = 1.5
    confidence_retry_enabled: bool = True
    confidence_retry_attempts: int = 3
    
    def __post_init__(self):
        """配置验证"""
        if self.max_attempts < 1:
            raise ConfigError(f"max_attempts must be at least 1, got {self.max_attempts}")
        
        if self.delay < 0:
            raise ConfigError(f"delay must be non-negative, got {self.delay}")
        
        if self.backoff < 1.0:
            raise ConfigError(f"backoff must be at least 1.0, got {self.backoff}")
        
        if self.confidence_retry_attempts < 1:
            raise ConfigError(f"confidence_retry_attempts must be at least 1, got {self.confidence_retry_attempts}")


@dataclass
class ScaleConfig:
    """缩放配置"""
    auto_scale: bool = True
    base_scale: float = 1.0
    base_resolution: Tuple[int, int] = (1920, 1080)
    template_scale_info: Optional[Dict[str, Any]] = None
    target_scale_info: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """配置验证"""
        if self.base_scale <= 0:
            raise ConfigError(f"base_scale must be positive, got {self.base_scale}")
        
        if len(self.base_resolution) != 2:
            raise ConfigError(f"base_resolution must be a tuple of 2 integers, got {self.base_resolution}")
        
        width, height = self.base_resolution
        if width <= 0 or height <= 0:
            raise ConfigError(f"base_resolution width and height must be positive, got {self.base_resolution}")


@dataclass
class ImageConfig:
    """图像处理配置"""
    format_validation: bool = True
    auto_rgb_to_bgr: bool = True
    quality_analysis: bool = True
    coordinate_system: str = "system"  # "system" 或 "base"
    
    def __post_init__(self):
        """配置验证"""
        valid_coordinate_systems = ["system", "base"]
        if self.coordinate_system not in valid_coordinate_systems:
            raise ConfigError(f"coordinate_system must be one of {valid_coordinate_systems}, got {self.coordinate_system}")


@dataclass
class ClickConfig:
    """点击操作配置"""
    click_delay: float = 0.1
    double_click_delay: float = 0.1
    right_click_delay: float = 0.1
    drag_delay: float = 0.1
    
    def __post_init__(self):
        """配置验证"""
        if self.click_delay < 0:
            raise ConfigError(f"click_delay must be non-negative, got {self.click_delay}")
        
        if self.double_click_delay < 0:
            raise ConfigError(f"double_click_delay must be non-negative, got {self.double_click_delay}")
        
        if self.right_click_delay < 0:
            raise ConfigError(f"right_click_delay must be non-negative, got {self.right_click_delay}")
        
        if self.drag_delay < 0:
            raise ConfigError(f"drag_delay must be non-negative, got {self.drag_delay}")


@dataclass
class KeyboardConfig:
    """键盘操作配置"""
    key_delay: float = 0.1
    text_delay: float = 0.05
    hotkey_delay: float = 0.1
    
    def __post_init__(self):
        """配置验证"""
        if self.key_delay < 0:
            raise ConfigError(f"key_delay must be non-negative, got {self.key_delay}")
        
        if self.text_delay < 0:
            raise ConfigError(f"text_delay must be non-negative, got {self.text_delay}")
        
        if self.hotkey_delay < 0:
            raise ConfigError(f"hotkey_delay must be non-negative, got {self.hotkey_delay}")


@dataclass
class WindowConfig:
    """窗口操作配置"""
    window_find_timeout: float = 5.0
    window_operation_delay: float = 0.5
    minimize_delay: float = 0.5
    maximize_delay: float = 0.5
    
    def __post_init__(self):
        """配置验证"""
        if self.window_find_timeout <= 0:
            raise ConfigError(f"window_find_timeout must be positive, got {self.window_find_timeout}")
        
        if self.window_operation_delay < 0:
            raise ConfigError(f"window_operation_delay must be non-negative, got {self.window_operation_delay}")
        
        if self.minimize_delay < 0:
            raise ConfigError(f"minimize_delay must be non-negative, got {self.minimize_delay}")
        
        if self.maximize_delay < 0:
            raise ConfigError(f"maximize_delay must be non-negative, got {self.maximize_delay}")


@dataclass
class OperationConfig:
    """操作配置"""
    click: ClickConfig = field(default_factory=ClickConfig)
    keyboard: KeyboardConfig = field(default_factory=KeyboardConfig)
    window: WindowConfig = field(default_factory=WindowConfig)


@dataclass
class GuiConfig:
    """GUI自动化工具主配置"""
    matching: MatchConfig = field(default_factory=MatchConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)
    scale: ScaleConfig = field(default_factory=ScaleConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    operations: OperationConfig = field(default_factory=OperationConfig)
    
    def __post_init__(self):
        """配置验证"""
        # 所有子配置的验证在各自的__post_init__中完成
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GuiConfig':
        """从字典创建配置"""
        try:
            # 递归创建子配置
            if 'matching' in data:
                data['matching'] = MatchConfig(**data['matching'])
            if 'retry' in data:
                data['retry'] = RetryConfig(**data['retry'])
            if 'scale' in data:
                data['scale'] = ScaleConfig(**data['scale'])
            if 'image' in data:
                data['image'] = ImageConfig(**data['image'])
            if 'operations' in data and isinstance(data['operations'], dict):
                ops_data = data['operations']
                if 'click' in ops_data:
                    ops_data['click'] = ClickConfig(**ops_data['click'])
                if 'keyboard' in ops_data:
                    ops_data['keyboard'] = KeyboardConfig(**ops_data['keyboard'])
                if 'window' in ops_data:
                    ops_data['window'] = WindowConfig(**ops_data['window'])
                data['operations'] = OperationConfig(**ops_data)
            
            return cls(**data)
        except Exception as e:
            raise ConfigError(f"Failed to create config from dict: {e}")


def validate_config(config: GuiConfig) -> bool:
    """
    验证配置的有效性
    
    Args:
        config: 要验证的配置
        
    Returns:
        bool: 配置是否有效
        
    Raises:
        ConfigError: 配置无效时抛出异常
    """
    try:
        # 配置验证在__post_init__中完成
        # 这里可以添加额外的跨配置验证逻辑
        return True
    except Exception as e:
        raise ConfigError(f"Config validation failed: {e}")


def load_config(config_path: Union[str, Path]) -> GuiConfig:
    """
    从文件加载配置
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        GuiConfig: 加载的配置
        
    Raises:
        ConfigError: 加载失败时抛出异常
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.suffix.lower() == '.json':
                data = json.load(f)
            elif config_path.suffix.lower() in ['.yml', '.yaml']:
                data = yaml.safe_load(f)
            else:
                raise ConfigError(f"Unsupported config file format: {config_path.suffix}")
        
        config = GuiConfig.from_dict(data)
        validate_config(config)
        logger.info(f"Config loaded successfully from {config_path}")
        return config
        
    except Exception as e:
        raise ConfigError(f"Failed to load config from {config_path}: {e}")


def save_config(config: GuiConfig, config_path: Union[str, Path], 
                format: str = 'json') -> None:
    """
    保存配置到文件
    
    Args:
        config: 要保存的配置
        config_path: 保存路径
        format: 保存格式 ('json' 或 'yaml')
        
    Raises:
        ConfigError: 保存失败时抛出异常
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        validate_config(config)
        data = config.to_dict()
        
        with open(config_path, 'w', encoding='utf-8') as f:
            if format.lower() == 'json':
                json.dump(data, f, indent=2, ensure_ascii=False)
            elif format.lower() in ['yml', 'yaml']:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
            else:
                raise ConfigError(f"Unsupported format: {format}")
        
        logger.info(f"Config saved successfully to {config_path}")
        
    except Exception as e:
        raise ConfigError(f"Failed to save config to {config_path}: {e}")


def create_default_config() -> GuiConfig:
    """
    创建默认配置
    
    Returns:
        GuiConfig: 默认配置
    """
    return GuiConfig()


def merge_configs(base_config: GuiConfig, override_config: Dict[str, Any]) -> GuiConfig:
    """
    合并配置
    
    Args:
        base_config: 基础配置
        override_config: 覆盖配置
        
    Returns:
        GuiConfig: 合并后的配置
    """
    try:
        base_dict = base_config.to_dict()
        
        def deep_merge(base: dict, override: dict) -> dict:
            """深度合并字典"""
            for key, value in override.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    base[key] = deep_merge(base[key], value)
                else:
                    base[key] = value
            return base
        
        merged_dict = deep_merge(base_dict, override_config)
        return GuiConfig.from_dict(merged_dict)
        
    except Exception as e:
        raise ConfigError(f"Failed to merge configs: {e}")
