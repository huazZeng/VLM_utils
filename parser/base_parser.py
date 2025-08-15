#!/usr/bin/env python3
"""
基础解析器类
定义所有解析器的通用接口，支持自动注册
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class BaseParser(ABC):
    """
    基础解析器抽象类
    所有具体的解析器都应该继承此类，支持自动注册
    """
    
    # Registry for all parser classes
    _registry = {}
    
    def __init_subclass__(cls, **kwargs):
        """Automatically register subclasses."""
        super().__init_subclass__(**kwargs)
        # Register by class name (e.g., SpectralParser -> spectral)
        class_name = cls.__name__
        
        # Convert class name to registry key
        registry_key = cls._convert_class_name_to_key(class_name)
        cls._registry[registry_key] = cls
    
    @staticmethod
    def _convert_class_name_to_key(class_name: str) -> str:
        """Convert class name to registry key."""
        # Remove 'Parser' suffix
        name = class_name.replace('Parser', '')
        
        # Convert camelCase to snake_case
        import re
        name = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', name).lower()
        
        return name
    
    def __init__(self, **kwargs):
        """
        初始化解析器
        
        Args:
            **kwargs: 解析器特定的配置参数
        """
        self.config = kwargs
    
    @abstractmethod
    def parse_to_print(self, raw_result: str, **kwargs) -> str:
        """
        将原始结果解析为可打印的格式
        
        Args:
            raw_result: 原始推理结果字符串
            **kwargs: 额外的解析参数
            
        Returns:
            str: 格式化后的可打印字符串
        """
        pass
    
    @abstractmethod
    def parse_to_save(self, raw_result: str, **kwargs) -> Dict[str, Any]:
        """
        将原始结果解析为可保存的结构化数据
        
        Args:
            raw_result: 原始推理结果字符串
            **kwargs: 额外的解析参数
            
        Returns:
            Dict[str, Any]: 结构化的数据字典，用于保存到文件
        """
        pass
    
    def validate_result(self, raw_result: str) -> bool:
        """
        验证原始结果是否有效
        
        Args:
            raw_result: 原始推理结果字符串
            
        Returns:
            bool: 结果是否有效
        """
        return raw_result is not None and len(raw_result.strip()) > 0
    
    def get_parser_name(self) -> str:
        """
        获取解析器名称
        
        Returns:
            str: 解析器类名
        """
        return self.__class__.__name__
    
    @classmethod
    def get_registry(cls) -> dict:
        """Get the registry of all parser classes."""
        return cls._registry.copy()
    
    @classmethod
    def create_parser(cls, parser_type: str, **kwargs) -> 'BaseParser':
        """Create parser instance based on parser type."""
        registry = cls.get_registry()
        parser_class = registry.get(parser_type)
        
        if parser_class is None:
            # Get available parsers for error message
            available_parsers = list(registry.keys())
            error_msg = f"No parser found for: {parser_type}\n"
            error_msg += f"Available parsers: {available_parsers}"
            raise ValueError(error_msg)
        
        return parser_class(**kwargs)
    
    @classmethod
    def get_available_parsers(cls) -> list:
        """Get list of available parser types."""
        return list(cls.get_registry().keys()) 