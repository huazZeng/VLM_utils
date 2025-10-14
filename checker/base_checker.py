#!/usr/bin/env python3
"""
基础检查器类
定义所有检查器的通用接口，支持自动注册
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseChecker(ABC):
    """
    基础检查器抽象类
    所有具体的检查器都应该继承此类，支持自动注册
    """
    
    # Registry for all checker classes
    _registry = {}
    
    def __init_subclass__(cls, **kwargs):
        """Automatically register subclasses."""
        super().__init_subclass__(**kwargs)
        # Register by class name (e.g., JSONChecker -> json)
        class_name = cls.__name__
        
        # Convert class name to registry key
        registry_key = cls._convert_class_name_to_key(class_name)
        cls._registry[registry_key] = cls
    
    @staticmethod
    def _convert_class_name_to_key(class_name: str) -> str:
        """Convert class name to registry key."""
        # Remove 'Checker' suffix
        name = class_name.replace('Checker', '')
        
        # Convert camelCase to snake_case
        import re
        name = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', name).lower()
        
        return name
    
    def __init__(self, **kwargs):
        """
        初始化检查器
        
        Args:
            **kwargs: 检查器特定的配置参数
        """
        self.config = kwargs
    
    @abstractmethod
    def check(self, data: Any, **kwargs) -> bool:
        """
        检查数据是否符合要求
        
        Args:
            data: 待检查的数据
            **kwargs: 额外的检查参数
            
        Returns:
            bool: 检查是否通过
        """
        pass
    
    def get_checker_name(self) -> str:
        """
        获取检查器名称
        
        Returns:
            str: 检查器类名
        """
        return self.__class__.__name__
    
    @classmethod
    def get_registry(cls) -> dict:
        """Get the registry of all checker classes."""
        return cls._registry.copy()
    
    @classmethod
    def create_checker(cls, checker_type: str, **kwargs) -> 'BaseChecker':
        """Create checker instance based on checker type."""
        registry = cls.get_registry()
        checker_class = registry.get(checker_type)
        
        if checker_class is None:
            # Get available checkers for error message
            available_checkers = list(registry.keys())
            error_msg = f"No checker found for: {checker_type}\n"
            error_msg += f"Available checkers: {available_checkers}"
            raise ValueError(error_msg)
        
        return checker_class(**kwargs)
    
    @classmethod
    def get_available_checkers(cls) -> list:
        """Get list of available checker types."""
        return list(cls.get_registry().keys())

