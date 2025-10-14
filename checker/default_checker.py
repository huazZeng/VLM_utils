#!/usr/bin/env python3
"""
默认检查器
默认实现：直接返回True
"""

from typing import Any
from .base_checker import BaseChecker


class DefaultChecker(BaseChecker):
    """
    默认检查器
    所有数据检查都返回True
    """
    
    def check(self, data: Any, **kwargs) -> bool:
        """
        检查数据是否符合要求
        默认实现：直接返回True
        
        Args:
            data: 待检查的数据
            **kwargs: 额外的检查参数
            
        Returns:
            bool: 始终返回True
        """
        return True

