#!/usr/bin/env python3
"""
JSON格式检查器
检查数据是否为有效的JSON格式
"""

import json
from typing import Any
from .base_checker import BaseChecker


class JSONChecker(BaseChecker):
    """
    JSON格式检查器
    检查输入数据是否为有效的JSON格式
    """
    
    def check(self, data: Any, **kwargs) -> bool:
        """
        检查数据是否为有效的JSON格式
        
        Args:
            data: 待检查的数据（字符串）
            **kwargs: 额外的检查参数
            
        Returns:
            bool: 如果是有效JSON格式返回True，否则返回False
        """
        # 传进来的肯定是字符串
        try:
            json.loads(data)
            return True
        except (json.JSONDecodeError, ValueError):
            return False

