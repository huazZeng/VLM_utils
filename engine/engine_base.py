#!/usr/bin/env python3
"""
通用推理引擎抽象基类
提供统一的推理接口，支持不同推理引擎的插件化实现
"""

import os
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from PIL import Image


class InferenceEngineBase(ABC):
    """
    推理引擎抽象基类
    定义所有推理引擎必须实现的通用接口
    """
    
    def __init__(self, **kwargs):
        """
        初始化推理引擎
        
        Args:
            **kwargs: 引擎特定的初始化参数
        """
        
    
    @abstractmethod
    def preprocess(self, image_path: str, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """
        预处理：准备推理所需的输入数据
        
        Args:
            image_path: 图像路径
            system_prompt: 系统提示词
            user_prompt: 用户提示词
            
        Returns:
            预处理后的输入数据字典
        """
        pass
    
    @abstractmethod
    def single_infer(self, image_path: str, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """
        单个推理：对单张图像进行推理
        
        Args:
            image_path: 图像路径
            system_prompt: 系统提示词
            user_prompt: 用户提示词
            
        Returns:
            推理结果字典，包含：
            - success: bool, 是否成功
            - prediction: str, 模型预测结果
            - error: str, 错误信息（如果失败）
            - metadata: Dict, 其他元数据
        """
        pass
    
    @abstractmethod
    def batch_infer(self, image_paths: List[str], system_prompt: str, user_prompt: str, **kwargs) -> List[Dict[str, Any]]:
        """
        批量推理：对多张图像进行推理
        
        Args:
            image_paths: 图像路径列表
            system_prompt: 系统提示词
            user_prompt: 用户提示词
            **kwargs: 其他参数，如batch_size等
            
        Returns:
            推理结果列表，每个元素包含：
            - success: bool, 是否成功
            - image_path: str, 图像路径
            - prediction: str, 模型预测结果
            - error: str, 错误信息（如果失败）
            - metadata: Dict, 其他元数据
        """
        pass
    
    def load_image(self, image_path: str) -> Image.Image:
        """
        加载图像（通用方法）
        
        Args:
            image_path: 图像路径
            
        Returns:
            PIL Image对象
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        image = Image.open(image_path)
        # 转换为RGB模式
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image
    
    def create_result_dict(self, success: bool, prediction: str = "", error: str = "", 
                          image_path: str = "", metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        创建标准化的结果字典
        
        Args:
            success: 是否成功
            prediction: 预测结果
            error: 错误信息
            image_path: 图像路径
            metadata: 元数据
            
        Returns:
            标准化的结果字典
        """
        result = {
            "success": success,
            "prediction": prediction,
            "image_path": image_path,
            "metadata": metadata or {}
        }
        return result 