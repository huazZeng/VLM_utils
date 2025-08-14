#!/usr/bin/env python3
"""
推理引擎工厂类
提供统一的接口来创建和管理不同的推理引擎
"""

from typing import Dict, Any, List, Optional
from .engine_base import InferenceEngineBase
from .transformer_engine import TransformerEngine
from .vllm_offline_engine import VLLMOfflineEngine
from .vllm_api_engine import VLLMAPIEngine


class InferenceEngineFactory:
    """
    推理引擎工厂类
    提供统一的接口来创建不同的推理引擎
    """
    
    # 支持的引擎类型
    SUPPORTED_ENGINES = {
        "transformer": TransformerEngine,
        "vllm_offline": VLLMOfflineEngine,
        "vllm_api": VLLMAPIEngine
    }
    
    @classmethod
    def create_engine(cls, engine_type: str, **kwargs) -> InferenceEngineBase:
        """
        创建推理引擎实例
        
        Args:
            engine_type: 引擎类型，支持 "transformer", "vllm_offline", "vllm_api"
            **kwargs: 引擎特定的初始化参数
            
        Returns:
            推理引擎实例
            
        Raises:
            ValueError: 当引擎类型不支持时
        """
        if engine_type not in cls.SUPPORTED_ENGINES:
            supported = ", ".join(cls.SUPPORTED_ENGINES.keys())
            raise ValueError(f"Unsupported engine type: {engine_type}. Supported types: {supported}")
        
        engine_class = cls.SUPPORTED_ENGINES[engine_type]
        return engine_class(**kwargs)
    
    @classmethod
    def get_supported_engines(cls) -> List[str]:
        """
        获取支持的引擎类型列表
        
        Returns:
            支持的引擎类型列表
        """
        return list(cls.SUPPORTED_ENGINES.keys())
    
    @classmethod
    def get_engine_info(cls, engine_type: str) -> Dict[str, Any]:
        """
        获取引擎信息
        
        Args:
            engine_type: 引擎类型
            
        Returns:
            引擎信息字典
        """
        if engine_type not in cls.SUPPORTED_ENGINES:
            return {"error": f"Unsupported engine type: {engine_type}"}
        
        engine_class = cls.SUPPORTED_ENGINES[engine_type]
        
        # 获取引擎类的文档字符串
        doc = engine_class.__doc__ or "No description available"
        
        # 获取初始化方法的参数信息
        init_signature = engine_class.__init__.__code__
        init_params = init_signature.co_varnames[1:init_signature.co_argcount]
        
        return {
            "name": engine_type,
            "class": engine_class.__name__,
            "description": doc.strip(),
            "parameters": init_params
        }


class InferenceManager:
    """
    推理管理器
    提供高级接口来管理推理任务
    """
    
    def __init__(self, engine_type: str, **engine_kwargs):
        """
        初始化推理管理器
        
        Args:
            engine_type: 引擎类型
            **engine_kwargs: 引擎初始化参数
        """
        self.engine = InferenceEngineFactory.create_engine(engine_type, **engine_kwargs)
        self.engine_type = engine_type
    
    def single_infer(self, image_path: str, system_prompt: str = None, user_prompt: str = None) -> Dict[str, Any]:
        """
        单个推理
        
        Args:
            image_path: 图像路径
            system_prompt: 系统提示词，如果为None则使用引擎默认的
            user_prompt: 用户提示词
            
        Returns:
            推理结果字典
        """
        if system_prompt is None:
            system_prompt = self.engine.system_prompt
        
        if user_prompt is None:
            user_prompt = "<image>Find the spectral images in the figure"
        
        return self.engine.single_infer(image_path, system_prompt, user_prompt)
    
    def batch_infer(self, image_paths: List[str], system_prompt: str = None, 
                   user_prompt: str = None, **kwargs) -> List[Dict[str, Any]]:
        """
        批量推理
        
        Args:
            image_paths: 图像路径列表
            system_prompt: 系统提示词，如果为None则使用引擎默认的
            user_prompt: 用户提示词
            **kwargs: 其他参数
            
        Returns:
            推理结果列表
        """
        if system_prompt is None:
            system_prompt = self.engine.system_prompt
        
        if user_prompt is None:
            user_prompt = "<image>Find the spectral images in the figure"
        
        return self.engine.batch_infer(image_paths, system_prompt, user_prompt, **kwargs)
    
    def batch_infer_with_custom_prompts(self, image_paths: List[str], prompts: List[str], 
                                      system_prompt: str = None, **kwargs) -> List[Dict[str, Any]]:
        """
        批量推理（每个样本使用不同的prompt）
        
        Args:
            image_paths: 图像路径列表
            prompts: 每个样本对应的prompt列表
            system_prompt: 系统提示词，如果为None则使用引擎默认的
            **kwargs: 其他参数
            
        Returns:
            推理结果列表
        """
        if len(image_paths) != len(prompts):
            raise ValueError("image_paths and prompts must have the same length")
        
        if system_prompt is None:
            system_prompt = self.engine.system_prompt
        
        # 对于每个样本分别调用single_infer
        results = []
        for image_path, prompt in zip(image_paths, prompts):
            result = self.engine.single_infer(image_path, system_prompt, prompt)
            results.append(result)
        
        return results
    
    def get_engine_info(self) -> Dict[str, Any]:
        """
        获取当前引擎信息
        
        Returns:
            引擎信息字典
        """
        return InferenceEngineFactory.get_engine_info(self.engine_type) 