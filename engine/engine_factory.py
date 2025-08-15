#!/usr/bin/env python3
"""
推理引擎工厂类
提供统一的接口来创建和管理不同的推理引擎
"""

from typing import Dict, Any, List
from .engine_base import InferenceEngineBase
from .transformer_engine import TransformerEngine


class InferenceEngineFactory:
    """
    推理引擎工厂类
    提供统一的接口来创建不同的推理引擎
    """

    @classmethod
    def create_engine(cls, engine_type: str, **kwargs) -> InferenceEngineBase:
        """
        创建推理引擎实例
        """
        if engine_type == "transformer":
            return TransformerEngine(**kwargs)

        elif engine_type == "vllm_offline":
            from .vllm_offline_engine import VLLMOfflineEngine
            return VLLMOfflineEngine(**kwargs)

        elif engine_type == "vllm_api":
            from .vllm_api_engine import VLLMAPIEngine
            return VLLMAPIEngine(**kwargs)

        else:
            raise ValueError(f"Unsupported engine type: {engine_type}. "
                             f"Supported types: transformer, vllm_offline, vllm_api")

    @classmethod
    def get_supported_engines(cls) -> List[str]:
        """获取支持的引擎类型列表"""
        return ["transformer", "vllm_offline", "vllm_api"]

    @classmethod
    def get_engine_info(cls, engine_type: str) -> Dict[str, Any]:
        """获取引擎信息"""
        if engine_type == "transformer":
            engine_class = TransformerEngine
        elif engine_type == "vllm_offline":
            from .vllm_offline_engine import VLLMOfflineEngine
            engine_class = VLLMOfflineEngine
        elif engine_type == "vllm_api":
            from .vllm_api_engine import VLLMAPIEngine
            engine_class = VLLMAPIEngine
        else:
            return {"error": f"Unsupported engine type: {engine_type}"}

        doc = engine_class.__doc__ or "No description available"
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
        self.engine = InferenceEngineFactory.create_engine(engine_type, **engine_kwargs)
        self.engine_type = engine_type

    def single_infer(self, image_path: str, system_prompt: str = None, user_prompt: str = None) -> Dict[str, Any]:
        if system_prompt is None:
            system_prompt = self.engine.system_prompt
        if user_prompt is None:
            user_prompt = "<image>Find the spectral images in the figure"
        return self.engine.single_infer(image_path, system_prompt, user_prompt)

    def batch_infer(self, image_paths: List[str], system_prompt: str = None, 
                   user_prompt: str = None, **kwargs) -> List[Dict[str, Any]]:
        if system_prompt is None:
            system_prompt = self.engine.system_prompt
        if user_prompt is None:
            user_prompt = "<image>Find the spectral images in the figure"
        return self.engine.batch_infer(image_paths, system_prompt, user_prompt, **kwargs)

    def batch_infer_with_custom_prompts(self, image_paths: List[str], prompts: List[str], 
                                      system_prompt: str = None, **kwargs) -> List[Dict[str, Any]]:
        if len(image_paths) != len(prompts):
            raise ValueError("image_paths and prompts must have the same length")
        if system_prompt is None:
            system_prompt = self.engine.system_prompt
        results = []
        for image_path, prompt in zip(image_paths, prompts):
            results.append(self.engine.single_infer(image_path, system_prompt, prompt))
        return results

    def get_engine_info(self) -> Dict[str, Any]:
        return InferenceEngineFactory.get_engine_info(self.engine_type)
