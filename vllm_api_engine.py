#!/usr/bin/env python3
"""
vLLM API推理引擎实现
基于vLLM服务器的API调用推理
"""

import asyncio
import base64
import io
from PIL import Image
from qwen_vl_utils import smart_resize
from typing import Dict, Any, List
from tqdm import tqdm

from utils.api_infer import OpenAIChatClient
from .engine_base import InferenceEngineBase


class VLLMAPIEngine(InferenceEngineBase):
    """
    vLLM API推理引擎
    基于vLLM服务器的API调用推理
    """
    
    def __init__(self, base_url: str, model_name: str, api_key: str = "EMPTY", **kwargs):
        """
        初始化vLLM API推理引擎
        
        Args:
            base_url: vLLM服务器基础URL
            model_name: 模型名称
            api_key: API密钥
            **kwargs: 其他参数
        """
        super().__init__(**kwargs)
        
        self.base_url = base_url
        self.model_name = model_name
        self.api_key = api_key
        self.concurrency = kwargs.get("concurrency", 64)
        # 初始化客户端
        self.client = OpenAIChatClient(
            base_url=self.base_url,
            api_key=self.api_key,
            model_name=self.model_name
        )
        
        print(f"vLLM API client initialized with model: {model_name}")
    
    def image_to_data_uri(self, image_path: str) -> str:
        """
        将图像转换为data URI，使用smart_resize进行resize
        
        Args:
            image_path: 图像路径
            
        Returns:
            data URI字符串
        """
        # 打开图片后首先进行resize
        image = Image.open(image_path)
        original_width, original_height = image.size
        
        print(f"Original image size: {original_width}x{original_height}")
        
        # 使用smart_resize计算新的宽高
        new_width, new_height = smart_resize(original_width, original_height, 
                                           min_pixels=512*28*28, max_pixels=2048*28*28)
        
        print(f"Resized image size: {new_width}x{new_height}")
        
        # 调整图片大小
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # 转换为base64
        buffer = io.BytesIO()
        resized_image.save(buffer, format='JPEG', quality=95)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        return f"data:image/jpeg;base64,{encoded}"
    
    def preprocess(self, image_path: str, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """
        预处理：准备vLLM API推理所需的输入数据
        
        Args:
            image_path: 图像路径
            system_prompt: 系统提示词
            user_prompt: 用户提示词
            
        Returns:
            预处理后的输入数据字典
        """
        try:
            # 准备消息
            content = [
                {
                    "type": "image_url",
                    "image_url": {"url": self.image_to_data_uri(image_path)}
                },
                {
                    "type": "text",
                    "text": user_prompt
                }
            ]
            
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": content
                }
            ]
            
            return {
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": 1024
            }
            
        except Exception as e:
            raise Exception(f"Preprocessing failed: {e}")
    

    
    async def single_infer(self, image_path: str, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """
        单个推理：使用vLLM API对单张图像进行推理
        
        Args:
            image_path: 图像路径
            system_prompt: 系统提示词
            user_prompt: 用户提示词
            
        Returns:
            推理结果字典
        """
        try:
            # 预处理
            preprocessed_data = self.preprocess(image_path, system_prompt, user_prompt)
            
            # 创建事件循环并执行异步推理
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                prediction = loop.run_until_complete(
                    self.client.chat(
                        preprocessed_data["messages"], 
                        temperature=preprocessed_data["temperature"], 
                        max_tokens=preprocessed_data["max_tokens"]
                    )
                )
                
                return self.create_result_dict(
                    success=True,
                    prediction=prediction,
                    image_path=image_path,
                    metadata={
                        "model_name": self.model_name,
                        "base_url": self.base_url
                    }
                )
            finally:
                loop.close()
                
        except Exception as e:
            return self.create_result_dict(
                success=False,
                error=str(e),
                image_path=image_path,
                metadata={
                    "model_name": self.model_name,
                    "base_url": self.base_url
                }
            )
    
    async def batch_infer(self, image_paths: List[str], system_prompt: str, user_prompt: str, **kwargs) -> List[Dict[str, Any]]:
        """
        批量推理：使用vLLM API的并发推理功能
        
        Args:
            image_paths: 图像路径列表
            system_prompt: 系统提示词
            user_prompt: 用户提示词
            **kwargs: 其他参数，包括concurrency
            
        Returns:
            推理结果列表
        """
        concurrency = kwargs.get('concurrency', 64)
        
        # 使用信号量控制并发数
        semaphore = asyncio.Semaphore(concurrency)
        
        async def limited_single_infer(image_path: str) -> Dict[str, Any]:
            async with semaphore:
                try:
                    # 预处理
                    preprocessed_data = self.preprocess(image_path, system_prompt, user_prompt)
                    
                    # 异步推理
                    prediction = await self.client.chat(
                        preprocessed_data["messages"], 
                        temperature=preprocessed_data["temperature"], 
                        max_tokens=preprocessed_data["max_tokens"]
                    )
                    
                    return self.create_result_dict(
                        success=True,
                        prediction=prediction,
                        image_path=image_path,
                        metadata={
                            "model_name": self.model_name,
                            "base_url": self.base_url
                        }
                    )
                except Exception as e:
                    return self.create_result_dict(
                        success=False,
                        error=str(e),
                        image_path=image_path,
                        metadata={
                            "model_name": self.model_name,
                            "base_url": self.base_url
                        }
                    )
        
        # 创建事件循环并执行并发推理
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # 创建所有任务
            tasks = [limited_single_infer(image_path) for image_path in image_paths]
            
            # 并发执行所有任务
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理异常结果
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    # 如果任务抛出异常，创建错误结果
                    error_result = self.create_result_dict(
                        success=False,
                        error=str(result),
                        image_path=image_paths[i],
                        metadata={
                            "model_name": self.model_name,
                            "base_url": self.base_url
                        }
                    )
                    processed_results.append(error_result)
                else:
                    processed_results.append(result)
            
            return processed_results
            
        except Exception as e:
            print(f"Error in batch processing: {e}")
            # 如果批量处理失败，回退到单个处理
            print("Falling back to single inference...")
            results = []
            
            for image_path in image_paths:
                try:
                    result = await limited_single_infer(image_path)
                    results.append(result)
                except Exception as single_error:
                    error_result = self.create_result_dict(
                        success=False,
                        error=str(single_error),
                        image_path=image_path,
                        metadata={
                            "model_name": self.model_name,
                            "base_url": self.base_url
                        }
                    )
                    results.append(error_result)
            
            return results
        finally:
            loop.close() 