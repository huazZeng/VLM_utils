#!/usr/bin/env python3
"""
vLLM API推理引擎实现
基于vLLM服务器的API调用推理（纯异步实现，支持chat接口）
"""

import asyncio
import base64
import io
from PIL import Image
from qwen_vl_utils import smart_resize
from typing import Dict, Any, List
from tqdm.asyncio import tqdm as async_tqdm
from utils.api_infer import OpenAIChatClient
from engine.engine_base import InferenceEngineBase


class APIChatEngine(InferenceEngineBase):
    """
    vLLM API推理引擎
    纯异步实现，支持chat接口
    """

    def __init__(self, base_url: str, model_name: str, api_key: str = "EMPTY", **kwargs):
        super().__init__(**kwargs)
        self.base_url = base_url
        self.model_name = model_name
        self.api_key = api_key
        self.concurrency = kwargs.get("concurrency", 64)
        self.max_tokens = kwargs.get("max_tokens", 1024)
        self.temperature = kwargs.get("temperature", 0.0)
        self.timeout = kwargs.get("timeout", 60.0)
        self.max_retries = kwargs.get("max_retries", 3)
        self.client = OpenAIChatClient(
            base_url=self.base_url,
            api_key=self.api_key,
            model_name=self.model_name,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
        print(f"API client initialized with model: {model_name}")
        print(f"API client initialized with base_url: {base_url}")

    def image_to_data_uri(self, image_path: str) -> str:
        """将图像转换为data URI格式"""
        image = Image.open(image_path)
        # original_width, original_height = image.size
        # new_width, new_height = smart_resize(
        #     original_width, original_height,
        #     min_pixels=512*28*28, max_pixels=2048*28*28
        # )
        # resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=95)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"

    def preprocess(self, image_path: str, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """
        预处理：准备vLLM模型推理所需的输入数据
        
        Args:
            image_path: 图像路径
            system_prompt: 系统提示词
            user_prompt: 用户提示词
            
        Returns:
            预处理后的输入数据字典
        """
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
        if system_prompt:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content}
            ]
        else:
            messages = [
                {"role": "user", "content": content}
            ]
        return {
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

    def single_infer(self, image_path: str, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        return asyncio.run(self._single_infer(image_path, system_prompt, user_prompt))
    
    async def _single_infer(self, image_path: str, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """
        单个推理的异步方法
        
        Args:
            image_path: 图像路径
            system_prompt: 系统提示词
            user_prompt: 用户提示词
            
        Returns:
            推理结果字典
        """
        try:
            preprocessed_data = self.preprocess(image_path, system_prompt, user_prompt)
            prediction = await self.client.chat(
                preprocessed_data["messages"],
                temperature=preprocessed_data["temperature"],
                max_tokens=preprocessed_data["max_tokens"]
            )
            return self.create_result_dict(
                success=True,
                prediction=prediction,
                image_path=image_path,
            )
        except Exception as e:
            return self.create_result_dict(
                success=False,
                error=str(e),
                image_path=image_path,
            )
    def batch_infer(self, *args) -> List[Dict[str, Any]]:
        return asyncio.run(self._batch_infer(*args))
    
    # Replace the batch inference completion loop with this improved version
    async def _batch_infer(self, *args) -> List[Dict[str, Any]]:
        """
        批量推理的异步方法
        """
        from utils.data_processor import _prepare_samples

        samples = _prepare_samples(*args)
        concurrency = self.concurrency
        semaphore = asyncio.Semaphore(concurrency)

        async def limited_single_infer(idx: int, sample: Dict[str, Any]):
            async with semaphore:
                try:
                    r = await self._single_infer(
                        sample["image_path"],
                        sample["system_prompt"],
                        sample["user_prompt"],
                    )
                    return idx, r
                except Exception as e:
                    return idx, self.create_result_dict(
                        success=False,
                        error=str(e),
                        image_path=sample.get("image_path", ""),
                    )

        tasks = [limited_single_infer(i, s) for i, s in enumerate(samples)]
        processed_results = [None] * len(tasks)  # 预分配，保证顺序

        # 实时进度条 + 按 idx 写入结果
        for coro in async_tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="推理进度"):
            idx, r = await coro
            processed_results[idx] = r

        return processed_results
