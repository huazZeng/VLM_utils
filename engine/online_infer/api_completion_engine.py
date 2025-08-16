#!/usr/bin/env python3
"""
vLLM API推理引擎实现
基于vLLM服务器的API调用推理（纯异步实现）
"""

import asyncio
import base64
import io
from PIL import Image
from qwen_vl_utils import smart_resize
from typing import Dict, Any, List
from tqdm.asyncio import tqdm as async_tqdm
from vllm import SamplingParams
from utils.api_infer import OpenAIChatClient
from engine.engine_base import InferenceEngineBase


class APICompletionEngine(InferenceEngineBase):
    """
    vLLM API推理引擎
    纯异步实现，支持completion接口
    """

    def __init__(self, base_url: str, model_name: str, api_key: str = "EMPTY", **kwargs):
        super().__init__(**kwargs)
        self.base_url = base_url
        self.model_name = model_name
        self.api_key = api_key
        self.concurrency = kwargs.get("concurrency", 64)
        self.max_tokens = kwargs.get("max_tokens", 1024)
        self.temperature = kwargs.get("temperature", 0.0)
        self.client = OpenAIChatClient(
            base_url=self.base_url,
            api_key=self.api_key,
            model_name=self.model_name
        )
        print(f"vLLM API client initialized with model: {model_name}")
        print(f"vLLM API client initialized with base_url: {base_url}")

    def image_to_data_uri(self, image_path: str) -> str:
        """将图像转换为data URI格式"""
        image = Image.open(image_path)
        original_width, original_height = image.size
        new_width, new_height = smart_resize(
            original_width, original_height,
            min_pixels=512*28*28, max_pixels=2048*28*28
        )
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        resized_image.save(buffer, format='JPEG', quality=95)
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
        try:
            # 加载图像
            image = self.load_image(image_path)
            
            # 创建prompt
            full_prompt = (
                "<|im_start|>system\n" + system_prompt + "<|im_end|>\n"
                "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
                f"{user_prompt}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            
            # 设置采样参数
            sampling_params = SamplingParams(
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stop=["<|im_end|>"],
                skip_special_tokens=False
            )
            
            return {
                "prompt": full_prompt,
                "multi_modal_data": {"image": image},
                "sampling_params": sampling_params
            }
            
        except Exception as e:
            raise Exception(f"Preprocessing failed: {e}")

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
            prediction = await self.client.complete(
                preprocessed_data["prompt"],
                temperature=preprocessed_data["sampling_params"].temperature,
                max_tokens=preprocessed_data["sampling_params"].max_tokens
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
    
    async def _batch_infer(self, *args) -> List[Dict[str, Any]]:
        """
        批量推理的异步方法
        
        Args:
            *args: 可变参数，支持以下三种情况：
            1. list image_paths, str user_prompt, str system_prompt
            2. list image_paths, list user_prompts, list system_prompts
            3. list of dict [{"image_path":..., "user_prompt":..., "system_prompt":...}, ...]
            
        Returns:
            推理结果列表
        """
        from utils.data_processor import _prepare_samples
        
        samples = _prepare_samples(*args)
        concurrency = self.concurrency
        semaphore = asyncio.Semaphore(concurrency)

        async def limited_single_infer(sample: Dict[str, Any]):
            async with semaphore:
                return await self._single_infer(
                    sample["image_path"], 
                    sample["system_prompt"], 
                    sample["user_prompt"]
                )

        tasks = [limited_single_infer(sample) for sample in samples]
        processed_results = []

        # 使用 asyncio.as_completed 来实时更新进度
        for coro in async_tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="推理进度"):
            try:
                r = await coro
                processed_results.append(r)
            except Exception as e:
                idx = len(processed_results)
                processed_results.append(self.create_result_dict(
                    success=False,
                    error=str(e),
                    image_path=samples[idx]["image_path"] if idx < len(samples) else "",
                ))

        return processed_results