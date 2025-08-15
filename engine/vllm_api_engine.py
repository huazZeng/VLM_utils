#!/usr/bin/env python3
"""
vLLM API推理引擎实现
基于vLLM服务器的API调用推理（支持同步和异步）
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
from tqdm.asyncio import tqdm as async_tqdm  # 用异步版 tqdm


class VLLMAPIEngine(InferenceEngineBase):
    """
    vLLM API推理引擎
    支持同步调用与异步调用
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
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content}
        ]
        return {
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

    # ======== 异步接口 ========
    async def _single_infer(self, image_path: str, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
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
                metadata={"model_name": self.model_name, "base_url": self.base_url}
            )
        except Exception as e:
            return self.create_result_dict(
                success=False,
                error=str(e),
                image_path=image_path,
                metadata={"model_name": self.model_name, "base_url": self.base_url}
            )


    async def _batch_infer(self, image_paths: List[str], system_prompt: str, user_prompt: str) -> List[Dict[str, Any]]:
        concurrency = self.concurrency
        semaphore = asyncio.Semaphore(concurrency)

        async def limited_single_infer(image_path: str):
            async with semaphore:
                return await self._single_infer(image_path, system_prompt, user_prompt)

        tasks = [limited_single_infer(p) for p in image_paths]
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
                    image_path=image_paths[idx],
                    metadata={"model_name": self.model_name, "base_url": self.base_url}
                ))

        return processed_results


    # ======== 同步接口（安全调用 async） ========
    def single_infer(self, image_path: str, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        return self._safe_async_run(self._single_infer(image_path, system_prompt, user_prompt))

    def batch_infer(self, image_paths: List[str], system_prompt: str, user_prompt: str, **kwargs) -> List[Dict[str, Any]]:
        return self._safe_async_run(self._batch_infer(image_paths, system_prompt, user_prompt, **kwargs))

    # ======== 统一安全运行 async 方法 ========
    @staticmethod
    def _safe_async_run(awaitable):
        """
        安全运行 async 协程，兼容已有事件循环的环境
        """
        return asyncio.run(awaitable)