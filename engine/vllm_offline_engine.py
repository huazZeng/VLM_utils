#!/usr/bin/env python3
"""
vLLM离线推理引擎实现
基于vLLM库的本地模型推理
"""

from dataclasses import asdict
from vllm import LLM, EngineArgs, SamplingParams
from PIL import Image
from typing import Dict, Any, List
from tqdm import tqdm

from .engine_base import InferenceEngineBase


class VLLMOfflineEngine(InferenceEngineBase):
    """
    vLLM离线推理引擎
    基于vLLM库实现本地模型推理
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct", **kwargs):
        """
        初始化vLLM离线推理引擎
        
        Args:
            model_name: 模型名称
            **kwargs: 其他参数
        """
        super().__init__(**kwargs)
        self.batch_size = kwargs.get('batch_size', 16)
        self.model_name = model_name
        
        # 初始化vLLM模型
        print(f"Loading vLLM offline model: {model_name}")
        engine_args = EngineArgs(
            model=model_name,
            max_model_len=4096,
            max_num_seqs=4,  # 支持批量推理
            mm_processor_kwargs={
                "min_pixels": 401408,
                "max_pixels": 1003520,
            },
            limit_mm_per_prompt={"image": 1},
        )
        
        self.llm = LLM(**asdict(engine_args))
        print("vLLM offline model loaded successfully!")
    
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
                temperature=0.0,
                max_tokens=1024,
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
        """
        单个推理：使用vLLM模型对单张图像进行推理
        
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
            
            # 执行推理
            inputs = {
                "prompt": preprocessed_data["prompt"],
                "multi_modal_data": preprocessed_data["multi_modal_data"]
            }
            
            outputs = self.llm.generate(inputs, sampling_params=preprocessed_data["sampling_params"])
            
            # 提取生成的文本
            if outputs and outputs[0].outputs:
                prediction = outputs[0].outputs[0].text.strip()
            else:
                prediction = ""
            
            return self.create_result_dict(
                success=True,
                prediction=prediction,
                image_path=image_path,
                metadata={"model_name": self.model_name}
            )
            
        except Exception as e:
            return self.create_result_dict(
                success=False,
                error=str(e),
                image_path=image_path,
                metadata={"model_name": self.model_name}
            )
    
    def batch_infer(self, image_paths: List[str], system_prompt: str, user_prompt: str, **kwargs) -> List[Dict[str, Any]]:
        """
        批量推理：使用vLLM模型对多张图像进行推理
        
        Args:
            image_paths: 图像路径列表
            system_prompt: 系统提示词
            user_prompt: 用户提示词
            **kwargs: 其他参数，包括batch_size
            
        Returns:
            推理结果列表
        """
        results = []
        
        # 分批处理数据
        for batch_start in tqdm(range(0, len(image_paths), self.batch_size), desc="Processing batches"):
            batch_end = min(batch_start + self.batch_size, len(image_paths))
            batch_paths = image_paths[batch_start:batch_end]
            
            # 准备批量输入
            batch_inputs = []
            batch_indices = []
            
            for i, image_path in enumerate(batch_paths):
                try:
                    # 预处理
                    preprocessed_data = self.preprocess(image_path, system_prompt, user_prompt)
                    
                    # 添加到批量输入
                    batch_inputs.append({
                        "prompt": preprocessed_data["prompt"],
                        "multi_modal_data": preprocessed_data["multi_modal_data"]
                    })
                    batch_indices.append(batch_start + i)
                    
                except Exception as e:
                    # 如果预处理失败，添加错误结果
                    error_result = self.create_result_dict(
                        success=False,
                        error=str(e),
                        image_path=image_path,
                        metadata={"model_name": self.model_name}
                    )
                    results.append(error_result)
            
            if not batch_inputs:
                continue
            
            try:
                # 设置采样参数
                sampling_params = SamplingParams(
                    temperature=0.0,
                    max_tokens=1024,
                    stop=["<|im_end|>"],
                    skip_special_tokens=False
                )
                
                # 执行批量推理
                outputs = self.llm.generate(batch_inputs, sampling_params=sampling_params)
                
                # 处理批量输出
                for i, output in enumerate(outputs):
                    sample_idx = batch_indices[i]
                    image_path = batch_paths[i]
                    
                    # 提取生成的文本
                    if output.outputs:
                        prediction = output.outputs[0].text.strip()
                    else:
                        prediction = ""
                    
                    result = self.create_result_dict(
                        success=True,
                        prediction=prediction,
                        image_path=image_path,
                        metadata={"model_name": self.model_name}
                    )
                    results.append(result)
                
            except Exception as e:
                # 如果批量推理失败，为失败的批次添加错误结果
                for i, sample_idx in enumerate(batch_indices):
                    image_path = batch_paths[i]
                    error_result = self.create_result_dict(
                        success=False,
                        error=str(e),
                        image_path=image_path,
                        metadata={"model_name": self.model_name}
                    )
                    results.append(error_result)
        
        return results 