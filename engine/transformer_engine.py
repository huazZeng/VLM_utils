#!/usr/bin/env python3
"""
Transformer推理引擎实现
基于transformers库的本地模型推理
"""

import torch
from transformers import AutoTokenizer, AutoModelForVision2Seq, AutoProcessor, AutoConfig
from PIL import Image
from typing import Dict, Any, List
from tqdm import tqdm

from .engine_base import InferenceEngineBase


class TransformerEngine(InferenceEngineBase):
    """
    Transformer推理引擎
    基于transformers库实现本地模型推理
    """
    
    def __init__(self, model_name: str, device: str = "cuda", **kwargs):
        """
        初始化Transformer推理引擎
        
        Args:
            model_name: 模型名称或路径
            device: 设备类型
            **kwargs: 其他参数
        """
        super().__init__(**kwargs)
        
        self.device = device
        self.model_name = model_name
        self.max_tokens = kwargs.get("max_tokens", 1024)
        self.temperature = kwargs.get("temperature", 0.0)
        self.skip_special_token = kwargs.get("skip_special_token", False)
        
        print(f"Loading Transformer model: {model_name}")
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

        # 加载模型和tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
            config=config
        )
        
        print("Transformer model loaded successfully!")
    
    def preprocess(self, image_path: str, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """
        预处理：准备Transformer模型推理所需的输入数据
        
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
            
            # 准备消息格式
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_prompt
                        },
                        {
                            "image": image_path
                        }
                    ]
                }
            ]
            
            # 应用聊天模板
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt").to(self.device)
            
            return {
                "inputs": inputs,
                "image": image,
                "text": text
            }
            
        except Exception as e:
            raise Exception(f"Preprocessing failed: {e}")
    
    def single_infer(self, image_path: str, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """
        单个推理：使用Transformer模型对单张图像进行推理
        
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
            inputs = preprocessed_data["inputs"]
            
            # 生成输出
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs, 
                    max_new_tokens=self.max_tokens,
                    skip_special_tokens=self.skip_special_token,
                    temperature=self.temperature
                )
            
            # 解码输出
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
            prediction = self.processor.batch_decode(generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
            
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
    
    def batch_infer(self, image_paths: List[str], system_prompt: str, user_prompt: str) -> List[Dict[str, Any]]:
        """
        批量推理：使用Transformer模型对多张图像进行推理
        
        Args:
            image_paths: 图像路径列表
            system_prompt: 系统提示词
            user_prompt: 用户提示词
            **kwargs: 其他参数
            
        Returns:
            推理结果列表
        """
        results = []
        
        for i, image_path in enumerate(tqdm(image_paths, desc="Processing images")):
            try:
                result = self.single_infer(image_path, system_prompt, user_prompt)
                results.append(result)
            except Exception as e:
                # 如果单个推理失败，添加错误结果
                error_result = self.create_result_dict(
                    success=False,
                    error=str(e),
                    image_path=image_path,
                )
                results.append(error_result)
        
        return results 