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
from utils.data_processor import _prepare_samples
from engine.engine_base import InferenceEngineBase



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
        self.batch_size = kwargs.get("batch_size", 1)
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
            messages = []
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt
                    },
                    {
                        "type": "image",
                        "image": image_path
                    }
                ]
            })
            
            
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
                )
            # 解码输出
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
            prediction = self.processor.batch_decode(generated_ids, skip_special_tokens=self.skip_special_token, clean_up_tokenization_spaces=False)[0]
            
            return self.create_result_dict(
                success=True,
                prediction=prediction,
                image_path=image_path,
                metadata={"model_name": self.model_name, "device": self.device}
            )
            
        except Exception as e:
            return self.create_result_dict(
                success=False,
                error=str(e),
                image_path=image_path,
                metadata={"model_name": self.model_name, "device": self.device}
            )
    
    def _process_vision_info(self, messages_batch):
        """
        处理批量消息中的视觉信息
        
        Args:
            messages_batch: 批量消息列表
            
        Returns:
            image_inputs: 图像输入列表
            video_inputs: 视频输入列表（当前为空）
        """
        image_inputs = []
        video_inputs = []
        
        for messages in messages_batch:
            images = []
            videos = []
            
            for message in messages:
                if message["role"] == "user":
                    for content in message["content"]:
                        if content.get("type") == "image" and "image" in content:
                            image_path = content["image"]
                            try:
                                image = self.load_image(image_path)
                                images.append(image)
                            except Exception as e:
                                print(f"Failed to load image {image_path}: {e}")
                                images.append(None)
                        elif content.get("type") == "video" and "video" in content:
                            # 处理视频（当前为空实现）
                            videos.append(None)
            
            # 如果当前消息没有图像，添加None
            if not images:
                images = [None]
            if not videos:
                videos = [None]
                
            image_inputs.extend(images)
            video_inputs.extend(videos)
        
        return image_inputs, video_inputs

    def batch_infer(self, *args) -> List[Dict[str, Any]]:
        """
        批量推理：使用Transformer模型对多张图像进行真正的批量推理
        
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
        results = []
        
        # 提取批量数据
        image_paths = [sample["image_path"] for sample in samples]
        user_prompts = [sample["user_prompt"] for sample in samples]
        system_prompts = [sample["system_prompt"] for sample in samples]
        
        # 计算总批次数
        total_batches = (len(samples) + self.batch_size - 1) // self.batch_size
        
        # 按批次处理，添加进度条
        for batch_start in tqdm(range(0, len(samples), self.batch_size), 
                               total=total_batches, 
                               desc="Processing batches", 
                               unit="batch"):
            batch_end = min(batch_start + self.batch_size, len(samples))
            batch_image_paths = image_paths[batch_start:batch_end]
            batch_user_prompts = user_prompts[batch_start:batch_end]
            batch_system_prompts = system_prompts[batch_start:batch_end]
            
            try:
                # 准备批量消息格式
                messages_batch = []
                for image_path, user_prompt, system_prompt in zip(batch_image_paths, batch_user_prompts, batch_system_prompts):
                    messages = [
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image_path},
                                {"type": "text", "text": user_prompt},
                            ],
                        }
                    ]
                    messages_batch.append(messages)
                
                # 准备文本输入 - 对齐示例代码
                texts = [
                    self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    for messages in messages_batch
                ]
                
                # 处理视觉信息 - 对齐示例代码
                image_inputs, video_inputs = self._process_vision_info(messages_batch)
                
                # 准备批量输入 - 对齐示例代码
                inputs = self.processor(
                    text=texts,
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = inputs.to(self.device)
                
                # 执行批量推理 - 对齐示例代码
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **inputs, 
                        max_new_tokens=self.max_tokens,
                    )
                
                # 解码输出 - 对齐示例代码
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_texts = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=self.skip_special_token, 
                    clean_up_tokenization_spaces=False
                )
                
                # 构建当前批次的结果列表
                for i, (image_path, output_text) in enumerate(zip(batch_image_paths, output_texts)):
                    results.append(self.create_result_dict(
                        success=True,
                        prediction=output_text,
                        image_path=image_path,
                        metadata={"model_name": self.model_name, "device": self.device}
                    ))
                        
            except Exception as e:
                # 如果当前批次推理失败，为所有样本添加错误结果
                for image_path in batch_image_paths:
                    results.append(self.create_result_dict(
                        success=False,
                        error=f"Batch inference failed: {str(e)}",
                        image_path=image_path,
                        metadata={"model_name": self.model_name, "device": self.device}
                    ))
                print(f"Batch failed: {str(e)}")
        
        return results