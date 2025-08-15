#!/usr/bin/env python3
"""
统一的推理器，基于engine包实现
支持多种推理引擎：transformer, vllm_offline, vllm_api
"""

import os
import json
import argparse
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from engine.engine_factory import InferenceManager
from base_inference import BaseInference, BaseCLI

class UnifiedInference(BaseInference):
    """
    基于engine包实现，支持多种推理引擎
    """
    
    def __init__(self, engine_type: str, **engine_kwargs):
        """
        初始化统一推理器
        
        Args:
            engine_type: 引擎类型，支持 "transformer", "vllm_offline", "vllm_api"
            **engine_kwargs: 引擎特定的初始化参数
        """
        super().__init__()
        system_prompt_file = engine_kwargs.get("system_prompt_file", None)
        if system_prompt_file:
            with open(system_prompt_file, "r") as f:
                self.system_prompt = f.read()
        else:
            self.system_prompt = "You are a helpful assistant."


        user_prompt_file = engine_kwargs.get("user_prompt_file", None)
        if user_prompt_file:
            with open(user_prompt_file, "r") as f:
                self.user_prompt = f.read()
        else:
            self.user_prompt = "Describe the image in detail."

        # 初始化推理管理器
        self.manager = InferenceManager(engine_type, **engine_kwargs)
        self.engine_type = engine_type
        
        print(f"Unified inference initialized with engine: {engine_type}")
    
    def single_infer(self, image_path: str, prompt: str) -> str:
        """
        单个推理：直接输入prompt和image_path进行推理
        
        Args:
            image_path: 图像路径
            prompt: 提示文本
            
        Returns:
            模型输出文本
        """
        try:
            # 使用推理管理器进行推理
            result = self.manager.single_infer(image_path, system_prompt=self.system_prompt, user_prompt=prompt)
            
            if result["success"]:
                return result["prediction"]
            else:
                print(f"Error in inference: {result['error']}")
                return ""
                
        except Exception as e:
            print(f"Error in single inference for {image_path}: {e}")
            return ""
    
    def _process_json_batch(self, data: List[Dict], output_dir: str, save_mode: str, **kwargs) -> List[Dict]:
        """
        处理JSON文件的批量推理
        
        Args:
            data: JSON数据列表
            output_dir: 输出目录
            save_mode: 存储模式
            **kwargs: 其他参数
            
        Returns:
            推理结果列表
        """
        results = []
        
        # 准备批量推理数据
        image_paths = []
        sample_indices = []
        
        for i, sample in enumerate(data):
            if sample["images"]:
                image_paths.append(sample["images"][0])
                sample_indices.append(i)
            else:
                print(f"Warning: No image path for sample {i}")
                continue
        
        if not image_paths:
            print("No valid samples found!")
            return results
        
        try:
            # 使用engine的batch_infer进行批量推理
            # 对于JSON文件，我们使用默认的prompt，因为每个样本的prompt可能不同
            default_prompt = self.user_prompt
            batch_results = self.manager.batch_infer(
                image_paths, 
                system_prompt=self.system_prompt,
                user_prompt=default_prompt,
            )
            
            # 处理批量推理结果
            for i, (sample_idx, batch_result) in enumerate(zip(sample_indices, batch_results)):
                sample = data[sample_idx]
                
                if batch_result["success"]:
                    prediction = batch_result["prediction"]
                else:
                    prediction = ""
                    print(f"Error in batch inference for sample {sample_idx}: {batch_result['error']}")
                
                result = {
                    "image_path": sample["images"][0],
                    "prompt": sample["messages"][-2]["content"],  # 使用原始prompt
                    "ground_truth": sample["messages"][-1]["content"],  # assistant message
                    "prediction": prediction
                }
                results.append(result)
                
                # 立即保存单个结果
                if save_mode == "divided":
                    self._save_divided_results(result, output_dir, "json")
            
        except Exception as e:
            print(f"Error in batch inference: {e}")
            # 如果批量推理失败，回退到单个推理
            print("Falling back to single inference...")
            for i, sample_idx in enumerate(sample_indices):
                sample = data[sample_idx]
                image_path = sample["images"][0]
                prompt = sample["messages"][-2]["content"]
                
                try:
                    prediction = self.single_infer(image_path, prompt)
                except Exception as single_error:
                    prediction = ""
                    print(f"Error in single inference for sample {sample_idx}: {single_error}")
                
                result = {
                    "image_path": image_path,
                    "prompt": prompt,
                    "ground_truth": sample["messages"][-1]["content"],
                    "prediction": prediction
                }
                results.append(result)
                
                # 立即保存单个结果
                if save_mode == "divided":
                    self._save_divided_results(result, output_dir, "json")
        
        return results
    
    def _process_folder_batch(self, data: List[str], output_dir: str, save_mode: str, **kwargs) -> List[Dict]:
        """
        处理文件夹的批量推理
        
        Args:
            data: 图像路径列表
            output_dir: 输出目录
            save_mode: 存储模式
            **kwargs: 其他参数
            
        Returns:
            推理结果列表
        """
        results = []
        default_prompt = "Describe the image in detail."
        
        try:
            # 使用engine的batch_infer进行批量推理
            batch_results = self.manager.batch_infer(
                data,  # data就是image_paths列表
                system_prompt=self.system_prompt,
                user_prompt=default_prompt,
            )
            
            # 处理批量推理结果
            for i, batch_result in enumerate(batch_results):
                image_path = data[i]
                
                if batch_result["success"]:
                    prediction = batch_result["prediction"]
                else:
                    prediction = ""
                    print(f"Error in batch inference for image {i}: {batch_result['error']}")
                
                result = {
                    "image_path": image_path,
                    "prompt": default_prompt,
                    "prediction": prediction
                }
                results.append(result)
                
                # 立即保存单个结果
                if save_mode == "divided":
                    self._save_divided_results(result, output_dir, "folder")
            
            
        except Exception as e:
            print(f"Error in batch inference: {e}")
            # 如果批量推理失败，回退到单个推理
            print("Falling back to single inference...")
            for i, image_path in enumerate(data):
                try:
                    prediction = self.single_infer(image_path, default_prompt)
                except Exception as single_error:
                    prediction = ""
                    print(f"Error in single inference for image {i}: {single_error}")
                
                result = {
                    "image_path": image_path,
                    "prompt": default_prompt,
                    "prediction": prediction
                }
                results.append(result)
                
                # 立即保存单个结果
                if save_mode == "divided":
                    self._save_divided_results(result, output_dir, "folder")
        
        return results
    
    def batch_infer(self, input_path: str, output_dir: str, save_mode: str = "all", **kwargs):
        """
        批量推理：支持JSON文件和文件夹输入
        
        Args:
            input_path: 输入路径（JSON文件或文件夹）
            output_file: 输出文件路径
            save_mode: 存储模式，"divided"或"all"
            **kwargs: 其他参数
        """
        # 加载输入数据
        input_data = self.load_input_data(input_path)
        input_type = input_data["type"]
        data = input_data["data"]
        
        print(f"Processing {len(data)} samples from {input_type} input using {self.engine_type} engine...")
        
        
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")
        
        
        # 根据输入类型选择处理函数
        if input_type == "json":
            results = self._process_json_batch(data, output_dir, save_mode, **kwargs)
        elif input_type == "folder":
            results = self._process_folder_batch(data, output_dir, save_mode, **kwargs)
        else:
            raise ValueError(f"Unsupported input type: {input_type}")
        
        # 根据save_mode保存结果
        if save_mode == "divided":
            # divided模式已经在循环中保存了每个结果
            print(f"All individual results saved to: {output_dir}")
        else:  # save_mode == "all"
            # 保存所有结果到一个文件
            self._save_all_results(results, output_dir, input_type)
        
        print(f"Batch inference completed! Results saved with mode: {save_mode}")
        print(f"Processed {len(results)} samples successfully.")


def main():
    """主函数"""
    # 创建解析器
    parser, single_parser, batch_parser = BaseCLI.create_parser("Unified Model inference for spectral detection")
    
    # 添加引擎选择参数
    parser.add_argument("--engine_type", type=str, required=True, 
                       choices=["transformer", "vllm_offline", "vllm_api"], 
                       help="Inference engine type")
    
    # 添加引擎特定参数
    parser.add_argument("--skip_special_token", type=bool, default=False, help="Skip special token for inference")
    parser.add_argument("--model_name", type=str, help="Model name or path")
    parser.add_argument("--system_prompt", type=str, help="System prompt for inference")
    parser.add_argument("--user_prompt", type=str, help="User prompt for inference")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (for transformer)")
    parser.add_argument("--base_url", type=str, help="vLLM server base URL (for vllm_api)")
    parser.add_argument("--api_key", type=str, default="EMPTY", help="API key (for vllm_api)")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Max tokens for inference (for vllm_api)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for inference (for vllm_api)")
    batch_parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference")
    batch_parser.add_argument("--concurrency", type=int, default=64, help="Concurrency level for inference (for vllm_api)")
    
    args = parser.parse_args()
    
    # 根据引擎类型准备参数
    engine_kwargs = {}
    
    if args.engine_type == "transformer":
        if not args.model_name:
            raise ValueError("--model_name is required for transformer engine")
        engine_kwargs = {
            "model_name": args.model_name,
            "device": args.device,
            "skip_special_token": args.skip_special_token,
            "system_prompt": args.system_prompt
        }
    elif args.engine_type == "vllm_offline":
        engine_kwargs = {
            "model_name": args.model_name or "Qwen/Qwen2.5-VL-3B-Instruct",
            "batch_size": args.batch_size,
            "skip_special_token": args.skip_special_token,
            "system_prompt": args.system_prompt
        }
    elif args.engine_type == "vllm_api":
        if not args.base_url or not args.model_name:
            raise ValueError("--base_url and --model_name are required for vllm_api engine")
        engine_kwargs = {
            "base_url": args.base_url,
            "model_name": args.model_name,
            "api_key": args.api_key,
            "concurrency": args.concurrency,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "system_prompt": args.system_prompt
        }

    # 初始化推理器
    inference = UnifiedInference(args.engine_type, **engine_kwargs)
    
    if args.mode == 'single':
        # 单个推理模式
        try:
            BaseCLI.validate_single_args(args)
            BaseCLI.print_single_info(args)
            print(f"  Engine type: {args.engine_type}")
            
            result = inference.single_infer(args.image_path, args.prompt)
            print(f"\nInference result:")
            print(result)
            
            # 可视化（如果指定了save_path）
            if args.save_path:
                parsed_dict = inference.parser.parse_to_dict(result)

                
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return
        
    elif args.mode == 'batch':
        # 批量推理模式
        try:
            BaseCLI.validate_batch_args(args)
            BaseCLI.print_batch_info(args)
            print(f"  Engine type: {args.engine_type}")
            
            # 根据引擎类型设置批量参数
            batch_kwargs = {}
            if args.engine_type == "vllm_offline":
                batch_kwargs["batch_size"] = args.batch_size
            elif args.engine_type == "vllm_api":
                batch_kwargs["concurrency"] = args.concurrency
            
            inference.batch_infer(args.input_path, args.output_file, args.save_mode, **batch_kwargs)
            
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return
        
    else:
        print("Please specify inference mode: 'single' or 'batch'")
        parser.print_help()


if __name__ == "__main__":
    main() 