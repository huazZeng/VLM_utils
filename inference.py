#!/usr/bin/env python3
"""
统一的推理入口文件
支持多种推理引擎：online (api_chat, api_completion) 和 offline (vllm_offline, transformer)
"""

import argparse
import os
from typing import List, Dict, Any
from base_inference import BaseInference
from engine.engine_factory import InferenceEngineFactory


class UnifiedInference(BaseInference):
    """
    统一推理类，继承BaseInference，支持所有类型的推理引擎
    """
    
    def __init__(self, parser: str, engine_type: str, **engine_kwargs):
        """
        初始化统一推理器
        
        Args:
            parser: 解析器类型
            engine_type: 引擎类型，支持 "api_chat", "api_completion", "vllm_offline", "transformer"
            **engine_kwargs: 引擎特定的初始化参数
        """
        if parser is None:
            parser = "default"
        super().__init__(parser)
        
        # 设置系统提示词和用户提示词
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

        # 直接初始化推理引擎
        self.engine = InferenceEngineFactory.create_engine(engine_type, **engine_kwargs)
        self.engine_type = engine_type
        
        print(f"Unified inference initialized with engine: {engine_type}")
    
    def single_infer(self, image_path: str, prompt: str) -> str:
        """
        单个推理方法
        
        Args:
            image_path: 图像路径
            prompt: 提示文本
            
        Returns:
            模型输出文本
        """
        try:
            result = self.engine.single_infer(
                image_path=image_path,
                system_prompt=self.system_prompt,
                user_prompt=prompt
            )
            
            return result["prediction"]
                
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
            # 为每个样本准备对应的prompt和system_prompt
            prompts = []
            system_prompts = []
            
            for sample_idx in sample_indices:
                sample = data[sample_idx]
                messages = sample.get("messages", [])
                
                # 获取user prompt
                user_prompt = ""
                system_prompt = ""
                for msg in messages:
                    if msg.get("role") == "user":
                        content = msg.get("content", "")
                        user_prompt = content
                    if msg.get("role") == "system":
                        system_prompt = msg.get("content", "")
                        
                
                if not system_prompt:
                    system_prompt = ""
                prompts.append(user_prompt)
                system_prompts.append(system_prompt)
            
            # 使用引擎的batch_infer进行批量推理，每个样本使用自己的prompt
            batch_results = self.engine.batch_infer(
                image_paths, 
                prompts,
                system_prompts,
            )
            
            # 处理批量推理结果
            for i, (sample_idx, batch_result) in enumerate(zip(sample_indices, batch_results)):
                sample = data[sample_idx]
                
                prediction = batch_result["prediction"]
                messages = sample.get("messages", [])
                ground_truth = ""
                
                for msg in messages:
                    if msg.get("role") == "assistant":
                        ground_truth = msg.get("content", "")
                        break
                
                result = {
                    "image_path": sample["images"][0],
                    "prompt": prompts[i],  # 使用之前提取的prompt
                    "ground_truth": ground_truth,
                    "prediction": prediction
                }
                results.append(result)
                
            
        except Exception as e:
            print(f"Error in batch inference: {e}")
            # 如果批量推理失败，回退到单个推理
            print("Falling back to single inference...")
            for i, sample_idx in enumerate(sample_indices):
                sample = data[sample_idx]
                image_path = sample["images"][0]
                
                # 直接使用之前提取的prompt，只需要获取ground_truth
                messages = sample.get("messages", [])
                ground_truth = ""
                
                # 查找assistant消息作为ground_truth
                for msg in messages:
                    if msg.get("role") == "assistant":
                        ground_truth = msg.get("content", "")
                        break
                
                try:
                    prediction = self.single_infer(image_path, prompts[i])  # 使用之前提取的prompt
                except Exception as single_error:
                    prediction = ""
                    print(f"Error in single inference for sample {sample_idx}: {single_error}")
                
                result = {
                    "image_path": image_path,
                    "prompt": prompts[i],  # 使用之前提取的prompt
                    "ground_truth": ground_truth,
                    "prediction": prediction
                }
                results.append(result)
        
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
        default_prompt = self.user_prompt
        
        try:
            # 直接使用引擎的batch_infer进行批量推理
            batch_results = self.engine.batch_infer(
                data,  # data就是image_paths列表
                default_prompt,
                self.system_prompt,
            )
            
            # 处理批量推理结果
            for i, batch_result in enumerate(batch_results):
                image_path = data[i]
                
                prediction = batch_result["prediction"]
                
                result = {
                    "image_path": image_path,
                    "prompt": default_prompt,
                    "prediction": prediction
                }
                results.append(result)
            
            
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
        
        return results
    
    def batch_infer(self, input_path: str, output_dir: str, save_mode: str = "all", **kwargs):
        """
        批量推理：支持JSON文件和文件夹输入
        
        Args:
            input_path: 输入路径（JSON文件或文件夹）
            output_dir: 输出目录
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
            # 保存所有结果到分离的文件
            self._save_divided_results(results, output_dir, input_type)
            print(f"All individual results saved to: {output_dir}")
        else:  # save_mode == "all"
            # 保存所有结果到一个文件
            self._save_all_results(results, output_dir, input_type)
        
        print(f"Batch inference completed! Results saved with mode: {save_mode}")
        print(f"Processed {len(results)} samples successfully.")


def add_engine_specific_args(parser: argparse.ArgumentParser, engine_type: str):
    """
    根据引擎类型添加特定的参数
    
    Args:
        parser: 参数解析器
        engine_type: 引擎类型
    """
    if engine_type in ["api_chat", "api_completion"]:
        # Online API 引擎参数
        parser.add_argument("--base_url", type=str, required=True, help="API基础URL")
        parser.add_argument("--model_name", type=str, required=True, help="模型名称")
        parser.add_argument("--api_key", type=str, default="EMPTY", help="API密钥")
        parser.add_argument("--concurrency", type=int, default=64, help="并发数")
        parser.add_argument("--max_tokens", type=int, default=1024, help="最大token数")
        parser.add_argument("--temperature", type=float, default=0.0, help="温度参数")
        
    elif engine_type in ["vllm_offline", "transformer"]:
        # Offline 引擎参数
        parser.add_argument("--model_name", type=str, help="模型名称或路径")
        parser.add_argument("--skip_special_token", type=bool, default=False, help="跳过特殊token")
        if engine_type == "vllm_offline":
            parser.add_argument("--batch_size", type=int, default=16, help="批处理大小")


def main():
    """主函数"""
    # 创建解析器
    parser = argparse.ArgumentParser(description="统一推理工具，支持多种推理引擎")
    parser.add_argument("--parser", type=str, help="解析器类型")
    parser.add_argument("--engine_type", type=str, required=True, 
                       choices=["api_chat", "api_completion", "vllm_offline", "transformer"], 
                       help="推理引擎类型")
    parser.add_argument("--system_prompt_file", type=str, help="系统提示词文件")
    parser.add_argument("--user_prompt_file", type=str, help="用户提示词文件")
    
    # 解析引擎类型参数
    args, _ = parser.parse_known_args()
    
    # 根据引擎类型添加特定参数
    add_engine_specific_args(parser, args.engine_type)
    
    # 添加推理模式子解析器
    subparsers = parser.add_subparsers(dest='mode', help='推理模式')
        
    # 单个推理模式
    single_parser = subparsers.add_parser('single', help='单个推理模式')
    single_parser.add_argument("--image_path", type=str, required=True, help="图像文件路径")
    single_parser.add_argument("--prompt", type=str, default="<image>Find the spectral images in the figure", help="提示文本")
    single_parser.add_argument("--save_path", type=str, default=None, help="可视化保存路径")
        
    # 批量推理模式
    batch_parser = subparsers.add_parser('batch', help='批量推理模式')
    batch_parser.add_argument("--input_path", type=str, required=True, help="输入路径（JSON文件或文件夹）")
    batch_parser.add_argument("--output_dir", type=str, required=True, help="推理结果输出目录")
    batch_parser.add_argument("--save_mode", type=str, choices=["divided", "all"], default="all", help="保存模式：'divided'为分离文件，'all'为单个文件")
    batch_parser.add_argument("--batch_size", type=int, default=16, help="批处理大小")
    batch_parser.add_argument("--concurrency", type=int, default=64, help="并发数")
    # 重新解析所有参数
    args = parser.parse_args()

    # 根据引擎类型准备参数
    engine_kwargs = {}
    
    if args.engine_type in ["api_chat", "api_completion"]:
        engine_kwargs = {
            "base_url": args.base_url,
            "model_name": args.model_name,
            "api_key": args.api_key,
            "concurrency": args.concurrency,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
        }
    elif args.engine_type == "vllm_offline":
        engine_kwargs = {
            "model_name": args.model_name or "Qwen/Qwen2.5-VL-3B-Instruct",
            "batch_size": args.batch_size,
            "skip_special_token": args.skip_special_token,
        }
    elif args.engine_type == "transformer":
        if not args.model_name:
            raise ValueError("--model_name is required for transformer engine")
        engine_kwargs = {
            "model_name": args.model_name,
            "batch_size": args.batch_size,
            "skip_special_token": args.skip_special_token,
        }
    
    engine_kwargs["system_prompt_file"] = args.system_prompt_file
    engine_kwargs["user_prompt_file"] = args.user_prompt_file

    # 初始化推理器
    inference = UnifiedInference(args.parser, args.engine_type, **engine_kwargs)
    
    if args.mode == 'single':
        # 单个推理模式
        try:
            print(f"Engine type: {args.engine_type}")
            
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
        try:     
            inference.batch_infer(args.input_path, args.output_dir, args.save_mode)
            
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return
        
    else:
        print("Please specify inference mode: 'single' or 'batch'")
        parser.print_help()


if __name__ == "__main__":
    main() 