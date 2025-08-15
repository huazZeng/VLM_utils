#!/usr/bin/env python3
"""
统一的推理基类，提取transformer和vllm推理器的共同功能
"""

import json
import os
import argparse
from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod
from parser.base_parser import BaseParser

class BaseInference(ABC):
    """
    提取transformer和vllm推理器的共同功能
    """
    
    def __init__(self, parser: str):
        """初始化基类"""
        self.system_prompt = """
            You are a helpful assistant.
            """
        self.user_prompt = "Describe the image in detail."
        self.parser = BaseParser.create_parser(parser)
    def find_image_files(self, folder_path: str) -> List[str]:
        """
        遍历文件夹获取所有图片文件路径
        
        Args:
            folder_path: 文件夹路径
            
        Returns:
            图片文件路径列表
        """
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif', '.webp'}
        image_files = []
        
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        if not os.path.isdir(folder_path):
            raise ValueError(f"Path is not a directory: {folder_path}")
        
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1].lower()
                if file_ext in image_extensions:
                    image_files.append(file_path)
        
        return sorted(image_files)
    
    def load_input_data(self, input_path: str) -> Dict[str, Any]:
        """
        加载输入数据，支持JSON文件和文件夹
        
        Args:
            input_path: 输入路径（JSON文件或文件夹）
            
        Returns:
            包含数据类型的字典
        """
        if os.path.isfile(input_path) and input_path.endswith('.json'):
            # JSON文件模式
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return {
                "type": "json",
                "data": data
            }
        elif os.path.isdir(input_path):
            # 文件夹模式
            image_files = self.find_image_files(input_path)
            return {
                "type": "folder",
                "data": image_files
            }
        else:
            raise ValueError(f"Input path must be a JSON file or directory: {input_path}")
    
    @abstractmethod
    def single_infer(self, image_path: str, prompt: str) -> str:
        """
        单个推理的抽象方法
        
        Args:
            image_path: 图像路径
            prompt: 提示文本
            
        Returns:
            模型输出文本
        """
        pass
    
    @abstractmethod
    def batch_infer(self, input_path: str, output_file: str, save_mode: str = "all", **kwargs):
        """
        批量推理的抽象方法
        
        Args:
            input_path: 输入路径（JSON文件或文件夹）
            output_file: 输出文件路径
            save_mode: 存储模式，"divided"或"all"
            **kwargs: 其他参数
        """
        pass
    
    
    def _save_divided_results(self, results: List[Dict], output_dir: str, input_type: str):
        """
        保存为分离的文件模式
        
        Args:
            results: 推理结果列表
            output_dir: 输出目录路径
            input_type: 输入类型（"json"或"folder"）
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        for result in results:
            # 获取图像路径的basename
            image_path = result["image_path"]
            basename = os.path.basename(image_path).replace(".jpg", "").replace(".png", "").replace(".jpeg", "")
            
            # 解析预测结果
            prediction = result["prediction"]
            parsed_prediction = self.parser.parse_to_save(prediction)
            
            # 构建完整结果
            complete_result = {
                "prediction": parsed_prediction,
                "raw_prediction": prediction,
                "image_path": image_path,
                "prompt": result["prompt"]
            }
            
            # 如果是JSON模式，添加ground truth
            if input_type == "json":
                ground_truth = result["ground_truth"]
                parsed_ground_truth = self.parser.parse_to_save(ground_truth)
                complete_result.update({
                    "ground_truth": parsed_ground_truth,
                    "raw_ground_truth": ground_truth
                })
            
            # 保存到单独的文件
            file_path = os.path.join(output_dir, f"{basename}.json")
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(complete_result, f, indent=2, ensure_ascii=False)
    
    def _save_all_results(self, results: List[Dict], output_dir: str, input_type: str):
        """
        保存为单个文件模式
        
        Args:
            results: 推理结果列表
            output_dir: 输出目录路径
            input_type: 输入类型（"json"或"folder"）
        """
        # 创建输出目录
        output_file = os.path.join(output_dir, "all_results.json")
        # 构建结果字典
        all_results = {}
        
        for result in results:
            # 获取图像路径的basename（去掉扩展名）
            image_path = result["image_path"]
            basename = os.path.basename(image_path).replace(".jpg", "").replace(".png", "").replace(".jpeg", "")
            
            # 解析预测结果
            prediction = result["prediction"]
            parsed_prediction = self.parser.parse_to_save(prediction)
            
            # 构建完整结果
            complete_result = {
                "prediction": parsed_prediction,
                "raw_prediction": prediction,
                "image_path": image_path,
                "prompt": result["prompt"]
            }
            
            # 如果是JSON模式，添加ground truth
            if input_type == "json":
                ground_truth = result["ground_truth"]
                parsed_ground_truth = self.parser.parse_to_save(ground_truth)
                complete_result.update({
                    "ground_truth": parsed_ground_truth,
                    "raw_ground_truth": ground_truth
                })
            
            # 添加到结果字典
            all_results[basename] = complete_result
        
        # 保存到文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)


