#!/usr/bin/env python3
"""
Spectral Detection Parser
继承自BaseParser，用于解析光谱检测结果
"""

import re
import json
from typing import List, Dict, Any, Optional
from .base_parser import BaseParser


class SpectralParser(BaseParser):
    """
    光谱检测结果解析器
    继承自BaseParser，实现光谱检测结果的解析功能
    """
    
    def __init__(self, **kwargs):
        """
        初始化解析器
        
        Args:
            **kwargs: 解析器特定的配置参数
        """
        super().__init__(**kwargs)
    
    def parse_detection_none(self, text: str) -> bool:
        """
        检查是否为Detection_None
        
        Args:
            text: 模型输出文本
            
        Returns:
            是否为Detection_None
        """
        return "<|Detection_None|>" in text
    
    def parse_bbox(self, text: str) -> List[tuple]:
        """
        解析bbox坐标
        
        Args:
            text: 包含bbox信息的文本
            
        Returns:
            bbox列表 [(x1, y1, x2, y2), ...]
        """
        bboxes = []
        pattern = r'<\|box_start\|>\((\d+),(\d+)\),\((\d+),(\d+)\)<\|box_end\|>'
        matches = re.findall(pattern, text)
        
        for match in matches:
            x1, y1, x2, y2 = map(int, match)
            bboxes.append((x1, y1, x2, y2))
        
        return bboxes
    
    def parse_type(self, text: str) -> List[str]:
        """
        解析对象类型
        
        Args:
            text: 包含对象类型信息的文本
            
        Returns:
            对象类型列表
        """
        pattern = r'<\|object_ref_start\|>([^<]+)<\|object_ref_end\|>'
        matches = re.findall(pattern, text)
        return matches
    
    def parse_caption(self, text: str) -> List[str]:
        """
        解析caption
        
        Args:
            text: 包含caption信息的文本
            
        Returns:
            caption列表
        """
        pattern = r'<\|caption_start\|>([^<]+)<\|caption_end\|>'
        matches = re.findall(pattern, text)
        return matches
    
    def parse_name(self, text: str) -> List[str]:
        """
        解析name
        
        Args:
            text: 包含name信息的文本
            
        Returns:
            name列表
        """
        pattern = r'<\|name_start\|>([^<]+)<\|name_end\|>'
        matches = re.findall(pattern, text)
        return matches
    
    def parse_single_result(self, text: str) -> List[Dict[str, Any]]:
        """
        解析单个结果，将文本解析为结构化数据
        
        Args:
            text: 模型输出文本
            
        Returns:
            解析后的结果列表，每个元素包含type、bbox、caption、name等字段
        """
        # 检查是否为Detection_None
        if self.parse_detection_none(text):
            return []
        
        # 使用正则表达式分割文本，按bbox分割
        bbox_pattern = r'<\|box_start\|>\([^)]+\),\([^)]+\)<\|box_end\|>'
        bbox_matches = list(re.finditer(bbox_pattern, text))
        
        if not bbox_matches:
            return []
        
        results = []
        
        for i, bbox_match in enumerate(bbox_matches):
            # 确定当前bbox的文本范围
            start_pos = bbox_match.start()
            if i + 1 < len(bbox_matches):
                end_pos = bbox_matches[i + 1].start()
            else:
                end_pos = len(text)
            
            # 提取当前bbox对应的文本片段
            bbox_text = text[start_pos:end_pos]
            
            # 在文本片段前添加前面的内容（用于查找type）
            if i == 0:
                # 第一个bbox，包含前面的所有内容
                full_text = text[:end_pos]
            else:
                # 非第一个bbox，从上一个bbox结束到当前bbox结束
                prev_end = bbox_matches[i - 1].end()
                full_text = text[prev_end:end_pos]
            
            # 解析当前bbox的各个字段
            result = self._parse_bbox_segment(full_text, bbox_text)
            if result:
                results.append(result)
        
        return results
    
    def _parse_bbox_segment(self, full_text: str, bbox_text: str) -> Dict[str, Any]:
        """
        解析单个bbox片段
        
        Args:
            full_text: 包含当前bbox的完整文本片段
            bbox_text: 当前bbox的文本片段
            
        Returns:
            解析后的结果字典
        """
        # 解析type（在full_text中查找）
        type_pattern = r'<\|object_ref_start\|>([^<]+)<\|object_ref_end\|>'
        type_match = re.search(type_pattern, full_text)
        obj_type = type_match.group(1) if type_match else ""
        
        # 解析bbox坐标
        bbox_pattern = r'<\|box_start\|>\((\d+),(\d+)\),\((\d+),(\d+)\)<\|box_end\|>'
        bbox_match = re.search(bbox_pattern, bbox_text)
        if not bbox_match:
            return None
        
        x1, y1, x2, y2 = map(int, bbox_match.groups())
        bbox = (x1, y1, x2, y2)
        
        # 解析caption（在bbox_text中查找）
        caption_pattern = r'<\|caption_start\|>([^<]+)<\|caption_end\|>'
        caption_match = re.search(caption_pattern, bbox_text)
        caption = caption_match.group(1) if caption_match else None
        
        # 解析name（在bbox_text中查找）
        name_pattern = r'<\|name_start\|>([^<]+)<\|name_end\|>'
        name_match = re.search(name_pattern, bbox_text)
        name = name_match.group(1) if name_match else None
        
        # 构建结果
        result = {
            "type": obj_type,
            "bbox": bbox
        }
        
        if caption is not None:
            result["caption"] = caption
        
        if name is not None:
            result["name"] = name
        
        return result
    
    def parse_to_print(self, raw_result: str, **kwargs) -> str:
        """
        将原始结果解析为可打印的格式
        
        Args:
            raw_result: 原始推理结果字符串
            **kwargs: 额外的解析参数
            
        Returns:
            str: 格式化后的可打印字符串
        """
        results = self.parse_single_result(raw_result)
        
        if not results:
            return "Detection_None"
        
        output_lines = []
        for i, result in enumerate(results):
            lines = []
            lines.append(f"Detection {i+1}:")
            lines.append(f"  Type: {result['type']}")
            
            if result['bbox']:
                x1, y1, x2, y2 = result['bbox']
                lines.append(f"  Bbox: ({x1}, {y1}), ({x2}, {y2})")
            
            if 'caption' in result:
                lines.append(f"  Caption: {result['caption']}")
            
            if 'name' in result:
                lines.append(f"  Name: {result['name']}")
            
            output_lines.extend(lines)
            output_lines.append("")  # 空行分隔
        
        return "\n".join(output_lines)
    
    def parse_to_save(self, raw_result: str, **kwargs) -> Dict[str, Any]:
        """
        将原始结果解析为可保存的结构化数据
        
        Args:
            raw_result: 原始推理结果字符串
            **kwargs: 额外的解析参数
            
        Returns:
            Dict[str, Any]: 结构化的数据字典，用于保存到文件
        """
        return self.parse_to_json_format(raw_result)
    
    def parse_to_json_format(self, text: str) -> str:
        """
        解析为JSON格式输出（保持原有接口兼容性）
        
        Args:
            text: 模型输出文本
            
        Returns:
            JSON格式的字符串
        """
        results = self.parse_single_result(text)
        
        # 转换为JSON格式
        json_results = []
        for result in results:
            json_result = {
                "type": result["type"],
                "bbox": result["bbox"]
            }
            
            # 只在存在时添加caption和name字段
            if "caption" in result:
                json_result["caption"] = result["caption"]
            
            if "name" in result:
                json_result["name"] = result["name"]
            
            json_results.append(json_result)
        
        return json_results
    


def main():
    """测试解析器"""
    parser = SpectralParser()
    
    # 测试用例
    test_cases = [
        # 正常情况
        "<|object_ref_start|>nmr<|object_ref_end|> <|box_start|>(160,127),(1289,1544)<|box_end|> <|caption_start|>13C NMR spectrum of 2g\nS94<|caption_end|> <|name_start|>2g<|name_end|><|im_end|>",
        
        # 多个检测结果
        "<|object_ref_start|>nmr<|object_ref_end|> <|box_start|>(100,100),(500,500)<|box_end|> <|object_ref_start|>ms<|object_ref_end|> <|box_start|>(600,100),(1000,500)<|box_end|> <|caption_start|>MS spectrum<|caption_end|> <|name_start|>compound2<|name_end|>",
        
        # 没有caption和name
        "<|object_ref_start|>nmr<|object_ref_end|> <|box_start|>(200,200),(800,800)<|box_end|>",
        
        # Detection_None
        "<|Detection_None|>",
        
        # 空结果
        ""
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\n{'='*50}")
        print(f"Test Case {i+1}:")
        print(f"Input: {test_case}")
        print(f"\nPrint Format:")
        print(parser.parse_to_print(test_case))
        print(f"\nSave Format:")
        print(parser.parse_to_save(test_case))
        print(f"\nJSON Format:")
        print(parser.parse_to_json_format(test_case))
        print(f"\nDict Format:")


if __name__ == "__main__":
    main() 