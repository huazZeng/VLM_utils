#!/usr/bin/env python3
"""
Gradio前端界面 for VLM推理工具
支持单个推理和批量推理
"""

import gradio as gr
import os
import json
import tempfile
from typing import List, Dict, Any
from inference import UnifiedInference


class GradioInferenceApp:
    """
    Gradio推理应用类
    """
    
    def __init__(self):
        self.inference = None
        self.current_engine_type = None
        self.current_model_name = None
        
    def initialize_engine(self, engine_type: str, model_name: str, **kwargs):
        """
        初始化推理引擎
        
        Args:
            engine_type: 引擎类型
            model_name: 模型名称
            **kwargs: 其他参数
        """
        try:
            # 如果引擎类型或模型名称改变，重新初始化
            if (self.current_engine_type != engine_type or 
                self.current_model_name != model_name):
                
                engine_kwargs = {
                    "model_name": model_name,
                    **kwargs
                }
                
                self.inference = UnifiedInference(
                    parser=None,
                    engine_type=engine_type,
                    **engine_kwargs
                )
                
                self.current_engine_type = engine_type
                self.current_model_name = model_name
                
                return f"✅ 引擎初始化成功: {engine_type} - {model_name}"
            else:
                return f"ℹ️ 引擎已存在: {engine_type} - {model_name}"
                
        except Exception as e:
            return f"❌ 引擎初始化失败: {str(e)}"
    
    def single_inference(self, image, prompt: str, engine_type: str, model_name: str, 
                        system_prompt: str = "You are a helpful assistant.",
                        max_tokens: int = 1024, temperature: float = 0.0):
        """
        单个推理
        
        Args:
            image: 上传的图像
            prompt: 提示文本
            engine_type: 引擎类型
            model_name: 模型名称
            system_prompt: 系统提示词
            max_tokens: 最大token数
            temperature: 温度参数
            
        Returns:
            推理结果
        """
        if image is None:
            return "❌ 请上传图像"
        
        if not prompt.strip():
            return "❌ 请输入提示文本"
        
        try:
            # 初始化引擎
            init_result = self.initialize_engine(
                engine_type=engine_type,
                model_name=model_name,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            if "❌" in init_result:
                return init_result
            
            # 保存临时图像文件
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
                image.save(tmp_file.name)
                temp_image_path = tmp_file.name
            
            try:
                # 执行推理
                result = self.inference.single_infer(temp_image_path, prompt)
                
                # 清理临时文件
                os.unlink(temp_image_path)
                
                return f"✅ 推理完成\n\n**结果:**\n{result}"
                
            except Exception as e:
                # 清理临时文件
                if os.path.exists(temp_image_path):
                    os.unlink(temp_image_path)
                raise e
                
        except Exception as e:
            return f"❌ 推理失败: {str(e)}"
    
    def batch_inference(self, files, engine_type: str, model_name: str,
                       system_prompt: str = "You are a helpful assistant.",
                       user_prompt: str = "Describe the image in detail.",
                       max_tokens: int = 1024, temperature: float = 0.0,
                       batch_size: int = 16):
        """
        批量推理
        
        Args:
            files: 上传的文件列表
            engine_type: 引擎类型
            model_name: 模型名称
            system_prompt: 系统提示词
            user_prompt: 用户提示词
            max_tokens: 最大token数
            temperature: 温度参数
            batch_size: 批处理大小
            
        Returns:
            批量推理结果
        """
        if not files:
            return "❌ 请上传文件"
        
        try:
            # 初始化引擎
            init_result = self.initialize_engine(
                engine_type=engine_type,
                model_name=model_name,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                batch_size=batch_size
            )
            
            if "❌" in init_result:
                return init_result
            
            # 创建临时目录
            with tempfile.TemporaryDirectory() as temp_dir:
                # 保存上传的文件
                image_paths = []
                for file in files:
                    if file.name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                        # 图像文件
                        temp_path = os.path.join(temp_dir, os.path.basename(file.name))
                        with open(temp_path, 'wb') as f:
                            f.write(file.read())
                        image_paths.append(temp_path)
                    elif file.name.lower().endswith('.json'):
                        # JSON文件，直接使用
                        temp_path = os.path.join(temp_dir, os.path.basename(file.name))
                        with open(temp_path, 'wb') as f:
                            f.write(file.read())
                        image_paths.append(temp_path)
                
                if not image_paths:
                    return "❌ 没有找到有效的图像或JSON文件"
                
                # 执行批量推理
                output_dir = os.path.join(temp_dir, "results")
                os.makedirs(output_dir, exist_ok=True)
                
                if len(image_paths) == 1 and image_paths[0].endswith('.json'):
                    # JSON文件批量推理
                    self.inference.batch_infer(image_paths[0], output_dir, "all")
                else:
                    # 图像文件批量推理
                    self.inference.batch_infer(temp_dir, output_dir, "all")
                
                # 读取结果
                results = []
                for file in os.listdir(output_dir):
                    if file.endswith('.json'):
                        with open(os.path.join(output_dir, file), 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            if isinstance(data, list):
                                results.extend(data)
                            else:
                                results.append(data)
                
                # 格式化输出
                output = "✅ 批量推理完成\n\n"
                output += f"**处理文件数:** {len(image_paths)}\n"
                output += f"**结果数:** {len(results)}\n\n"
                
                for i, result in enumerate(results[:10]):  # 只显示前10个结果
                    output += f"**结果 {i+1}:**\n"
                    output += f"图像: {result.get('image_path', 'N/A')}\n"
                    output += f"提示: {result.get('prompt', 'N/A')}\n"
                    output += f"预测: {result.get('prediction', 'N/A')}\n"
                    if 'ground_truth' in result:
                        output += f"真值: {result.get('ground_truth', 'N/A')}\n"
                    output += "\n"
                
                if len(results) > 10:
                    output += f"... 还有 {len(results) - 10} 个结果\n"
                
                return output
                
        except Exception as e:
            return f"❌ 批量推理失败: {str(e)}"


def create_gradio_interface():
    """
    创建Gradio界面
    """
    app = GradioInferenceApp()
    
    # 引擎配置
    engine_types = ["transformer", "vllm_offline", "api_chat", "api_completion"]
    
    with gr.Blocks(title="VLM推理工具", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🤖 VLM推理工具")
        gr.Markdown("支持多种推理引擎的统一推理界面")
        
        with gr.Tab("单个推理"):
            with gr.Row():
                with gr.Column(scale=1):
                    # 引擎配置
                    gr.Markdown("### 🔧 引擎配置")
                    engine_type = gr.Dropdown(
                        choices=engine_types,
                        value="transformer",
                        label="推理引擎类型"
                    )
                    model_name = gr.Textbox(
                        value="/mnt/petrelfs/zenghuazheng/workspace/models/models/qwenvl2.5-3b",
                        label="模型名称/路径"
                    )
                    system_prompt = gr.Textbox(
                        value="You are a helpful assistant.",
                        label="系统提示词",
                        lines=2
                    )
                    
                    with gr.Row():
                        max_tokens = gr.Slider(
                            minimum=1, maximum=4096, value=1024, step=1,
                            label="最大Token数"
                        )
                        temperature = gr.Slider(
                            minimum=0.0, maximum=2.0, value=0.0, step=0.1,
                            label="温度参数"
                        )
                    
                    # 初始化按钮
                    init_btn = gr.Button("🚀 初始化引擎", variant="primary")
                    init_output = gr.Textbox(label="初始化状态", interactive=False)
                
                with gr.Column(scale=1):
                    # 推理输入
                    gr.Markdown("### 📸 推理输入")
                    image_input = gr.Image(
                        label="上传图像",
                        type="pil"
                    )
                    prompt_input = gr.Textbox(
                        value="<image>Find the spectral images in the figure",
                        label="提示文本",
                        lines=3
                    )
                    
                    # 推理按钮
                    infer_btn = gr.Button("🔍 开始推理", variant="primary", size="lg")
            
            # 推理结果
            gr.Markdown("### 📊 推理结果")
            result_output = gr.Markdown(label="推理结果")
        
        with gr.Tab("批量推理"):
            with gr.Row():
                with gr.Column(scale=1):
                    # 引擎配置
                    gr.Markdown("### 🔧 引擎配置")
                    batch_engine_type = gr.Dropdown(
                        choices=engine_types,
                        value="transformer",
                        label="推理引擎类型"
                    )
                    batch_model_name = gr.Textbox(
                        value="/mnt/petrelfs/zenghuazheng/workspace/models/models/qwenvl2.5-3b",
                        label="模型名称/路径"
                    )
                    batch_system_prompt = gr.Textbox(
                        value="You are a helpful assistant.",
                        label="系统提示词",
                        lines=2
                    )
                    batch_user_prompt = gr.Textbox(
                        value="Describe the image in detail.",
                        label="用户提示词",
                        lines=2
                    )
                    
                    with gr.Row():
                        batch_max_tokens = gr.Slider(
                            minimum=1, maximum=4096, value=1024, step=1,
                            label="最大Token数"
                        )
                        batch_temperature = gr.Slider(
                            minimum=0.0, maximum=2.0, value=0.0, step=0.1,
                            label="温度参数"
                        )
                        batch_size = gr.Slider(
                            minimum=1, maximum=64, value=16, step=1,
                            label="批处理大小"
                        )
                
                with gr.Column(scale=1):
                    # 批量输入
                    gr.Markdown("### 📁 批量输入")
                    gr.Markdown("支持上传多个图像文件或一个JSON文件")
                    files_input = gr.File(
                        label="上传文件",
                        file_count="multiple",
                        file_types=["image", "json"]
                    )
                    
                    # 批量推理按钮
                    batch_infer_btn = gr.Button("🚀 开始批量推理", variant="primary", size="lg")
            
            # 批量推理结果
            gr.Markdown("### 📊 批量推理结果")
            batch_result_output = gr.Markdown(label="批量推理结果")
        
        # 事件绑定
        init_btn.click(
            fn=app.initialize_engine,
            inputs=[engine_type, model_name, system_prompt, max_tokens, temperature],
            outputs=init_output
        )
        
        infer_btn.click(
            fn=app.single_inference,
            inputs=[image_input, prompt_input, engine_type, model_name, 
                   system_prompt, max_tokens, temperature],
            outputs=result_output
        )
        
        batch_infer_btn.click(
            fn=app.batch_inference,
            inputs=[files_input, batch_engine_type, batch_model_name,
                   batch_system_prompt, batch_user_prompt,
                   batch_max_tokens, batch_temperature, batch_size],
            outputs=batch_result_output
        )
        
        # 示例
        gr.Markdown("### 💡 使用说明")
        gr.Markdown("""
        1. **单个推理**: 上传一张图像，输入提示文本，点击推理
        2. **批量推理**: 上传多个图像文件或一个JSON文件进行批量处理
        3. **引擎配置**: 选择合适的推理引擎和模型
        4. **参数调整**: 根据需要调整温度、最大token数等参数
        """)
    
    return interface


if __name__ == "__main__":
    # 创建并启动Gradio界面
    interface = create_gradio_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=True
    ) 