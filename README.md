# VLM Utils - Vision Language Model Inference Framework

A comprehensive and modular framework for Vision Language Model (VLM) inference, supporting multiple inference engines and deployment scenarios.

## 🚀 Features

### Core Capabilities
- **Unified Interface**: Single entry point for all inference engines
- **Multi-Engine Support**: Support for online and offline inference backends
- **Flexible Input Formats**: Support for JSON datasets and folder-based image collections
- **Batch Processing**: Efficient batch inference with configurable batch sizes and progress tracking
- **Async Processing**: Full asynchronous support for high-performance API-based inference
- **Result Parsing**: Built-in parsers for structured output processing
- **Save Modes**: Multiple output formats (divided files or single consolidated file)
- **Progress Tracking**: Real-time progress bars and statistics for batch processing

### Supported Inference Engines

#### Offline Engines
- **Transformer**: Local inference using Hugging Face transformers library with batch processing
- **vLLM Offline**: High-performance local inference using vLLM framework

#### Online Engines  
- **API Chat**: OpenAI-compatible chat API (e.g., GPT-4, Claude)
- **API Completion**: OpenAI-compatible completion API 

### Input Types
- **JSON Files**: Structured datasets with images and prompts
- **Image Folders**: Direct folder traversal for image collections
- **Single Images**: Individual image inference with custom prompts
## 🎯 Key Benefits

### Unified Interface
- **Single Entry Point**: All inference engines accessible through one command-line interface
- **Consistent API**: Same parameters and output formats across all engines
- **Easy Switching**: Change engines without modifying code or scripts
- **Standardized Error Handling**: Consistent error reporting and recovery

### Performance Features
- **Batch Processing**: Efficient batch inference with configurable sizes
- **Progress Tracking**: Real-time progress bars and statistics
- **Memory Optimization**: Configurable batch sizes for different hardware
- **Error Recovery**: Graceful handling of failed samples

### Flexibility
- **Multiple Input Formats**: JSON datasets, image folders, single images
- **Custom Prompts**: Support for custom system and user prompts
- **Multiple Output Modes**: Single file or divided file outputs
- **Extensible Architecture**: Easy to add new engines and parsers

## Route Map

- [x] Unified inference interface
- [x] Progress tracking for batch processing
- [x] Memory-efficient batch inference
- [ ] Accelerate offline inference
- [ ] CLI reorganization
- [ ] Support multiple VLM preprocessing methods
- [ ] Development of data distillation tools
- [ ] Development of visualization tools



## 📖 Usage

### Unified Inference Interface

The framework provides a unified command-line interface for all inference engines through `inference.py`.

#### Single Image Inference

**Online API (Chat)**
```bash
python -m inference \
    --engine_type api_chat \
    --base_url "https://api.openai.com/v1" \
    --model_name "gpt-4o-mini" \
    --api_key "sk-your-api-key" \
    --max_tokens 1024 \
    --temperature 0.0 \
    single \
    --image_path "/path/to/image.jpg" \
    --prompt "Describe this image in detail"
```

**Online API (Completion)**
```bash
python -m inference \
    --engine_type api_completion \
    --base_url "https://api.openai.com/v1" \
    --model_name "gpt-4o-mini" \
    --api_key "sk-your-api-key" \
    --max_tokens 1024 \
    --temperature 0.0 \
    single \
    --image_path "/path/to/image.jpg" \
    --prompt "Describe this image in detail"
```

**Offline Transformer**
```bash
python -m inference \
    --engine_type transformer \
    --model_name "Qwen/Qwen2.5-VL-3B-Instruct" \
    --batch_size 1 \
    --skip_special_token False \
    single \
    --image_path "/path/to/image.jpg" \
    --prompt "Describe this image in detail"
```

**Offline vLLM**
```bash
python -m inference \
    --engine_type vllm_offline \
    --model_name "Qwen/Qwen2.5-VL-3B-Instruct" \
    --batch_size 16 \
    --skip_special_token False \
    single \
    --image_path "/path/to/image.jpg" \
    --prompt "Describe this image in detail"
```

#### Batch Inference

**Online API (Chat)**
```bash
python -m inference \
    --engine_type api_chat \
    --base_url "https://api.openai.com/v1" \
    --model_name "gpt-4o-mini" \
    --api_key "sk-your-api-key" \
    --concurrency 64 \
    --max_tokens 1024 \
    --temperature 0.0 \
    batch \
    --input_path "/path/to/dataset.json" \
    --output_dir "/path/to/output" \
    --save_mode "all"
```

**Offline Transformer (with Progress Tracking)**
```bash
python -m inference \
    --engine_type transformer \
    --model_name "Qwen/Qwen2.5-VL-3B-Instruct" \
    --batch_size 4 \
    --skip_special_token False \
    batch \
    --input_path "/path/to/dataset.json" \
    --output_dir "/path/to/output" \
    --save_mode "all"
```

**Offline vLLM**
```bash
python -m inference \
    --engine_type vllm_offline \
    --model_name "Qwen/Qwen2.5-VL-3B-Instruct" \
    --batch_size 16 \
    --skip_special_token False \
    batch \
    --input_path "/path/to/dataset.json" \
    --output_dir "/path/to/output" \
    --save_mode "all"
```

### Custom Prompts

You can use custom system and user prompts by providing prompt files:

```bash
python -m inference \
    --engine_type transformer \
    --model_name "Qwen/Qwen2.5-VL-3B-Instruct" \
    --system_prompt_file "/path/to/system_prompt.txt" \
    --user_prompt_file "/path/to/user_prompt.txt" \
    single \
    --image_path "/path/to/image.jpg" \
    --prompt "Custom prompt"
```

## 🔧 Configuration

### Engine Parameters

#### Transformer Engine
- `model_name`: Path to local model or Hugging Face model ID (required)
- `batch_size`: Batch size for inference (default: 1)
- `skip_special_token`: Whether to skip special tokens in processing (default: False)
- `max_tokens`: Maximum tokens to generate (default: 1024)
- `temperature`: Sampling temperature (default: 0.0)

#### vLLM Offline Engine
- `model_name`: Path to local model (default: "Qwen/Qwen2.5-VL-3B-Instruct")
- `batch_size`: Batch size for inference (default: 16)
- `skip_special_token`: Whether to skip special tokens (default: False)

#### API Engines (Chat & Completion)
- `base_url`: API endpoint URL (required)
- `model_name`: Model identifier (required)
- `api_key`: Authentication key (default: "EMPTY")
- `concurrency`: Number of concurrent requests (default: 64)
- `max_tokens`: Maximum tokens in response (default: 1024)
- `temperature`: Sampling temperature (default: 0.0)

### Global Parameters
- `parser`: Parser type for result processing (optional)
- `system_prompt_file`: Path to system prompt file (optional)
- `user_prompt_file`: Path to user prompt file (optional)

### Save Modes

- **`all`**: Save all results to a single JSON file
- **`divided`**: Save each result as a separate file

### Progress Tracking

The Transformer engine includes real-time progress tracking for batch inference:
- Progress bar showing batch processing status
- Batch completion statistics
- Overall success rate and error reporting
- Memory-efficient processing with configurable batch sizes

### Input Formats

#### JSON Format
```json
[
  {
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "<image>Describe this image"},
      {"role": "assistant", "content": "Ground truth response"}
    ],
    "images": ["/path/to/image.jpg"]
  }
]
```

#### Folder Structure
```
image_folder/
├── image1.jpg
├── image2.png
└── subfolder/
    └── image3.jpg
```






## 📄 License

MIT

## 🆘 Support

### Common Issues

**Memory Issues with Large Models**
```bash
# Reduce batch size for memory-constrained environments
python -m inference --engine_type transformer --model_name large-model --batch_size 1
```

**Slow Batch Processing**
```bash
# Increase batch size for better GPU utilization (if memory allows)
python -m inference --engine_type transformer --model_name model --batch_size 8
```

**API Rate Limiting**
```bash
# Reduce concurrency for API engines
python -m inference --engine_type api_chat --concurrency 10
```

### Getting Help

For issues and questions:
1. Check the existing documentation
2. Review error messages and logs
3. Check the progress output for batch processing issues
4. Create an issue with detailed information including:
   - Engine type and model name
   - Batch size and memory configuration
   - Error messages and stack traces
   - System specifications (GPU, memory, etc.)

### Performance Tips

- **Transformer Engine**: Start with `batch_size=1` and increase based on available memory
- **vLLM Engine**: Use larger batch sizes (8-16) for better GPU utilization
- **API Engines**: Adjust concurrency based on API rate limits and network capacity
- **Progress Monitoring**: Use the built-in progress tracking to monitor batch processing

---
**Note**: This framework is designed for research and production use cases involving Vision Language Models. Ensure you have appropriate licenses and permissions for the models and APIs you use. 