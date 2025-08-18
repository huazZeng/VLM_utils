# VLM Utils - Vision Language Model Inference Framework

A comprehensive and modular framework for Vision Language Model (VLM) inference, supporting multiple inference engines and deployment scenarios.

## 🚀 Features

### Core Capabilities
- **Multi-Engine Support**: Unified interface for different inference backends
- **Flexible Input Formats**: Support for JSON datasets and folder-based image collections
- **Batch Processing**: Efficient batch inference with configurable batch sizes and concurrency
- **Async Processing**: Full asynchronous support for high-performance API-based inference
- **Result Parsing**: Built-in parsers for structured output processing
- **Save Modes**: Multiple output formats (divided files or single consolidated file)

### Supported Inference Engines

#### Offline Engines
- **Transformer**: Local inference using Hugging Face transformers library
- **vLLM Offline**: High-performance local inference using vLLM framework

#### Online Engines  
- **API Chat**: OpenAI-compatible chat API  (e.g., GPT-4, Claude)
- **API Completion**: OpenAI-compatible completion API 

### Input Types
- **JSON Files**: Structured datasets with images and prompts
- **Image Folders**: Direct folder traversal for image collections
- **Single Images**: Individual image inference with custom prompts
## Route Map

- [ ] Accelerate offline inference
- [ ] CLI reorg
- [x] unify the two inference
- [ ] Support multiple VLM preprocessing methods
- [ ] Development of data distillation tools
- [ ] Development of visualization tools



## 📖 Usage

### 1. Offline Inference
#### Single Inferencer
* online 
```bash
INPUT_PATH="path contains images"
OUTPUT_PATH=""
BASE_URL=""
MODEL_NAME="gpt-4o-mini"
API_KEY="sk-"
PARSER="default"
SAVE_MODE="all"
CONCURRENCY=2
MAX_TOKENS=1024
TEMPERATURE=0.0

python -m inference \
    --engine_type api_chat \
    --base_url "$BASE_URL" \
    --model_name "$MODEL_NAME" \
    --api_key "$API_KEY" \
    --parser "$PARSER" \
    --concurrency "$CONCURRENCY" \
    --max_tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    single \
    --image_path "" \
    --prompt "" \
    --save_path ""
```

* offline
```
MODEL_NAME=""
python -m inference \
    --engine_type --engine_type "transformer" or "vllm_offline" \
    --model_name $MODEL_NAME \
    single \
    --image_path "" \
    --prompt "" \
    --save_path ""

```

#### Batch Inferencer
* online
```bash
INPUT_PATH="path contains images"
OUTPUT_PATH=""
BASE_URL=""
MODEL_NAME="gpt-4o-mini"
API_KEY="sk-"
PARSER="default"
SAVE_MODE="all"
CONCURRENCY=2
MAX_TOKENS=1024
TEMPERATURE=0.0

python -m inference \
    --engine_type api_chat \
    --base_url "$BASE_URL" \
    --model_name "$MODEL_NAME" \
    --api_key "$API_KEY" \
    --parser "$PARSER" \
    --concurrency "$CONCURRENCY" \
    --max_tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    batch \
    --input_path "$INPUT_PATH" \
    --output_dir "$OUTPUT_PATH" \
    --save_mode "$SAVE_MODE"
```
* offline
```
MODEL_NAME=""
python -m inference \
    --engine_type "transformer" or "vllm_offline" \
    --model_name $MODEL_NAME \
    batch \
    --input_path "/mnt/petrelfs/zenghuazheng/workspace/data/spectral/test_data.json" \
    --output_dir "/mnt/petrelfs/zenghuazheng/workspace/vlm_utils/dataset/eval/vllm_all/" \
    --save_mode "all" \
```

## 🔧 Configuration

### Engine Parameters

#### Transformer Engine
- `model_name`: Path to local model or Hugging Face model ID
- `skip_special_token`: Whether to skip special tokens in processing

#### vLLM Offline Engine
- `model_name`: Path to local model
- `batch_size`: Batch size for inference
- `skip_special_token`: Whether to skip special tokens

#### API Engines
- `base_url`: API endpoint URL
- `model_name`: Model identifier
- `api_key`: Authentication key
- `concurrency`: Number of concurrent requests
- `max_tokens`: Maximum tokens in response
- `temperature`: Sampling temperature

### Save Modes

- **`all`**: Save all results to a single JSON file
- **`divided`**: Save each result as a separate file

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

## 🔍 Advanced Features

### Custom Parsers
Implement custom parsers by extending `BaseParser`:

```python
from parser.base_parser import BaseParser

class CustomParser(BaseParser):
    def parse_to_dict(self, text: str) -> Dict[str, Any]:
        # Custom parsing logic
        return parsed_result
```

### Engine Factory
Create custom engines using the factory pattern:

```python
from engine.engine_factory import InferenceEngineFactory

engine = InferenceEngineFactory.create_engine("transformer", model_name="your-model")
```

### Async Processing
For high-performance scenarios, use async methods:

```python
import asyncio
from online_infer import OnlineInference

inference = OnlineInference("default", "api_chat", **kwargs)
result = await inference.single_infer(image_path, prompt)
```




## 📄 License

MIT

## 🆘 Support

For issues and questions:
1. Check the existing documentation
2. Review error messages and logs
3. Create an issue with detailed information
4. Include system information and error traces

---
**Note**: This framework is designed for research and production use cases involving Vision Language Models. Ensure you have appropriate licenses and permissions for the models and APIs you use. 