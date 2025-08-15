# 统一推理器使用说明

## 概述

[UnifiedInference](file:///mnt/petrelfs/zenghuazheng/workspace/vlm_utils/unified_inference.py#L15-L278) 是一个统一的推理引擎，支持多种推理后端，包括 `transformer`、`vllm_offline` 和 `vllm_api`。它支持单张图片推理和批量推理两种模式，适用于视觉语言模型的推理任务，目前仅支持单图推理。

## 命令行接口

该工具使用命令行接口，基本使用格式如下：

```bash
python unified_inference.py --engine_type [引擎类型] [模式相关参数]
```

### 通用参数

- `--engine_type`：指定推理引擎类型，可选值包括 `transformer`、`vllm_offline`、`vllm_api`
- `--parser`：指定解析器类型，用于处理模型输出，无则直接返回模型输出,详见[paser docment](./parser/Parser.md)
- `--model_name`：模型名称或路径,vllm_offline、transformer为模型名称或路径，vllm_api为vllm部署的服务model_name
- `--system_prompt`：系统提示词文件路径
- `--user_prompt`：用户提示词文件路径
- `--skip_special_token`：是否跳过特殊标记，默认为 False，仅对vllm_offline,transformer有效,vllm_api目前仍为chat模式，不支持输出特殊字符

### 不同引擎的特定参数

#### Transformer 引擎
- `--model_name`（必需）：模型名称或路径
- `--skip_special_token`

#### VLLM 离线引擎 (vllm_offline)
- `--model_name`：模型名称或路径，默认为 "Qwen/Qwen2.5-VL-3B-Instruct"
- `--batch_size`：批处理大小，默认为 16
- `--skip_special_token` : 默认为 False
#### VLLM API 引擎 (vllm_api)
- `--base_url`（必需）：VLLM 服务器基础 URL,也可推理openai接口的模型，需要apikey
- `--model_name`（必需）：模型名称
- `--api_key`：API 密钥，默认为 "EMPTY"
- `--concurrency`：并发级别，默认为 64
- `--max_tokens`：最大生成标记数，默认为 1024
- `--temperature`：温度参数，默认为 0.0

## 推理模式

### 单张图片推理模式

使用 `single` 子命令进行单张图片推理：

```bash
python unified_inference.py --engine_type [引擎类型] single \
    --image_path [图片路径] \
    --prompt [提示词]
```

参数说明：
- `--image_path`（必需）：图片文件路径
- `--prompt`：提示词文本，默认为 "<image>Find the spectral images in the figure"

示例：
```bash
python unified_inference.py --engine_type transformer \
    --model_name Qwen/Qwen2.5-VL-3B-Instruct \
    single \
    --image_path ./example.jpg \
    --prompt "请描述这张图片的内容"
```

### 批量推理模式

使用 `batch` 子命令进行批量推理：

```bash
python unified_inference.py --engine_type [引擎类型] batch \
    --input_path [输入路径] \
    --output_file [输出文件路径]
```

参数说明：
- `--input_path`（必需）：输入路径，可以是 JSON 文件或包含图片的文件夹
- `--output_file`（必需）：输出文件路径
- `--save_mode`：保存模式，可选 "divided"（分文件保存）或 "all"（保存到单个文件），默认为 "all"
- `--batch_size`：批处理大小（仅适用于 vllm_offline），默认为 16
- `--concurrency`：并发级别（仅适用于 vllm_api），默认为 64

示例：
```bash
python unified_inference.py --engine_type vllm_offline \
    --model_name Qwen/Qwen2.5-VL-3B-Instruct \
    batch \
    --input_path ./images_folder \
    --output_file ./results \
    --save_mode all
```

## 输入数据格式

### JSON 文件格式

当使用 JSON 文件作为输入时，应写为以下格式，默认为eval模式，有着assitant的回答作为gt
```json
{
    "messages": [
      {
        "role": "system",
        "content": ""
      },
      {
        "role": "user",
        "content": "<image>xxxxx"
      },
      {
        "role": "assistant",
        "content": "xxxxx"
      }
    ],
    "images": [
      "jpg/10.1002-anie.202412299_37.jpg"
    ]
  },
```


### 文件夹模式

当输入路径是文件夹时，系统会自动遍历文件夹中的所有图片文件（支持 .jpg, .jpeg, .png, .bmp, .tiff, .tif, .gif, .webp 等格式）。

## 输出格式

### 统一输出文件格式

当使用 `--save_mode all` 时，所有结果将保存在一个名为 `all_results.json` 的文件中，格式如下：

```json
{
  "image_basename": {
    "prediction": "解析后的预测结果",
    "raw_prediction": "原始预测结果",
    "image_path": "图片路径",
    "prompt": "使用的提示词",
    "ground_truth": "真实答案（如果输入是JSON格式）",
    "raw_ground_truth": "原始真实答案（如果输入是JSON格式）"
  }
}
```

### 分文件保存格式

当使用 `--save_mode divided` 时，每个图片的结果将保存在单独的 JSON 文件中，文件名为图片的基本名称。

## 错误处理

- 如果批量推理失败，系统会自动回退到单张图片推理模式
- 推理过程中出现的错误会被记录并显示在控制台中
- 对于推理失败的样本，结果中的 prediction 字段将为空字符串

## 使用建议

1. 对于大量图片的推理任务，推荐使用 `vllm_offline` 或 `vllm_api` 引擎以获得更好的性能
2. 根据系统资源调整 `--batch_size` 或 `--concurrency` 参数
3. 合理使用 `--save_mode` 参数，如果需要对每个样本进行单独分析，使用 `divided` 模式；如果需要整体分析，使用 `all` 模式