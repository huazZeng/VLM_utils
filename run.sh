#!/bin/bash

MODEL_NAME="/mnt/dhwfile/MinerU4S/zenghuazheng/models/SpectralDetection-nospecialtoken/v1-20250812-125322/checkpoint-245"
python -m offline_inference \
    --engine_type "vllm_offline" \
    --model_name $MODEL_NAME \
    --skip_special_token True \
    --max_tokens 1024 \
    --temperature 0.0 \
    batch \
    --input_path "/mnt/petrelfs/zenghuazheng/workspace/data/spectral/test_data.json" \
    --output_file "/mnt/petrelfs/zenghuazheng/workspace/SpectralDetection/dataset/eval/vllm_all/" \
    --save_mode "all" \
    --batch_size 16

# srun -p mineru4s --gres=gpu:1 \
#  apptainer exec  --cleanenv   --nv   --bind  /mnt:/mnt  \
#  /mnt/dhwfile/MinerU4S/zenghuazheng/apptrainer/ms-swift-3.7.sif \
#  /mnt/petrelfs/zenghuazheng/workspace/vlm_utils/run.sh