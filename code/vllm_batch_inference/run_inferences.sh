#!/bin/bash

export HF_HOME=/home/user/.cache
export HF_HUB_ENABLE_HF_TRANSFER=1
export VLLM_USE_V1=1
configs=(
    "--model google/gemma-3-4b-it --tensor_parallel_size 2 --max_tokens 1024 --max_model_len 2048 --sampling_params qwen"
    "--model google/gemma-3-12b-it --tensor_parallel_size 4 --max_tokens 1024 --max_model_len 2048 --sampling_params qwen"
    "--model google/gemma-3-27b-it --tensor_parallel_size 4 --max_tokens 1024 --max_model_len 2048 --sampling_params qwen"
    # add more models as needed
)
for config in "${configs[@]}"; do
    model_name=$(echo $config | awk -F' ' '{for(i=1;i<=NF;i++){if($i=="--model"){print $(i+1)}}}')

    echo "Running inference with: $config"
    python3 run_vllm.py $config
    
    echo "Finished processing for $model_name, deleting cache."

    rm -rf /home/user/.cache/hub/*
    echo "Deleted cache."
    echo "----------------------"
    sleep 30
done