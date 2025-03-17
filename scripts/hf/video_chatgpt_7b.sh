#!/bin/bash

# Add vllm to PYTHONPATH
export PYTHONPATH="/home/username/Projects/DrivingVllmBench:$PYTHONPATH"

# Activate mamba environment
# mamba init && mamba activate driving_vllm_bench_minicpmv

export CUDA_VISIBLE_DEVICES=1
export HF_HOME="/home/username/.cache/huggingface"

# Run the evaluation script with parameters
/home/username/mambaforge/envs/driving_vllm_bench_videochatgpt/bin/python /home/username/Projects/DrivingVllmBench/driving_vllm_arena/benchmarking/run_evaluation.py \
    --model video_chatgpt_7b \
    --config /home/username/Projects/DrivingVllmBench/all_experiments/table1_main_performance/model_configs/hf/config_video_chatgpt_7b.py \
    --question_path "/home/username/Projects/DrivingVllmBench/all_experiments/table1_main_performance/question_bank/question_bank_1000.jsonl" \
    --batch_size 1 \
    --placement none \
    --output_dir "/home/username/Projects/DrivingVllmBench/all_experiments/table1_main_performance/ai_responses"