#!/bin/bash

# Add vllm to PYTHONPATH
export PYTHONPATH="/home/username/Projects/DrivingVllmBench:$PYTHONPATH"

# Activate mamba environment
# mamba init && mamba activate driving_vllm_bench_minicpmv

export CUDA_VISIBLE_DEVICES=0,1
export HF_HOME="/stat-nx1/stat/home/grad/tongzeng/.cache/huggingface"

# Run the evaluation script with parameters
/home/username/mambaforge/envs/driving_vllm_bench_vllm_latest/bin/python /home/username/Projects/DrivingVllmBench/driving_vllm_arena/benchmarking/run_evaluation.py \
    --model qwen2_vl_72b \
    --config /home/username/Projects/DrivingVllmBench/all_experiments/table1_main_performance/model_configs/vllm/config_qwen2_vl_72b.py \
    --question_path "/home/username/Projects/DrivingVllmBench/all_experiments/table1_main_performance/question_bank/question_bank_1000.jsonl" \
    --batch_size 1 \
    --placement all_and_none \
    --output_dir "/home/username/Projects/DrivingVllmBench/all_experiments/table1_main_performance/ai_responses"