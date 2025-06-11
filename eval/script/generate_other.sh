#!/bin/bash
models=()

datasets=()
batch_size=1
for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        echo "Running with model: $model and dataset: $dataset"
        CUDA_VISIBLE_DEVICES=0 python ../generate_responses_result_batch.py --model "$model" --dataset "$dataset" --save_path "./generate_results/llama3/generation_${batch_size}" --batch_size $batch_size
    done
done
