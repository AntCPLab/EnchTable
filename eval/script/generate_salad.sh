#!/bin/bash
models=(
       )

for model in "${models[@]}"; do
    echo "Running with model: $model and dataset: $dataset"
    CUDA_VISIBLE_DEVICES=0 python ../generate_response_salad.py --model_id "$model" --generation_method "greedy" --input_file "sampled_salad_dataset.json" --save_path "./test_salad/generation"
done
