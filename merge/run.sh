export CUDA_VISIBLE_DEVICES=0

task_model=(
    'ajibawa-2023/Code-Llama-3-8B'
    # 'TIGER-Lab/MAmmoTH2-8B-Plus'
    # 'ContactDoctor/Bio-Medical-Llama-3-8B'
         )
task_model_pre=(
    'meta-llama/Meta-Llama-3-8B-Instruct'

                )
safety_model=(
    'meta-llama/Meta-Llama-3-8B-Instruct'
              )
# harmful_model
safety_model_pre=(
    './safety_distillation/LLaMA-Factory/saves/llama3-8b-beavertail_harmful/attention/sft_ntk_linear_e4'
)
save_path=(
    './merged_models/Code-Llama-3-8B_aligned'
)
methods=(
    'enchtable'
)

length=${#task_model[@]}
for ((i=0; i<length; i++))
do
    echo "Running experiment $((i+1))/$length"
    echo "Parameters: task_model=${task_model[$i]}, task_model_pre=${task_model_pre[$i]}, safety_model=${safety_model[$i]}, safety_model_pre=${safety_model_pre[$i]}, save_path=${save_path[$i]}"
    for method in "${methods[@]}"; do
        python merge.py --task_model ${task_model[$i]} \
            --task_model_pre ${task_model_pre[$i]} \
            --safety_model ${safety_model[$i]} \
            --safety_model_pre ${safety_model_pre[$i]} \
            --save_path ${save_path[$i]}'_'$method \
            --method $method \
            --ada \
            --ada_alpha 0.1
    done
    echo "-----------------------------"
done

