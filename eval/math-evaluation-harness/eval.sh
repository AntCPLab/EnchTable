export CUDA_VISIBLE_DEVICES=0

model_list=(
    "meta-llama/Meta-Llama-3-8B"
)

# 循环遍历模型列表并进行实验
for model in "${model_list[@]}"; do
    bash scripts/run_eval.sh tora "$model"
done