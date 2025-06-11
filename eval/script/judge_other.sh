export CUDA_VISIBLE_DEVICES=0
input_path="input.json"
output_path="output.json"
echo "Evaluating responses with input_file: $input_path and output_file: $output_path"
python ../judge_MD_other.py \
    --input_file "$input_path" \
    --judge_model_path "OpenSafetyLab/MD-Judge-v0_2-internlm2_7b" \
    --output_file "$output_path" \
    --batch_size 16
