export HF_ALLOW_CODE_EVAL=0

device='cuda'
task=medmcqa
batch_size=8 

model_list=()
for model in "${model_list[@]}"; do
    lm_eval --model hf \
        --model_args pretrained="$model" \
        --tasks $task \
        --device $device \
        --batch_size $batch_size \
        --confirm_run_unsafe_code \
        --output_path "./test_leh" \
        --trust_remote_code
done
