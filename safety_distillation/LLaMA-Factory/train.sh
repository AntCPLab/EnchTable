#!/bin/bash

# 定义 CUDA 设备
export CUDA_DEVICES="6,7"
# export CUDA_LAUNCH_BLOCKING=1 
# 定义配置文件列表
CONFIG_FILES=(
    "./examples/extras/llama_pro/llama3_freeze_sft.yaml"
)

# 遍历每个配置文件并运行训练命令
for CONFIG_FILE in "${CONFIG_FILES[@]}"; do
    echo "正在运行训练命令: CUDA_VISIBLE_DEVICES=$CUDA_DEVICES llamafactory-cli train $CONFIG_FILE"
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICES llamafactory-cli train $CONFIG_FILE
    
    # 检查上一个命令是否成功执行
    if [ $? -eq 0 ]; then
        echo "训练完成: $CONFIG_FILE"
    else
        echo "训练失败: $CONFIG_FILE"
        exit 1
    fi
done

echo "所有训练任务已完成。"
