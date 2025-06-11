#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# 运行评估命令
evalplus.evaluate --model "YOUR MODEL" \
                  --dataset humaneval \
                  --backend hf \
                  --greedy
