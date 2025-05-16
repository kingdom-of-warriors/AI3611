#!/bin/bash
# 运行第二个训练和评估任务
python main.py train_evaluate --config_file configs/resnet8k_256.yaml

# 运行第三个训练和评估任务
python main.py train_evaluate --config_file configs/resnet8k_512.yaml

echo "所有任务已完成。"