#!/bin/bash

# 基础路径
BASE_DIR="/root/autodl-tmp/AI3611"
REFERENCE_FILE="${BASE_DIR}/img_cap/data/flickr8k/caption.txt"

# 查找所有 resnet 和 vit 文件夹下的 json 文件并计算结果
find "${BASE_DIR}/img_cap/experiments/resnet" "${BASE_DIR}/img_cap/experiments/vit" -type d | while read dir; do
    # 查找该目录下的 json 文件
    json_file=$(find "$dir" -maxdepth 1 -name "*.json" | head -n 1)
    
    # 如果找到 json 文件
    if [ -n "$json_file" ]; then
        # 设置输出文件路径
        output_file="${dir}/result.txt"
        
        echo "处理文件: $json_file"
        echo "输出结果到: $output_file"
        
        # 执行评估命令
        python ${BASE_DIR}/img_cap/evaluate.py \
            --prediction_file "$json_file" \
            --reference_file "$REFERENCE_FILE" \
            --output_file "$output_file"
    fi
done

echo "所有评估完成！"