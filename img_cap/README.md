安装 requirements.txt 中的包, 准备好环境。


跑 baseline:
```bash
sbatch run.sh
```

过程中如果提示缺少资源，则按照指示下载（例如 'punkt_tab'）:
```python
>>> import nltk
>>> nltk.download('punkt_tab')
```

注: 
evaluate.py 用于计算指标，预测结果 `prediction.json` 写成这样的形式:
```json
[
    {
        "img_id":"1386964743_9e80d96b05.jpg",
        "prediction":[
            "young boy is standing in a field of grass"
        ]
    },
    {
        "img_id":"3523559027_a65619a34b.jpg",
        "prediction":[
            "young boy is standing in a field of grass"
        ]
    },
    ......
]
```
调用方法：
```bash
python evaluate.py --prediction_file experiments/vit/8k_embed300/vit_b128_emd300_predictions.json \
                   --reference_file data/flickr8k/caption.txt \
                   --output_file experiments/vit/8k_embed300/result_30k.txt
```

## 扩展数据集
在网上搜查后得知，flickr还有一个30k的大数据集版本，于是我将其下载下来整合为了本项目需要的格式，并上传到了huggingface上。下载方式如下：
```bash
huggingface-cli download jiarui1/flickr flickr30k.zip --repo-type dataset --local-dir data # 在img_cap路径下运行
unzip data/flickr30k.zip # 解压文件
rm -rf data/flickr30k.zip # 删除压缩文件
```
然后修改yaml文件中的数据集路径即可。最后训练出的结果如图：

## 获取最佳结果
### 训练最佳模型
```bash
python main.py train_evaluate --config_file configs/vit8k_best.yaml # 训练flickr8k数据集的最佳结果
python main.py train_evaluate --config_file configs/vit30k_best.yaml # 训练flickr30k数据集的最佳结果
```
运行第二个命令需要先完成数据集的扩展。

### 获取最佳结果
如果不想再跑一遍训练，可以直接从huggingface上下载.pt文件来进行验证。
```bash
huggingface-cli download jiarui1/AI3611 vit30k.pt --local-dir ckpt/ --local-dir-use-symlinks False # 下载flickr30k的pt文件
huggingface-cli download jiarui1/AI3611 vit8k.pt --local-dir ckpt/ --local-dir-use-symlinks False # 下载flickr8k的pt文件
python eval.py evaluate --config_file configs/vit8k_best.yaml # 获取flickr8k数据集的最佳结果
python eval.py evaluate --config_file configs/vit30k_best.yaml # 获取flickr30k数据集的最佳结果
```
结果文件会存在 `results/` 目录下，分别有预测句子的 `json` 文件和结果指标的 `txt` 文件。