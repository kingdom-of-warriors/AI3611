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
python evaluate.py --prediction_file experiments/vit/30k_embed300/vit_b128_emd300_predictions.json \
                   --reference_file data/flickr8k/caption.txt \
                   --output_file experiments/vit/30k_embed300/result.txt
```

## 扩展数据集
在网上搜查后得知，flickr还有一个30k的大数据集版本，于是我将其下载下来整合为了本项目需要的格式，并上传到了huggingface上。下载方式如下：
```bash
huggingface-cli download jiarui1/flickr flickr30k.zip --repo-type dataset --local-dir data # 在img_cap路径下运行
unzip data/flickr30k.zip # 解压文件
rm -rf data/flickr30k.zip # 删除压缩文件
```
然后修改yaml文件中的数据集路径即可。最后训练出的结果如图：