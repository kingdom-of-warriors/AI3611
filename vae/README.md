# VAE Project
## Main Requirement
```bash
pip install torch numpy matplotlib scikit-learn tqdm
```

## Reproduce
```bash
python main.py # dim(z)=1, images generating in folder ./images/z{dim}
python min_loss.py # dim(z)=64, to minimize the reconstructed loss in MNIST test dataset
```

## Best result sheet
```bash 
python min_loss.py
```
This code will return the lowest loss, save loss curve and plot latent space in `images/z64` like following, and the lowest loss is $100.35$:
<div style="display: flex; justify-content: center; align-items: center; gap: 40px;">
    <div style="flex: 1; text-align: center;">
        <img src="loss.jpg" alt="Loss curve" style="width: 400px; height: 200px; object-fit: cover;">
        <p>Loss curve</p>
    </div>
    <div style="flex: 1; text-align: center;">
        <img src="z64.png" alt="latent space" style="width: 400px; height: 200px; object-fit: cover;">
        <p>latent space</p>
    </div>
</div>