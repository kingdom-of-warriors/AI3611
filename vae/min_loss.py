import os
import torch
from torch.optim import Adam
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt # <--- 导入 matplotlib

# Assume model.py contains VAE, train_one_step, evaluate
from model import VAE, train_one_step, evaluate, plot_latent_space

def train_model(latent_dim, train_loader, test_loader, device, num_epochs=200, batch_size=128, lr=1e-3):
    """Trains a VAE model with specified latent dim, logs to wandb, and plots losses.""" # <--- 更新文档字符串

    # Create directories if they don't exist
    save_dir = f'./images/z{latent_dim}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Initialize Model and Optimizer
    print(f"Training VAE with latent_dim = {latent_dim}")
    model = VAE(latent_dim=latent_dim)
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr)

    # --- 新增：用于存储每个 epoch 损失的列表 ---
    history = {
        'train_loss': [],
        'recon_loss': [],
        'kl_loss': []
    }
    # --- 结束新增 ---

    # Training Loop
    for epoch in tqdm(range(num_epochs), desc=f"Training Z={latent_dim}"):
        model.train() # Set model to training mode
        total_train_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0

        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device) # <--- 确保数据移动到设备
            # Assuming train_one_step returns: total_loss, recon_loss, kl_loss for the batch
            # Ensure these are scalar values (e.g., already .item()'d in train_one_step or here)
            batch_loss, batch_recon_loss, batch_kl_loss = train_one_step(model, data, optimizer, device)

            # Accumulate *scalar* batch losses
            total_train_loss += batch_loss # Assume these are already scalars
            total_recon_loss += batch_recon_loss
            total_kl_loss += batch_kl_loss

        # Calculate average losses for the epoch (per sample)
        avg_train_loss = total_train_loss / len(train_loader.dataset)
        avg_recon_loss = total_recon_loss / len(train_loader.dataset)
        avg_kl_loss = total_kl_loss / len(train_loader.dataset)

        # --- 新增：记录当前 epoch 的平均损失 ---
        history['train_loss'].append(avg_train_loss)
        history['recon_loss'].append(avg_recon_loss)
        history['kl_loss'].append(avg_kl_loss)
        # --- 结束新增 ---

        # Evaluate and generate images periodically
        if (epoch + 1) % 20 == 0:
            val_loss, val_recon, val_kl = evaluate(model, test_loader, device)
            print(f'\nEpoch: {epoch+1} Val Loss: {val_loss:.4f} Recon: {val_recon:.4f} KL: {val_kl:.4f}')

        # Print average training loss (maybe less frequently or combined with eval)
        # Keep your original printing logic or adjust as needed
        if (epoch + 1) % 5 == 0:
             print(f'Epoch: {epoch+1}, Avg Train Loss: {avg_train_loss:.4f} (Recon: {avg_recon_loss:.4f}, KL: {avg_kl_loss:.4f})')

    print("Generating loss plot...")
    # 创建 x 轴的 epoch 列表 (从 1 到 num_epochs)
    epochs_range = range(1, num_epochs + 1)

    plt.figure(figsize=(12, 6)) # 创建图形

    # 绘制三条损失曲线
    plt.plot(epochs_range, history['train_loss'], label='Total Training Loss', color='blue')
    plt.plot(epochs_range, history['recon_loss'], label='Reconstruction Loss', color='green', linestyle='--')
    plt.plot(epochs_range, history['kl_loss'], label='KL Divergence Loss', color='red', linestyle=':')

    # 添加标题和标签
    plt.title(f'VAE Training Losses (Latent Dim = {latent_dim})')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss per Sample')
    plt.legend() # 显示图例
    plt.grid(True) # 显示网格

    # 保存图形到文件
    plot_save_path = os.path.join(save_dir, f'training_losses_z{latent_dim}.png')
    try:
        plt.savefig(plot_save_path)
        print(f"Loss plot saved to {plot_save_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    # plt.show() # 如果你想在运行时显示图形（需要图形界面环境），取消这行注释
    plt.close() # 关闭图形，释放内存

    # plot_latent_space(model, train_loader, device, save_path=save_dir)
    print(f"Finished training VAE with latent_dim = {latent_dim}")

# The main() function remains the same as before
def main():
    # 设置设备
    # 确保 NPU 可用，否则可能需要回退到 CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        print("Cuda not available, using CPU.")
        device = torch.device('cpu')
    print(f"Using device: {device}")


    # 加载数据
    print("Loading MNIST dataset...")
    # For NPU/GPU, pin_memory=True can speed up data transfer
    # num_workers can be adjusted based on your system's cores/IO
    # If running on CPU, set num_workers=0 potentially
    pin_memory_flag = True if device.type != 'cpu' else False
    num_workers_flag = 4 # Adjust based on system

    train_dataset = datasets.MNIST(root='./data/', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root='./data/', train=False, transform=transforms.ToTensor(), download=True)

    batch_size = 128
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers_flag, pin_memory=pin_memory_flag, drop_last=True) # Added drop_last
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers_flag, pin_memory=pin_memory_flag)
    print("Dataset loaded.")

    # Train VAE with latent_dim = 64, to minimize the valid loss
    train_model(latent_dim=64,
                train_loader=train_loader,
                test_loader=test_loader,
                device=device,
                num_epochs=200, # Keep original num_epochs
                batch_size=batch_size, # batch_size is used by loader, not needed here
                lr=1e-3)

if __name__ == "__main__":
    main()
