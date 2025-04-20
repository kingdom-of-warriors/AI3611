import os
import torch
from torch.optim import Adam
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from vae import VAE, train_one_step, evaluate, generate_and_save_images

def main():
    # 创建图像保存目录
    if not os.path.exists('./images'):
        os.makedirs('./images')
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载数据
    train_dataset = datasets.MNIST(root='./data/', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root='./data/', train=False, transform=transforms.ToTensor(), download=False)

    batch_size = 128
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    # 训练隐层维度为1的VAE
    print("训练隐层维度为1的VAE")
    vae_z1 = VAE(latent_dim=1)
    vae_z1 = vae_z1.to(device)
    optimizer_z1 = Adam(vae_z1.parameters(), lr=1e-3)
    
    # 训练循环
    num_epochs = 200
    for epoch in tqdm(range(num_epochs)):
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            batch_loss, recon_loss, kl_loss = train_one_step(vae_z1, data, optimizer_z1, device)
            train_loss += batch_loss
        
        avg_loss = train_loss / len(train_loader.dataset)
        print(f'Epoch: {epoch+1}, Loss: {avg_loss:.4f}')
        
        # 每10个epoch评估和生成图像
        if (epoch + 1) % 10 == 0:
            val_loss, val_recon, val_kl = evaluate(vae_z1, test_loader, device)
            print(f'验证集 - 总损失: {val_loss:.4f}, 重建损失: {val_recon:.4f}, KL损失: {val_kl:.4f}')
            generate_and_save_images(vae_z1, test_loader, epoch, device, save_dir='./images/z1')
    
    # 训练隐层维度为2的VAE
    print("训练隐层维度为2的VAE")
    vae_z2 = VAE(latent_dim=2)
    vae_z2 = vae_z2.to(device)
    optimizer_z2 = Adam(vae_z2.parameters(), lr=1e-3)
    
    # 训练循环
    for epoch in tqdm(range(num_epochs)):
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            batch_loss, recon_loss, kl_loss = train_one_step(vae_z2, data, optimizer_z2, device)
            train_loss += batch_loss
        
        avg_loss = train_loss / len(train_loader.dataset)
        print(f'Epoch: {epoch+1}, Loss: {avg_loss:.4f}')
        
        # 每10个epoch评估和生成图像
        if (epoch + 1) % 10 == 0:
            val_loss, val_recon, val_kl = evaluate(vae_z2, test_loader, device)
            print(f'验证集 - 总损失: {val_loss:.4f}, 重建损失: {val_recon:.4f}, KL损失: {val_kl:.4f}')
            generate_and_save_images(vae_z2, test_loader, epoch, device, save_dir='./images/z2')

if __name__ == "__main__":
    main()