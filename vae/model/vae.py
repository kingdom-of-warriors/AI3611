import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np

from sklearn.manifold import TSNE

class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=2):
        super(VAE, self).__init__()
        
        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        
        # mean and std
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        
        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # 因为MNIST数据范围是[0,1]
        )
        
        self.latent_dim = latent_dim
        
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        # 将图像flatten为向量
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decode(z)
        
        return reconstruction, mu, log_var
    
    def sample(self, num_samples, device):
        """从潜在空间采样生成新图像"""
        z = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self.decode(z)
        return samples
    
    def interpolate(self, x1, x2, steps=10):
        """在潜在空间中的两点之间进行插值"""
        x1 = x1.view(1, -1)
        x2 = x2.view(1, -1)
        
        # 编码到潜在空间
        mu1, _ = self.encode(x1)
        mu2, _ = self.encode(x2)
        
        # 在潜在空间中线性插值
        vectors = []
        for i in range(steps + 1):
            alpha = i / steps
            z = mu1 * (1 - alpha) + mu2 * alpha
            vectors.append(z)
        
        vectors = torch.cat(vectors, dim=0)
        interpolations = self.decode(vectors)
        return interpolations
    
    def generate_from_latent(self, z):
        """从给定的潜在向量生成图像"""
        return self.decode(z)


def train_one_step(model, data, optimizer, device):
    """训练VAE模型一个批次"""
    model.train()
    data = data.to(device)
    optimizer.zero_grad()
    
    reconstruction, mu, log_var = model(data)
    recon_loss = F.binary_cross_entropy(reconstruction, data.view(-1, 784), reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    # 总损失
    loss = recon_loss + kl_loss
    loss.backward()
    optimizer.step()
    
    return loss.item(), recon_loss.item(), kl_loss.item()

def evaluate(model, test_loader, device):
    """评估VAE模型在测试集上的表现"""
    model.eval()
    val_loss = 0
    recon_loss_total = 0
    kl_loss_total = 0
    
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            reconstruction, mu, log_var = model(data)
            
            # 重建损失
            recon_loss = F.binary_cross_entropy(reconstruction, data.view(-1, 784), reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            val_loss += (recon_loss + kl_loss).item()
            recon_loss_total += recon_loss.item()
            kl_loss_total += kl_loss.item()
    
    val_loss /= len(test_loader.dataset)
    recon_loss_total /= len(test_loader.dataset)
    kl_loss_total /= len(test_loader.dataset)
    
    return val_loss, recon_loss_total, kl_loss_total

def generate_and_save_images(model, test_loader, epoch, device, save_dir='./images'):
    """生成并保存重建图像和从潜在空间采样的图像"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    model.eval()
    with torch.no_grad():
        # 重建测试集图像
        test_data, _ = next(iter(test_loader))
        test_data = test_data.to(device)
        reconstruction, _, _ = model(test_data)
        
        # 保存重建图像
        comparison = torch.cat([test_data.view(-1, 1, 28, 28)[:10],
                            reconstruction.view(-1, 1, 28, 28)[:10]])
        save_image(comparison.cpu(), f'{save_dir}/reconstruction_{epoch+1}.png', nrow=10)
        
        # 从正态分布采样并生成图像
        if model.latent_dim == 1:
            # 对于1维潜在空间，生成不同z值的图像
            z_values = torch.linspace(-3, 3, 10).to(device).view(10, 1)
            sample = model.generate_from_latent(z_values)
            save_image(sample.view(10, 1, 28, 28).cpu(),
                    f'{save_dir}/sample_z1dim_{epoch+1}.png', nrow=10)
            
        elif model.latent_dim == 2:
            # 对于2维潜在空间，创建网格并生成
            x = torch.linspace(-5, 5, 10).to(device)
            y = torch.linspace(-5, 5, 10).to(device)
            
            grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
            z = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
            
            samples = model.generate_from_latent(z)
            save_image(samples.view(100, 1, 28, 28).cpu(),
                    f'{save_dir}/sample_z2dim_{epoch+1}.png', nrow=10)
            

def plot_latent_space(model, data_loader, device, save_path='./images/latent_space.png'):
    """
    Visualizes the latent space distribution. If latent_dim > 2, uses t-SNE
    to reduce dimensionality to 2D for plotting.

    Args:
        model: The trained VAE model.
        data_loader: DataLoader for the dataset (e.g., test_loader).
        device: The device to run the model on ('cuda' or 'cpu').
        save_path: Path to save the generated plot.
    """
    if model.latent_dim < 2:
        print(f"Skipping latent space plot: Function requires latent_dim >= 2, but model has latent_dim={model.latent_dim}.")
        return

    model.eval()
    latent_vectors = []
    labels = []
    plot_title = f"Latent Space Distribution (MNIST Test Set, d={model.latent_dim})"

    print("Encoding data points to latent space...")
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            batch_size = data.size(0)
            data_flat = data.view(batch_size, -1)
            mu, _ = model.encode(data_flat)
            latent_vectors.append(mu.cpu().numpy())
            labels.append(target.cpu().numpy())

    latent_vectors = np.concatenate(latent_vectors, axis=0)
    labels = np.concatenate(labels, axis=0)
    print(f"Encoded {len(labels)} data points.")

    # --- Dimensionality Reduction if latent_dim > 2 ---
    if model.latent_dim > 2:
        print(f"Latent dimension is {model.latent_dim}. Applying t-SNE to reduce to 2 dimensions...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=300, init='pca', learning_rate='auto') # Added init and lr for newer sklearn versions
        latent_vectors_2d = tsne.fit_transform(latent_vectors)
        plot_title = f"t-SNE Visualization of Latent Space (d={model.latent_dim} -> 2D)"
        print("t-SNE finished.")
    else: # latent_dim == 2
        latent_vectors_2d = latent_vectors # Use original 2D vectors
    # --- ---

    # Create the plot using the 2D vectors
    print("Generating plot...")
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latent_vectors_2d[:, 0], latent_vectors_2d[:, 1], c=labels, cmap='tab10', s=5, alpha=0.7)

    # Add legend
    handles, _ = scatter.legend_elements(prop='colors', alpha=1.0)
    unique_labels = np.unique(labels)
    legend_labels = [str(l) for l in unique_labels]
    try:
        sorted_indices = np.argsort([int(lbl) for lbl in legend_labels])
        handles = [handles[i] for i in sorted_indices]
        legend_labels = [legend_labels[i] for i in sorted_indices]
    except ValueError:
        pass
    plt.legend(handles, legend_labels, title="Digits")

    # Add labels and title
    if model.latent_dim > 2:
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
    else:
        plt.xlabel("Latent Dimension 1 (x)")
        plt.ylabel("Latent Dimension 2 (y)")
        # Optionally set fixed limits for the d=2 case if needed

    plt.title(plot_title)
    plt.grid(True)

    # Save the plot
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Latent space plot saved to {save_path}")
    plt.close()