import os
import torch
from torch.optim import Adam
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Assume model.py contains VAE, train_one_step, evaluate
from model import VAE, train_one_step, evaluate, plot_latent_space

def train_model(latent_dim, train_loader, test_loader, device, num_epochs=200, batch_size=128, lr=1e-3):
    """Trains a VAE model with specified latent dim and logs to wandb."""

    # Create directories if they don't exist
    save_dir = f'./images/z{latent_dim}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Initialize Model and Optimizer
    print(f"Training VAE with latent_dim = {latent_dim}")
    model = VAE(latent_dim=latent_dim)
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr) # Use wandb config

    # Training Loop
    for epoch in tqdm(num_epochs, desc=f"Training Z={latent_dim}"):
        model.train() # Set model to training mode
        total_train_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0

        for batch_idx, (data, _) in enumerate(train_loader):
            # Assuming train_one_step returns: total_loss, recon_loss, kl_loss for the batch
            # Ensure these are detached tensors before calling .item() if they require grad
            batch_loss, batch_recon_loss, batch_kl_loss = train_one_step(model, data, optimizer, device)

            # Use .item() to get scalar value and avoid memory leaks
            total_train_loss += batch_loss
            total_recon_loss += batch_recon_loss
            total_kl_loss += batch_kl_loss

        # Calculate average losses for the epoch (per sample)
        avg_train_loss = total_train_loss / len(train_loader.dataset)
        avg_recon_loss = total_recon_loss / len(train_loader.dataset)
        avg_kl_loss = total_kl_loss / len(train_loader.dataset)

        # Evaluate and generate images periodically
        if (epoch + 1) % 20 == 0:
            # Assuming evaluate returns: avg_val_loss, avg_val_recon_loss, avg_val_kl_loss
            val_loss, val_recon, val_kl = evaluate(model, test_loader, device)
            print(f'\nEpoch: {epoch+1} Val Loss: {val_loss:.4f} Recon: {val_recon:.4f} KL: {val_kl:.4f}')

        if (epoch + 1) % 5 == 0: # Print average training loss less frequently
            print(f'Epoch: {epoch+1}, Avg Train Loss: {avg_train_loss:.4f} (Recon: {avg_recon_loss:.4f}, KL: {avg_kl_loss:.4f})')

        if (epoch + 1) % 50 == 0:
            plot_latent_space(model, train_loader, device, save_path=save_dir)
    
    print(f"Finished training VAE with latent_dim = {latent_dim}")

# The main() function remains the same as before
def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载数据
    print("Loading MNIST dataset...")
    train_dataset = datasets.MNIST(root='./data/', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root='./data/', train=False, transform=transforms.ToTensor(), download=True)

    batch_size = 128
    # Consider adding drop_last=True to train_loader if dataset size is not divisible by batch_size
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    print("Dataset loaded.")

    # Train VAE with latent_dim = 64, to minimize the valid loss
    train_model(latent_dim=64,
                train_loader=train_loader,
                test_loader=test_loader,
                device=device,
                num_epochs=200,
                batch_size=batch_size,
                lr=1e-3)

if __name__ == "__main__":
    main()