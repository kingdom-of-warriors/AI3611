import os
import torch
from torch.optim import Adam
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import VAE, train_one_step, evaluate, generate_and_save_images, plot_latent_space
import wandb

# --- Wandb Configuration ---
WANDB_PROJECT = "vae-mnist-example" # Change to your project name
WANDB_ENTITY = None

def train_model_with_wandb(latent_dim, train_loader, test_loader, device, num_epochs=200, batch_size=128, lr=1e-3):
    """Trains a VAE model with specified latent dim and logs to wandb."""

    # --- Initialize Wandb ---
    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=f"vae_latent_dim_{latent_dim}", # Give each run a descriptive name
        config={
            "learning_rate": lr,
            "epochs": num_epochs,
            "batch_size": batch_size,
            "latent_dim": latent_dim,
            "dataset": "MNIST",
        }
    )

    # Create directories if they don't exist
    save_dir = f'./images/z{latent_dim}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Initialize Model and Optimizer
    print(f"Training VAE with latent_dim = {latent_dim}")
    model = VAE(latent_dim=latent_dim)
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=wandb.config.learning_rate) # Use wandb config

    # --- Watch Model (Optional: logs gradients and parameters) ---
    wandb.watch(model, log="gradients", log_freq=100) # Log gradients every 100 batches

    # Training Loop
    for epoch in tqdm(range(wandb.config.epochs), desc=f"Training Z={latent_dim}"):
        model.train() # Set model to training mode
        total_train_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0

        for batch_idx, (data, _) in enumerate(train_loader):
            batch_loss, batch_recon_loss, batch_kl_loss = train_one_step(model, data, optimizer, device)
            total_train_loss += batch_loss
            total_recon_loss += batch_recon_loss
            total_kl_loss += batch_kl_loss

        # Calculate average losses for the epoch (per sample)
        avg_train_loss = total_train_loss / len(train_loader.dataset)
        avg_recon_loss = total_recon_loss / len(train_loader.dataset)
        avg_kl_loss = total_kl_loss / len(train_loader.dataset)

        # --- Modified log_dict structure ---
        log_dict = {
            "train/losses": {
                "total": avg_train_loss,
                "reconstruction": avg_recon_loss,
                "kl_divergence": avg_kl_loss,
            },
             "epoch": epoch + 1
        }
        # --- End Modification ---

        # Evaluate and generate images periodically
        if (epoch + 1) % 20 == 0:
            val_loss, val_recon, val_kl = evaluate(model, test_loader, device)
            print(f'\nEpoch: {epoch+1} Val Loss: {val_loss:.4f} Recon: {val_recon:.4f} KL: {val_kl:.4f}')

            # Group validation losses under "val/losses"
            log_dict.update({
                 "val/losses": {
                    "total": val_loss,
                    "reconstruction": val_recon,
                    "kl_divergence": val_kl,
                 }
            })
            generate_and_save_images(model, test_loader, epoch, device, save_dir=save_dir)


        # Log metrics to wandb for the current epoch
        wandb.log(log_dict) # wandb automatically uses the step associated with the run

        if (epoch+1) % 5 == 0: # Print average training loss less frequently
             print(f'Epoch: {epoch+1}, Avg Train Loss: {avg_train_loss:.4f} (Recon: {avg_recon_loss:.4f}, KL: {avg_kl_loss:.4f})')
        # plot latent space every 50 epochs when latent_dim=2
        if ((epoch + 1) % 50 == 0) & (latent_dim == 2):
            plot_latent_space(model, train_loader, device, save_path=save_dir)
    
    run.finish()
    print(f"Finished training VAE with latent_dim = {latent_dim}")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading MNIST dataset...")
    train_dataset = datasets.MNIST(root='./data/', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root='./data/', train=False, transform=transforms.ToTensor(), download=True) # Set download=True just in case

    batch_size = 128
    # Consider adding drop_last=True to train_loader if dataset size is not divisible by batch_size
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    print("Dataset loaded.")

    # Train VAE with latent_dim = 1
    train_model_with_wandb(latent_dim=1,
                           train_loader=train_loader,
                           test_loader=test_loader,
                           device=device,
                           num_epochs=200, # Can be overridden by wandb config if needed
                           batch_size=batch_size,
                           lr=1e-3)

    # Train VAE with latent_dim = 2
    train_model_with_wandb(latent_dim=2,
                           train_loader=train_loader,
                           test_loader=test_loader,
                           device=device,
                           num_epochs=200,
                           batch_size=batch_size,
                           lr=1e-3)

if __name__ == "__main__":
    main()