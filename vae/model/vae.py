import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np

from sklearn.manifold import TSNE

class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) model for MNIST.
    Consists of an encoder, a reparameterization step, and a decoder.
    """
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=2):
        """
        Initializes the VAE model layers.

        Args:
            input_dim (int): Dimension of the input data (e.g., 784 for flattened MNIST).
            hidden_dim (int): Dimension of the hidden layers.
            latent_dim (int): Dimension of the latent space.
        """
        super(VAE, self).__init__()

        # Encoder: Maps input to hidden representation
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )

        # Latent space parameters: Mean (mu) and log-variance (log_var)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)

        # Decoder: Maps latent representation back to input space
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # Use Sigmoid for MNIST pixel values in [0, 1]
        )

        self.latent_dim = latent_dim

    def encode(self, x):
        """
        Encodes the input data into latent space parameters (mu and log_var).

        Args:
            x (torch.Tensor): Input data tensor.

        Returns:
            tuple: (mu, log_var) tensors representing the latent distribution parameters.
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        """
        Applies the reparameterization trick to sample from the latent distribution.

        Args:
            mu (torch.Tensor): Mean of the latent distribution.
            log_var (torch.Tensor): Log-variance of the latent distribution.

        Returns:
            torch.Tensor: Sampled latent vector z.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std) # Sample from standard normal distribution
        z = mu + eps * std
        return z

    def decode(self, z):
        """
        Decodes a latent vector z back into the input space.

        Args:
            z (torch.Tensor): Latent vector.

        Returns:
            torch.Tensor: Reconstructed data.
        """
        return self.decoder(z)

    def forward(self, x):
        """
        Performs the forward pass of the VAE.

        Args:
            x (torch.Tensor): Input data tensor.

        Returns:
            tuple: (reconstruction, mu, log_var) tensors.
        """
        # Flatten input image
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        # Encode to get latent parameters
        mu, log_var = self.encode(x)
        # Sample latent vector using reparameterization trick
        z = self.reparameterize(mu, log_var)
        # Decode latent vector to reconstruct input
        reconstruction = self.decode(z)

        return reconstruction, mu, log_var

    def sample(self, num_samples, device):
        """
        Generates new images by sampling from the latent space prior (standard normal).

        Args:
            num_samples (int): Number of samples to generate.
            device (torch.device): Device to perform computation on.

        Returns:
            torch.Tensor: Generated samples.
        """
        # Sample random latent vectors from N(0, I)
        z = torch.randn(num_samples, self.latent_dim).to(device)
        # Decode samples
        samples = self.decode(z)
        return samples

    def interpolate(self, x1, x2, steps=10):
        """
        Performs linear interpolation between two points in the latent space.

        Args:
            x1 (torch.Tensor): First input image.
            x2 (torch.Tensor): Second input image.
            steps (int): Number of interpolation steps.

        Returns:
            torch.Tensor: Decoded interpolated images.
        """
        x1 = x1.view(1, -1)
        x2 = x2.view(1, -1)

        # Encode inputs to their latent means
        mu1, _ = self.encode(x1)
        mu2, _ = self.encode(x2)

        # Linearly interpolate between the means
        vectors = []
        for i in range(steps + 1):
            alpha = i / steps
            z = mu1 * (1 - alpha) + mu2 * alpha
            vectors.append(z)

        vectors = torch.cat(vectors, dim=0)
        # Decode interpolated latent vectors
        interpolations = self.decode(vectors)
        return interpolations

    def generate_from_latent(self, z):
        """
        Generates images from given latent vectors.

        Args:
            z (torch.Tensor): Latent vectors.

        Returns:
            torch.Tensor: Decoded images.
        """
        return self.decode(z)


def train_one_step(model, data, optimizer, device):
    """
    Performs a single training step (forward pass, loss calculation, backward pass).

    Args:
        model (VAE): The VAE model.
        data (torch.Tensor): Batch of input data.
        optimizer (torch.optim.Optimizer): Optimizer instance.
        device (torch.device): Device to perform computation on.

    Returns:
        tuple: (total_loss, reconstruction_loss, kl_divergence_loss) for the batch.
    """
    model.train()
    data = data.to(device)
    optimizer.zero_grad()

    reconstruction, mu, log_var = model(data)

    # Reconstruction loss (Binary Cross Entropy for Sigmoid output)
    recon_loss = F.binary_cross_entropy(reconstruction, data.view(-1, 784), reduction='sum')
    # KL divergence loss (analytical form for Gaussian)
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    # Total loss (ELBO)
    loss = recon_loss + kl_loss
    loss.backward()
    optimizer.step()

    return loss.item(), recon_loss.item(), kl_loss.item()

def evaluate(model, test_loader, device):
    """
    Evaluates the VAE model on the test set.

    Args:
        model (VAE): The VAE model.
        test_loader (DataLoader): DataLoader for the test set.
        device (torch.device): Device to perform computation on.

    Returns:
        tuple: Average (total_loss, reconstruction_loss, kl_divergence_loss) per sample.
    """
    model.eval()
    val_loss = 0
    recon_loss_total = 0
    kl_loss_total = 0

    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            reconstruction, mu, log_var = model(data)

            # Calculate losses for the batch
            recon_loss = F.binary_cross_entropy(reconstruction, data.view(-1, 784), reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            val_loss += (recon_loss + kl_loss).item()
            recon_loss_total += recon_loss.item()
            kl_loss_total += kl_loss.item()

    # Calculate average losses per sample
    val_loss /= len(test_loader.dataset)
    recon_loss_total /= len(test_loader.dataset)
    kl_loss_total /= len(test_loader.dataset)

    return val_loss, recon_loss_total, kl_loss_total

def generate_and_save_images(model, test_loader, epoch, device, save_dir='./images'):
    """
    Generates and saves reconstructed images and samples from the latent space.

    Args:
        model (VAE): The trained VAE model.
        test_loader (DataLoader): DataLoader for the test set (to get sample data).
        epoch (int): Current epoch number (for file naming).
        device (torch.device): Device to perform computation on.
        save_dir (str): Directory to save the generated images.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model.eval()
    with torch.no_grad():
        # Get a batch of test data for reconstruction
        test_data, _ = next(iter(test_loader))
        test_data = test_data.to(device)
        reconstruction, _, _ = model(test_data)

        # Save comparison of original and reconstructed images
        comparison = torch.cat([test_data.view(-1, 1, 28, 28)[:10],
                            reconstruction.view(-1, 1, 28, 28)[:10]])
        save_image(comparison.cpu(), f'{save_dir}/reconstruction_{epoch+1}.png', nrow=10)

        # Generate and save images by sampling the latent space
        if model.latent_dim == 1:
            # For 1D latent space, sample along a line
            z_values = torch.linspace(-3, 3, 10).to(device).view(10, 1)
            sample = model.generate_from_latent(z_values)
            save_image(sample.view(10, 1, 28, 28).cpu(),
                    f'{save_dir}/sample_z1dim_{epoch+1}.png', nrow=10)

        elif model.latent_dim == 2:
            # For 2D latent space, sample on a grid
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
        model (VAE): The trained VAE model.
        data_loader (DataLoader): DataLoader for the dataset (e.g., test_loader).
        device (torch.device): Device to run the model on ('cuda' or 'cpu').
        save_path (str): Path to save the generated plot.
    """
    # Plotting requires at least 2 dimensions
    if model.latent_dim < 2:
        print(f"Skipping latent space plot: Function requires latent_dim >= 2, but model has latent_dim={model.latent_dim}.")
        return

    model.eval()
    latent_vectors = []
    labels = []
    plot_title = f"Latent Space Distribution (MNIST Test Set, d={model.latent_dim})"

    print("Encoding data points to latent space...")
    # Encode all data points in the data_loader
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            batch_size = data.size(0)
            data_flat = data.view(batch_size, -1)
            mu, _ = model.encode(data_flat) # Use the mean (mu) as the latent representation
            latent_vectors.append(mu.cpu().numpy())
            labels.append(target.cpu().numpy())

    latent_vectors = np.concatenate(latent_vectors, axis=0)
    labels = np.concatenate(labels, axis=0)
    print(f"Encoded {len(labels)} data points.")

    # Reduce dimensionality using t-SNE if latent_dim > 2
    if model.latent_dim > 2:
        print(f"Latent dimension is {model.latent_dim}. Applying t-SNE to reduce to 2 dimensions...")
        # Initialize t-SNE (adjust parameters as needed)
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=300, init='pca', learning_rate='auto')
        latent_vectors_2d = tsne.fit_transform(latent_vectors)
        plot_title = f"t-SNE Visualization of Latent Space (d={model.latent_dim} -> 2D)"
        print("t-SNE finished.")
    else: # latent_dim == 2
        latent_vectors_2d = latent_vectors # Use original 2D vectors

    # Create the scatter plot using the 2D vectors
    print("Generating plot...")
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latent_vectors_2d[:, 0], latent_vectors_2d[:, 1], c=labels, cmap='tab10', s=5, alpha=0.7)

    # Add legend with digit labels
    handles, _ = scatter.legend_elements(prop='colors', alpha=1.0)
    unique_labels = np.unique(labels)
    legend_labels = [str(l) for l in unique_labels]
    # Sort legend numerically if possible
    try:
        sorted_indices = np.argsort([int(lbl) for lbl in legend_labels])
        handles = [handles[i] for i in sorted_indices]
        legend_labels = [legend_labels[i] for i in sorted_indices]
    except ValueError:
        pass # Keep original order if labels are not purely numeric
    plt.legend(handles, legend_labels, title="Digits")

    # Add labels and title based on whether t-SNE was used
    if model.latent_dim > 2:
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
    else: # latent_dim == 2
        plt.xlabel("Latent Dimension 1 (x)")
        plt.ylabel("Latent Dimension 2 (y)")
        # Optionally set fixed limits for the d=2 case if needed
        # plt.xlim(-5, 5)
        # plt.ylim(-5, 5)

    plt.title(plot_title)
    plt.grid(True)

    # Save the plot
    # Ensure the directory exists before saving
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Latent space plot saved to {save_path}")
    plt.close() # Close the figure to free memory