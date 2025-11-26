"""
GAN training loop with moment matching and early stopping.

This module implements the training procedure for the Generator and
Discriminator, including loss computation, gradient clipping, and
convergence monitoring.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
import os

from .models import Generator, Discriminator, create_gan
from .config import TrainingConfig


def set_seed(seed: int = 307):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_loss_verge(g_loss: float, d_loss: float) -> float:
    """
    Calculate loss verge for early stopping.

    The verge is the Euclidean distance from origin in loss space.
    When this stabilizes, training has converged.
    """
    return np.sqrt(g_loss ** 2 + d_loss ** 2)


def compute_moment_loss(real: torch.Tensor, fake: torch.Tensor,
                        order: int = 3) -> torch.Tensor:
    """
    Compute moment matching loss.

    Matches mean, variance, and optionally skewness between
    real and generated distributions.

    Args:
        real: Real data tensor
        fake: Generated data tensor
        order: Maximum moment order (1=mean, 2=var, 3=skew)

    Returns:
        Moment matching loss
    """
    loss = 0.0

    # First moment (mean)
    loss += torch.mean((real.mean(dim=1) - fake.mean(dim=1)) ** 2)

    if order >= 2:
        # Second moment (variance)
        loss += torch.mean((real.var(dim=1) - fake.var(dim=1)) ** 2)

    if order >= 3:
        # Third moment (skewness proxy)
        real_centered = real - real.mean(dim=1, keepdim=True)
        fake_centered = fake - fake.mean(dim=1, keepdim=True)
        loss += torch.mean((real_centered.pow(3).mean(dim=1) -
                           fake_centered.pow(3).mean(dim=1)) ** 2)

    return loss


def compute_derivative_loss(real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    """
    Compute derivative matching loss.

    Matches the first differences (temporal dynamics) between
    real and generated sequences.
    """
    real_diff = real[:, 1:, :] - real[:, :-1, :]
    fake_diff = fake[:, 1:, :] - fake[:, :-1, :]

    return torch.mean((real_diff.mean(dim=1) - fake_diff.mean(dim=1)) ** 2)


class GANTrainer:
    """
    Trainer class for GAN models.

    Handles the training loop, loss computation, gradient clipping,
    and early stopping.

    Example:
        >>> trainer = GANTrainer()
        >>> results = trainer.train(train_loader, val_loader, epochs=200)
        >>> trainer.save('models/0050/')
    """

    def __init__(self, config: Optional[TrainingConfig] = None):
        """
        Initialize trainer.

        Args:
            config: Training configuration (uses defaults if None)
        """
        self.config = config or TrainingConfig()

        # Set device
        if self.config.device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Initialize models
        self.generator, self.discriminator = create_gan()
        self.generator.to(self.device)
        self.discriminator.to(self.device)

        # Loss function
        self.criterion = nn.MSELoss()

        # Optimizers
        self.optimizer_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.config.lr_generator,
            betas=(self.config.beta1, self.config.beta2)
        )
        self.optimizer_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.config.lr_discriminator,
            betas=(self.config.beta1, self.config.beta2)
        )

        # Training history
        self.train_g_losses: List[float] = []
        self.train_d_losses: List[float] = []
        self.val_g_losses: List[float] = []
        self.val_d_losses: List[float] = []

    def train_step(self, real_data: torch.Tensor) -> Tuple[float, float]:
        """
        Single training step.

        Args:
            real_data: Batch of real order book sequences

        Returns:
            Tuple of (generator_loss, discriminator_loss)
        """
        batch_size = real_data.size(0)
        seq_len = real_data.size(1)
        n_features = real_data.size(2)

        # Labels
        real_labels = torch.ones(batch_size, 1).to(self.device)
        fake_labels = torch.zeros(batch_size, 1).to(self.device)

        # =====================
        # Train Discriminator
        # =====================
        self.discriminator.zero_grad()

        # Real data
        real_output = self.discriminator(real_data)
        d_loss_real = self.criterion(real_output, real_labels)

        # Fake data
        noise = torch.randn(batch_size, seq_len, n_features).to(self.device)
        fake_data = self.generator(noise)
        fake_output = self.discriminator(fake_data.detach())
        d_loss_fake = self.criterion(fake_output, fake_labels)

        # Combined discriminator loss
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.discriminator.parameters(),
            self.config.clip_grad_discriminator
        )

        self.optimizer_d.step()

        # =====================
        # Train Generator
        # =====================
        self.generator.zero_grad()

        # Generate new fake data
        noise = torch.randn(batch_size, seq_len, n_features).to(self.device)
        fake_data = self.generator(noise)

        # Generator wants discriminator to output 1 (real)
        fake_output = self.discriminator(fake_data)
        g_loss_adv = self.criterion(fake_output, real_labels)

        # Moment matching loss
        g_loss_moment = compute_moment_loss(real_data, fake_data, self.config.moment_order)

        # Derivative matching loss
        g_loss_deriv = compute_derivative_loss(real_data, fake_data)

        # Combined generator loss
        g_loss = g_loss_adv + 0.1 * g_loss_moment + 0.05 * g_loss_deriv
        g_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.generator.parameters(),
            self.config.clip_grad_generator
        )

        self.optimizer_g.step()

        return g_loss.item(), d_loss.item()

    def evaluate(self, data_loader: DataLoader) -> Tuple[float, float]:
        """
        Evaluate on validation data.

        Args:
            data_loader: Validation data loader

        Returns:
            Tuple of (avg_g_loss, avg_d_loss)
        """
        self.generator.eval()
        self.discriminator.eval()

        g_losses = []
        d_losses = []

        with torch.no_grad():
            for data in data_loader:
                data = data.to(self.device)
                batch_size = data.size(0)
                seq_len = data.size(1)
                n_features = data.size(2)

                # Labels
                real_labels = torch.ones(batch_size, 1).to(self.device)
                fake_labels = torch.zeros(batch_size, 1).to(self.device)

                # Discriminator evaluation
                real_output = self.discriminator(data)
                d_loss_real = self.criterion(real_output, real_labels)

                noise = torch.randn(batch_size, seq_len, n_features).to(self.device)
                fake_data = self.generator(noise)
                fake_output = self.discriminator(fake_data)
                d_loss_fake = self.criterion(fake_output, fake_labels)

                d_losses.append((d_loss_real + d_loss_fake).item())

                # Generator evaluation
                g_loss = self.criterion(fake_output, real_labels)
                g_losses.append(g_loss.item())

        self.generator.train()
        self.discriminator.train()

        return np.mean(g_losses), np.mean(d_losses)

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              epochs: Optional[int] = None, verbose: bool = True) -> Dict:
        """
        Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs (uses config default if None)
            verbose: Whether to print progress

        Returns:
            Dictionary with training history
        """
        set_seed(self.config.seed)

        epochs = epochs or self.config.epochs
        best_verge = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            epoch_g_losses = []
            epoch_d_losses = []

            for data in train_loader:
                data = data.to(self.device)
                g_loss, d_loss = self.train_step(data)
                epoch_g_losses.append(g_loss)
                epoch_d_losses.append(d_loss)

            avg_g_loss = np.mean(epoch_g_losses)
            avg_d_loss = np.mean(epoch_d_losses)
            self.train_g_losses.extend(epoch_g_losses)
            self.train_d_losses.extend(epoch_d_losses)

            # Validation
            val_g_loss, val_d_loss = self.evaluate(val_loader)
            self.val_g_losses.append(val_g_loss)
            self.val_d_losses.append(val_d_loss)

            # Early stopping check
            current_verge = get_loss_verge(val_g_loss, val_d_loss)

            if current_verge < best_verge - self.config.min_delta:
                best_verge = current_verge
                patience_counter = 0
            else:
                patience_counter += 1

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | "
                      f"G Loss: {avg_g_loss:.4f} | D Loss: {avg_d_loss:.4f} | "
                      f"Val G: {val_g_loss:.4f} | Val D: {val_d_loss:.4f}")

            if self.config.early_stopping and patience_counter >= self.config.patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                break

        return {
            'train_g_losses': self.train_g_losses,
            'train_d_losses': self.train_d_losses,
            'val_g_losses': self.val_g_losses,
            'val_d_losses': self.val_d_losses,
            'epochs_completed': epoch + 1
        }

    def save(self, path: str, stock: str):
        """Save trained models."""
        os.makedirs(path, exist_ok=True)
        torch.save(self.generator, f"{path}/{stock}_generator.pth")
        torch.save(self.discriminator, f"{path}/{stock}_discriminator.pth")

    def load(self, path: str, stock: str):
        """Load trained models."""
        self.generator = torch.load(f"{path}/{stock}_generator.pth", weights_only=False)
        self.discriminator = torch.load(f"{path}/{stock}_discriminator.pth", weights_only=False)
        self.generator.to(self.device)
        self.discriminator.to(self.device)
