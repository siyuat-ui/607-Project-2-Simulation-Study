"""Training and inference for engression networks.

This module provides functions for training EngressionNet models
and generating samples from trained models.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from methods import EngressionNet, engression_loss, get_device


class EngressionTrainer:
    """Trainer for EngressionNet models with early stopping."""
    
    def __init__(self, batch_size=128, learning_rate=1e-4, num_epochs=200, 
                 m=50, patience=20, device=None):
        """Initialize the trainer.
        
        Parameters
        ----------
        batch_size : int, default=128
            Batch size for training
        learning_rate : float, default=1e-4
            Learning rate for Adam optimizer
        num_epochs : int, default=200
            Maximum number of training epochs
        m : int, default=50
            Number of epsilon samples per X_i in loss computation
        patience : int, default=20
            Number of epochs to wait for improvement before early stopping
        device : torch.device, optional
            Device for computation. If None, uses get_device()
        """
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.m = m
        self.patience = patience
        self.device = device if device is not None else get_device()
        
        self.training_history = {
            'loss': [],
            'term1': [],
            'term2': []
        }
    
    def train(self, X, model=None, verbose=True):
        """Train an engression network.
        
        Parameters
        ----------
        X : torch.Tensor or np.ndarray
            Training data of shape (n_samples, output_dim)
        model : EngressionNet, optional
            Pre-initialized model. If None, creates a new one
        verbose : bool, default=True
            Whether to print training progress
            
        Returns
        -------
        EngressionNet
            Trained model
        dict
            Training history with loss, term1, and term2
        """
        # Convert to tensor if needed
        if not isinstance(X, torch.Tensor):
            X = torch.from_numpy(X).float()
        else:
            X = X.float()
        
        # Move to device
        X = X.to(self.device)
        
        # Initialize model if not provided
        if model is None:
            output_dim = X.shape[1]
            model = EngressionNet(input_dim=2, output_dim=output_dim).to(self.device)
        
        # Initialize optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        
        # Prepare DataLoader
        dataset = TensorDataset(X)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Early stopping variables
        best_loss = float('inf')
        epochs_no_improve = 0
        best_model_state = None
        
        # Training loop with early stopping
        for epoch in range(1, self.num_epochs + 1):
            epoch_loss = 0.0
            epoch_term1 = 0.0
            epoch_term2 = 0.0
            
            for batch in dataloader:
                X_batch = batch[0]
                
                optimizer.zero_grad()
                loss, t1, t2 = engression_loss(model, X_batch, m=self.m, device=self.device)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * X_batch.size(0)
                epoch_term1 += t1.item() * X_batch.size(0)
                epoch_term2 += t2.item() * X_batch.size(0)
            
            # Average over dataset
            epoch_loss /= len(dataset)
            epoch_term1 /= len(dataset)
            epoch_term2 /= len(dataset)
            
            # Store history
            self.training_history['loss'].append(epoch_loss)
            self.training_history['term1'].append(epoch_term1)
            self.training_history['term2'].append(epoch_term2)
            
            # Print progress
            if verbose and (epoch % 10 == 0 or epoch == 1):
                print(f"Epoch {epoch:03d} | Loss: {epoch_loss:.4f} | "
                      f"Term1: {epoch_term1:.4f} | Term2: {epoch_term2:.4f}")
            
            # Early stopping check
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                epochs_no_improve = 0
                best_model_state = model.state_dict().copy()
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch} | Best Loss: {best_loss:.4f}")
                    model.load_state_dict(best_model_state)
                    break
        
        return model, self.training_history


def generate_samples(model, num_samples=1000, input_dim=2, device=None):
    """Generate samples from a trained engression network.
    
    Parameters
    ----------
    model : EngressionNet
        Trained engression network
    num_samples : int, default=1000
        Number of samples to generate
    input_dim : int, default=2
        Dimension of the epsilon (uniform random) input
    device : torch.device, optional
        Device for computation. If None, infers from model's device
        
    Returns
    -------
    torch.Tensor
        Generated samples of shape (num_samples, output_dim)
    """
    if device is None:
        # Infer device from model's first parameter
        device = next(model.parameters()).device
    
    model.eval()
    eps = torch.rand(num_samples, input_dim, device=device)
    
    with torch.no_grad():
        samples = model(eps)
    
    return samples


def train_and_generate(X, num_samples=1000, batch_size=128, learning_rate=1e-4,
                       num_epochs=200, m=50, patience=20, input_dim=2, 
                       verbose=True, device=None):
    """Convenience function to train a model and generate samples.
    
    Parameters
    ----------
    X : torch.Tensor or np.ndarray
        Training data of shape (n_samples, output_dim)
    num_samples : int, default=1000
        Number of samples to generate after training
    batch_size : int, default=128
        Batch size for training
    learning_rate : float, default=1e-4
        Learning rate for optimizer
    num_epochs : int, default=200
        Maximum number of training epochs
    m : int, default=50
        Number of epsilon samples per X_i
    patience : int, default=20
        Epochs to wait for improvement before early stopping
    input_dim : int, default=2
        Dimension of epsilon input
    verbose : bool, default=True
        Whether to print training progress
    device : torch.device, optional
        Device for computation. If None, uses get_device()
        
    Returns
    -------
    tuple
        (trained_model, training_history, generated_samples)
    """
    if device is None:
        device = get_device()
    
    # Initialize trainer
    trainer = EngressionTrainer(
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        m=m,
        patience=patience,
        device=device
    )
    
    # Train model
    model, history = trainer.train(X, verbose=verbose)
    
    # Generate samples
    samples = generate_samples(model, num_samples=num_samples, 
                              input_dim=input_dim, device=device)
    
    return model, history, samples
