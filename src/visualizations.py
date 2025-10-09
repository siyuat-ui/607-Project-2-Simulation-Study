"""Visualization functions for engression analysis.

This module provides functions to create visualizations for training
diagnostics and distribution comparisons.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns


def ensure_results_dir():
    """Ensure the results/figures directory exists."""
    results_dir = Path("results/figures")
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def plot_training_loss(training_history, save_name=None, show=True):
    """Plot training loss curves including all three terms.
    
    The engression loss consists of:
    - Total loss = term1 - 0.5 * term2
    - term1: E[||X - g(eps)||_2] (reconstruction term)
    - term2: E[||g(eps) - g(eps')||_2] (diversity term)
    
    Parameters
    ----------
    training_history : dict
        Dictionary with keys 'loss', 'term1', 'term2', each containing
        a list of values across epochs
    save_name : str, optional
        Filename to save the plot. If None, uses 'training_loss.png'
    show : bool, default=True
        Whether to display the plot
        
    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    results_dir = ensure_results_dir()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    epochs = range(1, len(training_history['loss']) + 1)
    
    # Plot total loss
    axes[0].plot(epochs, training_history['loss'], linewidth=2, color='blue')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Total Loss', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Plot term1 (reconstruction)
    axes[1].plot(epochs, training_history['term1'], linewidth=2, color='green')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Term1', fontsize=12)
    axes[1].set_title('Term1: E[||X - g(ε)||]', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # Plot term2 (diversity)
    axes[2].plot(epochs, training_history['term2'], linewidth=2, color='red')
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Term2', fontsize=12)
    axes[2].set_title('Term2: E[||g(ε) - g(ε\')||]', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    if save_name is None:
        save_name = 'training_loss.png'
    save_path = results_dir / save_name
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training loss plot saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_density_comparison(X_original, X_generated, generator_name=None, 
                           save_name=None, show=True):
    """Plot density comparisons between original and generated samples.
    
    Creates separate density plots for each dimension, comparing the
    original and generated distributions.
    
    Parameters
    ----------
    X_original : np.ndarray or torch.Tensor
        Original samples, shape (n_original, dim)
    X_generated : np.ndarray or torch.Tensor
        Generated samples, shape (n_generated, dim)
    generator_name : str, optional
        Name of the data generator (e.g., 'Normal(μ=0, σ=1)')
        If provided, will be included in the title
    save_name : str, optional
        Filename to save the plot. If None, uses 'density_comparison.png'
    show : bool, default=True
        Whether to display the plot
        
    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    results_dir = ensure_results_dir()
    
    # Convert to numpy if needed
    if hasattr(X_original, 'cpu'):
        X_original = X_original.cpu().numpy()
    if hasattr(X_generated, 'cpu'):
        X_generated = X_generated.cpu().numpy()
    
    n_dims = X_original.shape[1]
    
    # Create subplots
    n_cols = min(n_dims, 3)
    n_rows = (n_dims + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    
    # Handle single subplot case
    if n_dims == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    axes_flat = axes.flatten() if n_dims > 1 else axes
    
    for d in range(n_dims):
        ax = axes_flat[d]
        
        # Plot densities using KDE
        ax.hist(X_original[:, d], bins=50, alpha=0.5, density=True, 
                label='Original', color='blue', edgecolor='black', linewidth=0.5)
        ax.hist(X_generated[:, d], bins=50, alpha=0.5, density=True,
                label='Generated', color='red', edgecolor='black', linewidth=0.5)
        
        # Add KDE curves for smoother visualization
        try:
            from scipy.stats import gaussian_kde
            kde_orig = gaussian_kde(X_original[:, d])
            kde_gen = gaussian_kde(X_generated[:, d])
            
            x_range = np.linspace(
                min(X_original[:, d].min(), X_generated[:, d].min()),
                max(X_original[:, d].max(), X_generated[:, d].max()),
                200
            )
            
            ax.plot(x_range, kde_orig(x_range), 'b-', linewidth=2, alpha=0.8)
            ax.plot(x_range, kde_gen(x_range), 'r-', linewidth=2, alpha=0.8)
        except:
            pass  # Skip KDE if it fails
        
        ax.set_xlabel(f'Dimension {d+1}', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'Dimension {d+1}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for d in range(n_dims, len(axes_flat)):
        axes_flat[d].axis('off')
    
    # Add overall title
    if generator_name:
        fig.suptitle(f'Density Comparison: {generator_name}', 
                    fontsize=16, fontweight='bold', y=1.02)
    else:
        fig.suptitle('Density Comparison: Original vs Generated', 
                    fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save figure
    if save_name is None:
        save_name = 'density_comparison.png'
    save_path = results_dir / save_name
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Density comparison plot saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_scatter_comparison(X_original, X_generated, generator_name=None,
                           save_name=None, show=True):
    """Plot scatter plots comparing original and generated samples.
    
    For 1D data, creates a strip plot with jitter. For 2D data, creates 
    a single scatter plot. For higher dimensions, creates a grid of 
    pairwise scatter plots.
    
    Parameters
    ----------
    X_original : np.ndarray or torch.Tensor
        Original samples, shape (n_original, dim)
    X_generated : np.ndarray or torch.Tensor
        Generated samples, shape (n_generated, dim)
    generator_name : str, optional
        Name of the data generator
    save_name : str, optional
        Filename to save the plot. If None, uses 'scatter_comparison.png'
    show : bool, default=True
        Whether to display the plot
        
    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    results_dir = ensure_results_dir()
    
    # Convert to numpy if needed
    if hasattr(X_original, 'cpu'):
        X_original = X_original.cpu().numpy()
    if hasattr(X_generated, 'cpu'):
        X_generated = X_generated.cpu().numpy()
    
    n_dims = X_original.shape[1]
    
    if n_dims == 1:
        # For 1D data, create a strip plot with jitter
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Add jitter for visualization
        np.random.seed(42)
        jitter_orig = np.random.normal(0, 0.02, size=len(X_original))
        jitter_gen = np.random.normal(0, 0.02, size=len(X_generated))
        
        ax.scatter(X_original[:, 0], np.zeros(len(X_original)) + jitter_orig,
                  alpha=0.5, s=30, label='Original', color='blue')
        ax.scatter(X_generated[:, 0], np.ones(len(X_generated)) + jitter_gen,
                  alpha=0.5, s=30, label='Generated', color='red')
        
        ax.set_xlabel('Value', fontsize=12)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Original', 'Generated'], fontsize=11)
        ax.set_ylim(-0.5, 1.5)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='x')
        
        if generator_name:
            ax.set_title(f'1D Distribution: {generator_name}', 
                        fontsize=14, fontweight='bold')
        else:
            ax.set_title('1D Distribution: Original vs Generated',
                        fontsize=14, fontweight='bold')
    
    elif n_dims == 2:
        # Simple 2D scatter plot
        fig, ax = plt.subplots(figsize=(8, 8))
        
        ax.scatter(X_original[:, 0], X_original[:, 1], 
                  alpha=0.5, s=20, label='Original', color='blue')
        ax.scatter(X_generated[:, 0], X_generated[:, 1],
                  alpha=0.5, s=20, label='Generated', color='red')
        
        ax.set_xlabel('Dimension 1', fontsize=12)
        ax.set_ylabel('Dimension 2', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        if generator_name:
            ax.set_title(f'Scatter Plot: {generator_name}', 
                        fontsize=14, fontweight='bold')
        else:
            ax.set_title('Scatter Plot: Original vs Generated',
                        fontsize=14, fontweight='bold')
    
    else:
        # Pairwise scatter plots for higher dimensions
        # Limit to first 4 dimensions to keep plot manageable
        plot_dims = min(n_dims, 4)
        fig, axes = plt.subplots(plot_dims, plot_dims, 
                                figsize=(3*plot_dims, 3*plot_dims))
        
        for i in range(plot_dims):
            for j in range(plot_dims):
                ax = axes[i, j]
                
                if i == j:
                    # Diagonal: show histograms
                    ax.hist(X_original[:, i], bins=30, alpha=0.5, 
                           color='blue', density=True)
                    ax.hist(X_generated[:, i], bins=30, alpha=0.5,
                           color='red', density=True)
                    ax.set_ylabel('Density' if j == 0 else '')
                else:
                    # Off-diagonal: scatter plots
                    ax.scatter(X_original[:, j], X_original[:, i],
                             alpha=0.3, s=10, color='blue')
                    ax.scatter(X_generated[:, j], X_generated[:, i],
                             alpha=0.3, s=10, color='red')
                
                if i == plot_dims - 1:
                    ax.set_xlabel(f'Dim {j+1}', fontsize=10)
                else:
                    ax.set_xticklabels([])
                
                if j == 0 and i != j:
                    ax.set_ylabel(f'Dim {i+1}', fontsize=10)
                elif j != 0:
                    ax.set_yticklabels([])
                
                ax.grid(True, alpha=0.2)
        
        if generator_name:
            fig.suptitle(f'Pairwise Scatter Plots: {generator_name}',
                        fontsize=16, fontweight='bold')
        else:
            fig.suptitle('Pairwise Scatter Plots: Original vs Generated',
                        fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    if save_name is None:
        save_name = 'scatter_comparison.png'
    save_path = results_dir / save_name
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Scatter comparison plot saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_all_visualizations(X_original, X_generated, training_history,
                           generator_name=None, prefix=None, show=False):
    """Create all visualization plots at once.
    
    Parameters
    ----------
    X_original : np.ndarray or torch.Tensor
        Original samples
    X_generated : np.ndarray or torch.Tensor
        Generated samples
    training_history : dict
        Training history with loss curves
    generator_name : str, optional
        Name of the data generator
    prefix : str, optional
        Prefix for saved filenames
    show : bool, default=False
        Whether to display plots
        
    Returns
    -------
    dict
        Dictionary of figure objects with keys 'loss', 'density', 'scatter'
    """
    prefix = prefix + "_" if prefix else ""
    
    figures = {}
    
    # Training loss
    figures['loss'] = plot_training_loss(
        training_history,
        save_name=f'{prefix}training_loss.png',
        show=show
    )
    
    # Density comparison
    figures['density'] = plot_density_comparison(
        X_original, X_generated,
        generator_name=generator_name,
        save_name=f'{prefix}density_comparison.png',
        show=show
    )
    
    # Scatter comparison
    figures['scatter'] = plot_scatter_comparison(
        X_original, X_generated,
        generator_name=generator_name,
        save_name=f'{prefix}scatter_comparison.png',
        show=show
    )
    
    print(f"\nAll visualizations saved to results/figures/")
    
    return figures