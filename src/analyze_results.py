"""Analysis and visualization of simulation results.

This module provides functions to load simulation results and create
diagnostic and publication-quality visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob


def load_latest_results(results_dir='results/raw'):
    """Load the most recent simulation results.
    
    Parameters
    ----------
    results_dir : str, default='results/raw'
        Directory containing results CSV files
        
    Returns
    -------
    pd.DataFrame
        Results dataframe
    """
    results_path = Path(results_dir)
    csv_files = list(results_path.glob('simulation_results_*.csv'))
    
    if not csv_files:
        raise FileNotFoundError(f"No results found in {results_dir}")
    
    # Get most recent file
    latest_file = max(csv_files, key=lambda p: p.stat().st_mtime)
    print(f"Loading results from: {latest_file}")
    
    df = pd.read_csv(latest_file)
    return df


def create_diagnostic_heatmap(df, save_path='results/figures/diagnostic_heatmap.png'):
    """Create diagnostic heatmap showing mean MMD across conditions.
    
    Parameters
    ----------
    df : pd.DataFrame
        Results dataframe
    save_path : str
        Path to save figure
    """
    # Compute mean MMD by distribution and sample size
    pivot = df.pivot_table(
        values='mmd',
        index='generator',
        columns='sample_size',
        aggfunc='mean'
    )
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create heatmap
    sns.heatmap(
        pivot,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn_r',
        center=0.1,
        vmin=0,
        vmax=0.3,
        cbar_kws={'label': 'Mean MMD'},
        ax=ax
    )
    
    ax.set_xlabel('Sample Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Distribution', fontsize=12, fontweight='bold')
    ax.set_title('Diagnostic: Mean MMD Across All Conditions', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Diagnostic heatmap saved to: {save_path}")
    plt.close()


def create_publication_figure(df, save_path='results/figures/publication_figure.png'):
    """Create publication-quality multi-panel figure.
    
    Parameters
    ----------
    df : pd.DataFrame
        Results dataframe
    save_path : str
        Path to save figure
    """
    # Order distributions by expected difficulty
    dist_order = [
        'Normal(μ=0, σ=1)',
        'Uniform(0, 2)',
        'ChiSquare(df=5)',
        'Exponential(λ=1.00)',
        'Lognormal(μ=0, σ=1)'
    ]
    
    # Filter to only distributions that exist in the data
    dist_order = [d for d in dist_order if d in df['generator'].values]
    
    # Create figure with 2 panels
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Color palette for sample sizes
    colors = {'100': '#3498db', '500': '#2ecc71', '1000': '#e74c3c'}
    
    # Panel A: MMD by distribution
    ax1 = axes[0]
    
    for sample_size in sorted(df['sample_size'].unique()):
        subset = df[df['sample_size'] == sample_size]
        
        # Compute mean and 95% CI
        means = []
        cis_lower = []
        cis_upper = []
        
        for dist in dist_order:
            dist_data = subset[subset['generator'] == dist]['mmd']
            if len(dist_data) > 0:
                mean = dist_data.mean()
                se = dist_data.std() / np.sqrt(len(dist_data))
                ci = 1.96 * se  # 95% CI
                
                means.append(mean)
                cis_lower.append(mean - ci)
                cis_upper.append(mean + ci)
            else:
                means.append(np.nan)
                cis_lower.append(np.nan)
                cis_upper.append(np.nan)
        
        x_pos = np.arange(len(dist_order)) + (sample_size - 500) / 1000
        
        ax1.errorbar(
            x_pos, means,
            yerr=[np.array(means) - np.array(cis_lower), 
                  np.array(cis_upper) - np.array(means)],
            fmt='o-',
            capsize=5,
            label=f'n={sample_size}',
            color=colors.get(str(sample_size), 'gray'),
            linewidth=2,
            markersize=8
        )
    
    ax1.axhline(y=0.1, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Target (MMD<0.1)')
    ax1.set_xticks(range(len(dist_order)))
    ax1.set_xticklabels([d.split('(')[0] for d in dist_order], rotation=45, ha='right')
    ax1.set_ylabel('Maximum Mean Discrepancy (MMD)', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Distribution Type', fontsize=11, fontweight='bold')
    ax1.set_title('A. Distribution Matching Quality', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', frameon=True)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(bottom=0)
    
    # Panel B: Two-sample test p-values
    ax2 = axes[1]
    
    for sample_size in sorted(df['sample_size'].unique()):
        subset = df[df['sample_size'] == sample_size]
        
        means = []
        cis_lower = []
        cis_upper = []
        
        for dist in dist_order:
            dist_data = subset[subset['generator'] == dist]['two_sample_mean_p']
            if len(dist_data) > 0:
                mean = dist_data.mean()
                se = dist_data.std() / np.sqrt(len(dist_data))
                ci = 1.96 * se
                
                means.append(mean)
                cis_lower.append(mean - ci)
                cis_upper.append(mean + ci)
            else:
                means.append(np.nan)
                cis_lower.append(np.nan)
                cis_upper.append(np.nan)
        
        x_pos = np.arange(len(dist_order)) + (sample_size - 500) / 1000
        
        ax2.errorbar(
            x_pos, means,
            yerr=[np.array(means) - np.array(cis_lower), 
                  np.array(cis_upper) - np.array(means)],
            fmt='o-',
            capsize=5,
            label=f'n={sample_size}',
            color=colors.get(str(sample_size), 'gray'),
            linewidth=2,
            markersize=8
        )
    
    ax2.axhline(y=0.05, color='red', linestyle='--', linewidth=1, alpha=0.5, label='α=0.05')
    ax2.set_xticks(range(len(dist_order)))
    ax2.set_xticklabels([d.split('(')[0] for d in dist_order], rotation=45, ha='right')
    ax2.set_ylabel('Mean p-value (KS Test)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Distribution Type', fontsize=11, fontweight='bold')
    ax2.set_title('B. Two-Sample Test Results', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower right', frameon=True)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Publication figure saved to: {save_path}")
    plt.close()


def create_sample_size_scaling_plot(df, save_path='results/figures/sample_size_scaling.png'):
    """Create plot showing how performance scales with sample size.
    
    Parameters
    ----------
    df : pd.DataFrame
        Results dataframe
    save_path : str
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get unique distributions
    distributions = df['generator'].unique()
    
    for dist in sorted(distributions):
        subset = df[df['generator'] == dist]
        
        sample_sizes = sorted(subset['sample_size'].unique())
        means = []
        stds = []
        
        for n in sample_sizes:
            data = subset[subset['sample_size'] == n]['mmd']
            means.append(data.mean())
            stds.append(data.std())
        
        ax.plot(sample_sizes, means, 'o-', label=dist.split('(')[0], linewidth=2, markersize=8)
        ax.fill_between(
            sample_sizes,
            np.array(means) - np.array(stds),
            np.array(means) + np.array(stds),
            alpha=0.2
        )
    
    ax.axhline(y=0.1, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Target (MMD<0.1)')
    ax.set_xlabel('Sample Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('MMD (mean ± std)', fontsize=12, fontweight='bold')
    ax.set_title('Sample Size Scaling Analysis', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', frameon=True)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Sample size scaling plot saved to: {save_path}")
    plt.close()


def create_success_rate_heatmap(df, threshold=0.1, save_path='results/figures/success_rate_heatmap.png'):
    """Create heatmap showing success rate (% replications with MMD < threshold).
    
    Parameters
    ----------
    df : pd.DataFrame
        Results dataframe
    threshold : float, default=0.1
        MMD threshold for success
    save_path : str
        Path to save figure
    """
    # Compute success rate
    df['success'] = df['mmd'] < threshold
    
    pivot = df.pivot_table(
        values='success',
        index='generator',
        columns='sample_size',
        aggfunc='mean'
    ) * 100  # Convert to percentage
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.heatmap(
        pivot,
        annot=True,
        fmt='.1f',
        cmap='RdYlGn',
        vmin=0,
        vmax=100,
        cbar_kws={'label': 'Success Rate (%)'},
        ax=ax
    )
    
    ax.set_xlabel('Sample Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Distribution', fontsize=12, fontweight='bold')
    ax.set_title(f'Success Rate: % Replications with MMD < {threshold}', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Success rate heatmap saved to: {save_path}")
    plt.close()


def generate_all_figures(results_dir='results/raw'):
    """Generate all diagnostic and publication figures.
    
    Parameters
    ----------
    results_dir : str
        Directory containing results CSV files
    """
    print("=" * 70)
    print("Generating Analysis Figures")
    print("=" * 70)
    
    # Load results
    df = load_latest_results(results_dir)
    print(f"\nLoaded {len(df)} experiments")
    print(f"Distributions: {df['generator'].nunique()}")
    print(f"Sample sizes: {sorted(df['sample_size'].unique())}")
    print(f"Replications per condition: {df.groupby(['generator', 'sample_size']).size().min()}")
    
    print("\nGenerating figures...")
    
    # Create all figures
    create_diagnostic_heatmap(df)
    create_publication_figure(df)
    create_sample_size_scaling_plot(df)
    create_success_rate_heatmap(df)
    
    print("\n" + "=" * 70)
    print("All figures generated successfully!")
    print("=" * 70)


if __name__ == "__main__":
    generate_all_figures()