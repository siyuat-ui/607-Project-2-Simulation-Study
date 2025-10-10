"""Main simulation orchestration for engression experiments.

This module provides functions to run complete simulation experiments
that compare engression-based sample generation across different distributions.
"""

import numpy as np
import torch
import pandas as pd
from pathlib import Path
import time
from datetime import datetime

from dgps import (
    NormalGenerator,
    ExponentialGenerator,
    UniformGenerator,
    LognormalGenerator,
    ChiSquareGenerator
)
from train_and_inference import train_and_generate
from metrics import compute_all_metrics
from visualizations import plot_all_visualizations

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


class SimulationExperiment:
    """Class to orchestrate simulation experiments.
    
    Parameters
    ----------
    generators : list of DataGenerator
        List of data generators to test
    sample_sizes : list of int
        List of sample sizes to test
    n_replications : int, default=10
        Number of replications per configuration
    training_params : dict, optional
        Parameters for training (num_epochs, batch_size, etc.)
    save_results : bool, default=True
        Whether to save results to files
    results_dir : str or Path, default='results'
        Directory to save results
    verbose : bool, default=True
        Whether to print progress
    """
    
    def __init__(self, generators, sample_sizes, n_replications=10,
                 training_params=None, save_results=True,
                 results_dir='results', verbose=True):
        self.generators = generators
        self.sample_sizes = sample_sizes
        self.n_replications = n_replications
        self.save_results = save_results
        self.results_dir = Path(results_dir)
        self.verbose = verbose
        
        # Default training parameters
        self.training_params = {
            'num_epochs': 200,
            'batch_size': 128,
            'learning_rate': 1e-4,
            'm': 50,
            'patience': 20,
            'input_dim': 128,
        }
        if training_params is not None:
            self.training_params.update(training_params)
        
        # Storage for results
        self.results = []
        
        # Create results directories
        if self.save_results:
            self.figures_dir = self.results_dir / 'figures'
            self.raw_dir = self.results_dir / 'raw'
            self.figures_dir.mkdir(parents=True, exist_ok=True)
            self.raw_dir.mkdir(parents=True, exist_ok=True)
    
    def run_single_experiment(self, generator, sample_size, replication):
        """Run a single experiment configuration.
        
        Parameters
        ----------
        generator : DataGenerator
            Data generator to use
        sample_size : int
            Number of samples to generate
        replication : int
            Replication number
            
        Returns
        -------
        dict
            Dictionary containing all experiment results
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Generator: {generator.name}")
            print(f"Sample size: {sample_size}")
            print(f"Replication: {replication + 1}/{self.n_replications}")
            print(f"{'='*70}")
        
        start_time = time.time()
        
        # Generate original data
        X_original_np = generator.generate(sample_size)
        if X_original_np.ndim == 1:
            X_original_np = X_original_np.reshape(-1, 1)
        X_original = torch.from_numpy(X_original_np).float()
        
        # Train model and generate samples
        model, history, X_generated = train_and_generate(
            X_original,
            num_samples=sample_size,
            verbose=self.verbose,
            **self.training_params
        )
        
        training_time = time.time() - start_time
        
        # Compute metrics
        metrics = compute_all_metrics(
            X_original, X_generated,
            verbose=self.verbose
        )
        
        # Create visualizations if saving results
        if self.save_results:
            prefix = f"{generator.name.replace('(', '_').replace(')', '_').replace(', ', '_').replace('=', '')}_n{sample_size}_rep{replication}"
            plot_all_visualizations(
                X_original,
                X_generated,
                history,
                generator_name=generator.name,
                prefix=prefix,
                show=False
            )
        
        # Compile results
        result = {
            'generator': generator.name,
            'sample_size': sample_size,
            'replication': replication,
            'training_time': training_time,
            'final_loss': history['loss'][-1],
            'final_term1': history['term1'][-1],
            'final_term2': history['term2'][-1],
            'n_epochs': len(history['loss']),
            'mmd': metrics['mmd'],
            'two_sample_min_p': metrics['two_sample_test']['min_p_value'],
            'two_sample_mean_p': metrics['two_sample_test']['mean_p_value'],
            'two_sample_rejected': metrics['two_sample_test']['num_rejected'],
            'mean_distance': metrics['mean_distance'],
            'cov_frobenius': metrics['cov_frobenius'],
            'cov_trace': metrics['cov_trace'],
        }
        
        if self.verbose:
            print(f"\nCompleted in {training_time:.2f} seconds")
            print(f"Final loss: {result['final_loss']:.6f}")
            print(f"MMD: {result['mmd']:.6f}")
        
        return result
    
    def run_all_experiments(self):
        """Run all experiment configurations.
        
        Returns
        -------
        pd.DataFrame
            DataFrame containing all results
        """
        total_experiments = (len(self.generators) * 
                           len(self.sample_sizes) * 
                           self.n_replications)
        
        if self.verbose:
            print(f"\n{'#'*70}")
            print(f"Starting simulation with {total_experiments} experiments")
            print(f"Generators: {len(self.generators)}")
            print(f"Sample sizes: {self.sample_sizes}")
            print(f"Replications: {self.n_replications}")
            print(f"{'#'*70}\n")
        
        experiment_count = 0
        start_time = time.time()
        
        for generator in self.generators:
            for sample_size in self.sample_sizes:
                for replication in range(self.n_replications):
                    experiment_count += 1
                    
                    if self.verbose:
                        print(f"\nExperiment {experiment_count}/{total_experiments}")
                    
                    result = self.run_single_experiment(
                        generator, sample_size, replication
                    )
                    self.results.append(result)
        
        total_time = time.time() - start_time
        
        # Convert to DataFrame
        results_df = pd.DataFrame(self.results)
        
        if self.verbose:
            print(f"\n{'#'*70}")
            print(f"Simulation completed!")
            print(f"Total time: {total_time/60:.2f} minutes")
            print(f"Average time per experiment: {total_time/total_experiments:.2f} seconds")
            print(f"{'#'*70}\n")
        
        # Save results
        if self.save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = self.raw_dir / f"simulation_results_{timestamp}.csv"
            results_df.to_csv(results_file, index=False)
            print(f"Results saved to: {results_file}")
        
        return results_df
    
    def summarize_results(self, results_df=None):
        """Create summary statistics from results.
        
        Parameters
        ----------
        results_df : pd.DataFrame, optional
            Results dataframe. If None, uses stored results.
            
        Returns
        -------
        pd.DataFrame
            Summary statistics grouped by generator and sample size
        """
        if results_df is None:
            if not self.results:
                raise ValueError("No results available. Run experiments first.")
            results_df = pd.DataFrame(self.results)
        
        # Group by generator and sample size
        summary = results_df.groupby(['generator', 'sample_size']).agg({
            'training_time': ['mean', 'std'],
            'final_loss': ['mean', 'std'],
            'n_epochs': ['mean', 'std'],
            'mmd': ['mean', 'std'],
            'two_sample_mean_p': ['mean', 'std'],
            'two_sample_rejected': ['mean', 'std'],
            'mean_distance': ['mean', 'std'],
            'cov_frobenius': ['mean', 'std'],
        }).round(6)
        
        # Flatten column names
        summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
        summary = summary.reset_index()
        
        if self.save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_file = self.raw_dir / f"simulation_summary_{timestamp}.csv"
            summary.to_csv(summary_file, index=False)
            print(f"Summary saved to: {summary_file}")
        
        return summary


def run_quick_simulation(n_replications=3, sample_sizes=[100, 500], 
                        verbose=True, save_results=True):
    """Run a quick simulation with default settings.
    
    Parameters
    ----------
    n_replications : int, default=3
        Number of replications per configuration
    sample_sizes : list of int, default=[100, 500]
        Sample sizes to test
    verbose : bool, default=True
        Whether to print progress
    save_results : bool, default=True
        Whether to save results
        
    Returns
    -------
    tuple
        (results_df, summary_df)
    """
    # Define generators to test
    generators = [
        NormalGenerator(loc=0, scale=1),
        ExponentialGenerator(scale=1),
        UniformGenerator(low=0, high=2),
    ]
    
    # Quick training parameters
    training_params = {
        'num_epochs': 50,
        'batch_size': 64,
        'learning_rate': 1e-3,
        'patience': 10,
        'input_dim': 2,  # Small input_dim for quick testing
    }
    
    # Run simulation
    sim = SimulationExperiment(
        generators=generators,
        sample_sizes=sample_sizes,
        n_replications=n_replications,
        training_params=training_params,
        save_results=save_results,
        verbose=verbose
    )
    
    results_df = sim.run_all_experiments()
    summary_df = sim.summarize_results(results_df)
    
    return results_df, summary_df


def run_full_simulation(n_replications=10, sample_sizes=[100, 500, 1000],
                       verbose=True, save_results=True):
    """Run a full simulation with all generators.
    
    Parameters
    ----------
    n_replications : int, default=10
        Number of replications per configuration
    sample_sizes : list of int, default=[100, 500, 1000]
        Sample sizes to test
    verbose : bool, default=True
        Whether to print progress
    save_results : bool, default=True
        Whether to save results
        
    Returns
    -------
    tuple
        (results_df, summary_df)
    """
    # Define all generators to test
    generators = [
        NormalGenerator(loc=0, scale=1),
        ExponentialGenerator(scale=1),
        UniformGenerator(low=0, high=2),
        LognormalGenerator(mean=0, sigma=1),
        ChiSquareGenerator(df=5),
    ]
    
    # Full simulation uses default training params (including input_dim=128)
    # Run simulation
    sim = SimulationExperiment(
        generators=generators,
        sample_sizes=sample_sizes,
        n_replications=n_replications,
        save_results=save_results,
        verbose=verbose
    )
    
    results_df = sim.run_all_experiments()
    summary_df = sim.summarize_results(results_df)
    
    return results_df, summary_df