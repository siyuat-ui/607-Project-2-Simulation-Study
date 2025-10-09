"""Tests for visualization functions.

This module tests the visualization functions to ensure they work
correctly and produce expected outputs.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from visualizations import (
    ensure_results_dir,
    plot_training_loss,
    plot_density_comparison,
    plot_scatter_comparison,
    plot_all_visualizations
)
from dgps import NormalGenerator
from train_and_inference import train_and_generate


@pytest.fixture
def training_history():
    """Create sample training history."""
    return {
        'loss': [1.0, 0.8, 0.6, 0.5, 0.4],
        'term1': [1.2, 1.0, 0.9, 0.8, 0.7],
        'term2': [0.4, 0.4, 0.6, 0.6, 0.6]
    }


@pytest.fixture
def sample_data_2d():
    """Create sample 2D data."""
    np.random.seed(42)
    X_original = np.random.randn(100, 2)
    X_generated = np.random.randn(100, 2) + 0.5
    return X_original, X_generated


@pytest.fixture
def sample_data_1d():
    """Create sample 1D data."""
    np.random.seed(42)
    X_original = np.random.randn(100, 1)
    X_generated = np.random.randn(100, 1) + 0.5
    return X_original, X_generated


@pytest.fixture
def sample_data_high_dim():
    """Create sample high-dimensional data."""
    np.random.seed(42)
    X_original = np.random.randn(100, 5)
    X_generated = np.random.randn(100, 5) + 0.5
    return X_original, X_generated


@pytest.fixture(autouse=True)
def cleanup_plots():
    """Clean up matplotlib figures after each test."""
    yield
    plt.close('all')


@pytest.fixture(scope="module", autouse=True)
def cleanup_generated_figures():
    """Clean up generated figure files after all tests complete."""
    yield
    # This runs after all tests in the module
    results_dir = Path('results/figures')
    if results_dir.exists():
        # Remove test-generated files
        patterns = [
            'test_*.png', 
            'actual_test_*.png', 
            'exp1_*.png',
            'training_loss.png',
            'density_comparison.png',
            'scatter_comparison.png'
        ]
        for pattern in patterns:
            for file in results_dir.glob(pattern):
                try:
                    file.unlink()
                    print(f"Cleaned up: {file}")
                except Exception as e:
                    print(f"Could not remove {file}: {e}")


class TestEnsureResultsDir:
    """Test directory creation."""
    
    def test_ensure_results_dir_creates_directory(self):
        results_dir = ensure_results_dir()
        assert results_dir.exists()
        assert results_dir.is_dir()
        assert results_dir.name == "figures"
    
    def test_ensure_results_dir_idempotent(self):
        # Should work even if called multiple times
        dir1 = ensure_results_dir()
        dir2 = ensure_results_dir()
        assert dir1 == dir2


class TestPlotTrainingLoss:
    """Test training loss plotting."""
    
    def test_plot_training_loss_creates_figure(self, training_history):
        fig = plot_training_loss(training_history, show=False)
        assert isinstance(fig, plt.Figure)
    
    def test_plot_training_loss_has_three_subplots(self, training_history):
        fig = plot_training_loss(training_history, show=False)
        assert len(fig.axes) == 3
    
    def test_plot_training_loss_saves_file(self, training_history, tmp_path):
        # Change to temp directory
        import os
        original_dir = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            plot_training_loss(training_history, save_name='test_loss.png', show=False)
            saved_file = tmp_path / 'results' / 'figures' / 'test_loss.png'
            assert saved_file.exists()
        finally:
            os.chdir(original_dir)
    
    def test_plot_training_loss_with_default_name(self, training_history, tmp_path):
        import os
        original_dir = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            plot_training_loss(training_history, show=False)
            saved_file = tmp_path / 'results' / 'figures' / 'training_loss.png'
            assert saved_file.exists()
        finally:
            os.chdir(original_dir)
    
    def test_plot_training_loss_with_long_history(self):
        long_history = {
            'loss': list(np.linspace(1.0, 0.1, 100)),
            'term1': list(np.linspace(1.2, 0.3, 100)),
            'term2': list(np.linspace(0.4, 0.4, 100))
        }
        fig = plot_training_loss(long_history, show=False)
        assert isinstance(fig, plt.Figure)
    
    def test_plot_training_loss_with_single_epoch(self):
        single_epoch = {
            'loss': [1.0],
            'term1': [1.2],
            'term2': [0.4]
        }
        fig = plot_training_loss(single_epoch, show=False)
        assert isinstance(fig, plt.Figure)


class TestPlotDensityComparison:
    """Test density comparison plotting."""
    
    def test_plot_density_comparison_creates_figure(self, sample_data_2d):
        X_orig, X_gen = sample_data_2d
        fig = plot_density_comparison(X_orig, X_gen, show=False)
        assert isinstance(fig, plt.Figure)
    
    def test_plot_density_comparison_2d(self, sample_data_2d):
        X_orig, X_gen = sample_data_2d
        fig = plot_density_comparison(X_orig, X_gen, show=False)
        # Should have 2 subplots for 2D data
        assert len(fig.axes) >= 2
    
    def test_plot_density_comparison_1d(self, sample_data_1d):
        X_orig, X_gen = sample_data_1d
        fig = plot_density_comparison(X_orig, X_gen, show=False)
        # Should have 1 subplot for 1D data
        assert len(fig.axes) >= 1
    
    def test_plot_density_comparison_high_dim(self, sample_data_high_dim):
        X_orig, X_gen = sample_data_high_dim
        fig = plot_density_comparison(X_orig, X_gen, show=False)
        # Should have 5 subplots for 5D data
        assert len(fig.axes) >= 5
    
    def test_plot_density_comparison_with_generator_name(self, sample_data_2d):
        X_orig, X_gen = sample_data_2d
        fig = plot_density_comparison(
            X_orig, X_gen, 
            generator_name='Normal(μ=0, σ=1)',
            show=False
        )
        assert isinstance(fig, plt.Figure)
    
    def test_plot_density_comparison_with_torch_tensors(self):
        torch.manual_seed(42)
        X_orig = torch.randn(100, 2)
        X_gen = torch.randn(100, 2)
        fig = plot_density_comparison(X_orig, X_gen, show=False)
        assert isinstance(fig, plt.Figure)
    
    def test_plot_density_comparison_saves_file(self, sample_data_2d, tmp_path):
        import os
        original_dir = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            X_orig, X_gen = sample_data_2d
            plot_density_comparison(
                X_orig, X_gen,
                save_name='test_density.png',
                show=False
            )
            saved_file = tmp_path / 'results' / 'figures' / 'test_density.png'
            assert saved_file.exists()
        finally:
            os.chdir(original_dir)


class TestPlotScatterComparison:
    """Test scatter comparison plotting."""
    
    def test_plot_scatter_comparison_1d(self, sample_data_1d):
        X_orig, X_gen = sample_data_1d
        fig = plot_scatter_comparison(X_orig, X_gen, show=False)
        assert isinstance(fig, plt.Figure)
        # Should create a strip plot for 1D
        assert len(fig.axes) == 1
    
    def test_plot_scatter_comparison_2d(self, sample_data_2d):
        X_orig, X_gen = sample_data_2d
        fig = plot_scatter_comparison(X_orig, X_gen, show=False)
        assert isinstance(fig, plt.Figure)
        # Should create a single scatter plot for 2D
        assert len(fig.axes) == 1
    
    def test_plot_scatter_comparison_high_dim(self, sample_data_high_dim):
        X_orig, X_gen = sample_data_high_dim
        fig = plot_scatter_comparison(X_orig, X_gen, show=False)
        assert isinstance(fig, plt.Figure)
        # Should create a grid of scatter plots
        # For 5D data limited to 4 dims, expect 4x4=16 subplots
        assert len(fig.axes) == 16
    
    def test_plot_scatter_comparison_with_generator_name(self, sample_data_2d):
        X_orig, X_gen = sample_data_2d
        fig = plot_scatter_comparison(
            X_orig, X_gen,
            generator_name='Exponential(λ=1.00)',
            show=False
        )
        assert isinstance(fig, plt.Figure)
    
    def test_plot_scatter_comparison_with_torch_tensors(self):
        torch.manual_seed(42)
        X_orig = torch.randn(100, 2)
        X_gen = torch.randn(100, 2)
        fig = plot_scatter_comparison(X_orig, X_gen, show=False)
        assert isinstance(fig, plt.Figure)
    
    def test_plot_scatter_comparison_saves_file(self, sample_data_2d, tmp_path):
        import os
        original_dir = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            X_orig, X_gen = sample_data_2d
            plot_scatter_comparison(
                X_orig, X_gen,
                save_name='test_scatter.png',
                show=False
            )
            saved_file = tmp_path / 'results' / 'figures' / 'test_scatter.png'
            assert saved_file.exists()
        finally:
            os.chdir(original_dir)
    
    def test_plot_scatter_comparison_3d(self):
        np.random.seed(42)
        X_orig = np.random.randn(100, 3)
        X_gen = np.random.randn(100, 3)
        fig = plot_scatter_comparison(X_orig, X_gen, show=False)
        assert isinstance(fig, plt.Figure)
        # Should create 3x3 grid
        assert len(fig.axes) == 9


class TestPlotAllVisualizations:
    """Test the convenience function that creates all plots."""
    
    def test_plot_all_visualizations_creates_three_figures(
        self, sample_data_2d, training_history, tmp_path
    ):
        import os
        original_dir = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            X_orig, X_gen = sample_data_2d
            figures = plot_all_visualizations(
                X_orig, X_gen, training_history,
                show=False
            )
            
            assert isinstance(figures, dict)
            assert 'loss' in figures
            assert 'density' in figures
            assert 'scatter' in figures
            assert isinstance(figures['loss'], plt.Figure)
            assert isinstance(figures['density'], plt.Figure)
            assert isinstance(figures['scatter'], plt.Figure)
        finally:
            os.chdir(original_dir)
    
    def test_plot_all_visualizations_with_prefix(
        self, sample_data_2d, training_history, tmp_path
    ):
        import os
        original_dir = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            X_orig, X_gen = sample_data_2d
            plot_all_visualizations(
                X_orig, X_gen, training_history,
                prefix='exp1',
                show=False
            )
            
            # Check that files exist with prefix
            results_dir = tmp_path / 'results' / 'figures'
            assert (results_dir / 'exp1_training_loss.png').exists()
            assert (results_dir / 'exp1_density_comparison.png').exists()
            assert (results_dir / 'exp1_scatter_comparison.png').exists()
        finally:
            os.chdir(original_dir)
    
    def test_plot_all_visualizations_with_generator_name(
        self, sample_data_2d, training_history, tmp_path
    ):
        import os
        original_dir = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            X_orig, X_gen = sample_data_2d
            figures = plot_all_visualizations(
                X_orig, X_gen, training_history,
                generator_name='Uniform(0, 2)',
                show=False
            )
            
            assert isinstance(figures, dict)
            assert len(figures) == 3
        finally:
            os.chdir(original_dir)
    
    def test_plot_all_visualizations_1d(self, sample_data_1d, training_history, tmp_path):
        import os
        original_dir = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            X_orig, X_gen = sample_data_1d
            figures = plot_all_visualizations(
                X_orig, X_gen, training_history,
                show=False
            )
            
            assert isinstance(figures, dict)
            assert len(figures) == 3
        finally:
            os.chdir(original_dir)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_small_sample_size(self, training_history):
        X_orig = np.random.randn(5, 2)
        X_gen = np.random.randn(5, 2)
        
        fig_density = plot_density_comparison(X_orig, X_gen, show=False)
        fig_scatter = plot_scatter_comparison(X_orig, X_gen, show=False)
        
        assert isinstance(fig_density, plt.Figure)
        assert isinstance(fig_scatter, plt.Figure)
    
    def test_different_sample_sizes(self, training_history):
        X_orig = np.random.randn(100, 2)
        X_gen = np.random.randn(50, 2)
        
        fig_density = plot_density_comparison(X_orig, X_gen, show=False)
        fig_scatter = plot_scatter_comparison(X_orig, X_gen, show=False)
        
        assert isinstance(fig_density, plt.Figure)
        assert isinstance(fig_scatter, plt.Figure)
    
    def test_very_high_dimensional(self):
        X_orig = np.random.randn(100, 10)
        X_gen = np.random.randn(100, 10)
        
        # Should handle gracefully
        fig_density = plot_density_comparison(X_orig, X_gen, show=False)
        fig_scatter = plot_scatter_comparison(X_orig, X_gen, show=False)
        
        assert isinstance(fig_density, plt.Figure)
        assert isinstance(fig_scatter, plt.Figure)
    
    def test_empty_generator_name(self, sample_data_2d):
        X_orig, X_gen = sample_data_2d
        fig = plot_density_comparison(X_orig, X_gen, generator_name="", show=False)
        assert isinstance(fig, plt.Figure)
    
    def test_single_sample(self):
        X_orig = np.random.randn(1, 2)
        X_gen = np.random.randn(1, 2)
        
        # Should not crash
        fig = plot_scatter_comparison(X_orig, X_gen, show=False)
        assert isinstance(fig, plt.Figure)


class TestActualTrainingVisualization:
    """Test visualization with actual trained model (integration test)."""
    
    def test_full_pipeline_visualization(self):
        """Test complete pipeline: generate data, train model, visualize.
        
        This is an integration test that creates actual meaningful
        visualizations from a trained model. Figures are saved to
        results/figures/ directory.
        """
        # Generate original data
        generator = NormalGenerator(loc=0, scale=1)
        X_original_np = generator.generate(500)
        
        # Ensure it's 2D
        if X_original_np.ndim == 1:
            X_original_np = X_original_np.reshape(-1, 1)
        
        X_original = torch.from_numpy(X_original_np).float()
        
        # Train and generate samples (small epoch for speed)
        model, history, X_generated = train_and_generate(
            X_original,
            num_samples=500,
            num_epochs=10,
            batch_size=64,
            learning_rate=1e-3,
            patience=5,
            verbose=False
        )
        
        # Create all visualizations (saved to results/figures/)
        figures = plot_all_visualizations(
            X_original,
            X_generated,
            history,
            generator_name=generator.name,
            prefix="actual_test",
            show=False
        )
        
        # Verify figures were created
        assert isinstance(figures, dict)
        assert len(figures) == 3
        
        # Verify files were saved
        results_dir = Path('results/figures')
        assert (results_dir / 'actual_test_training_loss.png').exists()
        assert (results_dir / 'actual_test_density_comparison.png').exists()
        assert (results_dir / 'actual_test_scatter_comparison.png').exists()
        
        # Verify training history has reasonable values
        assert len(history['loss']) > 0
        assert len(history['term1']) > 0
        assert len(history['term2']) > 0
        
        # Verify loss decreased (at least somewhat)
        assert history['loss'][-1] <= history['loss'][0]
        
        # Verify data shapes match
        assert X_original.shape[0] == 500
        assert X_generated.shape[0] == 500
        assert X_original.shape[1] == X_generated.shape[1]
        
        print("\n" + "="*70)
        print("Integration test passed! Actual visualizations saved to:")
        print(f"{results_dir.absolute()}")
        print(f"Data shape: {X_original.shape}")
        print("="*70)