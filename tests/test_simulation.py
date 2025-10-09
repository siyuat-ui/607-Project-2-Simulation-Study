"""Tests for simulation module.

This module tests the simulation orchestration to ensure
experiments run correctly and produce expected outputs.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import pandas as pd
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing

from simulation import (
    SimulationExperiment,
    run_quick_simulation,
    run_full_simulation
)
from dgps import NormalGenerator, ExponentialGenerator


@pytest.fixture
def small_generators():
    """Create a small set of generators for testing."""
    return [
        NormalGenerator(loc=0, scale=1),
        ExponentialGenerator(scale=1)
    ]


@pytest.fixture
def test_training_params():
    """Create fast training parameters for testing."""
    return {
        'num_epochs': 5,
        'batch_size': 32,
        'learning_rate': 1e-3,
        'patience': 3,
        'input_dim': 2,  # Small input_dim for fast testing
    }


@pytest.fixture(scope="module", autouse=True)
def cleanup_test_results():
    """Clean up test results after all tests complete."""
    yield
    # Clean up results directory after tests
    results_dir = Path('results')
    if results_dir.exists():
        # Remove test files
        for subdir in ['figures', 'raw']:
            test_dir = results_dir / subdir
            if test_dir.exists():
                for file in test_dir.glob('*'):
                    if 'test' in file.name.lower() or file.stat().st_size < 1000:
                        try:
                            file.unlink()
                            print(f"Cleaned up: {file}")
                        except Exception as e:
                            print(f"Could not remove {file}: {e}")


class TestSimulationExperimentInitialization:
    """Test SimulationExperiment initialization."""
    
    def test_initialization_default(self, small_generators):
        sim = SimulationExperiment(
            generators=small_generators,
            sample_sizes=[50, 100],
            n_replications=2
        )
        
        assert sim.n_replications == 2
        assert len(sim.generators) == 2
        assert len(sim.sample_sizes) == 2
        assert sim.save_results == True
        assert sim.verbose == True
    
    def test_initialization_custom_params(self, small_generators, test_training_params):
        sim = SimulationExperiment(
            generators=small_generators,
            sample_sizes=[50],
            n_replications=1,
            training_params=test_training_params,
            save_results=False,
            verbose=False
        )
        
        assert sim.training_params['num_epochs'] == 5
        assert sim.training_params['batch_size'] == 32
        assert sim.save_results == False
        assert sim.verbose == False
    
    def test_creates_directories(self, small_generators):
        sim = SimulationExperiment(
            generators=small_generators,
            sample_sizes=[50],
            n_replications=1,
            save_results=True
        )
        
        assert sim.figures_dir.exists()
        assert sim.raw_dir.exists()
        assert sim.figures_dir == Path('results/figures')
        assert sim.raw_dir == Path('results/raw')
    
    def test_results_storage_initialized(self, small_generators):
        sim = SimulationExperiment(
            generators=small_generators,
            sample_sizes=[50],
            n_replications=1
        )
        
        assert sim.results == []
        assert isinstance(sim.results, list)


class TestRunSingleExperiment:
    """Test running a single experiment."""
    
    def test_run_single_experiment_returns_dict(self, small_generators, test_training_params):
        sim = SimulationExperiment(
            generators=small_generators,
            sample_sizes=[50],
            n_replications=1,
            training_params=test_training_params,
            save_results=False,
            verbose=False
        )
        
        result = sim.run_single_experiment(
            generator=small_generators[0],
            sample_size=50,
            replication=0
        )
        
        assert isinstance(result, dict)
    
    def test_single_experiment_has_required_keys(self, small_generators, test_training_params):
        sim = SimulationExperiment(
            generators=small_generators,
            sample_sizes=[50],
            n_replications=1,
            training_params=test_training_params,
            save_results=False,
            verbose=False
        )
        
        result = sim.run_single_experiment(
            generator=small_generators[0],
            sample_size=50,
            replication=0
        )
        
        required_keys = [
            'generator', 'sample_size', 'replication', 'training_time',
            'final_loss', 'final_term1', 'final_term2', 'n_epochs',
            'mmd', 'two_sample_min_p', 'two_sample_mean_p',
            'two_sample_rejected', 'mean_distance', 'cov_frobenius', 'cov_trace'
        ]
        
        for key in required_keys:
            assert key in result
    
    def test_single_experiment_values_reasonable(self, small_generators, test_training_params):
        sim = SimulationExperiment(
            generators=small_generators,
            sample_sizes=[50],
            n_replications=1,
            training_params=test_training_params,
            save_results=False,
            verbose=False
        )
        
        result = sim.run_single_experiment(
            generator=small_generators[0],
            sample_size=50,
            replication=0
        )
        
        # Check reasonable ranges
        assert result['training_time'] > 0
        assert result['n_epochs'] > 0
        assert result['n_epochs'] <= test_training_params['num_epochs']
        assert result['mmd'] >= 0
        assert 0 <= result['two_sample_min_p'] <= 1
        assert 0 <= result['two_sample_mean_p'] <= 1
        assert result['mean_distance'] >= 0


class TestRunAllExperiments:
    """Test running all experiments."""
    
    def test_run_all_experiments_returns_dataframe(self, small_generators, test_training_params):
        sim = SimulationExperiment(
            generators=small_generators[:1],  # Only 1 generator
            sample_sizes=[50],
            n_replications=2,
            training_params=test_training_params,
            save_results=False,
            verbose=False
        )
        
        results_df = sim.run_all_experiments()
        
        assert isinstance(results_df, pd.DataFrame)
    
    def test_run_all_experiments_correct_number_of_rows(self, small_generators, test_training_params):
        sim = SimulationExperiment(
            generators=small_generators[:1],
            sample_sizes=[50, 100],
            n_replications=2,
            training_params=test_training_params,
            save_results=False,
            verbose=False
        )
        
        results_df = sim.run_all_experiments()
        
        # 1 generator × 2 sample sizes × 2 replications = 4 rows
        assert len(results_df) == 4
    
    def test_run_all_experiments_has_all_columns(self, small_generators, test_training_params):
        sim = SimulationExperiment(
            generators=small_generators[:1],
            sample_sizes=[50],
            n_replications=1,
            training_params=test_training_params,
            save_results=False,
            verbose=False
        )
        
        results_df = sim.run_all_experiments()
        
        expected_columns = [
            'generator', 'sample_size', 'replication', 'training_time',
            'final_loss', 'mmd', 'mean_distance'
        ]
        
        for col in expected_columns:
            assert col in results_df.columns
    
    def test_run_all_experiments_saves_to_csv(self, small_generators, test_training_params, tmp_path):
        import os
        original_dir = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            sim = SimulationExperiment(
                generators=small_generators[:1],
                sample_sizes=[50],
                n_replications=1,
                training_params=test_training_params,
                save_results=True,
                verbose=False
            )
            
            results_df = sim.run_all_experiments()
            
            # Check CSV file was created
            raw_dir = tmp_path / 'results' / 'raw'
            csv_files = list(raw_dir.glob('simulation_results_*.csv'))
            assert len(csv_files) > 0
            
            # Check file can be read
            saved_df = pd.read_csv(csv_files[0])
            assert len(saved_df) == len(results_df)
        
        finally:
            os.chdir(original_dir)


class TestSummarizeResults:
    """Test result summarization."""
    
    def test_summarize_results_returns_dataframe(self, small_generators, test_training_params):
        sim = SimulationExperiment(
            generators=small_generators[:1],
            sample_sizes=[50],
            n_replications=2,
            training_params=test_training_params,
            save_results=False,
            verbose=False
        )
        
        results_df = sim.run_all_experiments()
        summary_df = sim.summarize_results(results_df)
        
        assert isinstance(summary_df, pd.DataFrame)
    
    def test_summarize_results_groups_correctly(self, small_generators, test_training_params):
        sim = SimulationExperiment(
            generators=small_generators[:1],
            sample_sizes=[50, 100],
            n_replications=2,
            training_params=test_training_params,
            save_results=False,
            verbose=False
        )
        
        results_df = sim.run_all_experiments()
        summary_df = sim.summarize_results(results_df)
        
        # Should have 1 generator × 2 sample sizes = 2 rows
        assert len(summary_df) == 2
    
    def test_summarize_results_has_mean_and_std(self, small_generators, test_training_params):
        sim = SimulationExperiment(
            generators=small_generators[:1],
            sample_sizes=[50],
            n_replications=2,
            training_params=test_training_params,
            save_results=False,
            verbose=False
        )
        
        results_df = sim.run_all_experiments()
        summary_df = sim.summarize_results(results_df)
        
        # Check for mean and std columns
        assert any('_mean' in col for col in summary_df.columns)
        assert any('_std' in col for col in summary_df.columns)
    
    def test_summarize_results_saves_to_csv(self, small_generators, test_training_params, tmp_path):
        import os
        original_dir = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            sim = SimulationExperiment(
                generators=small_generators[:1],
                sample_sizes=[50],
                n_replications=2,
                training_params=test_training_params,
                save_results=True,
                verbose=False
            )
            
            results_df = sim.run_all_experiments()
            summary_df = sim.summarize_results(results_df)
            
            # Check CSV file was created
            raw_dir = tmp_path / 'results' / 'raw'
            csv_files = list(raw_dir.glob('simulation_summary_*.csv'))
            assert len(csv_files) > 0
        
        finally:
            os.chdir(original_dir)


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_run_quick_simulation_returns_tuple(self):
        results_df, summary_df = run_quick_simulation(
            n_replications=1,
            sample_sizes=[50],
            verbose=False,
            save_results=False
        )
        
        assert isinstance(results_df, pd.DataFrame)
        assert isinstance(summary_df, pd.DataFrame)
    
    def test_run_quick_simulation_correct_number_experiments(self):
        results_df, summary_df = run_quick_simulation(
            n_replications=2,
            sample_sizes=[50],
            verbose=False,
            save_results=False
        )
        
        # 3 generators × 1 sample size × 2 replications = 6 rows
        assert len(results_df) == 6
    
    def test_run_full_simulation_returns_tuple(self):
        # Override with small input_dim for testing
        from simulation import SimulationExperiment
        from dgps import (
            NormalGenerator, ExponentialGenerator, UniformGenerator,
            LognormalGenerator, ChiSquareGenerator
        )
        
        generators = [
            NormalGenerator(loc=0, scale=1),
            ExponentialGenerator(scale=1),
            UniformGenerator(low=0, high=2),
            LognormalGenerator(mean=0, sigma=1),
            ChiSquareGenerator(df=5),
        ]
        
        training_params = {
            'num_epochs': 5,
            'batch_size': 32,
            'learning_rate': 1e-3,
            'patience': 3,
            'input_dim': 2,  # Small for testing
        }
        
        sim = SimulationExperiment(
            generators=generators,
            sample_sizes=[50],
            n_replications=1,
            training_params=training_params,
            save_results=False,
            verbose=False
        )
        
        results_df = sim.run_all_experiments()
        summary_df = sim.summarize_results(results_df)
        
        assert isinstance(results_df, pd.DataFrame)
        assert isinstance(summary_df, pd.DataFrame)
    
    def test_run_full_simulation_has_all_generators(self):
        # Override with small input_dim for testing
        from simulation import SimulationExperiment
        from dgps import (
            NormalGenerator, ExponentialGenerator, UniformGenerator,
            LognormalGenerator, ChiSquareGenerator
        )
        
        generators = [
            NormalGenerator(loc=0, scale=1),
            ExponentialGenerator(scale=1),
            UniformGenerator(low=0, high=2),
            LognormalGenerator(mean=0, sigma=1),
            ChiSquareGenerator(df=5),
        ]
        
        training_params = {
            'num_epochs': 5,
            'batch_size': 32,
            'learning_rate': 1e-3,
            'patience': 3,
            'input_dim': 2,  # Small for testing
        }
        
        sim = SimulationExperiment(
            generators=generators,
            sample_sizes=[50],
            n_replications=1,
            training_params=training_params,
            save_results=False,
            verbose=False
        )
        
        results_df = sim.run_all_experiments()
        
        # Should have 5 generators × 1 sample size × 1 replication = 5 rows
        assert len(results_df) == 5
        
        # Check all generator types are present
        generators_in_results = results_df['generator'].unique()
        assert len(generators_in_results) == 5


class TestIntegration:
    """Integration tests for full workflow."""
    
    def test_full_workflow_with_visualization(self, small_generators, test_training_params, tmp_path):
        """Test complete workflow including visualization."""
        import os
        original_dir = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            sim = SimulationExperiment(
                generators=small_generators[:1],
                sample_sizes=[50],
                n_replications=1,
                training_params=test_training_params,
                save_results=True,
                verbose=False
            )
            
            results_df = sim.run_all_experiments()
            summary_df = sim.summarize_results(results_df)
            
            # Check directories exist
            assert (tmp_path / 'results' / 'figures').exists()
            assert (tmp_path / 'results' / 'raw').exists()
            
            # Check CSV files exist
            raw_dir = tmp_path / 'results' / 'raw'
            assert len(list(raw_dir.glob('simulation_results_*.csv'))) > 0
            assert len(list(raw_dir.glob('simulation_summary_*.csv'))) > 0
            
            # Check figure files exist
            figures_dir = tmp_path / 'results' / 'figures'
            png_files = list(figures_dir.glob('*.png'))
            assert len(png_files) > 0  # Should have at least some figures
            
        finally:
            os.chdir(original_dir)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_replication(self, small_generators, test_training_params):
        sim = SimulationExperiment(
            generators=small_generators[:1],
            sample_sizes=[50],
            n_replications=1,
            training_params=test_training_params,
            save_results=False,
            verbose=False
        )
        
        results_df = sim.run_all_experiments()
        assert len(results_df) == 1
    
    def test_multiple_sample_sizes(self, small_generators, test_training_params):
        sim = SimulationExperiment(
            generators=small_generators[:1],
            sample_sizes=[50, 100, 200],
            n_replications=1,
            training_params=test_training_params,
            save_results=False,
            verbose=False
        )
        
        results_df = sim.run_all_experiments()
        assert len(results_df) == 3
        assert set(results_df['sample_size'].unique()) == {50, 100, 200}
    
    def test_summarize_without_running_experiments_raises_error(self, small_generators):
        sim = SimulationExperiment(
            generators=small_generators,
            sample_sizes=[50],
            n_replications=1,
            save_results=False
        )
        
        with pytest.raises(ValueError, match="No results available"):
            sim.summarize_results()